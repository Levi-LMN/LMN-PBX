# services/ari_agent.py
"""
ARI-based agent service using Azure Voice Live API for ultra-low latency voice.

Architecture:
  Caller → Asterisk → ARI ExternalMedia → RTP UDP socket (this service)
         ↕ PCM audio (μ-law 8kHz)
  This service ↔ Azure Voice Live WebSocket (PCM16 8kHz or 16kHz)
  ↓ audio back → RTP → Asterisk → Caller

Why Azure Voice Live API?
  Old (OpenAI Realtime): stream RTP → OpenAI WS (PCM16 24kHz required) → stream back
  New (Azure Voice Live): stream RTP → Azure WS (PCM16 8/16kHz, G.711 supported)
                          → stream back with Azure neural voices + semantic VAD

Key differences from OpenAI Realtime:
  - WebSocket endpoint: wss://<resource>.services.ai.azure.com/voice-live/realtime?api-version=2025-10-01
  - Auth header: "api-key: <key>"  OR  "Authorization: Bearer <token>"
  - session.update shape is flatter (no nested audio.input/audio.output blocks)
  - input_audio_sampling_rate: 16000 or 24000 (we upsample from 8kHz to 16kHz)
  - voice is an object: {"name": "en-US-AvaNeural", "type": "azure-standard"}
  - turn_detection type: "azure_semantic_vad" (superior to server_vad for telephony)
  - Extra features: noise suppression, echo cancellation, filler-word removal
  - Audio events: response.audio.delta (same as OpenAI) → PCM16 at output rate
  - Transcript event: conversation.item.input_audio_transcription.completed (same key)
"""

import asyncio
import aiohttp
from aiohttp.web_exceptions import HTTPUnprocessableEntity
import aioari
import os
import socket
import struct
import audioop
import base64
import inspect
import json
import logging
import websockets
from datetime import datetime
from flask import Flask

# Allow overriding the Azure host suffix via env var so that older
# CognitiveServices resources (*.cognitiveservices.azure.com) work without
# code changes.  AI-Foundry resources use the default below.
# Set AZURE_VOICE_LIVE_HOST_SUFFIX=cognitiveservices.azure.com in .env for
# legacy resources.
AZURE_VOICE_LIVE_HOST_SUFFIX = os.getenv(
    "AZURE_VOICE_LIVE_HOST_SUFFIX", "services.ai.azure.com"
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Azure Voice Live API endpoint template.
# Substitute your AI Foundry resource name via env var AZURE_VOICE_LIVE_RESOURCE.
# For older (CognitiveServices) resources use the .cognitiveservices.azure.com host.
AZURE_VOICE_LIVE_API_VERSION = "2025-10-01"
AZURE_VOICE_LIVE_MODEL       = "gpt-realtime"   # or "gpt-realtime-mini"

def _build_azure_ws_url(resource_name: str, model: str) -> str:
    return (
        f"wss://{resource_name}.{AZURE_VOICE_LIVE_HOST_SUFFIX}"
        f"/voice-live/realtime"
        f"?api-version={AZURE_VOICE_LIVE_API_VERSION}"
        f"&model={model}"
    )


def _resolve_hostname(hostname: str) -> bool:
    """
    Return True if *hostname* resolves via DNS, False otherwise.
    Used to give a clear error before attempting a WebSocket connect.
    """
    try:
        socket.getaddrinfo(hostname, None)
        return True
    except socket.gaierror:
        return False

# Audio parameters
ASTERISK_SAMPLE_RATE  = 8000    # Asterisk sends μ-law 8 kHz
AZURE_SAMPLE_RATE     = 16000   # Azure Voice Live minimum; 8000 also works but 16k is cleaner
RTP_PACKET_MS         = 20      # 20 ms RTP packets from Asterisk (160 samples @ 8 kHz)
RTP_HEADER_SIZE       = 12      # Standard RTP header

# RTP port range for ExternalMedia channels
RTP_LISTEN_HOST = "0.0.0.0"
RTP_PORT_START  = 20000
RTP_PORT_END    = 20100


def _ws_header_kwargs(headers: dict) -> dict:
    """
    Return the correct kwarg for passing extra headers to websockets.connect(),
    compatible with both websockets < 14 (extra_headers) and >= 14 (additional_headers).
    """
    try:
        params = inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):
        params = inspect.signature(websockets.connect.__init__).parameters

    if "additional_headers" in params:
        return {"additional_headers": headers}
    return {"extra_headers": headers}


def _ws_is_open(ws) -> bool:
    if ws is None:
        return False
    if hasattr(ws, "closed"):
        return not ws.closed
    if hasattr(ws, "state"):
        return ws.state == websockets.State.OPEN
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ARIAgent — top-level service
# ─────────────────────────────────────────────────────────────────────────────

class ARIAgent:
    """Manages the ARI connection and spawns an AzureVoiceLiveCallSession per call."""

    def __init__(self, app_config, flask_app=None):
        self.config    = app_config
        self.flask_app = flask_app
        self.running   = False
        self.active_calls: dict[str, "AzureVoiceLiveCallSession"] = {}
        self.total_calls = 0

        # ARI
        self.ari_url      = os.getenv("ARI_URL",      "http://localhost:8088/ari")
        self.ari_base     = os.getenv("ARI_BASE",     "http://localhost:8088")
        self.ari_username = os.getenv("ARI_USERNAME", "asterisk")
        self.ari_password = os.getenv("ARI_PASSWORD", "your_ari_password")
        self.ari_app      = os.getenv("ARI_APP",      "ai-agent")

        # Azure Voice Live API
        self.azure_resource  = os.getenv("AZURE_VOICE_LIVE_RESOURCE", "")
        self.azure_api_key   = os.getenv("AZURE_SPEECH_KEY", "")
        self.azure_voice_name= os.getenv("AZURE_VOICE_NAME", "en-US-AvaNeural")
        self.azure_voice_type= os.getenv("AZURE_VOICE_TYPE", "azure-standard")
        self.system_prompt   = os.getenv("DEFAULT_SYSTEM_PROMPT") or self._default_prompt()

        # Port allocator for ExternalMedia RTP sockets
        self._next_rtp_port = RTP_PORT_START
        self._port_lock     = asyncio.Lock()

        self.ari_client = None

        # Pre-validate Azure config so we can reject calls cleanly instead of
        # surfacing a cryptic DNS error mid-call.
        self._config_ok = bool(self.azure_resource and self.azure_api_key)

        logger.info("ARIAgent (Azure Voice Live) initialised")

    def _default_prompt(self):
        return (
            "You are a phone assistant for Jubilee Insurance. "
            "CRITICAL RULES you must follow on every single response: "
            "(1) MAXIMUM 1-2 SHORT SENTENCES — never more, non-negotiable. "
            "(2) NEVER list things. If asked about products, name one category and ask which interests them. "
            "(3) NEVER open with Certainly, Of course, Great question, or similar filler. "
            "(4) Be direct and conversational, like a human on a phone call. "
            "(5) Never say you are an AI. "
            "(6) If the caller wants a human, say you will transfer them now. "
            "GOOD example: We cover life, health, motor and home — which interests you? "
            "BAD example: Certainly! I would be happy to provide an overview of our wide range..."
        )

    async def _alloc_rtp_port(self) -> int:
        async with self._port_lock:
            port = self._next_rtp_port
            self._next_rtp_port = port + 2
            if self._next_rtp_port > RTP_PORT_END:
                self._next_rtp_port = RTP_PORT_START
            return port

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        self.running = True
        logger.info("=" * 60)
        logger.info("🤖 ARI Agent starting (Azure Voice Live API)")
        logger.info(f"   Resource : {self.azure_resource}")
        logger.info(f"   Voice    : {self.azure_voice_name} ({self.azure_voice_type})")
        logger.info(f"   Model    : {AZURE_VOICE_LIVE_MODEL}")
        logger.info("=" * 60)

        if not self.azure_resource:
            logger.error("❌ AZURE_VOICE_LIVE_RESOURCE not set — cannot start")
            logger.error("   Set AZURE_VOICE_LIVE_RESOURCE=<your-resource-name> in .env")
            return
        if not self.azure_api_key:
            logger.error("❌ AZURE_SPEECH_KEY not set — cannot start")
            logger.error("   Set AZURE_SPEECH_KEY=<your-api-key> in .env")
            return

        # Build and log the full WS URL so it can be verified at startup.
        _ws_url = _build_azure_ws_url(self.azure_resource, AZURE_VOICE_LIVE_MODEL)
        logger.info(f"   WS URL   : {_ws_url}")

        # DNS pre-check — fail fast with a clear message rather than surfacing
        # a cryptic socket.gaierror when the first call arrives.
        _azure_hostname = f"{self.azure_resource}.{AZURE_VOICE_LIVE_HOST_SUFFIX}"
        logger.info(f"   Checking DNS for {_azure_hostname} …")
        if not _resolve_hostname(_azure_hostname):
            logger.error(f"❌ Cannot resolve '{_azure_hostname}'")
            logger.error("   Possible causes:")
            logger.error("     1. AZURE_VOICE_LIVE_RESOURCE is misspelled "
                         f"(currently: '{self.azure_resource}')")
            logger.error("     2. This server has no outbound internet DNS "
                         "(check /etc/resolv.conf and firewall rules)")
            logger.error("     3. The resource lives under cognitiveservices.azure.com "
                         "— set AZURE_VOICE_LIVE_HOST_SUFFIX=cognitiveservices.azure.com "
                         "in .env")
            logger.error("   Agent will start but all calls will be rejected until "
                         "connectivity is restored.")
            self._config_ok = False
            # Don't return — keep the ARI listener running so calls get a
            # polite rejection rather than ringing forever.

        try:
            logger.info(f"Connecting to ARI at {self.ari_base} …")
            self.ari_client = await aioari.connect(
                self.ari_base, self.ari_username, self.ari_password
            )
            logger.info("✅ ARI connected")

            self.ari_client.on_event("StasisStart",           self._handle_stasis_start)
            self.ari_client.on_event("StasisEnd",             self._handle_stasis_end)
            self.ari_client.on_event("ChannelHangupRequest",  self._handle_hangup_request)

            logger.info("🎙️  READY — waiting for calls")
            logger.info("=" * 60)
            await self.ari_client.run(apps=self.ari_app)

        except Exception as e:
            logger.error(f"❌ ARI error: {e}")
            self.running = False

    async def stop(self):
        self.running = False
        for session in list(self.active_calls.values()):
            await session.close()
        if self.ari_client:
            try:
                await self.ari_client.close()
            except Exception:
                pass
        logger.info("ARIAgent stopped")

    # ── ARI event handlers ────────────────────────────────────────────────────

    def _handle_stasis_start(self, event):
        channel      = event.get("channel", {})
        channel_name = channel.get("name", "")
        if channel_name.startswith("UnicastRTP/"):
            logger.debug(f"StasisStart for ExternalMedia channel — ignoring")
            return
        asyncio.create_task(self._process_call(event))

    def _handle_stasis_end(self, event):
        channel_id = event.get("channel", {}).get("id")
        if channel_id in self.active_calls:
            logger.info(f"📴 Channel {channel_id[:12]} left Stasis")
            self.active_calls[channel_id].caller_hung_up = True

    def _handle_hangup_request(self, event):
        channel_id = event.get("channel", {}).get("id")
        if channel_id in self.active_calls:
            self.active_calls[channel_id].caller_hung_up = True

    async def _process_call(self, event):
        channel_id = event.get("channel", {}).get("id")
        if not channel_id:
            return
        try:
            channel = await self.ari_client.channels.get(channelId=channel_id)
            await self._handle_call(channel)
        except Exception as e:
            logger.error(f"❌ Call processing error: {e}")

    async def _handle_call(self, channel):
        caller_number = channel.json.get("caller", {}).get("number", "Unknown")
        logger.info(f"📞 Incoming call from {caller_number}")

        if not self._config_ok:
            logger.error(
                f"❌ Rejecting call from {caller_number} — "
                "Azure Voice Live is not reachable (check DNS / credentials). "
                "See startup logs for details."
            )
            try:
                await channel.answer()
                # Give Asterisk a moment then hang up so the caller hears
                # a busy/disconnect tone rather than ringing indefinitely.
                await asyncio.sleep(0.5)
                await channel.hangup()
            except Exception:
                pass
            return

        rtp_port = await self._alloc_rtp_port()

        ws_url = _build_azure_ws_url(self.azure_resource, AZURE_VOICE_LIVE_MODEL)

        session = AzureVoiceLiveCallSession(
            channel       = channel,
            ari_client    = self.ari_client,
            azure_api_key = self.azure_api_key,
            azure_ws_url  = ws_url,
            voice_name    = self.azure_voice_name,
            voice_type    = self.azure_voice_type,
            system_prompt = self.system_prompt,
            rtp_port      = rtp_port,
            ari_url       = self.ari_url,
            ari_username  = self.ari_username,
            ari_password  = self.ari_password,
            flask_app     = self.flask_app,
        )

        self.active_calls[channel.id] = session
        self.total_calls += 1
        self._db_log_call_start(channel.id, caller_number)

        try:
            await session.run()
        except Exception as e:
            logger.error(f"❌ Session error: {e}")
            self._db_log_call_error(channel.id, str(e))
        finally:
            self.active_calls.pop(channel.id, None)
            await session.close()
            self._db_log_call_end(session)

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _db_log_call_start(self, call_id, caller_number):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                db.session.add(Call(
                    call_id       = call_id,
                    caller_number = caller_number,
                    status        = "active",
                    started_at    = datetime.utcnow(),
                ))
                db.session.commit()
        except Exception as e:
            logger.error(f"DB log start error: {e}")

    def _db_log_call_error(self, call_id, error_msg):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                c = Call.query.filter_by(call_id=call_id).first()
                if c:
                    c.status   = "error"
                    c.ended_at = datetime.utcnow()
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB log error error: {e}")

    def _db_log_call_end(self, session: "AzureVoiceLiveCallSession"):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                c = Call.query.filter_by(call_id=session.channel_id).first()
                if c:
                    c.status    = "escalated" if session.escalated else "completed"
                    c.escalated = session.escalated
                    c.ended_at  = datetime.utcnow()
                    if c.started_at:
                        c.duration_seconds = int(
                            (c.ended_at - c.started_at).total_seconds()
                        )
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB log end error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# AzureVoiceLiveCallSession
# ─────────────────────────────────────────────────────────────────────────────

class AzureVoiceLiveCallSession:
    """
    One session per inbound call using Azure Voice Live API.

    Audio pipeline:
      Asterisk (μ-law 8kHz RTP) → ulaw2lin → ratecv 8→16kHz → base64
        → Azure Voice Live WS (input_audio_buffer.append)
      Azure Voice Live WS (response.audio.delta, PCM16 16kHz) → base64 decode
        → ratecv 16→8kHz → lin2ulaw → RTP → Asterisk

    Key Azure Voice Live differences vs OpenAI Realtime:
      - Auth: "api-key" header (or Bearer token)
      - No "OpenAI-Beta" header needed
      - session.update is flatter: instructions, voice{}, turn_detection{},
        input_audio_sampling_rate, input_audio_noise_reduction, etc.
        at session root — NOT nested under audio.input / audio.output
      - turn_detection.type = "azure_semantic_vad" for best telephony results
      - voice = {"name": "en-US-AvaNeural", "type": "azure-standard"}
      - Audio delta event: "response.audio.delta"  (not response.output_audio.delta)
      - All other events (session.created, response.done, error,
        conversation.item.input_audio_transcription.completed) are identical
    """

    TRANSFER_KEYWORDS = {
        "speak", "talk", "human", "person", "agent",
        "representative", "manager", "supervisor", "someone",
        "transfer", "escalate", "real person",
    }

    def __init__(self, *, channel, ari_client, azure_api_key, azure_ws_url,
                 voice_name, voice_type, system_prompt, rtp_port,
                 ari_url, ari_username, ari_password, flask_app):
        self.channel        = channel
        self.channel_id     = channel.id
        self.ari_client     = ari_client
        self.azure_api_key  = azure_api_key
        self.azure_ws_url   = azure_ws_url
        self.voice_name     = voice_name
        self.voice_type     = voice_type
        self.system_prompt  = system_prompt
        self.rtp_port       = rtp_port
        self.ari_url        = ari_url
        self.ari_username   = ari_username
        self.ari_password   = ari_password
        self.flask_app      = flask_app

        self.caller_hung_up = False
        self.escalated      = False
        self._closed        = False

        # RTP state
        self._udp_sock: socket.socket | None = None
        self._asterisk_rtp_addr: tuple | None = None
        self._rtp_seq  = 0
        self._rtp_ts   = 0
        self._rtp_ssrc = 0xDEADBEEF

        # Azure Voice Live WS
        self._azure_ws = None

        # Bridge & external media channel IDs
        self._bridge_id      = None
        self._ext_channel_id = None

        # Audio queues
        self._audio_to_azure:    asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        self._audio_to_asterisk: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)

        # For ratecv state (stateful sample-rate conversion)
        self._ratecv_state_up   = None   # 8→16 kHz converter state
        self._ratecv_state_down = None   # 16→8 kHz converter state

        # Logged once when first TTS audio is received from Azure
        self._first_audio_received = False

    # ── main run ──────────────────────────────────────────────────────────────

    async def run(self):
        """Answer → bridge → ExternalMedia → Azure Voice Live WS → pump."""
        try:
            # 1. Answer
            await self.channel.answer()
            logger.info(f"✅ [{self.channel_id[:12]}] Call answered")
            await asyncio.sleep(0.3)

            # 2. Create bridge
            bridge = await self.ari_client.bridges.create(type="mixing")
            self._bridge_id = bridge.id
            await bridge.addChannel(channel=self.channel_id)
            logger.info(f"🌉 [{self.channel_id[:12]}] Bridge created: {self._bridge_id}")

            # 3. Open UDP socket BEFORE creating ExternalMedia
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind((RTP_LISTEN_HOST, self.rtp_port))
            self._udp_sock.setblocking(False)
            logger.info(f"🔌 [{self.channel_id[:12]}] UDP RTP listening on port {self.rtp_port}")

            # 4. Create ExternalMedia channel
            ext_channel = await self.ari_client.channels.externalMedia(
                app             = os.getenv("ARI_APP", "ai-agent"),
                external_host   = f"127.0.0.1:{self.rtp_port}",
                format          = "ulaw",
                encapsulation   = "rtp",
                transport       = "udp",
                connection_type = "client",
                direction       = "both",
            )
            self._ext_channel_id = ext_channel.id
            logger.info(f"📡 [{self.channel_id[:12]}] ExternalMedia channel: {self._ext_channel_id}")

            await self._add_channel_to_bridge_with_retry(bridge, self._ext_channel_id)

            # 5. Connect to Azure Voice Live API
            await self._connect_azure()

            # 6. Run all pumps concurrently until call ends
            await asyncio.gather(
                self._recv_rtp_loop(),
                self._send_rtp_loop(),
                self._azure_recv_loop(),
                self._azure_send_loop(),
                return_exceptions=True,
            )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"❌ [{self.channel_id[:12]}] Session run error: {e}", exc_info=True)
        finally:
            await self.close()

    async def _add_channel_to_bridge_with_retry(self, bridge, channel_id,
                                                  retries: int = 5,
                                                  delay: float = 0.2):
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                await bridge.addChannel(channel=channel_id)
                if attempt > 1:
                    logger.info(
                        f"🌉 [{self.channel_id[:12]}] Channel {channel_id} "
                        f"added to bridge on attempt {attempt}"
                    )
                return
            except HTTPUnprocessableEntity as e:
                last_err = e
                logger.debug(
                    f"[{self.channel_id[:12]}] addChannel attempt {attempt}/{retries} "
                    f"not ready (422) — retrying in {delay}s …"
                )
                await asyncio.sleep(delay)

        logger.error(
            f"❌ [{self.channel_id[:12]}] Could not add channel {channel_id} "
            f"to bridge after {retries} attempts"
        )
        raise last_err

    # ── Azure Voice Live WebSocket ─────────────────────────────────────────────

    async def _connect_azure(self):
        """
        Connect to Azure Voice Live API.

        Authentication: api-key header (simplest; use Bearer token for prod).
        The session.update shape for Voice Live API is FLAT — properties like
        `input_audio_sampling_rate`, `voice`, `turn_detection`, and
        `input_audio_noise_reduction` all sit directly under `session`, not
        nested inside audio.input / audio.output blocks (which is the old OpenAI
        Realtime Beta shape).
        """
        # Extract the hostname from the WS URL for pre-flight DNS check.
        # e.g. "wss://myresource.services.ai.azure.com/..." → "myresource.services.ai.azure.com"
        try:
            from urllib.parse import urlparse
            _parsed_host = urlparse(self.azure_ws_url).hostname or ""
        except Exception:
            _parsed_host = ""

        if _parsed_host and not _resolve_hostname(_parsed_host):
            raise ConnectionError(
                f"Cannot resolve Azure Voice Live hostname '{_parsed_host}'. "
                "Check AZURE_VOICE_LIVE_RESOURCE (currently "
                f"'{_parsed_host.split('.')[0]}') and outbound DNS/firewall. "
                "If this is a CognitiveServices resource, set "
                "AZURE_VOICE_LIVE_HOST_SUFFIX=cognitiveservices.azure.com in .env"
            )

        headers = {
            "api-key": self.azure_api_key,
        }
        self._azure_ws = await websockets.connect(
            self.azure_ws_url,
            ping_interval=20,
            ping_timeout=30,
            **_ws_header_kwargs(headers),
        )
        logger.info(f"🔗 [{self.channel_id[:12]}] Azure Voice Live WS connected")
        logger.info(f"   URL: {self.azure_ws_url}")

        # Configure session with Azure Voice Live API shape.
        #
        # Notable differences from OpenAI Realtime:
        #   - `voice` is an object, not a string
        #   - `input_audio_sampling_rate` replaces nested audio.input.format.rate
        #   - `turn_detection.type` = "azure_semantic_vad" for semantic EOU
        #     (handles natural pauses better than server_vad — ideal for telephony)
        #   - `input_audio_noise_reduction` enables server-side noise suppression
        #   - `input_audio_echo_cancellation` prevents the mic picking up AI voice
        #   - `modalities` uses ["text", "audio"] (same as OpenAI)
        await self._azure_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": self.system_prompt + " CRITICAL: You MUST respond in English only, regardless of any ambiguity in the caller's speech. Never switch to any other language.",
                "modalities":   ["text", "audio"],

                # ── Voice (Azure TTS) ──────────────────────────────────────
                "voice": {
                    "name": self.voice_name,   # e.g. "en-US-AvaNeural"
                    "type": self.voice_type,   # "azure-standard" or "azure-custom"
                },

                # ── Input audio ────────────────────────────────────────────
                # We upsample μ-law 8kHz → PCM16 16kHz before sending.
                # Azure Voice Live supports 16000 and 24000; 16kHz halves
                # bandwidth vs 24kHz while remaining well above telephony quality.
                "input_audio_sampling_rate": AZURE_SAMPLE_RATE,

                # Azure deep noise suppression — removes call background noise
                "input_audio_noise_reduction": {
                    "type": "azure_deep_noise_suppression"
                },

                # Server-side echo cancellation — prevents AI audio loopback
                "input_audio_echo_cancellation": {
                    "type": "server_echo_cancellation"
                },

                # ── Transcription (for DB logging & escalation detection) ──
                # en-KE = English as spoken in Kenya; improves recognition of
                # East African accents significantly over plain "en".
                "input_audio_transcription": {
                    "model": "azure-speech",
                    "language": "en-KE",
                },

                # ── Turn detection ─────────────────────────────────────────
                # "azure_semantic_vad" understands natural speech pauses and
                # filler words — much better than volume-based "server_vad"
                # for real phone calls. Works with all models (not just gpt-realtime).
                "turn_detection": {
                    "type":                "azure_semantic_vad",
                    "threshold":           0.4,    # lower = more sensitive to barge-in
                    "silence_duration_ms": 300,    # faster response after caller stops
                    "prefix_padding_ms":   200,
                    "remove_filler_words": True,   # ignore "umm", "uh", etc.
                    "interrupt_response":  True,   # allow caller to barge in mid-response
                    "create_response":     True,
                },
            },
        }))
        logger.info(f"⚙️  [{self.channel_id[:12]}] Azure Voice Live session configured")
        logger.info(f"   Voice : {self.voice_name} ({self.voice_type})")
        logger.info(f"   VAD   : azure_semantic_vad | noise suppression ON | echo cancel ON")

    # ── RTP receive (Asterisk → queue → Azure) ────────────────────────────────

    async def _recv_rtp_loop(self):
        """
        Read μ-law RTP from Asterisk, convert to PCM16 16kHz, enqueue for Azure.
        Uses stateful ratecv for smooth resampling across packet boundaries.
        """
        loop = asyncio.get_running_loop()
        while not self.caller_hung_up and not self._closed:
            try:
                data, addr = await loop.run_in_executor(
                    None, self._udp_sock.recvfrom, 4096
                )
                if not self._asterisk_rtp_addr:
                    self._asterisk_rtp_addr = addr
                    logger.info(
                        f"📻 [{self.channel_id[:12]}] "
                        f"Asterisk RTP source: {addr[0]}:{addr[1]}"
                    )

                if len(data) <= RTP_HEADER_SIZE:
                    continue

                ulaw_payload = data[RTP_HEADER_SIZE:]

                # μ-law → PCM16 @ 8 kHz
                pcm8 = audioop.ulaw2lin(ulaw_payload, 2)

                # Upsample 8 kHz → 16 kHz (stateful for smooth interpolation)
                pcm16, self._ratecv_state_up = audioop.ratecv(
                    pcm8, 2, 1,
                    ASTERISK_SAMPLE_RATE, AZURE_SAMPLE_RATE,
                    self._ratecv_state_up
                )

                await self._audio_to_azure.put(pcm16)

            except BlockingIOError:
                await asyncio.sleep(0.005)
            except Exception as e:
                if not self._closed:
                    logger.debug(f"RTP recv error: {e}")
                break

    # ── Azure send (queue → WS) ───────────────────────────────────────────────

    async def _azure_send_loop(self):
        """Pull PCM16 chunks from queue and send to Azure Voice Live as base64."""
        while not self.caller_hung_up and not self._closed:
            try:
                chunk = await asyncio.wait_for(
                    self._audio_to_azure.get(), timeout=0.5
                )
                if self._azure_ws and _ws_is_open(self._azure_ws):
                    await self._azure_ws.send(json.dumps({
                        "type":  "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not self._closed:
                    logger.debug(f"Azure send error: {e}")
                break

    # ── Azure receive (WS → queue / events) ──────────────────────────────────

    async def _azure_recv_loop(self):
        """
        Receive events from Azure Voice Live API.

        Key events:
          response.audio.delta
              PCM16 audio from Azure TTS — convert back to μ-law 8kHz for Asterisk.
              NOTE: Azure uses "response.audio.delta" (not "response.output_audio.delta"
              which was the OpenAI Realtime Beta name).

          conversation.item.input_audio_transcription.completed
              Caller speech transcript — same event name as OpenAI Realtime.
              Used for escalation detection and DB logging.

          session.created / session.updated
              Lifecycle events — logged only.

          response.done
              Response finished — log token usage.

          error
              API error — log and continue.
        """
        while not self.caller_hung_up and not self._closed:
            try:
                if not self._azure_ws:
                    await asyncio.sleep(0.1)
                    continue

                raw   = await asyncio.wait_for(self._azure_ws.recv(), timeout=1.0)
                event = json.loads(raw)
                etype = event.get("type", "")

                # ── TTS audio from Azure ───────────────────────────────────
                if etype == "response.audio.delta":
                    delta_b64 = event.get("delta", "")
                    if not delta_b64:
                        continue
                    pcm_out = base64.b64decode(delta_b64)

                    # Azure Voice Live TTS output is PCM16 at 24 kHz regardless
                    # of the input_audio_sampling_rate we requested.
                    # Downsample 24 kHz → 8 kHz for Asterisk μ-law.
                    # (If Azure ever returns 16 kHz the ratecv still works fine;
                    #  the only effect is a slight pitch shift.  To be safe we
                    #  always convert from 24 kHz since that is the documented
                    #  default output rate for gpt-realtime.)
                    AZURE_OUTPUT_RATE = 24000
                    pcm8, self._ratecv_state_down = audioop.ratecv(
                        pcm_out, 2, 1,
                        AZURE_OUTPUT_RATE, ASTERISK_SAMPLE_RATE,
                        self._ratecv_state_down
                    )

                    # PCM16 → μ-law
                    ulaw = audioop.lin2ulaw(pcm8, 2)
                    await self._audio_to_asterisk.put(ulaw)

                # ── Caller speech transcript ───────────────────────────────
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    logger.info(f"👤 [{self.channel_id[:12]}] Caller: {transcript}")
                    self._db_log_transcript("caller", transcript, 1.0)
                    if self._detect_transfer_intent(transcript):
                        logger.info(f"🔀 [{self.channel_id[:12]}] Transfer intent detected")
                        await self._handle_escalation(transcript)

                # ── AI response transcript (streamed) ──────────────────────
                elif etype in ("response.audio_transcript.delta",
                               "response.output_audio_transcript.delta"):
                    # Accumulate the AI's spoken text for logging
                    delta = event.get("delta", "")
                    if delta:
                        self._ai_transcript_buf = getattr(self, "_ai_transcript_buf", "") + delta

                elif etype in ("response.audio_transcript.done",
                               "response.output_audio_transcript.done"):
                    full = getattr(self, "_ai_transcript_buf", "").strip()
                    if full:
                        logger.info(f"🤖 [{self.channel_id[:12]}] AI said: {full}")
                        self._db_log_transcript("agent", full, 1.0)
                    self._ai_transcript_buf = ""

                # ── Response finished ──────────────────────────────────────
                elif etype == "response.done":
                    usage = event.get("response", {}).get("usage", {})
                    logger.debug(
                        f"🤖 [{self.channel_id[:12]}] Response done "
                        f"(tokens: {usage.get('total_tokens', '?')})"
                    )

                # ── Session lifecycle ──────────────────────────────────────
                elif etype == "session.created":
                    logger.info(
                        f"✅ [{self.channel_id[:12]}] Azure Voice Live session created: "
                        f"{event.get('session', {}).get('id', '')}"
                    )

                elif etype == "session.updated":
                    logger.info(f"⚙️  [{self.channel_id[:12]}] Session updated by server")
                    # Trigger the AI to speak a greeting immediately.
                    # Without this, Azure waits for the caller to speak first,
                    # leaving dead silence and confusing the caller.
                    if not getattr(self, "_greeting_sent", False):
                        self._greeting_sent = True
                        await self._azure_ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "instructions": (
                                    "Say ONLY: 'Thank you for calling Jubilee Insurance, how can I help?' "
                                    "Nothing else. One sentence maximum."
                                ),
                            },
                        }))
                        logger.info(f"👋 [{self.channel_id[:12]}] Greeting triggered")

                # ── Errors ────────────────────────────────────────────────
                elif etype == "error":
                    logger.error(
                        f"❌ Azure Voice Live error: {event.get('error', event)}"
                    )

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                if not self._closed:
                    logger.warning(f"⚠️  [{self.channel_id[:12]}] Azure WS closed")
                break
            except Exception as e:
                if not self._closed:
                    logger.debug(f"Azure recv error: {e}")
                break

    # ── RTP send (queue → Asterisk) ───────────────────────────────────────────

    # Asterisk expects exactly 160 μ-law samples (= 20 ms @ 8 kHz) per RTP
    # packet.  Azure Voice Live returns audio in large, variable-size bursts
    # (often 100–500 ms at once).  Sending those bursts as a single oversized
    # RTP packet causes Asterisk's jitter-buffer to discard or distort the
    # audio — the caller hears breaking / robot voice / silence.
    #
    # Fix: accumulate raw μ-law bytes in a local buffer and drain it in
    # exactly ULAW_PACKET_BYTES-sized chunks, pacing one packet every
    # PACKET_INTERVAL_S seconds so Asterisk's playout clock stays smooth.
    ULAW_PACKET_BYTES  = 160          # 20 ms @ 8 kHz, 1 byte/sample
    PACKET_INTERVAL_S  = 0.020        # 20 ms between packets

    async def _send_rtp_loop(self):
        """
        Pull μ-law audio from the queue, re-packetise into strict 160-byte
        (20 ms) RTP frames, and pace them at 20 ms intervals to Asterisk.
        """
        loop       = asyncio.get_running_loop()
        buf        = bytearray()          # accumulation buffer
        next_send  = loop.time()          # absolute send deadline

        while not self.caller_hung_up and not self._closed:
            # ── drain the queue into our buffer (non-blocking after first) ──
            try:
                chunk = await asyncio.wait_for(
                    self._audio_to_asterisk.get(), timeout=0.5
                )
                buf.extend(chunk)
            except asyncio.TimeoutError:
                # Nothing arrived — if we have a partial buffer keep waiting,
                # but don't spin; just loop back.
                continue
            except Exception as e:
                if not self._closed:
                    logger.debug(f"RTP send queue error: {e}")
                break

            # Drain any additional chunks already queued (burst draining)
            while True:
                try:
                    buf.extend(self._audio_to_asterisk.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # ── emit as many complete 160-byte packets as we can ────────────
            while len(buf) >= self.ULAW_PACKET_BYTES:
                if not self._asterisk_rtp_addr or not self._udp_sock:
                    break

                payload  = bytes(buf[:self.ULAW_PACKET_BYTES])
                del buf[:self.ULAW_PACKET_BYTES]

                rtp_pkt          = self._build_rtp_packet(payload)
                self._rtp_ts    += self.ULAW_PACKET_BYTES   # 1 sample = 1 byte for μ-law
                self._rtp_seq    = (self._rtp_seq + 1) & 0xFFFF

                # Pace: sleep until the next scheduled send time
                now = loop.time()
                if next_send > now:
                    await asyncio.sleep(next_send - now)
                next_send = max(loop.time(), next_send) + self.PACKET_INTERVAL_S

                try:
                    await loop.run_in_executor(
                        None,
                        self._udp_sock.sendto,
                        rtp_pkt,
                        self._asterisk_rtp_addr,
                    )
                except Exception as e:
                    if not self._closed:
                        logger.debug(f"RTP sendto error: {e}")
                    return

    def _build_rtp_packet(self, payload: bytes) -> bytes:
        """Minimal RTP/AVP header (RFC 3550) + μ-law payload (PT=0)."""
        header = struct.pack(
            "!BBHII",
            0x80,
            0x00,           # PT=0 = PCMU / μ-law
            self._rtp_seq,
            self._rtp_ts,
            self._rtp_ssrc,
        )
        return header + payload

    # ── Transfer / escalation ─────────────────────────────────────────────────

    def _detect_transfer_intent(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in self.TRANSFER_KEYWORDS)

    async def _handle_escalation(self, transcript: str):
        intent = self._classify_intent(transcript)
        dept   = self._get_department_for_intent(intent)

        if dept:
            logger.info(
                f"🔀 [{self.channel_id[:12]}] "
                f"Transferring to {dept.name} (ext {dept.extension})"
            )
            self.escalated = True
            try:
                await self.channel.continueInDialplan(
                    context   = "from-internal",
                    extension = dept.extension,
                    priority  = 1,
                )
            except Exception as e:
                logger.error(f"Transfer error: {e}")
        else:
            logger.warning(f"⚠️  [{self.channel_id[:12]}] No department found for intent '{intent}'")

    def _classify_intent(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ["buy", "quote", "new policy", "coverage"]):
            return "sales"
        if any(w in lower for w in ["claim", "accident", "damage"]):
            return "claims"
        if any(w in lower for w in ["bill", "payment", "pay", "invoice"]):
            return "billing"
        return "support"

    def _get_department_for_intent(self, intent_type: str):
        if not self.flask_app:
            return None
        try:
            with self.flask_app.app_context():
                from models import Department, RoutingRule
                rule = RoutingRule.query.filter_by(
                    intent_type=intent_type, is_active=True
                ).order_by(RoutingRule.priority.desc()).first()
                if rule and rule.department:
                    return rule.department
                name_map = {
                    "sales": "Sales", "claims": "Claims",
                    "billing": "Billing", "support": "Support",
                }
                if intent_type in name_map:
                    d = Department.query.filter_by(
                        name=name_map[intent_type], is_active=True
                    ).first()
                    if d:
                        return d
                return Department.query.filter_by(is_active=True).order_by(
                    Department.priority.desc()
                ).first()
        except Exception as e:
            logger.error(f"Dept lookup error: {e}")
            return None

    # ── DB transcript helper ──────────────────────────────────────────────────

    def _db_log_transcript(self, speaker: str, text: str, confidence: float):
        if not self.flask_app or not text:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call, CallTranscript
                call = Call.query.filter_by(call_id=self.channel_id).first()
                if call:
                    db.session.add(CallTranscript(
                        call_id    = call.id,
                        speaker    = speaker,
                        text       = text,
                        confidence = confidence,
                        timestamp  = datetime.utcnow(),
                    ))
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB transcript error: {e}")

    # ── cleanup ───────────────────────────────────────────────────────────────

    async def close(self):
        if self._closed:
            return
        self._closed = True

        if self._azure_ws:
            try:
                await self._azure_ws.close()
            except Exception:
                pass

        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass

        if self._ext_channel_id:
            try:
                await self.ari_client.channels.hangup(channelId=self._ext_channel_id)
            except Exception:
                pass

        if self._bridge_id:
            try:
                await self.ari_client.bridges.destroy(bridgeId=self._bridge_id)
            except Exception:
                pass

        if not self.escalated and not self.caller_hung_up:
            try:
                await self.channel.hangup()
            except Exception:
                pass

        logger.info(f"🔒 [{self.channel_id[:12]}] Session closed")


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat stubs
# ─────────────────────────────────────────────────────────────────────────────

class SoundCache:
    """No longer used — Azure Voice Live streams audio directly."""
    pass


class FileSystemAccess:
    """No longer used — no TTS files needed."""
    pass


class AzureSpeechTranscriber:
    """No longer used — transcription handled by Azure Voice Live semantic VAD."""
    pass


# Alias so any code importing RealtimeCallSession still works
RealtimeCallSession = AzureVoiceLiveCallSession