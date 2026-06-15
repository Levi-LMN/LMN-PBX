# services/ari_agent.py
"""
ARI-based agent service using Azure Voice Live API for ultra-low latency voice.

Architecture:
  Caller → Asterisk → ARI ExternalMedia → RTP UDP socket (this service)
         ↕ PCM audio (μ-law 8kHz)
  This service ↔ Azure Voice Live WebSocket (PCM16 16kHz)
  ↓ audio back → RTP → Asterisk → Caller

Escalation flow:
  1. Caller says transfer keyword  OR  AI says it will transfer
  2. _handle_escalation() fires
  3. ExternalMedia channel removed from bridge (stops AI audio)
  4. channel.continueInDialplan() sends caller back into FreePBX dialplan
     at the department extension (e.g. 1000 → Sales ring group)
  5. Softphone rings

Why Azure Voice Live API?
  - WebSocket endpoint: wss://<resource>.services.ai.azure.com/voice-live/realtime
  - Auth header: "api-key: <key>"
  - session.update shape is FLAT (no nested audio.input/audio.output blocks)
  - input_audio_sampling_rate: 16000 (we upsample from Asterisk's 8kHz)
  - voice is an object: {"name": "en-US-AvaNeural", "type": "azure-standard"}
  - turn_detection type: "azure_semantic_vad" — best for telephony
  - Built-in noise suppression, echo cancellation, filler-word removal
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
AZURE_VOICE_LIVE_HOST_SUFFIX = os.getenv(
    "AZURE_VOICE_LIVE_HOST_SUFFIX", "services.ai.azure.com"
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AZURE_VOICE_LIVE_API_VERSION = "2025-10-01"
AZURE_VOICE_LIVE_MODEL       = "gpt-realtime"

def _build_azure_ws_url(resource_name: str, model: str) -> str:
    return (
        f"wss://{resource_name}.{AZURE_VOICE_LIVE_HOST_SUFFIX}"
        f"/voice-live/realtime"
        f"?api-version={AZURE_VOICE_LIVE_API_VERSION}"
        f"&model={model}"
    )


def _resolve_hostname(hostname: str) -> bool:
    """Return True if hostname resolves via DNS."""
    try:
        socket.getaddrinfo(hostname, None)
        return True
    except socket.gaierror:
        return False


# Audio parameters
ASTERISK_SAMPLE_RATE = 8000   # Asterisk sends μ-law 8 kHz
AZURE_SAMPLE_RATE    = 16000  # We upsample to 16 kHz for Azure
AZURE_OUTPUT_RATE    = 24000  # Azure TTS always outputs PCM16 @ 24 kHz
RTP_HEADER_SIZE      = 12     # Standard RTP header bytes

# RTP port range for ExternalMedia channels (one pair per concurrent call)
RTP_LISTEN_HOST = "0.0.0.0"
RTP_PORT_START  = 20000
RTP_PORT_END    = 20100


def _ws_header_kwargs(headers: dict) -> dict:
    """
    Compatible extra-headers kwarg for websockets < 14 and >= 14.
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
# ARIAgent — top-level service, one per process
# ─────────────────────────────────────────────────────────────────────────────

class ARIAgent:
    """
    Manages the ARI WebSocket connection and spawns one
    AzureVoiceLiveCallSession per inbound call.
    """

    def __init__(self, app_config, flask_app=None):
        self.config    = app_config
        self.flask_app = flask_app
        self.running   = False
        self.active_calls: dict[str, "AzureVoiceLiveCallSession"] = {}
        self.total_calls = 0

        # ARI connection settings
        self.ari_url      = os.getenv("ARI_URL",      "http://localhost:8088/ari")
        self.ari_base     = os.getenv("ARI_BASE",     "http://localhost:8088")
        self.ari_username = os.getenv("ARI_USERNAME", "asterisk")
        self.ari_password = os.getenv("ARI_PASSWORD", "your_ari_password")
        self.ari_app      = os.getenv("ARI_APP",      "ai-agent")

        # Azure Voice Live settings
        self.azure_resource   = os.getenv("AZURE_VOICE_LIVE_RESOURCE", "")
        self.azure_api_key    = os.getenv("AZURE_SPEECH_KEY", "")
        self.azure_voice_name = os.getenv("AZURE_VOICE_NAME", "en-US-AvaNeural")
        self.azure_voice_type = os.getenv("AZURE_VOICE_TYPE", "azure-standard")
        self.system_prompt    = os.getenv("DEFAULT_SYSTEM_PROMPT") or self._default_prompt()

        # Port allocator — one UDP port pair per concurrent call
        self._next_rtp_port = RTP_PORT_START
        self._port_lock     = asyncio.Lock()

        self.ari_client = None
        self._config_ok = bool(self.azure_resource and self.azure_api_key)

        logger.info("ARIAgent (Azure Voice Live) initialised")

    def _default_prompt(self):
        return (
            "You are Ari, a friendly and knowledgeable phone assistant for Jubilee Insurance Kenya. "
            "You have access to detailed information about our products: motor, medical, life, last expense, "
            "home, travel insurance, claims processes, and payment methods. "
            "RULES: "
            "(1) Respond in 2-3 natural sentences — enough to be helpful, short enough for a phone call. "
            "(2) Never read out lists or bullet points — weave information conversationally. "
            "(3) Be warm but efficient, like a knowledgeable human agent. "
            "(4) If asked about products broadly, mention the relevant category and invite them to ask more. "
            "(5) Use specific details from what you know — KES amounts, timelines, M-PESA paybill numbers — when relevant. "
            "(6) Never say you are an AI. "
            "(7) Never use filler like Certainly, Of course, or Great question. "
            "(8) ONLY transfer to a human if the caller EXPLICITLY says they want to speak to a human, agent, or manager. "
            "(9) If you cannot help or the caller requests a human, say: 'Let me transfer you to one of our agents right away.' "
            "GOOD example: We cover your home structure and contents against fire, theft, and natural disasters, "
            "up to KES 10 million. Premiums are based on your property value — do you own or rent?"
        )

    async def _alloc_rtp_port(self) -> int:
        async with self._port_lock:
            port = self._next_rtp_port
            self._next_rtp_port = port + 2
            if self._next_rtp_port > RTP_PORT_END:
                self._next_rtp_port = RTP_PORT_START
            return port

    # ── Lifecycle ─────────────────────────────────────────────────────────────

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
            return
        if not self.azure_api_key:
            logger.error("❌ AZURE_SPEECH_KEY not set — cannot start")
            return

        _ws_url = _build_azure_ws_url(self.azure_resource, AZURE_VOICE_LIVE_MODEL)
        logger.info(f"   WS URL   : {_ws_url}")

        _azure_hostname = f"{self.azure_resource}.{AZURE_VOICE_LIVE_HOST_SUFFIX}"
        logger.info(f"   Checking DNS for {_azure_hostname} …")
        if not _resolve_hostname(_azure_hostname):
            logger.error(f"❌ Cannot resolve '{_azure_hostname}'")
            logger.error("   Check AZURE_VOICE_LIVE_RESOURCE spelling, DNS, and firewall.")
            self._config_ok = False

        try:
            logger.info(f"Connecting to ARI at {self.ari_base} …")
            self.ari_client = await aioari.connect(
                self.ari_base, self.ari_username, self.ari_password
            )
            logger.info("✅ ARI connected")

            self.ari_client.on_event("StasisStart",          self._handle_stasis_start)
            self.ari_client.on_event("StasisEnd",            self._handle_stasis_end)
            self.ari_client.on_event("ChannelHangupRequest", self._handle_hangup_request)

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
        channel_name = event.get("channel", {}).get("name", "")
        if channel_name.startswith("UnicastRTP/"):
            logger.debug("StasisStart for ExternalMedia channel — ignoring")
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
                "Azure Voice Live not reachable. See startup logs."
            )
            try:
                await channel.answer()
                await asyncio.sleep(0.5)
                await channel.hangup()
            except Exception:
                pass
            return

        rtp_port = await self._alloc_rtp_port()
        ws_url   = _build_azure_ws_url(self.azure_resource, AZURE_VOICE_LIVE_MODEL)

        # Enrich the system prompt with live KB content
        enriched_prompt = self.system_prompt + self._load_knowledge_context()

        session = AzureVoiceLiveCallSession(
            channel       = channel,
            ari_client    = self.ari_client,
            azure_api_key = self.azure_api_key,
            azure_ws_url  = ws_url,
            voice_name    = self.azure_voice_name,
            voice_type    = self.azure_voice_type,
            system_prompt = enriched_prompt,
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

    # ── Knowledge base loader ─────────────────────────────────────────────────

    def _load_knowledge_context(self) -> str:
        """Load active KB entries and return as a formatted string for the system prompt."""
        if not self.flask_app:
            return ""
        try:
            with self.flask_app.app_context():
                from models import KnowledgeBase, db
                entries = (
                    KnowledgeBase.query
                    .filter_by(is_active=True)
                    .order_by(KnowledgeBase.priority.desc())
                    .all()
                )
                if not entries:
                    return ""

                parts = ["\n\nKNOWLEDGE BASE — use this to answer callers accurately:"]
                for e in entries:
                    parts.append(f"\n[{e.category.upper()}] {e.title}:\n{e.content}")
                    e.increment_usage()
                db.session.commit()

                logger.info(f"📚 Loaded {len(entries)} knowledge base entries into session prompt")
                return "\n".join(parts)
        except Exception as exc:
            logger.error(f"KB load error: {exc}")
            return ""

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
            logger.error(f"DB log error: {e}")

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
# AzureVoiceLiveCallSession — one per inbound call
# ─────────────────────────────────────────────────────────────────────────────

class AzureVoiceLiveCallSession:
    """
    Manages a single call:
      - Asterisk RTP ↔ Azure Voice Live WebSocket (audio bridge)
      - Escalation detection on both caller and AI transcripts
      - Transfer via continueInDialplan → FreePBX softphone ring group
    """

    # Keywords in the CALLER's speech that trigger a transfer
    TRANSFER_KEYWORDS = {
        "speak to", "talk to", "human", "person", "agent",
        "representative", "manager", "supervisor", "someone else",
        "transfer", "escalate", "real person", "actual person",
    }

    # Phrases in the AI's speech that mean it has committed to a transfer
    AI_TRANSFER_PHRASES = [
        "transfer you",
        "transferring you",
        "connect you",
        "put you through",
        "one of our agents",
        "speak to an agent",
        "right away",            # e.g. "Let me transfer you to an agent right away"
    ]

    # RTP pacing constants — Asterisk needs exactly 160 μ-law bytes per packet
    ULAW_PACKET_BYTES = 160       # 20 ms @ 8 kHz, 1 byte/sample (μ-law)
    PACKET_INTERVAL_S = 0.020     # 20 ms between packets

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

        # State flags
        self.caller_hung_up = False
        self.escalated      = False
        self._closed        = False
        self._greeting_sent = False
        self._ai_transcript_buf = ""

        # RTP socket state
        self._udp_sock:           socket.socket | None = None
        self._asterisk_rtp_addr:  tuple | None         = None
        self._rtp_seq  = 0
        self._rtp_ts   = 0
        self._rtp_ssrc = 0xDEADBEEF

        # Azure WebSocket
        self._azure_ws = None

        # Bridge and ExternalMedia channel IDs (needed for cleanup/transfer)
        self._bridge_id      = None
        self._ext_channel_id = None

        # Audio queues
        self._audio_to_azure:    asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        self._audio_to_asterisk: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)

        # Stateful sample-rate converters (audioop.ratecv)
        self._ratecv_state_up   = None  # 8 kHz → 16 kHz
        self._ratecv_state_down = None  # 24 kHz → 8 kHz

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self):
        """Answer call → set up bridge → connect Azure WS → pump audio."""
        try:
            # 1. Answer
            await self.channel.answer()
            logger.info(f"✅ [{self.channel_id[:12]}] Call answered")
            await asyncio.sleep(0.3)

            # 2. Create mixing bridge
            bridge = await self.ari_client.bridges.create(type="mixing")
            self._bridge_id = bridge.id
            await bridge.addChannel(channel=self.channel_id)
            logger.info(f"🌉 [{self.channel_id[:12]}] Bridge created: {self._bridge_id}")

            # 3. Bind UDP socket BEFORE creating ExternalMedia
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind((RTP_LISTEN_HOST, self.rtp_port))
            self._udp_sock.setblocking(False)
            logger.info(f"🔌 [{self.channel_id[:12]}] UDP RTP listening on port {self.rtp_port}")

            # 4. Create ExternalMedia channel — Asterisk streams caller audio to our port
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

            # 5. Connect to Azure Voice Live
            await self._connect_azure()

            # 6. Run all pumps concurrently until call ends or escalation fires
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
        """Retry addChannel because ExternalMedia may not be ready immediately."""
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
        Connect to Azure Voice Live API and configure the session.

        The session.update payload for Azure Voice Live is FLAT — all properties
        sit directly under "session", not nested inside audio.input/audio.output
        blocks like the old OpenAI Realtime Beta format.
        """
        try:
            from urllib.parse import urlparse
            _host = urlparse(self.azure_ws_url).hostname or ""
        except Exception:
            _host = ""

        if _host and not _resolve_hostname(_host):
            raise ConnectionError(
                f"Cannot resolve Azure Voice Live hostname '{_host}'. "
                "Check AZURE_VOICE_LIVE_RESOURCE and DNS/firewall."
            )

        self._azure_ws = await websockets.connect(
            self.azure_ws_url,
            ping_interval = 20,
            ping_timeout  = 30,
            **_ws_header_kwargs({"api-key": self.azure_api_key}),
        )
        logger.info(f"🔗 [{self.channel_id[:12]}] Azure Voice Live WS connected")
        logger.info(f"   URL: {self.azure_ws_url}")

        await self._azure_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": (
                    self.system_prompt
                    + " CRITICAL: Always respond in English only."
                ),
                "modalities": ["text", "audio"],

                # Azure TTS voice
                "voice": {
                    "name": self.voice_name,
                    "type": self.voice_type,
                },

                # We upsample μ-law 8 kHz → PCM16 16 kHz before sending
                "input_audio_sampling_rate": AZURE_SAMPLE_RATE,

                # Server-side noise suppression (removes background noise)
                "input_audio_noise_reduction": {
                    "type": "azure_deep_noise_suppression"
                },

                # Server-side echo cancellation (prevents AI audio loopback)
                "input_audio_echo_cancellation": {
                    "type": "server_echo_cancellation"
                },

                # Transcription — used for escalation detection and DB logging
                "input_audio_transcription": {
                    "model":    "azure-speech",
                    "language": "en-US",
                },

                # Semantic VAD — understands natural speech pauses better than
                # volume-based server_vad; ideal for telephony
                "turn_detection": {
                    "type":                "azure_semantic_vad",
                    "threshold":           0.4,
                    "silence_duration_ms": 400,
                    "prefix_padding_ms":   200,
                    "remove_filler_words": True,
                    "interrupt_response":  True,
                    "create_response":     True,
                },
            },
        }))
        logger.info(f"⚙️  [{self.channel_id[:12]}] Azure Voice Live session configured")
        logger.info(f"   Voice : {self.voice_name} ({self.voice_type})")
        logger.info(f"   VAD   : azure_semantic_vad | noise suppression ON | echo cancel ON")

    # ── RTP receive (Asterisk → queue) ────────────────────────────────────────

    async def _recv_rtp_loop(self):
        """
        Read μ-law RTP packets from Asterisk, convert to PCM16 16 kHz,
        enqueue for Azure.  Uses stateful ratecv for smooth resampling.
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

                # Upsample 8 kHz → 16 kHz (stateful for smooth resampling)
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
        """Pull PCM16 chunks from the queue and stream to Azure as base64."""
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

        Events handled:
          response.audio.delta
              TTS audio from Azure — PCM16 24 kHz, convert to μ-law 8 kHz for Asterisk.

          conversation.item.input_audio_transcription.completed
              Caller's speech transcript — check for transfer keywords.

          response.audio_transcript.done / response.output_audio_transcript.done
              AI's full spoken response — check if AI announced a transfer.

          session.updated
              Trigger greeting on first fire.

          response.done, session.created, error
              Logged only.
        """
        while not self.caller_hung_up and not self._closed:
            try:
                if not self._azure_ws:
                    await asyncio.sleep(0.1)
                    continue

                raw   = await asyncio.wait_for(self._azure_ws.recv(), timeout=1.0)
                event = json.loads(raw)
                etype = event.get("type", "")

                # ── TTS audio from Azure → Asterisk ───────────────────────
                if etype == "response.audio.delta":
                    delta_b64 = event.get("delta", "")
                    if not delta_b64:
                        continue
                    pcm_out = base64.b64decode(delta_b64)

                    # Azure TTS always outputs PCM16 @ 24 kHz
                    # Downsample 24 kHz → 8 kHz for Asterisk
                    pcm8, self._ratecv_state_down = audioop.ratecv(
                        pcm_out, 2, 1,
                        AZURE_OUTPUT_RATE, ASTERISK_SAMPLE_RATE,
                        self._ratecv_state_down
                    )
                    ulaw = audioop.lin2ulaw(pcm8, 2)
                    await self._audio_to_asterisk.put(ulaw)

                # ── Caller speech transcript ───────────────────────────────
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    logger.info(f"👤 [{self.channel_id[:12]}] Caller: {transcript}")
                    self._db_log_transcript("caller", transcript, 1.0)

                    # Check if caller explicitly asked for a human
                    if not self.escalated and self._detect_transfer_intent(transcript):
                        logger.info(f"🔀 [{self.channel_id[:12]}] Caller requested transfer")
                        await self._handle_escalation(transcript)

                # ── AI response transcript (streamed delta) ────────────────
                elif etype in ("response.audio_transcript.delta",
                               "response.output_audio_transcript.delta"):
                    delta = event.get("delta", "")
                    if delta:
                        self._ai_transcript_buf += delta

                # ── AI response transcript complete ────────────────────────
                elif etype in ("response.audio_transcript.done",
                               "response.output_audio_transcript.done"):
                    full = self._ai_transcript_buf.strip()
                    self._ai_transcript_buf = ""

                    if full:
                        logger.info(f"🤖 [{self.channel_id[:12]}] AI said: {full}")
                        self._db_log_transcript("agent", full, 1.0)

                        # If the AI announced a transfer, execute it after a short delay
                        # so the caller hears the complete sentence before being transferred
                        if not self.escalated and any(
                            phrase in full.lower() for phrase in self.AI_TRANSFER_PHRASES
                        ):
                            logger.info(
                                f"🔀 [{self.channel_id[:12]}] "
                                "AI announced transfer — scheduling in 2.5s"
                            )
                            asyncio.create_task(
                                self._delayed_escalation(full, delay=2.5)
                            )

                # ── Response finished ──────────────────────────────────────
                elif etype == "response.done":
                    usage = event.get("response", {}).get("usage", {})
                    logger.debug(
                        f"🤖 [{self.channel_id[:12]}] Response done "
                        f"(tokens: {usage.get('total_tokens', '?')})"
                    )

                # ── Session created ────────────────────────────────────────
                elif etype == "session.created":
                    logger.info(
                        f"✅ [{self.channel_id[:12]}] Azure session created: "
                        f"{event.get('session', {}).get('id', '')}"
                    )

                # ── Session updated → send greeting ────────────────────────
                elif etype == "session.updated":
                    logger.info(f"⚙️  [{self.channel_id[:12]}] Session updated by server")
                    if not self._greeting_sent:
                        self._greeting_sent = True
                        await asyncio.sleep(0.5)
                        await self._azure_ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "instructions": (
                                    "Say ONLY: 'Thank you for calling Jubilee Insurance, "
                                    "how can I help?' — nothing else."
                                ),
                            },
                        }))
                        logger.info(f"👋 [{self.channel_id[:12]}] Greeting triggered")

                # ── API errors ────────────────────────────────────────────
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

    async def _send_rtp_loop(self):
        """
        Pull μ-law audio from the queue, re-packetise into strict 160-byte
        (20 ms) RTP frames, and pace at 20 ms intervals to Asterisk.

        Azure returns audio in large bursts (100-500 ms at a time).  Sending
        those bursts as one oversized packet causes Asterisk's jitter buffer to
        discard or distort them.  We re-chunk into exactly 160-byte packets.
        """
        loop      = asyncio.get_running_loop()
        buf       = bytearray()
        next_send = loop.time()

        while not self.caller_hung_up and not self._closed:
            try:
                chunk = await asyncio.wait_for(
                    self._audio_to_asterisk.get(), timeout=0.5
                )
                buf.extend(chunk)
            except asyncio.TimeoutError:
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

            # Emit as many complete 160-byte packets as we have data for
            while len(buf) >= self.ULAW_PACKET_BYTES:
                if not self._asterisk_rtp_addr or not self._udp_sock:
                    break

                payload = bytes(buf[:self.ULAW_PACKET_BYTES])
                del buf[:self.ULAW_PACKET_BYTES]

                rtp_pkt       = self._build_rtp_packet(payload)
                self._rtp_ts += self.ULAW_PACKET_BYTES
                self._rtp_seq = (self._rtp_seq + 1) & 0xFFFF

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
        """Build a minimal RTP/AVP header (RFC 3550) with μ-law payload (PT=0)."""
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
        """Return True if the caller's text contains an explicit transfer request."""
        lower = text.lower()
        return any(kw in lower for kw in self.TRANSFER_KEYWORDS)

    async def _delayed_escalation(self, transcript: str, delay: float = 2.5):
        """
        Wait for the AI to finish speaking its transfer announcement,
        then execute the actual transfer.
        """
        await asyncio.sleep(delay)
        if not self._closed and not self.escalated:
            await self._handle_escalation(transcript)

    async def _handle_escalation(self, transcript: str):
        """
        Transfer the caller to the appropriate department softphone.

        Steps:
          1. Classify intent from transcript to pick the right department
          2. Remove ExternalMedia from bridge (stops AI audio immediately)
          3. Close Azure WS and UDP socket
          4. Call continueInDialplan → FreePBX routes to dept extension
             (e.g. extension 1000 → Sales ring group → softphone rings)
        """
        if self.escalated:
            return  # Prevent double-transfer

        intent = self._classify_intent(transcript)
        dept   = self._get_department_for_intent(intent)

        dept_name = dept.name      if dept else "Support"
        dept_ext  = dept.extension if dept else "1005"

        logger.info(
            f"🔀 [{self.channel_id[:12]}] "
            f"Escalating → {dept_name} (ext {dept_ext}, intent: {intent})"
        )

        self.escalated = True

        # Step 1: Stop audio pumps by marking closed (they check this flag)
        self._closed = True

        # Step 2: Remove ExternalMedia from bridge so AI audio stops immediately
        if self._bridge_id and self._ext_channel_id:
            try:
                bridge = await self.ari_client.bridges.get(bridgeId=self._bridge_id)
                await bridge.removeChannel(channel=self._ext_channel_id)
                logger.info(f"🔇 [{self.channel_id[:12]}] ExternalMedia removed from bridge")
            except Exception as e:
                logger.debug(f"removeChannel error (non-fatal): {e}")

        # Step 3: Close Azure WS and UDP socket
        if self._azure_ws:
            try:
                await self._azure_ws.close()
            except Exception:
                pass
            self._azure_ws = None

        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None

        # Step 4: Hang up ExternalMedia channel
        if self._ext_channel_id:
            try:
                await self.ari_client.channels.hangup(channelId=self._ext_channel_id)
            except Exception:
                pass

        # Step 5: Destroy bridge (caller channel exits bridge, stays alive)
        if self._bridge_id:
            try:
                await self.ari_client.bridges.destroy(bridgeId=self._bridge_id)
            except Exception:
                pass

        # Step 6: Send caller into FreePBX dialplan at the department extension
        # from-internal is FreePBX's standard context for internal extension dialling.
        # Your ring groups / queues / softphones must be reachable at dept_ext in this context.
        transfer_contexts = ["from-internal", "default"]
        transferred = False

        for ctx in transfer_contexts:
            try:
                await self.channel.continueInDialplan(
                    context   = ctx,
                    extension = dept_ext,
                    priority  = 1,
                )
                logger.info(
                    f"✅ [{self.channel_id[:12]}] Transfer sent → "
                    f"{dept_name} ext {dept_ext} (context: {ctx})"
                )
                transferred = True
                break
            except Exception as e:
                logger.warning(f"continueInDialplan failed for context '{ctx}': {e}")

        if not transferred:
            logger.error(
                f"❌ [{self.channel_id[:12]}] All transfer attempts failed for ext {dept_ext}. "
                f"Check that extension {dept_ext} exists in FreePBX "
                f"and is reachable from context 'from-internal'."
            )
            # Last resort: hang up cleanly rather than leave caller in silence
            try:
                await self.channel.hangup()
            except Exception:
                pass

    def _classify_intent(self, text: str) -> str:
        """
        Keyword-based intent classification.
        Used to pick which department to transfer to.
        """
        lower = text.lower()
        if any(w in lower for w in ["buy", "quote", "new policy", "purchase", "sign up"]):
            return "sales"
        if any(w in lower for w in ["claim", "accident", "damage", "report", "incident"]):
            return "claims"
        if any(w in lower for w in ["bill", "payment", "pay", "invoice", "mpesa", "premium"]):
            return "billing"
        return "support"

    def _get_department_for_intent(self, intent_type: str):
        """
        Look up the Department for a given intent.
        Tries RoutingRule first, then Department name, then highest-priority dept.
        Returns None if DB is unavailable.
        """
        if not self.flask_app:
            return None
        try:
            with self.flask_app.app_context():
                from models import Department, RoutingRule

                # 1. Try routing rules (allows admin to configure custom mappings)
                rule = (
                    RoutingRule.query
                    .filter_by(intent_type=intent_type, is_active=True)
                    .order_by(RoutingRule.priority.desc())
                    .first()
                )
                if rule and rule.department and rule.department.is_active:
                    logger.info(
                        f"🗂️  Routing rule matched: {intent_type} → "
                        f"{rule.department.name} (ext {rule.department.extension})"
                    )
                    return rule.department

                # 2. Fall back to department name matching
                name_map = {
                    "sales":   "Sales",
                    "claims":  "Claims",
                    "billing": "Billing",
                    "support": "Support",
                }
                dept_name = name_map.get(intent_type)
                if dept_name:
                    dept = Department.query.filter_by(
                        name=dept_name, is_active=True
                    ).first()
                    if dept:
                        logger.info(
                            f"🗂️  Department name matched: {dept_name} "
                            f"(ext {dept.extension})"
                        )
                        return dept

                # 3. Fall back to highest-priority active department
                dept = (
                    Department.query
                    .filter_by(is_active=True)
                    .order_by(Department.priority.desc())
                    .first()
                )
                if dept:
                    logger.info(
                        f"🗂️  Fallback department: {dept.name} (ext {dept.extension})"
                    )
                return dept

        except Exception as e:
            logger.error(f"Department lookup error: {e}")
            return None

    # ── DB transcript logging ─────────────────────────────────────────────────

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

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def close(self):
        """
        Clean up all resources for this session.
        Called both on normal call end and after escalation.
        After a successful escalation, the caller channel is left alive
        (it's being handled by FreePBX), so we skip hanging it up.
        """
        if self._closed and not self.escalated:
            # Already cleaned up (escalation handles its own teardown)
            # Only skip if we truly already ran close() non-escalation path
            pass

        # Close Azure WS
        if self._azure_ws:
            try:
                await self._azure_ws.close()
            except Exception:
                pass
            self._azure_ws = None

        # Close UDP socket
        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None

        # Hang up ExternalMedia channel
        if self._ext_channel_id:
            try:
                await self.ari_client.channels.hangup(channelId=self._ext_channel_id)
            except Exception:
                pass
            self._ext_channel_id = None

        # Destroy bridge
        if self._bridge_id:
            try:
                await self.ari_client.bridges.destroy(bridgeId=self._bridge_id)
            except Exception:
                pass
            self._bridge_id = None

        # Hang up caller only if we're not escalating (escalation keeps caller alive)
        if not self.escalated and not self.caller_hung_up:
            try:
                await self.channel.hangup()
            except Exception:
                pass

        self._closed = True
        logger.info(f"🔒 [{self.channel_id[:12]}] Session closed")


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat stubs (no longer used)
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