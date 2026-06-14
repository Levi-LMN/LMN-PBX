# services/realtime_agent.py
"""
Azure OpenAI Realtime API agent for low-latency voice calls via Asterisk ARI.

Architecture:
  Caller <──RTP──> Asterisk ExternalMedia <──UDP──> RTPBridge <──PCM──> WebSocket <──> Azure gpt-realtime

Why this is faster than the old pipeline:
  Old: record file → STT (Azure Speech) → chat completions (OpenAI) → TTS (Azure Speech) → file copy → play
       4-5 sequential HTTP calls ≈ 2-4 seconds per turn

  New: raw 8-kHz PCM in → gpt-realtime WebSocket → 24-kHz PCM out (resampled back to 8k for Asterisk)
       Single persistent WebSocket ≈ 300-800 ms per turn

GA model names (as shown in Azure AI Foundry):
    gpt-realtime          – full model
    gpt-realtime-mini     – cheaper/faster
    gpt-realtime-1.5      – latest version

Config keys (same .env as before, just update the deployment name):
    AZURE_OPENAI_ENDPOINT              – your resource endpoint
    AZURE_OPENAI_KEY                   – api key
    AZURE_OPENAI_REALTIME_DEPLOYMENT   – name you gave the deployment in Foundry
                                         (e.g. gpt-realtime or gpt-realtime-mini)
    AZURE_OPENAI_API_VERSION           – 2025-04-01-preview
"""

import asyncio
import aioari
import os
import socket
import struct
import logging
import json
import base64
import audioop          # stdlib: 8-kHz ↔ 24-kHz resampling (no extra deps)
import time
import websockets
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
ASTERISK_RATE   = 8_000    # Hz – Asterisk default RTP codec (PCMU/PCMA)
REALTIME_RATE   = 24_000   # Hz – Azure Realtime API native rate
FRAME_MS        = 20       # ms per RTP frame
ASTERISK_SAMPLES = ASTERISK_RATE * FRAME_MS // 1000   # 160 samples
REALTIME_SAMPLES = REALTIME_RATE * FRAME_MS // 1000   # 480 samples
BYTES_PER_SAMPLE = 2       # 16-bit PCM


# ---------------------------------------------------------------------------
# RTP bridge – UDP socket that talks to Asterisk ExternalMedia
# ---------------------------------------------------------------------------
class RTPBridge:
    """
    Bidirectional RTP bridge between Asterisk and Azure Realtime.

    Asterisk ExternalMedia opens a UDP port on this host; we bind a local
    port and get told the remote Asterisk port via the channel variable
    UNICASTRTP_LOCAL_ADDRESS / UNICASTRTP_LOCAL_PORT.
    """

    RTP_HEADER_SIZE = 12   # bytes (fixed header, no CSRC/extensions)

    def __init__(self):
        self.sock: socket.socket | None = None
        self.asterisk_addr: tuple | None = None   # (host, port)
        self.local_port: int = 0
        self._seq = 0
        self._timestamp = 0
        self._ssrc = int.from_bytes(os.urandom(4), 'big')

    def open(self):
        """Bind a UDP socket on an ephemeral port."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', 0))
        self.sock.setblocking(False)
        self.local_port = self.sock.getsockname()[1]
        logger.info(f"RTP bridge bound on UDP port {self.local_port}")

    def set_remote(self, host: str, port: int):
        self.asterisk_addr = (host, int(port))
        logger.info(f"RTP remote set to {host}:{port}")

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _make_rtp_header(self, payload_type=0) -> bytes:
        """Build a minimal RTP header (RFC 3550)."""
        self._seq = (self._seq + 1) & 0xFFFF
        self._timestamp = (self._timestamp + ASTERISK_SAMPLES) & 0xFFFFFFFF
        # V=2, P=0, X=0, CC=0, M=0, PT=payload_type
        header = struct.pack(
            '!BBHII',
            0x80,                  # V=2, P=0, X=0, CC=0
            payload_type & 0x7F,   # M=0, PT
            self._seq,
            self._timestamp,
            self._ssrc
        )
        return header

    def send_pcm(self, pcm_24k: bytes):
        """
        Resample 24-kHz PCM16 from Realtime → 8-kHz, wrap in RTP, send to Asterisk.
        """
        if not self.sock or not self.asterisk_addr:
            return

        # Downsample 24k → 8k  (factor 3)
        pcm_8k, _ = audioop.ratecv(
            pcm_24k, BYTES_PER_SAMPLE, 1,
            REALTIME_RATE, ASTERISK_RATE, None
        )

        # Asterisk expects µ-law (PCMU, payload type 0)
        ulaw = audioop.lin2ulaw(pcm_8k, BYTES_PER_SAMPLE)

        header = self._make_rtp_header(payload_type=0)
        try:
            self.sock.sendto(header + ulaw, self.asterisk_addr)
        except Exception as e:
            logger.debug(f"RTP send error: {e}")

    async def recv_pcm_24k(self) -> bytes | None:
        """
        Non-blocking receive from Asterisk RTP → upsample to 24-kHz PCM16.
        Returns None if no data is ready.
        """
        if not self.sock:
            return None

        loop = asyncio.get_event_loop()
        try:
            data = await loop.sock_recv(self.sock, 4096)
        except BlockingIOError:
            return None
        except Exception:
            return None

        if len(data) <= self.RTP_HEADER_SIZE:
            return None

        payload = data[self.RTP_HEADER_SIZE:]
        payload_type = data[1] & 0x7F

        # Detect µ-law (PT=0) vs linear (PT=11 or raw)
        if payload_type == 0:
            pcm_8k = audioop.ulaw2lin(payload, BYTES_PER_SAMPLE)
        else:
            pcm_8k = payload   # assume already linear PCM

        # Upsample 8k → 24k
        pcm_24k, _ = audioop.ratecv(
            pcm_8k, BYTES_PER_SAMPLE, 1,
            ASTERISK_RATE, REALTIME_RATE, None
        )
        return pcm_24k


# ---------------------------------------------------------------------------
# Azure Realtime WebSocket session
# ---------------------------------------------------------------------------
class RealtimeSession:
    """
    Manages a single Azure OpenAI Realtime WebSocket connection for one call.

    Event flow (simplified):
        session.created  → send session.update (set voice, instructions, VAD)
        input_audio_buffer.append  → we feed mic audio
        response.audio.delta       → we get playback audio chunks
        response.audio.done        → one turn complete
        response.done              → ready for next user turn
    """

    def __init__(self, endpoint: str, api_key: str, deployment: str,
                 api_version: str, system_prompt: str):
        # GA models (gpt-realtime, gpt-realtime-mini, gpt-realtime-1.5) use:
        #   wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>
        # Preview models used the older path with api-version + deployment params.
        # We always use the GA path; api_version is kept for the header only.
        base = endpoint.rstrip('/').replace('https://', 'wss://')
        self.uri = f"{base}/openai/v1/realtime?model={deployment}"
        self.deployment = deployment
        self.api_version = api_version
        self.headers = {
            "api-key": api_key,
            # NOTE: Do NOT send "OpenAI-Beta: realtime=v1" on the GA
            # /openai/v1/realtime endpoint — Azure's GA Realtime API
            # rejects the handshake with HTTP 400 if this header is
            # present. It was only required for the old preview path
            # (/openai/realtime?api-version=...).
        }
        self.system_prompt = system_prompt
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()   # outbound audio chunks
        self._transcript_log: list[dict] = []
        self._connected = False
        self._session_ready = asyncio.Event()

    async def connect(self):
        """Open WebSocket and wait for session to be ready."""
        logger.info(f"Connecting to Azure Realtime: {self.uri}")
        self.ws = await websockets.connect(
            self.uri,
            additional_headers=self.headers,
            ping_interval=20,
            ping_timeout=30,
        )
        self._connected = True
        logger.info("✅ Realtime WebSocket connected")

    async def configure(self):
        """Send session.update to configure voice, VAD, and instructions (GA schema)."""
        config = {
            "type": "session.update",
            "session": {
                # GA requires an explicit session "type". For speech-to-speech
                # this must be "realtime" (the other option is "transcription").
                "type": "realtime",
                "model": self.deployment,
                "instructions": self.system_prompt,
                "output_modalities": ["audio", "text"],
                "audio": {
                    "input": {
                        "format": "pcm16",
                        "transcription": {
                            # whisper-1 is deprecated for Realtime in GA;
                            # gpt-4o-mini-transcribe is the supported model.
                            "model": "gpt-4o-mini-transcribe"
                        },
                        "turn_detection": {
                            "type": "server_vad",          # server-side VAD – no need to manage silence timers
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 600,    # 600 ms silence = end of user turn
                            "create_response": True        # auto-respond after each turn
                        }
                    },
                    "output": {
                        "format": "pcm16",
                        "voice": "alloy",   # Azure supported voices: alloy, echo, fable, onyx, nova, shimmer
                    }
                },
                "max_output_tokens": 150  # keep responses brief for phone calls
            }
        }
        await self._send(config)
        logger.info("Realtime session configured")

    async def send_audio(self, pcm_24k: bytes):
        """Append audio to the input buffer (caller's voice)."""
        if not self._connected or not self.ws:
            return
        chunk = base64.b64encode(pcm_24k).decode()
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": chunk
        })

    async def receive_loop(
        self,
        on_audio: callable,           # async fn(pcm_bytes: bytes)
        on_transcript: callable,      # async fn(speaker: str, text: str)
        on_transfer_intent: callable, # async fn(text: str) → bool (True = transfer happened)
        stop_event: asyncio.Event,
    ):
        """
        Main receive loop – processes all events from Azure Realtime until stop_event is set.
        """
        async for raw in self.ws:
            if stop_event.is_set():
                break

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "session.created":
                logger.info("✅ Realtime session created")
                await self.configure()
                self._session_ready.set()

            elif etype == "session.updated":
                logger.debug("Session updated")

            elif etype in ("response.audio.delta", "response.output_audio.delta"):
                # Streaming audio chunk back from the model
                delta_b64 = event.get("delta", "")
                if delta_b64:
                    pcm = base64.b64decode(delta_b64)
                    await on_audio(pcm)

            elif etype in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                # Streaming text transcript of the AI's speech
                pass  # we log on .done

            elif etype in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                text = event.get("transcript", "").strip()
                if text:
                    logger.info(f"🤖 AI: {text}")
                    await on_transcript("assistant", text)
                    # Check if the AI itself said it will transfer
                    transfer_words = ["transfer", "connecting you", "put you through", "one moment"]
                    if any(w in text.lower() for w in transfer_words):
                        await on_transfer_intent(text)

            elif etype == "conversation.item.input_audio_transcription.completed":
                text = event.get("transcript", "").strip()
                if text:
                    logger.info(f"👤 User: {text}")
                    await on_transcript("caller", text)
                    # Check user-side transfer keywords
                    transfer_keywords = [
                        'speak', 'talk', 'human', 'person', 'agent',
                        'representative', 'manager', 'supervisor', 'transfer'
                    ]
                    if any(kw in text.lower() for kw in transfer_keywords):
                        await on_transfer_intent(text)

            elif etype == "response.done":
                logger.debug("AI response turn complete")

            elif etype == "error":
                err = event.get("error", {})
                logger.error(f"Realtime API error: {err.get('message', event)}")

            else:
                logger.debug(f"Realtime event: {etype}")

    async def _send(self, payload: dict):
        if self.ws and self._connected:
            await self.ws.send(json.dumps(payload))

    async def close(self):
        self._connected = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# RealtimeCallInstance – one call using the Realtime API
# ---------------------------------------------------------------------------
class RealtimeCallInstance:
    """
    Handles a single phone call using Azure Realtime API.

    Replaces the old CallInstance's record→STT→LLM→TTS→play loop with:
      - ARI ExternalMedia channel for raw RTP
      - RealtimeSession for speech-in / speech-out
      - RTPBridge to shuttle UDP frames
    """

    def __init__(self, channel, ari_client, config: dict, flask_app=None, agent=None):
        self.channel = channel
        self.ari_client = ari_client
        self.config = config
        self.flask_app = flask_app
        self.agent = agent

        self.id = channel.id
        self.active = True
        self.user_hung_up = False
        self.escalated = False
        self.escalated_to_dept_id = None
        self.escalation_reason = None
        self.turn_count = 0

        self._stop = asyncio.Event()
        self._rtp = RTPBridge()
        self._ext_channel = None   # ExternalMedia channel
        self._bridge = None        # ARI mixing bridge

        system_prompt = config.get('DEFAULT_SYSTEM_PROMPT', '') or self._default_prompt()

        self._session = RealtimeSession(
            endpoint=config.get('AZURE_OPENAI_ENDPOINT', '').rstrip('/'),
            api_key=config.get('AZURE_OPENAI_KEY', ''),
            deployment=config.get('AZURE_OPENAI_REALTIME_DEPLOYMENT',
                                  config.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-realtime')),
            api_version=config.get('AZURE_OPENAI_API_VERSION', '2025-04-01-preview'),
            system_prompt=system_prompt,
        )

    def _default_prompt(self) -> str:
        return (
            "You are a professional phone assistant for Jubilee Insurance. "
            "Be brief – this is a phone call. "
            "Keep every response under 25 words. "
            "Never mention being an AI. "
            "If the caller wants a human, say you will transfer them immediately."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def process(self):
        try:
            await self.channel.answer()
            logger.info("✅ Call answered")

            # 1. Open our local UDP socket that will receive/send RTP
            self._rtp.open()

            # 2. Create a mixing bridge.
            bridge = await self.ari_client.bridges.create(type='mixing')
            self._bridge = bridge
            logger.info("✅ Bridge created")

            # 3. Create ExternalMedia channel (no channelId param — universally supported).
            #    Our StasisStart filter ignores ExternalMedia channels so this won't
            #    trigger a second call handler.
            ext = await self.ari_client.channels.externalMedia(
                app=self.config.get('ARI_APP', 'ai-agent'),
                external_host=f"127.0.0.1:{self._rtp.local_port}",
                format="ulaw",
                encapsulation="rtp",
                transport="udp",
                connection_type="client",
                direction="both",
            )
            self._ext_channel = ext

            # 3b. Wait for the ExternalMedia channel's own StasisStart event.
            #     externalMedia() returns as soon as the channel object exists,
            #     but Asterisk joins it to the Stasis app asynchronously.
            #     Calling bridge.addChannel() before that completes fails with
            #     422 "Channel not in Stasis application".
            if self.agent is not None:
                await self.agent.wait_for_ext_media_ready(ext.id)

            # 4. Add the ExternalMedia channel to the bridge FIRST.
            #    Asterisk's mixing bridge negotiates its media format from the
            #    first member added. Adding the caller channel first (which
            #    negotiates e.g. slin16) and then adding an ExternalMedia
            #    (ulaw/RTP) channel causes a format-negotiation failure that
            #    surfaces as "422 Unprocessable Entity" on addChannel. Adding
            #    ExternalMedia first avoids that mismatch.
            await bridge.addChannel(channel=ext.id)
            logger.info("✅ ExternalMedia channel bridged")

            # 5. Now add the caller channel to the same bridge.
            await bridge.addChannel(channel=self.id)
            logger.info("✅ Caller channel added to bridge")

            # 5. Read Asterisk's RTP address from channel variables.
            #    These are set on the ExternalMedia channel after it's created.
            ch_vars = ext.json.get('channelvars', {})
            asterisk_rtp_host = ch_vars.get('UNICASTRTP_LOCAL_ADDRESS', '127.0.0.1')
            asterisk_rtp_port = ch_vars.get('UNICASTRTP_LOCAL_PORT', 0)

            if not asterisk_rtp_port:
                # Fallback: the channel name encodes the address, e.g.
                # "UnicastRTP/127.0.0.1:12366-..."
                ch_name = ext.json.get('name', '')
                try:
                    # name is like: UnicastRTP/udp/127.0.0.1:PORT-...
                    addr_part = ch_name.split('/')[-1].split('-')[0]  # "127.0.0.1:PORT"
                    asterisk_rtp_host, port_str = addr_part.rsplit(':', 1)
                    asterisk_rtp_port = int(port_str)
                    logger.info(f"Parsed RTP addr from channel name: {asterisk_rtp_host}:{asterisk_rtp_port}")
                except Exception:
                    logger.warning(f"Could not parse RTP port from channel name: {ch_name!r}")

            self._rtp.set_remote(asterisk_rtp_host, int(asterisk_rtp_port or 0))
            logger.info(f"✅ RTP bridge ready — Asterisk:{asterisk_rtp_host}:{asterisk_rtp_port} ↔ us:{self._rtp.local_port}")

            # 6. Connect to Azure Realtime WebSocket
            await self._session.connect()

            # 7. Run all loops concurrently until call ends
            await asyncio.gather(
                self._audio_inbound_pump(),    # caller audio (UDP) → Azure Realtime
                self._audio_outbound_pump(),   # (keepalive — audio sent in receive_loop)
                self._session.receive_loop(    # Azure Realtime events + audio out
                    on_audio=self._on_realtime_audio,
                    on_transcript=self._on_transcript,
                    on_transfer_intent=self._on_transfer_intent,
                    stop_event=self._stop,
                ),
                self._watchdog(),              # hang-up / timeout detection
                return_exceptions=True,
            )

        except Exception as e:
            # aiohttp web exceptions (e.g. HTTPUnprocessableEntity) carry the
            # ARI server's JSON error reason in .text, but the default str()
            # only shows the generic HTTP reason phrase. Log the body too so
            # the actual Asterisk-side cause (e.g. "Channel not in Stasis
            # application", bad format negotiation, etc.) is visible.
            body = getattr(e, "text", None)
            if body:
                logger.error(f"RealtimeCallInstance.process error: {e} | ARI response: {body}", exc_info=True)
            else:
                logger.error(f"RealtimeCallInstance.process error: {e}", exc_info=True)
        finally:
            await self._teardown()

    # ------------------------------------------------------------------
    # Audio pumps
    # ------------------------------------------------------------------
    async def _audio_inbound_pump(self):
        """Continuously read RTP from Asterisk and forward to Realtime."""
        await self._session._session_ready.wait()  # don't send until session is configured

        while not self._stop.is_set():
            pcm_24k = await self._rtp.recv_pcm_24k()
            if pcm_24k:
                await self._session.send_audio(pcm_24k)
                self.turn_count += 1
            else:
                await asyncio.sleep(0.005)   # 5 ms back-off when no data

    _audio_output_queue: asyncio.Queue = None  # per-instance, set in __init__

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    # Override __init__ to also create the queue (done above already, but add here for clarity)
    # We put received audio into a queue and drain it in a separate pump so we don't block
    # the receive_loop coroutine with synchronous socket sends.

    async def _on_realtime_audio(self, pcm_24k: bytes):
        """Called by receive_loop when a new audio chunk arrives from Realtime."""
        # Send directly via RTP bridge (non-blocking)
        self._rtp.send_pcm(pcm_24k)

    async def _audio_outbound_pump(self):
        """Placeholder – outbound audio is sent synchronously in _on_realtime_audio."""
        while not self._stop.is_set():
            await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------
    async def _on_transcript(self, speaker: str, text: str):
        self._log_transcript(speaker, text, confidence=1.0)

    async def _on_transfer_intent(self, text: str):
        """Detect which department and perform the ARI transfer."""
        if self.escalated:
            return

        intent_type = self._classify_intent(text)
        department = self._get_department_for_intent(intent_type)

        if department:
            logger.info(f"🔀 Transferring to {department.name} (ext {department.extension})")
            try:
                await self.channel.continueInDialplan(
                    context='from-internal',
                    extension=department.extension,
                    priority=1,
                )
                self.escalated = True
                self.escalated_to_dept_id = department.id
                self.escalation_reason = f"Transfer to {department.name}"
                self._log_intent('escalation', 0.9, text)
                self._stop.set()
            except Exception as e:
                logger.error(f"Transfer failed: {e}")
        else:
            logger.warning("No department found for transfer intent")

    # ------------------------------------------------------------------
    # Watchdog – detect hung-up
    # ------------------------------------------------------------------
    async def _watchdog(self):
        while not self._stop.is_set():
            await asyncio.sleep(2)
            if self.user_hung_up:
                self._stop.set()
                break
            try:
                await self.ari_client.channels.get(channelId=self.id)
            except Exception:
                logger.info("📴 Channel gone – ending call")
                self.user_hung_up = True
                self._stop.set()
                break

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    async def _teardown(self):
        self._stop.set()
        await self._session.close()
        self._rtp.close()

        # Destroy bridge (removes all channels from it)
        if hasattr(self, '_bridge') and self._bridge:
            try:
                await self._bridge.destroy()
            except Exception:
                pass

        # Clean up ExternalMedia channel
        if self._ext_channel:
            try:
                await self._ext_channel.hangup()
            except Exception:
                pass

        # Hang up main channel if still active
        if not self.user_hung_up and not self.escalated:
            try:
                await self.channel.hangup()
            except Exception:
                pass

        self.active = False
        logger.info(f"📴 Call {self.id[:12]} cleaned up")

    # ------------------------------------------------------------------
    # Helpers (same logic as original CallInstance)
    # ------------------------------------------------------------------
    def _classify_intent(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ['buy', 'purchase', 'new policy', 'quote', 'coverage']):
            return 'sales'
        if any(w in lower for w in ['claim', 'accident', 'damage', 'file']):
            return 'claims'
        if any(w in lower for w in ['bill', 'payment', 'pay', 'invoice', 'charge']):
            return 'billing'
        return 'support'

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
                    'sales': 'Sales', 'claims': 'Claims',
                    'billing': 'Billing', 'support': 'Support'
                }
                if intent_type in name_map:
                    dept = Department.query.filter_by(
                        name=name_map[intent_type], is_active=True
                    ).first()
                    if dept:
                        return dept
                return Department.query.filter_by(is_active=True).order_by(
                    Department.priority.desc()
                ).first()
        except Exception as e:
            logger.error(f"get_department error: {e}")
            return None

    def _log_transcript(self, speaker: str, text: str, confidence: float):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call, CallTranscript
                call = Call.query.filter_by(call_id=self.id).first()
                if not call:
                    return
                db.session.add(CallTranscript(
                    call_id=call.id, speaker=speaker, text=text,
                    confidence=confidence, timestamp=datetime.utcnow()
                ))
                db.session.commit()
        except Exception as e:
            logger.error(f"log_transcript error: {e}")

    def _log_intent(self, intent_type: str, confidence: float, context: str):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call, CallIntent
                call = Call.query.filter_by(call_id=self.id).first()
                if not call:
                    return
                db.session.add(CallIntent(
                    call_id=call.id, intent_type=intent_type,
                    confidence=confidence, context=context,
                    detected_at=datetime.utcnow()
                ))
                db.session.commit()
        except Exception as e:
            logger.error(f"log_intent error: {e}")


# ---------------------------------------------------------------------------
# RealtimeARIAgent – drop-in replacement for ARIAgent
# ---------------------------------------------------------------------------
class RealtimeARIAgent:
    """
    Drop-in replacement for ARIAgent that uses the Azure OpenAI Realtime API.

    Usage in app.py – just swap the import:
        from services.realtime_agent import RealtimeARIAgent as ARIAgent
    """

    def __init__(self, app_config: dict, flask_app=None):
        self.config = app_config
        self.flask_app = flask_app
        self.running = False
        self.active_calls: dict = {}
        self.total_calls: int = 0
        # ExternalMedia channels fire their own StasisStart once Asterisk has
        # fully joined them to the Stasis app. We must wait for that event
        # before adding them to a bridge, or addChannel fails with
        # "Channel not in Stasis application" (422). Keyed by channel id.
        self._ext_media_ready: dict = {}

        self.ari_base     = os.getenv('ARI_BASE', app_config.get('ARI_BASE', 'http://localhost:8088'))
        self.ari_username = os.getenv('ARI_USERNAME', app_config.get('ARI_USERNAME', 'asterisk'))
        self.ari_password = os.getenv('ARI_PASSWORD', app_config.get('ARI_PASSWORD', ''))
        self.ari_app      = os.getenv('ARI_APP', app_config.get('ARI_APP', 'ai-agent'))

        self.ari_client = None

        # Validate Realtime config
        endpoint   = app_config.get('AZURE_OPENAI_ENDPOINT', '')
        api_key    = app_config.get('AZURE_OPENAI_KEY', '')
        deployment = app_config.get(
            'AZURE_OPENAI_REALTIME_DEPLOYMENT',
            app_config.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-realtime')
        )
        if not endpoint or not api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set for RealtimeARIAgent"
            )

        logger.info(f"RealtimeARIAgent configured: deployment={deployment}")

    async def start(self):
        self.running = True
        logger.info("=" * 60)
        logger.info("🤖 RealtimeARIAgent Starting (Azure OpenAI Realtime API)")
        logger.info("=" * 60)

        try:
            self.ari_client = await aioari.connect(
                self.ari_base, self.ari_username, self.ari_password
            )
            logger.info("✅ ARI connected")

            self.ari_client.on_event("StasisStart",      self._handle_stasis_start)
            self.ari_client.on_event("StasisEnd",        self._handle_stasis_end)
            self.ari_client.on_event("ChannelHangupRequest", self._handle_hangup_request)

            logger.info("🎙️ REALTIME AGENT READY – waiting for calls")
            logger.info("=" * 60)
            await self.ari_client.run(apps=self.ari_app)

        except Exception as e:
            logger.error(f"❌ ARI connection error: {e}")
            self.running = False

    async def stop(self):
        logger.info("Stopping RealtimeARIAgent...")
        self.running = False
        for call in list(self.active_calls.values()):
            call._stop.set()
        if self.ari_client:
            try:
                await self.ari_client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # ARI event handlers
    # ------------------------------------------------------------------
    def _handle_stasis_start(self, event):
        asyncio.create_task(self._process_call(event))

    def _handle_stasis_end(self, event):
        channel_id = event.get("channel", {}).get("id")
        if channel_id and channel_id in self.active_calls:
            call = self.active_calls[channel_id]
            call.user_hung_up = True
            call._stop.set()

    def _handle_hangup_request(self, event):
        channel_id = event.get("channel", {}).get("id")
        if channel_id and channel_id in self.active_calls:
            call = self.active_calls[channel_id]
            call.user_hung_up = True
            call._stop.set()

    async def wait_for_ext_media_ready(self, channel_id: str, timeout: float = 5.0):
        """
        Block until the ExternalMedia channel with the given id has fired its
        own StasisStart (i.e. Asterisk has fully joined it to our Stasis app).
        Adding it to a bridge before this happens fails with
        "Channel not in Stasis application" (422).
        """
        ev = self._ext_media_ready.get(channel_id)
        if ev is None:
            ev = asyncio.Event()
            self._ext_media_ready[channel_id] = ev
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timed out waiting for ExternalMedia channel {channel_id[:12]} StasisStart")
        finally:
            self._ext_media_ready.pop(channel_id, None)

    async def _process_call(self, event):
        channel_id = event.get("channel", {}).get("id")
        if not channel_id:
            return

        # ExternalMedia channels fire their own StasisStart but are not real
        # callers. Signal any waiter (process() waiting to bridge this
        # channel) that it's now in the Stasis app, then return.
        channel_name = event.get("channel", {}).get("name", "")
        if channel_name.startswith(("UnicastRTP/", "ExternalMedia/")):
            logger.debug(f"ExternalMedia channel ready: {channel_name} ({channel_id[:12]})")
            ev = self._ext_media_ready.get(channel_id)
            if ev:
                ev.set()
            else:
                # process() hasn't registered a waiter yet — pre-set so the
                # upcoming wait_for_ext_media_ready() returns immediately.
                ready = asyncio.Event()
                ready.set()
                self._ext_media_ready[channel_id] = ready
            return

        # Also skip if we're already handling this channel
        if channel_id in self.active_calls:
            logger.debug(f"Duplicate StasisStart for {channel_id[:12]} — ignoring")
            return

        try:
            channel = await self.ari_client.channels.get(channelId=channel_id)
            await self._handle_call(channel)
        except Exception as e:
            logger.error(f"Call processing error: {e}")

    async def _handle_call(self, channel):
        caller = channel.json.get('caller', {}).get('number', 'Unknown')
        logger.info(f"📞 Incoming call from {caller}")

        call = RealtimeCallInstance(
            channel=channel,
            ari_client=self.ari_client,
            config=self.config,
            flask_app=self.flask_app,
            agent=self,
        )

        self.active_calls[channel.id] = call
        self.total_calls += 1
        self._log_call_start(call.id, caller)

        try:
            await call.process()
        except Exception as e:
            logger.error(f"Call error: {e}")
            self._log_call_error(call.id, str(e))
        finally:
            self.active_calls.pop(channel.id, None)
            self._log_call_end(call)

    # ------------------------------------------------------------------
    # DB logging (identical to original ARIAgent)
    # ------------------------------------------------------------------
    def _log_call_start(self, call_id: str, caller_number: str):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                db.session.add(Call(
                    call_id=call_id, caller_number=caller_number,
                    status='active', started_at=datetime.utcnow()
                ))
                db.session.commit()
        except Exception as e:
            logger.error(f"log_call_start: {e}")

    def _log_call_error(self, call_id: str, error_msg: str):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                call = Call.query.filter_by(call_id=call_id).first()
                if call:
                    call.status = 'error'
                    call.ended_at = datetime.utcnow()
                    db.session.commit()
        except Exception as e:
            logger.error(f"log_call_error: {e}")

    def _log_call_end(self, call: RealtimeCallInstance):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                record = Call.query.filter_by(call_id=call.id).first()
                if record:
                    record.status = 'escalated' if call.escalated else 'completed'
                    record.escalated = call.escalated
                    record.escalated_to_department_id = call.escalated_to_dept_id
                    record.escalation_reason = call.escalation_reason
                    record.ended_at = datetime.utcnow()
                    if record.started_at:
                        record.duration_seconds = int(
                            (record.ended_at - record.started_at).total_seconds()
                        )
                    record.total_interactions = call.turn_count
                    db.session.commit()
        except Exception as e:
            logger.error(f"log_call_end: {e}")