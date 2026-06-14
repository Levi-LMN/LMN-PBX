# services/ari_agent.py
"""
ARI-based agent service using OpenAI Realtime API for ultra-low latency voice.

Architecture (new):
  Caller → Asterisk → ARI ExternalMedia → RTP UDP socket (this service)
         ↕ PCM audio (μ-law 8kHz)
  This service ↔ OpenAI Realtime WebSocket (PCM16 24kHz)
  ↓ audio back → RTP → Asterisk → Caller

Why Realtime API?
  Old: record → Azure STT → GPT chat → Azure TTS → play file  (~3-5 s latency)
  New: stream RTP → OpenAI Realtime (VAD + STT + LLM + TTS in one hop) → stream back (~300 ms)
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

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
REALTIME_MODEL = "gpt-4o-realtime-preview"

# Audio parameters
ASTERISK_SAMPLE_RATE = 8000    # Asterisk sends μ-law 8 kHz
OPENAI_SAMPLE_RATE   = 24000   # OpenAI Realtime expects PCM16 24 kHz
RTP_PACKET_MS        = 20      # 20 ms RTP packets from Asterisk (160 samples @ 8 kHz)
RTP_HEADER_SIZE      = 12      # Standard RTP header

# Default RTP port range for ExternalMedia channels
RTP_LISTEN_HOST = "0.0.0.0"
RTP_PORT_START  = 20000
RTP_PORT_END    = 20100


def _ws_header_kwargs(headers: dict) -> dict:
    """
    Return the correct kwarg for passing extra headers to websockets.connect(),
    depending on the installed websockets version:

      - websockets >= 14 (new asyncio client) → `additional_headers`
      - websockets <  14 (legacy client)      → `extra_headers`

    Passing the wrong kwarg raises:
      TypeError: ... got an unexpected keyword argument 'additional_headers'
    (or 'extra_headers' on newer versions), so detect at runtime instead of
    hardcoding one or the other.
    """
    try:
        params = inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):
        params = inspect.signature(websockets.connect.__init__).parameters

    if "additional_headers" in params:
        return {"additional_headers": headers}
    return {"extra_headers": headers}


# ─────────────────────────────────────────────────────────────────────────────
# ARIAgent — top-level service
# ─────────────────────────────────────────────────────────────────────────────

class ARIAgent:
    """Manages the ARI connection and spawns a RealtimeCallSession per call."""

    def __init__(self, app_config, flask_app=None):
        self.config     = app_config
        self.flask_app  = flask_app
        self.running    = False
        self.active_calls: dict[str, "RealtimeCallSession"] = {}
        self.total_calls = 0

        # ARI
        self.ari_url      = os.getenv("ARI_URL",      "http://localhost:8088/ari")
        self.ari_base     = os.getenv("ARI_BASE",     "http://localhost:8088")
        self.ari_username = os.getenv("ARI_USERNAME", "asterisk")
        self.ari_password = os.getenv("ARI_PASSWORD", "your_ari_password")
        self.ari_app      = os.getenv("ARI_APP",      "ai-agent")

        # OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_voice   = os.getenv("OPENAI_VOICE", "alloy")  # alloy / echo / shimmer / ash / coral / verse
        self.system_prompt  = os.getenv("DEFAULT_SYSTEM_PROMPT") or self._default_prompt()

        # Port allocator for ExternalMedia RTP sockets
        self._next_rtp_port = RTP_PORT_START
        self._port_lock     = asyncio.Lock()

        self.ari_client = None
        logger.info("ARIAgent (Realtime) initialised")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _default_prompt(self):
        return (
            "You are a professional phone assistant for Jubilee Insurance. "
            "RULES: respond in 20 words or fewer — this is a phone call, be brief. "
            "Never say you are an AI. Be empathetic and professional. "
            "When the caller wants a human agent, say you will transfer them now."
        )

    async def _alloc_rtp_port(self) -> int:
        async with self._port_lock:
            port = self._next_rtp_port
            self._next_rtp_port = port + 2  # step by 2 (RTP + RTCP)
            if self._next_rtp_port > RTP_PORT_END:
                self._next_rtp_port = RTP_PORT_START
            return port

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        self.running = True
        logger.info("=" * 60)
        logger.info("🤖 ARI Realtime Agent starting")
        logger.info(f"   Model  : {REALTIME_MODEL}")
        logger.info(f"   Voice  : {self.openai_voice}")
        logger.info("=" * 60)

        if not self.openai_api_key:
            logger.error("❌ OPENAI_API_KEY not set — cannot start")
            return

        try:
            logger.info(f"Connecting to ARI at {self.ari_base} …")
            self.ari_client = await aioari.connect(
                self.ari_base, self.ari_username, self.ari_password
            )
            logger.info("✅ ARI connected")

            self.ari_client.on_event("StasisStart",        self._handle_stasis_start)
            self.ari_client.on_event("StasisEnd",          self._handle_stasis_end)
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
        channel = event.get("channel", {})
        channel_name = channel.get("name", "")

        # ExternalMedia channels (created via channels.externalMedia for
        # RTP I/O) also enter this Stasis app — they belong to an existing
        # RealtimeCallSession's media leg, NOT a new inbound call. If we
        # don't filter these out, _process_call() tries to treat the
        # ExternalMedia channel as a brand-new call (answer it, spin up
        # another session, etc.), which fails with "Not Found" since that
        # channel can't be answered/handled like a SIP channel.
        if channel_name.startswith("UnicastRTP/"):
            logger.debug(
                f"StasisStart for ExternalMedia channel "
                f"{channel.get('id', '')[:12]} — ignoring (not a new call)"
            )
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

        rtp_port = await self._alloc_rtp_port()

        session = RealtimeCallSession(
            channel       = channel,
            ari_client    = self.ari_client,
            openai_api_key= self.openai_api_key,
            openai_voice  = self.openai_voice,
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
                    c.status     = "error"
                    c.ended_at   = datetime.utcnow()
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB log error error: {e}")

    def _db_log_call_end(self, session: "RealtimeCallSession"):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                c = Call.query.filter_by(call_id=session.channel_id).first()
                if c:
                    c.status            = "escalated" if session.escalated else "completed"
                    c.escalated         = session.escalated
                    c.ended_at          = datetime.utcnow()
                    if c.started_at:
                        c.duration_seconds = int(
                            (c.ended_at - c.started_at).total_seconds()
                        )
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB log end error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# RealtimeCallSession
# ─────────────────────────────────────────────────────────────────────────────

class RealtimeCallSession:
    """
    One session per inbound call.

    Flow:
      1. Answer the call via ARI.
      2. Create a mixing bridge.
      3. Create an ExternalMedia channel → Asterisk will RTP audio to us on `rtp_port`.
      4. Open a UDP socket to receive that RTP stream.
      5. Open a WebSocket to OpenAI Realtime API.
      6. Pump audio: UDP → base64 → OpenAI WS  (caller speech)
                     OpenAI WS → base64 → UDP   (AI speech back)
      7. Parse Realtime events for transcripts, VAD boundaries, and function calls.
      8. On transfer intent detected → continueInDialplan.
    """

    TRANSFER_KEYWORDS = {
        "speak", "talk", "human", "person", "agent",
        "representative", "manager", "supervisor", "someone",
        "transfer", "escalate", "real person",
    }

    def __init__(self, *, channel, ari_client, openai_api_key, openai_voice,
                 system_prompt, rtp_port, ari_url, ari_username, ari_password,
                 flask_app):
        self.channel        = channel
        self.channel_id     = channel.id
        self.ari_client     = ari_client
        self.openai_api_key = openai_api_key
        self.openai_voice   = openai_voice
        self.system_prompt  = system_prompt
        self.rtp_port       = rtp_port
        self.ari_url        = ari_url
        self.ari_username   = ari_username
        self.ari_password   = ari_password
        self.flask_app      = flask_app

        self.caller_hung_up  = False
        self.escalated       = False
        self._closed         = False

        # RTP state
        self._udp_sock: socket.socket | None = None
        self._asterisk_rtp_addr: tuple | None = None  # (ip, port) — learned from first packet
        self._rtp_seq   = 0
        self._rtp_ts    = 0
        self._rtp_ssrc  = 0xDEADBEEF

        # OpenAI WS
        self._openai_ws = None

        # Bridge & external media channel IDs
        self._bridge_id    = None
        self._ext_channel_id = None

        # Queues
        self._audio_to_openai: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        self._audio_to_asterisk: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)

    # ── main run ──────────────────────────────────────────────────────────────

    async def run(self):
        """Answer → bridge → ExternalMedia → Realtime WebSocket → pump."""
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

            # 4. Create ExternalMedia channel — Asterisk will push RTP here
            ext_channel = await self.ari_client.channels.externalMedia(
                app           = os.getenv("ARI_APP", "ai-agent"),
                external_host = f"127.0.0.1:{self.rtp_port}",
                format        = "ulaw",   # μ-law 8 kHz — what Asterisk prefers
                encapsulation = "rtp",
                transport     = "udp",
                connection_type = "client",
                direction     = "both",
            )
            self._ext_channel_id = ext_channel.id
            logger.info(f"📡 [{self.channel_id[:12]}] ExternalMedia channel: {self._ext_channel_id}")

            # Newly created ExternalMedia channels need a brief moment to
            # fully register with the Stasis app on the Asterisk side.
            # Adding them to a bridge immediately can raise 422
            # Unprocessable Entity ("Channel not in Stasis application"),
            # so retry briefly with a short backoff.
            await self._add_channel_to_bridge_with_retry(bridge, self._ext_channel_id)

            # 5. Connect to OpenAI Realtime
            await self._connect_openai()

            # 6. Run all pumps concurrently until call ends
            await asyncio.gather(
                self._recv_rtp_loop(),
                self._send_rtp_loop(),
                self._openai_recv_loop(),
                self._openai_send_loop(),
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
        """
        Add a channel to a bridge, retrying briefly if Asterisk responds
        with 422 Unprocessable Entity ("Channel not in Stasis application").

        This happens when bridge.addChannel() is called immediately after
        channels.externalMedia() returns — the channel exists, but Asterisk
        hasn't finished registering it with the Stasis app yet.
        """
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
                    f"[{self.channel_id[:12]}] addChannel({channel_id}) "
                    f"attempt {attempt}/{retries} not ready yet "
                    f"(422) — retrying in {delay}s …"
                )
                await asyncio.sleep(delay)

        logger.error(
            f"❌ [{self.channel_id[:12]}] Could not add channel {channel_id} "
            f"to bridge after {retries} attempts"
        )
        raise last_err

    # ── OpenAI WebSocket ──────────────────────────────────────────────────────

    async def _connect_openai(self):
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Beta":   "realtime=v1",
        }
        self._openai_ws = await websockets.connect(
            REALTIME_URL,
            ping_interval=20,
            ping_timeout=30,
            **_ws_header_kwargs(headers),
        )
        logger.info(f"🔗 [{self.channel_id[:12]}] OpenAI Realtime WS connected")

        # Configure session
        await self._openai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities":      ["audio", "text"],
                "voice":           self.openai_voice,
                "instructions":    self.system_prompt,
                "input_audio_format":  "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type":               "server_vad",
                    "threshold":          0.5,
                    "prefix_padding_ms":  300,
                    "silence_duration_ms": 600,
                },
                "temperature": 0.7,
            },
        }))
        logger.info(f"⚙️  [{self.channel_id[:12]}] Realtime session configured")

    # ── RTP receive (Asterisk → queue → OpenAI) ───────────────────────────────

    async def _recv_rtp_loop(self):
        """Read μ-law RTP from Asterisk, convert to PCM16 24kHz, enqueue for OpenAI."""
        loop = asyncio.get_running_loop()
        while not self.caller_hung_up and not self._closed:
            try:
                # Non-blocking read from UDP socket
                data, addr = await loop.run_in_executor(
                    None, self._udp_sock.recvfrom, 4096
                )
                if not self._asterisk_rtp_addr:
                    self._asterisk_rtp_addr = addr
                    logger.info(
                        f"📻 [{self.channel_id[:12]}] "
                        f"Asterisk RTP source: {addr[0]}:{addr[1]}"
                    )

                # Strip 12-byte RTP header
                if len(data) <= RTP_HEADER_SIZE:
                    continue
                ulaw_payload = data[RTP_HEADER_SIZE:]

                # μ-law → PCM16 @ 8 kHz
                pcm8 = audioop.ulaw2lin(ulaw_payload, 2)

                # Upsample 8 kHz → 24 kHz (3× linear interpolation)
                pcm24, _ = audioop.ratecv(pcm8, 2, 1, ASTERISK_SAMPLE_RATE, OPENAI_SAMPLE_RATE, None)

                await self._audio_to_openai.put(pcm24)

            except BlockingIOError:
                await asyncio.sleep(0.005)
            except Exception as e:
                if not self._closed:
                    logger.debug(f"RTP recv error: {e}")
                break

    # ── OpenAI send (queue → WS) ──────────────────────────────────────────────

    async def _openai_send_loop(self):
        """Pull PCM chunks from queue and send to OpenAI Realtime as base64."""
        while not self.caller_hung_up and not self._closed:
            try:
                chunk = await asyncio.wait_for(
                    self._audio_to_openai.get(), timeout=0.5
                )
                if self._openai_ws and not self._openai_ws.closed:
                    await self._openai_ws.send(json.dumps({
                        "type":  "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not self._closed:
                    logger.debug(f"OpenAI send error: {e}")
                break

    # ── OpenAI receive (WS → queue / events) ─────────────────────────────────

    async def _openai_recv_loop(self):
        """
        Receive events from OpenAI Realtime:
          - response.audio.delta  → enqueue PCM for sending back to Asterisk
          - conversation.item.input_audio_transcription.completed → check transfer intent
          - response.done / error → logging
        """
        while not self.caller_hung_up and not self._closed:
            try:
                if not self._openai_ws:
                    await asyncio.sleep(0.1)
                    continue

                raw = await asyncio.wait_for(
                    self._openai_ws.recv(), timeout=1.0
                )
                event = json.loads(raw)
                etype = event.get("type", "")

                if etype == "response.audio.delta":
                    # Decode PCM16 24kHz from OpenAI
                    pcm24 = base64.b64decode(event["delta"])
                    # Downsample 24 kHz → 8 kHz
                    pcm8, _ = audioop.ratecv(pcm24, 2, 1, OPENAI_SAMPLE_RATE, ASTERISK_SAMPLE_RATE, None)
                    # PCM16 → μ-law
                    ulaw = audioop.lin2ulaw(pcm8, 2)
                    await self._audio_to_asterisk.put(ulaw)

                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    logger.info(f"👤 [{self.channel_id[:12]}] Caller: {transcript}")
                    self._db_log_transcript("caller", transcript, 1.0)
                    # Check for transfer intent
                    if self._detect_transfer_intent(transcript):
                        logger.info(f"🔀 [{self.channel_id[:12]}] Transfer intent detected")
                        await self._handle_escalation(transcript)

                elif etype == "response.audio_transcript.delta":
                    # Streamed AI transcript — ignore or accumulate
                    pass

                elif etype == "response.done":
                    item = event.get("response", {})
                    usage = item.get("usage", {})
                    logger.debug(
                        f"🤖 [{self.channel_id[:12]}] Response done "
                        f"(tokens: {usage.get('total_tokens', '?')})"
                    )

                elif etype == "error":
                    logger.error(
                        f"❌ OpenAI Realtime error: {event.get('error', event)}"
                    )

                elif etype == "session.created":
                    logger.info(
                        f"✅ [{self.channel_id[:12]}] Realtime session created: "
                        f"{event.get('session', {}).get('id', '')}"
                    )

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                if not self._closed:
                    logger.warning(f"⚠️  [{self.channel_id[:12]}] OpenAI WS closed")
                break
            except Exception as e:
                if not self._closed:
                    logger.debug(f"OpenAI recv error: {e}")
                break

    # ── RTP send (queue → Asterisk) ───────────────────────────────────────────

    async def _send_rtp_loop(self):
        """
        Pull μ-law chunks from queue and send back to Asterisk via RTP.
        We build minimal RTP packets (sequence + timestamp + SSRC).
        """
        loop = asyncio.get_running_loop()
        while not self.caller_hung_up and not self._closed:
            try:
                ulaw_chunk = await asyncio.wait_for(
                    self._audio_to_asterisk.get(), timeout=0.5
                )
                if not self._asterisk_rtp_addr or not self._udp_sock:
                    continue

                # Build RTP packet
                # Each μ-law byte = 1 sample @ 8 kHz → 125 μs per sample
                n_samples = len(ulaw_chunk)
                rtp_pkt   = self._build_rtp_packet(ulaw_chunk)
                self._rtp_ts  += n_samples
                self._rtp_seq = (self._rtp_seq + 1) & 0xFFFF

                await loop.run_in_executor(
                    None,
                    self._udp_sock.sendto,
                    rtp_pkt,
                    self._asterisk_rtp_addr,
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not self._closed:
                    logger.debug(f"RTP send error: {e}")
                break

    def _build_rtp_packet(self, payload: bytes) -> bytes:
        """Minimal RTP/AVP header (RFC 3550) + μ-law payload (PT=0)."""
        header = struct.pack(
            "!BBHII",
            0x80,           # V=2, P=0, X=0, CC=0
            0x00,           # M=0, PT=0 (PCMU / μ-law)
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
        """Route caller to appropriate department via dialplan continue."""
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

        # Close OpenAI WebSocket
        if self._openai_ws:
            try:
                await self._openai_ws.close()
            except Exception:
                pass

        # Close UDP socket
        if self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass

        # Destroy ExternalMedia channel
        if self._ext_channel_id:
            try:
                await self.ari_client.channels.hangup(channelId=self._ext_channel_id)
            except Exception:
                pass

        # Destroy bridge
        if self._bridge_id:
            try:
                await self.ari_client.bridges.destroy(bridgeId=self._bridge_id)
            except Exception:
                pass

        # Hang up caller channel if still active and not transferred
        if not self.escalated and not self.caller_hung_up:
            try:
                await self.channel.hangup()
            except Exception:
                pass

        logger.info(f"🔒 [{self.channel_id[:12]}] Session closed")


# ─────────────────────────────────────────────────────────────────────────────
# Stubs retained for backward compatibility with blueprints that may import them
# ─────────────────────────────────────────────────────────────────────────────

class SoundCache:
    """No longer used — OpenAI Realtime streams audio directly."""
    pass


class FileSystemAccess:
    """No longer used — no TTS files needed."""
    pass


class AzureSpeechTranscriber:
    """No longer used — transcription handled by OpenAI Realtime VAD."""
    pass