# services/ari_agent.py
"""
ARI agent using Azure Voice Live API.

Architecture:
  Caller → Asterisk ARI ExternalMedia → UDP RTP (this service)
  This service ↔ Azure Voice Live WebSocket
  Azure audio → UDP RTP → Asterisk → Caller

Audio path (inbound):
  μ-law 8kHz RTP  →  ulaw2lin  →  PCM16 8kHz  →  base64  →  Azure (declared 8kHz)

Audio path (outbound):
  Azure PCM16 24kHz  →  ratecv 24→8kHz  →  lin2ulaw  →  RTP → Asterisk

Escalation:
  Caller says transfer keyword  OR  AI announces transfer
  → remove ExternalMedia from bridge → continueInDialplan → FreePBX dept ext
"""

import asyncio
import audioop
import base64
import inspect
import json
import logging
import os
import socket
import struct
import websockets
from aiohttp.web_exceptions import HTTPUnprocessableEntity
from datetime import datetime

import aioari

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

AZURE_API_VERSION   = "2025-10-01"
AZURE_MODEL         = "gpt-realtime"
AZURE_HOST_SUFFIX   = os.getenv("AZURE_VOICE_LIVE_HOST_SUFFIX", "services.ai.azure.com")

# Asterisk sends μ-law 8kHz; we tell Azure input is PCM16 8kHz (no upsample needed)
ASTERISK_RATE  = 8000
AZURE_IN_RATE  = 8000   # what we declare to Azure for input
AZURE_OUT_RATE = 24000  # Azure always outputs 24kHz PCM16

RTP_HEADER_SIZE   = 12
ULAW_FRAME_BYTES  = 160   # 20 ms @ 8kHz
FRAME_DURATION    = 0.020  # seconds

RTP_HOST  = "0.0.0.0"
RTP_START = 20000
RTP_END   = 20100


def _azure_ws_url(resource: str) -> str:
    return (
        f"wss://{resource}.{AZURE_HOST_SUFFIX}"
        f"/voice-live/realtime"
        f"?api-version={AZURE_API_VERSION}"
        f"&model={AZURE_MODEL}"
    )


def _ws_connect_kwargs(api_key: str) -> dict:
    """Return the right header kwarg for the installed websockets version."""
    try:
        params = inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):
        params = {}
    key = "additional_headers" if "additional_headers" in params else "extra_headers"
    return {key: {"api-key": api_key}}


def _ws_open(ws) -> bool:
    if ws is None:
        return False
    if hasattr(ws, "closed"):
        return not ws.closed
    if hasattr(ws, "state"):
        import websockets.connection
        return ws.state == websockets.connection.State.OPEN
    return True


# ── ARIAgent ──────────────────────────────────────────────────────────────────

class ARIAgent:
    """Top-level service: connects to ARI, spawns one session per inbound call."""

    def __init__(self, app_config, flask_app=None):
        self.flask_app  = flask_app
        self.running    = False
        self.total_calls = 0
        self.active_calls: dict[str, "CallSession"] = {}

        self.ari_base     = app_config.get("ARI_BASE",     "http://localhost:8088")
        self.ari_username = app_config.get("ARI_USERNAME", "asterisk")
        self.ari_password = app_config.get("ARI_PASSWORD", "your_ari_password")
        self.ari_app      = app_config.get("ARI_APP",      "ai-agent")
        self.ari_url      = app_config.get("ARI_URL",      "http://localhost:8088/ari")

        self.azure_resource   = app_config.get("AZURE_VOICE_LIVE_RESOURCE", "")
        self.azure_api_key    = app_config.get("AZURE_SPEECH_KEY", "")
        self.azure_voice_name = app_config.get("AZURE_VOICE_NAME", "en-KE-AsiliaNeural")
        self.azure_voice_type = app_config.get("AZURE_VOICE_TYPE", "azure-standard")
        self.system_prompt    = app_config.get("DEFAULT_SYSTEM_PROMPT") or self._default_prompt()

        self._port_lock     = asyncio.Lock()
        self._next_rtp_port = RTP_START
        self.ari_client     = None
        self._config_ok     = bool(self.azure_resource and self.azure_api_key)

    # ── Dashboard compatibility ───────────────────────────────────────────────

    @property
    def ai_client(self):
        return self.ari_client

    @property
    def is_connected(self) -> bool:
        return self.running and self.ari_client is not None

    @property
    def active_call_count(self) -> int:
        return len(self.active_calls)

    def get_status(self) -> dict:
        return {
            "connected":      self.is_connected,
            "running":        self.running,
            "config_ok":      self._config_ok,
            "active_calls":   self.active_call_count,
            "total_calls":    self.total_calls,
            "azure_resource": self.azure_resource or "not set",
            "voice_name":     self.azure_voice_name,
            "voice_type":     self.azure_voice_type,
            "model":          AZURE_MODEL,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        self.running = True
        logger.info("=" * 60)
        logger.info(f"   Resource : {self.azure_resource}")
        logger.info(f"   Voice    : {self.azure_voice_name} ({self.azure_voice_type})")
        logger.info(f"   Model    : {AZURE_MODEL}")
        logger.info("=" * 60)

        if not self.azure_resource or not self.azure_api_key:
            logger.error("❌ AZURE_VOICE_LIVE_RESOURCE / AZURE_SPEECH_KEY not set")
            return

        try:
            logger.info(f"Connecting to ARI at {self.ari_base} …")
            self.ari_client = await aioari.connect(
                self.ari_base, self.ari_username, self.ari_password
            )
            logger.info("✅ ARI connected")

            self.ari_client.on_event("StasisStart",          self._on_stasis_start)
            self.ari_client.on_event("StasisEnd",            self._on_stasis_end)
            self.ari_client.on_event("ChannelHangupRequest", self._on_hangup_request)

            logger.info("🎙️  READY — waiting for calls")
            logger.info("=" * 60)
            await self.ari_client.run(apps=self.ari_app)

        except Exception as e:
            logger.error(f"❌ ARI error: {e}")
            self.running = False

    async def stop(self):
        self.running = False
        for s in list(self.active_calls.values()):
            await s.close()
        if self.ari_client:
            try:
                await self.ari_client.close()
            except Exception:
                pass

    # ── ARI events ────────────────────────────────────────────────────────────

    def _on_stasis_start(self, event):
        name = event.get("channel", {}).get("name", "")
        if name.startswith("UnicastRTP/"):
            return
        asyncio.create_task(self._handle_call(event))

    def _on_stasis_end(self, event):
        name = event.get("channel", {}).get("name", "")
        cid  = event.get("channel", {}).get("id", "")
        if name.startswith("UnicastRTP/"):
            return
        sess = self.active_calls.get(cid)
        if sess:
            logger.info(f"📴 [{cid[:12]}] Caller hung up (StasisEnd)")
            sess.caller_hung_up = True
            sess._closed = True

    def _on_hangup_request(self, event):
        name = event.get("channel", {}).get("name", "")
        cid  = event.get("channel", {}).get("id", "")
        if name.startswith("UnicastRTP/"):
            return
        sess = self.active_calls.get(cid)
        if sess:
            logger.info(f"📴 [{cid[:12]}] Caller hanging up (HangupRequest)")
            sess.caller_hung_up = True
            sess._closed = True

    async def _handle_call(self, event):
        cid = event.get("channel", {}).get("id")
        if not cid:
            return
        try:
            channel = await self.ari_client.channels.get(channelId=cid)
        except Exception as e:
            logger.error(f"❌ Could not get channel {cid}: {e}")
            return

        caller = channel.json.get("caller", {}).get("number", "Unknown")
        logger.info(f"📞 Incoming call from {caller}")

        if not self._config_ok:
            logger.error("❌ Rejecting call — Azure not configured")
            try:
                await channel.answer()
                await asyncio.sleep(0.3)
                await channel.hangup()
            except Exception:
                pass
            return

        rtp_port = await self._alloc_rtp_port()
        prompt   = self.system_prompt + self._load_kb() + self._load_caller_context(caller)

        sess = CallSession(
            channel        = channel,
            ari_client     = self.ari_client,
            ari_app        = self.ari_app,
            azure_ws_url   = _azure_ws_url(self.azure_resource),
            azure_api_key  = self.azure_api_key,
            voice_name     = self.azure_voice_name,
            voice_type     = self.azure_voice_type,
            system_prompt  = prompt,
            rtp_port       = rtp_port,
            flask_app      = self.flask_app,
            caller_number  = caller,
        )

        self.active_calls[cid] = sess
        self.total_calls += 1
        self._db_call_start(cid, caller)

        try:
            await sess.run()
        except Exception as e:
            logger.error(f"❌ Session error: {e}", exc_info=True)
            self._db_call_error(cid)
        finally:
            self.active_calls.pop(cid, None)
            await sess.close()
            self._db_call_end(sess)

    async def _alloc_rtp_port(self) -> int:
        async with self._port_lock:
            port = self._next_rtp_port
            self._next_rtp_port = port + 2
            if self._next_rtp_port > RTP_END:
                self._next_rtp_port = RTP_START
            return port

    # ── Knowledge base ────────────────────────────────────────────────────────

    def _load_kb(self) -> str:
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
                parts = ["\n\nKNOWLEDGE BASE:"]
                for e in entries:
                    parts.append(f"\n[{e.category.upper()}] {e.title}:\n{e.content}")
                    e.increment_usage()
                db.session.commit()
                logger.info(f"📚 Loaded {len(entries)} KB entries")
                return "\n".join(parts)
        except Exception as e:
            logger.error(f"KB load error: {e}")
            return ""

    # ── Caller context (customer / claims / tickets lookup) ──────────────────

    def _load_caller_context(self, caller_number: str) -> str:
        """
        Look up the caller by phone number and build a short context block
        the AI can reference — existing policies, open claims, open tickets.
        Returns "" if no flask_app, no match, or unknown caller ID.
        """
        if not self.flask_app or not caller_number or caller_number == "Unknown":
            return ""
        try:
            with self.flask_app.app_context():
                from models import Customer
                customer = Customer.find_by_phone(caller_number)
                if not customer:
                    logger.info(f"👤 Caller {caller_number} not found in customer records")
                    return (
                        "\n\nCALLER CONTEXT:\nThis caller's number is not linked to an "
                        "existing customer record. Do not claim to know their policy "
                        "or claim details — ask for their name and policy number if needed."
                    )

                ctx = customer.to_context_dict()
                logger.info(f"👤 Caller matched: {ctx['name']} "
                            f"({len(ctx['open_claims'])} open claims, "
                            f"{len(ctx['open_tickets'])} open tickets)")

                parts = [f"\n\nCALLER CONTEXT:\nThis caller is {ctx['name']}, "
                         f"an existing customer. Greet them by first name after they "
                         f"confirm identity, but do not blurt out personal details unprompted."]

                if ctx["policies"]:
                    pol_lines = ", ".join(
                        f"{p['type']} policy {p['number']} ({p['status']})"
                        for p in ctx["policies"]
                    )
                    parts.append(f"Active policies: {pol_lines}.")

                if ctx["open_claims"]:
                    claim_lines = "; ".join(
                        f"{c['claim_number']} ({c['type']}, status: {c['status']}, "
                        f"filed {c['filed']})"
                        for c in ctx["open_claims"]
                    )
                    parts.append(f"Open claims: {claim_lines}.")
                else:
                    parts.append("No open claims on file.")

                if ctx["open_tickets"]:
                    ticket_lines = "; ".join(
                        f"{t['ticket_number']} — {t['subject']} (status: {t['status']})"
                        for t in ctx["open_tickets"]
                    )
                    parts.append(f"Open support tickets: {ticket_lines}.")

                parts.append(
                    "If they ask about a claim or ticket status, use the information "
                    "above. If they ask about something not listed here, say you don't "
                    "have that on file rather than guessing."
                )
                return "\n".join(parts)

        except Exception as e:
            logger.error(f"Caller context lookup error: {e}")
            return ""

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _db_call_start(self, call_id, caller):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                db.session.add(Call(
                    call_id=call_id, caller_number=caller,
                    status="active", started_at=datetime.utcnow(),
                ))
                db.session.commit()
        except Exception as e:
            logger.error(f"DB call start error: {e}")

    def _db_call_error(self, call_id):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                c = Call.query.filter_by(call_id=call_id).first()
                if c:
                    c.status = "error"
                    c.ended_at = datetime.utcnow()
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB call error log: {e}")

    def _db_call_end(self, sess: "CallSession"):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                c = Call.query.filter_by(call_id=sess.channel_id).first()
                if c:
                    c.status    = "escalated" if sess.escalated else "completed"
                    c.escalated = sess.escalated
                    c.ended_at  = datetime.utcnow()
                    if c.started_at:
                        c.duration_seconds = int((c.ended_at - c.started_at).total_seconds())
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB call end error: {e}")

    def _default_prompt(self):
        return (
            "You are Ari, a friendly and knowledgeable phone assistant for Jubilee Insurance Kenya. "
            "You have detailed knowledge of our products: motor, medical, life, last expense, "
            "home, and travel insurance, plus claims processes and payment methods. "
            "OUTPUT LANGUAGE — ABSOLUTE RULE: Your output is ALWAYS English. No exceptions. "
            "Even if the caller speaks Swahili or any other language, reply in English only. "
            "RESPONSE LENGTH — CRITICAL FOR PHONE: "
            "Keep every response to 2 sentences maximum — roughly 30 to 50 words, hard limit 60. "
            "This is a phone call: give one clear piece of information, then ask ONE short follow-up question. "
            "Never dump everything you know at once. Reveal information one layer at a time across turns. "
            "RULES: "
            "(1) 2 sentences max. 30–50 words. Hard limit: 60 words. "
            "(2) Never read bullet points — one fact at a time, spoken naturally. "
            "(3) Be warm and conversational, like a Kenyan insurance agent on the phone. "
            "(4) If asked broadly about products, name them in one sentence and ask which they want to know more about. "
            "(5) Use specific details — KES amounts, M-PESA paybill numbers — one detail per turn only. "
            "(6) Never say you are an AI. "
            "(7) Never use filler like 'Certainly', 'Of course', 'Absolutely', or 'Great question'. "
            "(8) Only transfer if the caller EXPLICITLY asks to speak to a person or agent. "
            "(9) When transferring say exactly: 'Let me transfer you to one of our agents right away.'"
        )


# ── CallSession ───────────────────────────────────────────────────────────────

class CallSession:
    """Manages one inbound call: RTP ↔ Azure Voice Live WebSocket."""

    TRANSFER_KEYWORDS = {
        "speak to", "talk to", "human", "person", "agent",
        "representative", "manager", "supervisor", "someone else",
        "transfer", "escalate", "real person", "actual person",
        "real agent", "sales agent", "customer service",
    }
    AI_TRANSFER_PHRASES = [
        "transfer you to", "transferring you to", "connect you to",
        "put you through to", "one of our agents", "speak to an agent",
    ]

    def __init__(self, *, channel, ari_client, ari_app, azure_ws_url,
                 azure_api_key, voice_name, voice_type, system_prompt,
                 rtp_port, flask_app, caller_number=None):
        self.channel        = channel
        self.channel_id     = channel.id
        self.ari_client     = ari_client
        self.ari_app        = ari_app
        self.azure_ws_url   = azure_ws_url
        self.azure_api_key  = azure_api_key
        self.voice_name     = voice_name
        self.voice_type     = voice_type
        self.system_prompt  = system_prompt
        self.rtp_port       = rtp_port
        self.flask_app      = flask_app
        self.caller_number  = caller_number
        self._customer_id   = None  # resolved lazily on first ticket creation

        self.caller_hung_up     = False
        self.escalated          = False
        self._closed            = False
        self._greeting_sent     = False
        self._ai_buf            = ""

        self._udp_sock          = None
        self._asterisk_addr     = None
        self._rtp_seq           = 0
        self._rtp_ts            = 0
        self._rtp_ssrc          = 0xDEADBEEF

        self._azure_ws          = None
        self._bridge_id         = None
        self._ext_channel_id    = None

        self._to_azure    = asyncio.Queue(maxsize=500)
        self._to_asterisk = asyncio.Queue(maxsize=500)
        self._ratecv_down = None   # 24kHz → 8kHz state

    # ── Main run ──────────────────────────────────────────────────────────────

    async def run(self):
        try:
            await self.channel.answer()
            logger.info(f"✅ [{self.channel_id[:12]}] Answered")
            await asyncio.sleep(0.1)  # reduced: just enough for channel to settle

            # Create bridge
            bridge = await self.ari_client.bridges.create(type="mixing")
            self._bridge_id = bridge.id
            await bridge.addChannel(channel=self.channel_id)
            logger.info(f"🌉 [{self.channel_id[:12]}] Bridge: {self._bridge_id}")

            # Bind UDP socket for RTP (blocking with timeout for clean shutdown)
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind((RTP_HOST, self.rtp_port))
            self._udp_sock.settimeout(0.05)
            logger.info(f"🔌 [{self.channel_id[:12]}] RTP port {self.rtp_port}")

            # Create ExternalMedia channel
            ext = await self.ari_client.channels.externalMedia(
                app             = self.ari_app,
                external_host   = f"127.0.0.1:{self.rtp_port}",
                format          = "ulaw",
                encapsulation   = "rtp",
                transport       = "udp",
                connection_type = "client",
                direction       = "both",
            )
            self._ext_channel_id = ext.id
            logger.info(f"📡 [{self.channel_id[:12]}] ExternalMedia: {self._ext_channel_id}")

            # Add ExternalMedia to bridge (with retry — channel may not be ready yet)
            for attempt in range(1, 6):
                try:
                    await bridge.addChannel(channel=self._ext_channel_id)
                    if attempt > 1:
                        logger.info(f"🌉 [{self.channel_id[:12]}] ExtMedia added on attempt {attempt}")
                    break
                except HTTPUnprocessableEntity:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(0.2)

            # Connect to Azure Voice Live
            await self._connect_azure()

            # Run all loops concurrently
            await asyncio.gather(
                self._rtp_recv_loop(),
                self._rtp_send_loop(),
                self._azure_recv_loop(),
                self._azure_send_loop(),
                return_exceptions=True,
            )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"❌ [{self.channel_id[:12]}] run() error: {e}", exc_info=True)
        finally:
            await self.close()

    # ── Azure connection & session config ─────────────────────────────────────

    async def _connect_azure(self):
        self._azure_ws = await websockets.connect(
            self.azure_ws_url,
            ping_interval = 20,
            ping_timeout  = 30,
            **_ws_connect_kwargs(self.azure_api_key),
        )
        logger.info(f"🔗 [{self.channel_id[:12]}] Azure WS connected")
        logger.info(f"   {self.azure_ws_url}")

        # Send session config — one message, only documented fields
        await self._azure_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": self.system_prompt,
                "modalities":   ["text", "audio"],
                "voice": {
                    "name": self.voice_name,
                    "type": self.voice_type,
                },
                # ── Audio formats ────────────────────────────────────────────
                # Asterisk sends μ-law 8kHz. We decode to PCM16 and tell Azure
                # it's 8kHz — no resampling needed on the inbound path.
                # Azure always outputs PCM16 @ 24kHz; we resample on output.
                "input_audio_format":        "pcm16",
                "output_audio_format":       "pcm16",
                "input_audio_sampling_rate": AZURE_IN_RATE,   # 8000
                # ── Noise & echo ─────────────────────────────────────────────
                "input_audio_noise_reduction": {
                    "type": "azure_deep_noise_suppression"
                },
                "input_audio_echo_cancellation": {
                    "type": "server_echo_cancellation"
                },
                # ── STT ──────────────────────────────────────────────────────
                # en-KE: best accuracy for Kenyan English accents.
                # Do NOT leave blank — blank triggers multilingual mode which
                # causes the model to mirror the detected language.
                "input_audio_transcription": {
                    "model":    "azure-speech",
                    "language": "en-KE",
                },
                # ── VAD ──────────────────────────────────────────────────────
                "turn_detection": {
                    "type":                "azure_semantic_vad",
                    "threshold":           0.5,
                    "silence_duration_ms": 500,
                    "prefix_padding_ms":   300,
                    "remove_filler_words": True,
                    "interrupt_response":  True,
                    "create_response":     True,
                },
                "temperature": 0.7,
                "tools": [
                    {
                        "type": "function",
                        "name": "create_ticket",
                        "description": (
                            "Log a support ticket / issue for this caller. Use this when "
                            "the caller wants something tracked or followed up on but does "
                            "NOT explicitly ask to be transferred to a person right now — "
                            "e.g. a callback request, a billing dispute, a complaint, or "
                            "asking someone to look into a delayed claim. Confirm back to "
                            "the caller in one short sentence that you've logged it, "
                            "including you'll mention it's noted, after calling this."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "subject": {
                                    "type": "string",
                                    "description": "Short one-line summary of the issue, "
                                                    "e.g. 'Requesting callback about delayed motor claim'.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Fuller detail from the conversation: what "
                                                    "the caller said, any reference numbers mentioned.",
                                },
                                "category": {
                                    "type": "string",
                                    "enum": ["claims", "billing", "policy", "complaint",
                                             "callback_request", "general"],
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "normal", "high", "urgent"],
                                    "description": "urgent/high only if caller signals real "
                                                    "distress or a time-critical issue.",
                                },
                            },
                            "required": ["subject", "category"],
                        },
                    },
                ],
                "tool_choice": "auto",
            },
        }))
        logger.info(f"⚙️  [{self.channel_id[:12]}] Session config sent")
        logger.info(f"   Voice : {self.voice_name} | Audio in: PCM16 {AZURE_IN_RATE}Hz | VAD: semantic")

    # ── RTP receive: Asterisk → Azure queue ───────────────────────────────────

    async def _rtp_recv_loop(self):
        loop = asyncio.get_running_loop()
        while not self._closed:
            try:
                data, addr = await loop.run_in_executor(
                    None, self._udp_sock.recvfrom, 4096
                )
            except OSError:
                # Socket timeout (every 50ms) or closed — check flags and loop
                continue
            except Exception as e:
                if not self._closed:
                    logger.debug(f"RTP recv: {e}")
                break

            if self._asterisk_addr is None:
                self._asterisk_addr = addr
                logger.info(f"📻 [{self.channel_id[:12]}] Asterisk RTP: {addr[0]}:{addr[1]}")

            if len(data) <= RTP_HEADER_SIZE:
                continue

            # μ-law → PCM16 (no resampling — 8kHz in, 8kHz declared to Azure)
            try:
                pcm16 = audioop.ulaw2lin(data[RTP_HEADER_SIZE:], 2)
            except audioop.error:
                continue

            if not self._to_azure.full():
                self._to_azure.put_nowait(pcm16)

    # ── Azure send: queue → WebSocket ─────────────────────────────────────────

    async def _azure_send_loop(self):
        while not self._closed:
            try:
                chunk = await asyncio.wait_for(self._to_azure.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            if not _ws_open(self._azure_ws):
                break
            try:
                await self._azure_ws.send(json.dumps({
                    "type":  "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))
            except Exception as e:
                if not self._closed:
                    logger.debug(f"Azure send: {e}")
                break

    # ── Azure receive: events + audio back to Asterisk ────────────────────────

    async def _azure_recv_loop(self):
        while not self._closed:
            if not _ws_open(self._azure_ws):
                await asyncio.sleep(0.05)
                continue
            try:
                raw   = await asyncio.wait_for(self._azure_ws.recv(), timeout=0.5)
                event = json.loads(raw)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                if not self._closed:
                    logger.warning(f"⚠️  [{self.channel_id[:12]}] Azure WS closed")
                break
            except Exception as e:
                if not self._closed:
                    logger.debug(f"Azure recv: {e}")
                break

            await self._handle_azure_event(event)

    async def _handle_azure_event(self, event: dict):
        etype = event.get("type", "")

        if etype == "response.audio.delta":
            # Azure outputs PCM16 @ 24kHz → downsample to 8kHz → μ-law → Asterisk
            b64 = event.get("delta", "")
            if not b64:
                return
            pcm24 = base64.b64decode(b64)
            pcm8, self._ratecv_down = audioop.ratecv(
                pcm24, 2, 1, AZURE_OUT_RATE, ASTERISK_RATE, self._ratecv_down
            )
            ulaw = audioop.lin2ulaw(pcm8, 2)
            if not self._to_asterisk.full():
                self._to_asterisk.put_nowait(ulaw)

        elif etype == "session.created":
            logger.info(f"✅ [{self.channel_id[:12]}] Azure session: {event.get('session', {}).get('id', '')}")

        elif etype == "session.updated":
            # Trigger greeting immediately — no extra sleep.
            # session.updated confirms Azure accepted the config, so it's ready.
            logger.info(f"⚙️  [{self.channel_id[:12]}] Session updated")
            if not self._greeting_sent:
                self._greeting_sent = True
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

        elif etype == "input_audio_buffer.speech_started":
            logger.info(f"🎤 [{self.channel_id[:12]}] Speech started")

        elif etype == "input_audio_buffer.speech_stopped":
            logger.info(f"🔇 [{self.channel_id[:12]}] Speech stopped")

        elif etype == "conversation.item.input_audio_transcription.completed":
            text = event.get("transcript", "").strip()
            if text:
                logger.info(f"👤 [{self.channel_id[:12]}] Caller: {text}")
                self._db_transcript("caller", text)
                if not self.escalated and self._wants_transfer(text):
                    await self._escalate(text)

        elif etype in ("response.audio_transcript.delta",
                       "response.output_audio_transcript.delta"):
            self._ai_buf += event.get("delta", "")

        elif etype in ("response.audio_transcript.done",
                       "response.output_audio_transcript.done"):
            full = self._ai_buf.strip()
            self._ai_buf = ""
            if full:
                logger.info(f"🤖 [{self.channel_id[:12]}] AI: {full}")
                self._db_transcript("agent", full)
                if not self.escalated and any(p in full.lower() for p in self.AI_TRANSFER_PHRASES):
                    logger.info(f"🔀 [{self.channel_id[:12]}] AI announced transfer — 2.5s delay")
                    asyncio.create_task(self._delayed_escalate(full))

        elif etype == "error":
            logger.error(f"❌ Azure error: {event.get('error', event)}")

        elif etype == "response.function_call_arguments.done":
            await self._handle_function_call(event)

        # All other events (response.created, response.done, content_part.*, etc.)
        # are silently ignored — we only care about the ones above.

    async def _handle_function_call(self, event: dict):
        """Azure Voice Live signals a completed function call — dispatch it,
        then send the result back so the model can continue the turn."""
        name      = event.get("name", "")
        call_id   = event.get("call_id", "")
        raw_args  = event.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}

        if name == "create_ticket":
            result = self._create_ai_ticket(args)
        else:
            result = {"error": f"Unknown function: {name}"}

        if not _ws_open(self._azure_ws):
            return
        try:
            await self._azure_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }))
            await self._azure_ws.send(json.dumps({"type": "response.create"}))
        except Exception as e:
            logger.error(f"Function call result send error: {e}")

    def _create_ai_ticket(self, args: dict) -> dict:
        """Create a Ticket row flagged is_ai_generated=True, linked to this
        call and (if resolvable) the calling customer."""
        if not self.flask_app:
            return {"error": "Ticketing unavailable in this environment"}

        subject     = (args.get("subject") or "Issue raised during AI call").strip()
        description = (args.get("description") or "").strip()
        category    = args.get("category", "general")
        priority    = args.get("priority", "normal")

        try:
            with self.flask_app.app_context():
                from models import db, Customer, Ticket, Call

                customer = None
                if self.caller_number:
                    customer = Customer.find_by_phone(self.caller_number)

                call = Call.query.filter_by(call_id=self.channel_id).first()

                if not customer:
                    # No matching customer record — still log the ticket so nothing
                    # is lost, but make that obvious to whoever reviews it.
                    logger.warning(
                        f"🎫 [{self.channel_id[:12]}] create_ticket with no matched "
                        f"customer (caller={self.caller_number}) — ticket unlinked"
                    )
                    return {"error": "No customer record found for this caller number; "
                                      "ask for their full name and policy number, then "
                                      "let them know an agent will follow up."}

                ticket = Ticket(
                    customer_id            = customer.id,
                    call_id                = call.id if call else None,
                    ticket_number          = Ticket.generate_ticket_number(),
                    subject                = subject,
                    description            = description,
                    category               = category,
                    priority               = priority,
                    status                 = "open",
                    is_ai_generated        = True,
                )
                db.session.add(ticket)
                db.session.commit()

                logger.info(f"🎫 [{self.channel_id[:12]}] AI created ticket "
                            f"{ticket.ticket_number} for {customer.full_name} "
                            f"({category}/{priority})")

                return {
                    "success": True,
                    "ticket_number": ticket.ticket_number,
                    "message": f"Ticket {ticket.ticket_number} logged for {customer.full_name}.",
                }

        except Exception as e:
            logger.error(f"create_ticket error: {e}")
            return {"error": "Failed to log ticket"}

    # ── RTP send: Asterisk ← Azure queue ─────────────────────────────────────

    async def _rtp_send_loop(self):
        """
        Drain the to_asterisk queue, packetise into 160-byte μ-law frames,
        and send at 20ms intervals. sendto() on a UDP socket is fast enough
        to call directly without run_in_executor.
        """
        loop      = asyncio.get_running_loop()
        buf       = bytearray()
        next_tick = loop.time()

        while not self._closed:
            # Pull whatever is in the queue without blocking long
            try:
                chunk = await asyncio.wait_for(self._to_asterisk.get(), timeout=0.1)
                buf.extend(chunk)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

            # Drain any additional queued chunks without awaiting
            while not self._to_asterisk.empty():
                try:
                    buf.extend(self._to_asterisk.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # Send as many complete 20ms frames as we have
            while len(buf) >= ULAW_FRAME_BYTES:
                if not self._asterisk_addr or not self._udp_sock:
                    del buf[:ULAW_FRAME_BYTES]
                    continue

                payload = bytes(buf[:ULAW_FRAME_BYTES])
                del buf[:ULAW_FRAME_BYTES]

                # Build RTP packet
                pkt = struct.pack(
                    "!BBHII",
                    0x80,           # V=2, no padding/ext/CC
                    0x00,           # M=0, PT=0 (PCMU)
                    self._rtp_seq,
                    self._rtp_ts,
                    self._rtp_ssrc,
                ) + payload
                self._rtp_seq  = (self._rtp_seq + 1) & 0xFFFF
                self._rtp_ts  += ULAW_FRAME_BYTES

                # Pace: wait until next_tick
                now = loop.time()
                if next_tick > now:
                    await asyncio.sleep(next_tick - now)
                next_tick = max(loop.time(), next_tick) + FRAME_DURATION

                try:
                    self._udp_sock.sendto(pkt, self._asterisk_addr)
                except OSError as e:
                    if not self._closed:
                        logger.debug(f"RTP sendto: {e}")
                    return

    # ── Transfer / escalation ─────────────────────────────────────────────────

    def _wants_transfer(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in self.TRANSFER_KEYWORDS)

    async def _delayed_escalate(self, text: str, delay: float = 2.5):
        await asyncio.sleep(delay)
        if not self._closed and not self.escalated:
            await self._escalate(text)

    async def _escalate(self, text: str):
        if self.escalated:
            return
        self.escalated = True
        self._closed   = True

        intent    = self._classify_intent(text)
        dept      = self._get_dept(intent)
        dept_name = dept.name      if dept else "Support"
        dept_ext  = dept.extension if dept else "1005"

        logger.info(f"🔀 [{self.channel_id[:12]}] → {dept_name} ext {dept_ext} (intent: {intent})")

        # Remove ExternalMedia from bridge (stops AI audio immediately)
        if self._bridge_id and self._ext_channel_id:
            try:
                bridge = await self.ari_client.bridges.get(bridgeId=self._bridge_id)
                await bridge.removeChannel(channel=self._ext_channel_id)
            except Exception:
                pass

        # Tear down Azure WS and UDP
        await self._close_media()

        # Transfer caller via dialplan
        for ctx in ["from-internal", "default"]:
            try:
                await self.channel.continueInDialplan(
                    context=ctx, extension=dept_ext, priority=1
                )
                logger.info(f"✅ [{self.channel_id[:12]}] Transfer → {dept_ext} ({ctx})")
                return
            except Exception as e:
                logger.warning(f"continueInDialplan ctx={ctx}: {e}")

        logger.error(f"❌ [{self.channel_id[:12]}] All transfer attempts failed")
        try:
            await self.channel.hangup()
        except Exception:
            pass

    def _classify_intent(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ["buy", "quote", "new policy", "purchase", "sign up",
                                     "sales", "sale", "enroll"]):
            return "sales"
        if any(w in lower for w in ["claim", "accident", "damage", "report", "incident", "loss"]):
            return "claims"
        if any(w in lower for w in ["bill", "payment", "pay", "invoice", "mpesa",
                                     "premium", "renew"]):
            return "billing"
        return "support"

    def _get_dept(self, intent: str):
        if not self.flask_app:
            return None
        try:
            with self.flask_app.app_context():
                from models import Department, RoutingRule
                rule = (
                    RoutingRule.query
                    .filter_by(intent_type=intent, is_active=True)
                    .order_by(RoutingRule.priority.desc())
                    .first()
                )
                if rule and rule.department and rule.department.is_active:
                    return rule.department
                name_map = {"sales": "Sales", "claims": "Claims",
                            "billing": "Billing", "support": "Support"}
                dept = Department.query.filter_by(
                    name=name_map.get(intent, "Support"), is_active=True
                ).first()
                if dept:
                    return dept
                return (Department.query
                        .filter_by(is_active=True)
                        .order_by(Department.priority.desc())
                        .first())
        except Exception as e:
            logger.error(f"Dept lookup: {e}")
            return None

    # ── DB transcript ─────────────────────────────────────────────────────────

    def _db_transcript(self, speaker: str, text: str):
        if not self.flask_app or not text:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call, CallTranscript
                call = Call.query.filter_by(call_id=self.channel_id).first()
                if call:
                    db.session.add(CallTranscript(
                        call_id=call.id, speaker=speaker, text=text,
                        confidence=1.0, timestamp=datetime.utcnow(),
                    ))
                    db.session.commit()
        except Exception as e:
            logger.error(f"DB transcript: {e}")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def _close_media(self):
        """Close Azure WS and UDP socket."""
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

    async def close(self):
        if self._closed and self.escalated:
            # Escalation already cleaned up everything; caller channel is live with FreePBX
            return

        self._closed = True
        await self._close_media()

        if self._ext_channel_id:
            try:
                await self.ari_client.channels.hangup(channelId=self._ext_channel_id)
            except Exception:
                pass
            self._ext_channel_id = None

        if self._bridge_id:
            try:
                await self.ari_client.bridges.destroy(bridgeId=self._bridge_id)
            except Exception:
                pass
            self._bridge_id = None

        if not self.escalated and not self.caller_hung_up:
            try:
                await self.channel.hangup()
            except Exception:
                pass

        logger.info(f"🔒 [{self.channel_id[:12]}] Session closed")