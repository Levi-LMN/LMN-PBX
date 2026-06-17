# services/ari_agent.py
"""
ARI (Asterisk REST Interface) agent that bridges inbound phone calls to
Azure's Voice Live real-time GPT API, enabling a live AI voice assistant
(Ari) to answer calls for Jubilee Insurance Kenya.

Call flow overview:
  1. Asterisk receives an inbound SIP call and routes it into the "ai-agent" Stasis app.
  2. ARIAgent catches the StasisStart event and spawns a CallSession.
  3. CallSession answers the call, creates a mixing bridge, and instructs Asterisk to
     open an ExternalMedia UDP channel that forwards raw RTP audio to a local port.
  4. Audio received from Asterisk (μ-law 8 kHz) is decoded and upsampled to 16 kHz,
     then streamed over a WebSocket to Azure Voice Live.
  5. Azure streams back PCM16 audio at 24 kHz, which is downsampled to 8 kHz,
     re-encoded as μ-law, and sent back to Asterisk via RTP.
  6. Azure's semantic VAD detects when the caller finishes speaking and generates
     a GPT response; transcripts are stored in the DB in real time.
  7. If the caller (or the AI itself) triggers a transfer, the session escalates the
     call via Asterisk's dialplan to the appropriate department extension.

CRITICAL AUDIO SAMPLING NOTE:
  Azure Voice Live only accepts input_audio_sampling_rate of 16000 or 24000 Hz.
  Declaring 8000 Hz (the native Asterisk μ-law rate) causes Azure's VAD to receive
  a mis-described stream — it either silently drops frames or cannot detect speech
  boundaries, resulting in partial transcripts and indefinite response hangs.
  Solution: upsample 8 kHz → 16 kHz before sending to Azure (done in _RTPProtocol),
  and tell Azure the rate is 16000.

Full audio pipeline:
  Inbound  : μ-law 8kHz RTP → ulaw2lin → PCM16 8kHz → ratecv 8→16kHz → base64 → Azure
  Outbound : Azure PCM16 24kHz → ratecv 24→8kHz → lin2ulaw → μ-law RTP → Asterisk
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

# ── Constants ──────────────────────────────────────────────────────────────────

# Azure Voice Live API version and model identifier (used in the WebSocket URL).
AZURE_API_VERSION = "2025-10-01"
AZURE_MODEL       = "gpt-realtime"
# The Azure resource subdomain; e.g. "my-resource" → wss://my-resource.services.ai.azure.com/...
AZURE_HOST_SUFFIX = os.getenv("AZURE_VOICE_LIVE_HOST_SUFFIX", "services.ai.azure.com")

# Asterisk sends and receives audio at 8 kHz (standard telephony μ-law rate).
ASTERISK_RATE  = 8000
# We upsample to 16 kHz before sending to Azure because 8 kHz is not a supported
# input rate — Azure's VAD breaks silently when the declared rate doesn't match.
AZURE_IN_RATE  = 16000
# Azure always synthesises and returns audio at 24 kHz regardless of input rate.
AZURE_OUT_RATE = 24000

# RTP packets from Asterisk carry a 12-byte fixed header before the audio payload.
RTP_HEADER_SIZE  = 12
# One RTP frame = 20 ms of μ-law audio at 8 kHz = 160 bytes.
ULAW_FRAME_BYTES = 160
FRAME_DURATION   = 0.020  # 20 ms in seconds; used to pace outbound RTP timing

# UDP port range reserved for ExternalMedia sockets (one port per concurrent call,
# incremented in steps of 2 so even/odd ports are never mixed across sessions).
RTP_HOST  = "0.0.0.0"
RTP_START = 20000
RTP_END   = 20100

# Instructions injected into the first Azure response.create call so the AI opens
# the conversation naturally rather than waiting for the caller to speak first.
GREETING_INSTRUCTIONS = (
    "Greet the caller warmly and ask how you can help. "
    "Keep it to one short natural sentence — do NOT read a scripted line. "
    "Example: 'Hi, thanks for calling Jubilee Insurance — how can I help you today?'"
)


# ── RTP DatagramProtocol ───────────────────────────────────────────────────────

class _RTPProtocol(asyncio.DatagramProtocol):
    """
    asyncio UDP protocol that receives raw RTP datagrams from Asterisk's
    ExternalMedia channel and converts them into 16 kHz PCM16 chunks
    suitable for streaming to Azure Voice Live.

    On each datagram:
      1. Strip the 12-byte RTP header.
      2. Decode μ-law payload → signed 16-bit PCM at 8 kHz.
      3. Upsample 8 kHz → 16 kHz using audioop.ratecv (stateful — preserves
         the resampling state between packets for clean interpolation).
      4. Push the resulting PCM chunk onto the async queue read by _azure_send_loop.

    Also captures the sender's (addr) on the first packet — this is the
    Asterisk RTP address we'll send outbound audio back to.
    """
    def __init__(self, queue: asyncio.Queue, on_addr_discovered):
        self._queue       = queue
        self._on_addr     = on_addr_discovered  # called once with the Asterisk RTP address
        self._addr_seen   = False
        self._ratecv_up   = None   # audioop.ratecv state for 8 kHz → 16 kHz upsampling
        self.transport    = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        # Capture the sender address on the first packet so we know where to
        # send outbound RTP back to Asterisk.
        if not self._addr_seen:
            self._addr_seen = True
            self._on_addr(addr)

        # Ignore malformed/empty packets that are only a header with no payload.
        if len(data) <= RTP_HEADER_SIZE:
            return
        try:
            # Strip the RTP header and decode μ-law → signed 16-bit PCM at 8 kHz.
            pcm8 = audioop.ulaw2lin(data[RTP_HEADER_SIZE:], 2)
            # Upsample from 8 kHz to 16 kHz. Azure's VAD only supports 16 kHz or
            # 24 kHz input; sending 8 kHz causes VAD to silently malfunction.
            pcm16k, self._ratecv_up = audioop.ratecv(
                pcm8, 2, 1, ASTERISK_RATE, AZURE_IN_RATE, self._ratecv_up
            )
        except audioop.error:
            # Drop corrupt frames rather than crashing the receive loop.
            return

        # Drop frames if the Azure send queue is full (back-pressure handling).
        if not self._queue.full():
            self._queue.put_nowait(pcm16k)

    def error_received(self, exc):
        pass

    def connection_lost(self, exc):
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _azure_ws_url(resource: str) -> str:
    """Build the Azure Voice Live WebSocket URL from the resource name."""
    return (
        f"wss://{resource}.{AZURE_HOST_SUFFIX}"
        f"/voice-live/realtime"
        f"?api-version={AZURE_API_VERSION}"
        f"&model={AZURE_MODEL}"
    )


def _ws_connect_kwargs(api_key: str) -> dict:
    """
    Return the correct header kwarg for websockets.connect() depending on the
    installed websockets library version. Older versions use 'extra_headers';
    newer versions renamed it to 'additional_headers'. We inspect the signature
    at runtime so this works across both without pinning a specific version.
    """
    try:
        params = inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):
        params = {}
    key = "additional_headers" if "additional_headers" in params else "extra_headers"
    return {key: {"api-key": api_key}}


def _ws_open(ws) -> bool:
    """
    Check whether a websockets connection is still open, regardless of the
    websockets library version. Different versions expose the state via either
    a .closed boolean or a .state enum — we handle both cases gracefully.
    Returns True if the socket appears open, False if it's None or closed.
    """
    if ws is None:
        return False
    if hasattr(ws, "closed"):
        return not ws.closed
    if hasattr(ws, "state"):
        import websockets.connection
        return ws.state == websockets.connection.State.OPEN
    return True


# ── ARIAgent ───────────────────────────────────────────────────────────────────

class ARIAgent:
    """
    Top-level service object. Connects to the Asterisk REST Interface (ARI) and
    listens for inbound call events. Spawns one CallSession per call to handle
    the full audio bridge lifecycle.

    Typical usage:
        agent = ARIAgent(config, flask_app)
        asyncio.run(agent.start())   # blocks until stopped
    """

    def __init__(self, app_config, flask_app=None):
        self.flask_app   = flask_app  # Flask app context used for DB access (lazy imports)
        self.running     = False
        self.total_calls = 0          # Lifetime call counter (for dashboard/stats)
        self.active_calls: dict[str, "CallSession"] = {}  # channel_id → session

        # ARI connection parameters (Asterisk REST API)
        self.ari_base     = app_config.get("ARI_BASE",     "http://localhost:8088")
        self.ari_username = app_config.get("ARI_USERNAME", "asterisk")
        self.ari_password = app_config.get("ARI_PASSWORD", "your_ari_password")
        self.ari_app      = app_config.get("ARI_APP",      "ai-agent")
        self.ari_url      = app_config.get("ARI_URL",      "http://localhost:8088/ari")

        # Azure Voice Live configuration
        self.azure_resource   = app_config.get("AZURE_VOICE_LIVE_RESOURCE", "")
        self.azure_api_key    = app_config.get("AZURE_SPEECH_KEY", "")
        self.azure_voice_name = app_config.get("AZURE_VOICE_NAME", "en-KE-AsiliaNeural")
        self.azure_voice_type = app_config.get("AZURE_VOICE_TYPE", "azure-standard")
        self.system_prompt    = app_config.get("DEFAULT_SYSTEM_PROMPT") or self._default_prompt()

        # Mutex + counter for allocating unique RTP ports to concurrent call sessions.
        self._port_lock     = asyncio.Lock()
        self._next_rtp_port = RTP_START
        self.ari_client     = None
        # If either Azure credential is missing, calls will be rejected at answer time.
        self._config_ok     = bool(self.azure_resource and self.azure_api_key)

    # ── Dashboard compatibility ────────────────────────────────────────────────
    # These properties expose ARIAgent state under the generic names the Flask
    # dashboard expects, so the dashboard doesn't need to know which agent type
    # is running.

    @property
    def ai_client(self):
        """Alias for ari_client — used by the dashboard's generic agent interface."""
        return self.ari_client

    @property
    def is_connected(self) -> bool:
        """True only when the agent is both marked running and has an active ARI connection."""
        return self.running and self.ari_client is not None

    @property
    def active_call_count(self) -> int:
        """Number of calls currently being handled."""
        return len(self.active_calls)

    def get_status(self) -> dict:
        return {
            "connected":       self.is_connected,
            "running":         self.running,
            "config_ok":       self._config_ok,
            "active_calls":    self.active_call_count,
            "total_calls":     self.total_calls,
            "azure_resource":  self.azure_resource or "not set",
            "voice_name":      self.azure_voice_name,
            "voice_type":      self.azure_voice_type,
            "model":           AZURE_MODEL,
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        self.running = True
        logger.info("=" * 60)
        logger.info(f"   Resource    : {self.azure_resource}")
        logger.info(f"   Voice       : {self.azure_voice_name} ({self.azure_voice_type})")
        logger.info(f"   Model       : {AZURE_MODEL}")
        logger.info(f"   Audio in    : PCM16 {AZURE_IN_RATE}Hz (upsampled from 8kHz)")
        logger.info(f"   Audio out   : PCM16 {AZURE_OUT_RATE}Hz → 8kHz μ-law → Asterisk")
        logger.info(f"   Greeting    : live via Azure (no cache)")
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

    # ── ARI events ─────────────────────────────────────────────────────────────

    def _on_stasis_start(self, event):
        """
        Fired when a channel enters the Stasis app. UnicastRTP channels are
        Asterisk's internal ExternalMedia plumbing — not real callers — so we
        skip those and only handle genuine inbound SIP/DAHDI channels.
        """
        name = event.get("channel", {}).get("name", "")
        if name.startswith("UnicastRTP/"):
            return
        asyncio.create_task(self._handle_call(event))

    def _on_stasis_end(self, event):
        """
        Fired when a channel leaves the Stasis app (caller hung up or was
        transferred). Marks the session as closed so all async loops exit cleanly.
        """
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
        """
        Fired when the caller presses hang-up before Asterisk has fully torn down
        the channel. We set the same flags as StasisEnd so the session doesn't try
        to hang up again in its finally block.
        """
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
        """
        Set up and run a CallSession for a new inbound call.
        Pulls the channel object from ARI, rejects the call if Azure isn't
        configured, allocates an RTP port, builds the system prompt (base +
        knowledge base + caller CRM context), and then hands off to sess.run().
        Cleans up from active_calls and writes final DB state when the call ends.
        """
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
        # Compose final system prompt: base personality + live KB entries + caller's CRM data
        prompt   = self.system_prompt + self._load_kb() + self._load_caller_context(caller)

        sess = CallSession(
            channel       = channel,
            ari_client    = self.ari_client,
            ari_app       = self.ari_app,
            azure_ws_url  = _azure_ws_url(self.azure_resource),
            azure_api_key = self.azure_api_key,
            voice_name    = self.azure_voice_name,
            voice_type    = self.azure_voice_type,
            system_prompt = prompt,
            rtp_port      = rtp_port,
            flask_app     = self.flask_app,
            caller_number = caller,
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
        """
        Thread-safe port allocator. Returns the next available even-numbered UDP
        port in the RTP_START–RTP_END range. Wraps back to RTP_START when exhausted.
        Ports are allocated in steps of 2 (RTP convention: even = data, odd = RTCP).
        """
        async with self._port_lock:
            port = self._next_rtp_port
            self._next_rtp_port = port + 2
            if self._next_rtp_port > RTP_END:
                self._next_rtp_port = RTP_START
            return port

    # ── Knowledge base ─────────────────────────────────────────────────────────

    def _load_kb(self) -> str:
        """
        Fetch all active KnowledgeBase entries from the DB (ordered by priority)
        and format them as a structured text block to append to the system prompt.
        Each entry is prefixed with its category so the AI can contextualise it.
        Also increments each entry's usage counter so admins can see what's referenced.
        Returns an empty string if no Flask app context is available or the table is empty.
        """
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

    # ── Caller context ─────────────────────────────────────────────────────────

    def _load_caller_context(self, caller_number: str) -> str:
        """
        Look up the caller's phone number in the Customer table and build a
        personalised context block for the system prompt.

        If matched: includes the customer's name, active policies, open claims,
        and open support tickets. The AI uses this to answer "what's the status
        of my claim?" without hallucinating.

        If not matched: instructs the AI not to claim knowledge of their account
        and to ask for their name and policy number instead.

        Returns empty string if no Flask app or if caller is "Unknown".
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

                if ctx.get("policies"):
                    pol_lines = ", ".join(
                        f"{p['type']} policy {p['number']} ({p['status']})"
                        for p in ctx["policies"]
                    )
                    parts.append(f"Active policies: {pol_lines}.")

                if ctx.get("open_claims"):
                    claim_lines = "; ".join(
                        f"{c['claim_number']} ({c['type']}, status: {c['status']}, "
                        f"filed {c['filed']})"
                        for c in ctx["open_claims"]
                    )
                    parts.append(f"Open claims: {claim_lines}.")
                else:
                    parts.append("No open claims on file.")

                if ctx.get("open_tickets"):
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

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _db_call_start(self, call_id, caller):
        """Insert a new Call row with status='active' when a call is answered."""
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
        """
        Mark an existing Call row as 'error' and record the end timestamp.
        Called when the session raises an unhandled exception.
        """
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
            logger.error(f"DB call error log: {e}")

    def _db_call_end(self, sess: "CallSession"):
        """
        Update the Call row on clean call termination. Sets status to 'escalated'
        if the call was transferred to a human agent, otherwise 'completed'.
        Also calculates and stores the total call duration in seconds.
        """
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


# ── CallSession ────────────────────────────────────────────────────────────────

class CallSession:
    """
    Manages the full lifecycle of a single inbound call:

      Asterisk RTP (μ-law 8kHz)
          ↓  _RTPProtocol.datagram_received → ulaw→PCM16 → upsample 8→16kHz → _to_azure queue
      _azure_send_loop → Azure Voice Live WebSocket
          ↓  Azure processes speech, runs GPT, streams PCM16 24kHz back
      _azure_recv_loop → downsample 24→8kHz → lin2ulaw → _to_asterisk queue
          ↓  _rtp_send_loop → UDP sendto → Asterisk ExternalMedia bridge

    Also handles: greeting injection, caller transcript capture, escalation/transfer,
    function calling (ticket creation), and DB transcript logging.
    """

    # Words/phrases in the caller's speech that should trigger a transfer to a human.
    TRANSFER_KEYWORDS = {
        "speak to", "talk to", "human", "person", "agent",
        "representative", "manager", "supervisor", "someone else",
        "transfer", "escalate", "real person", "actual person",
        "real agent", "sales agent", "customer service",
    }
    # Phrases the AI itself may say when it decides to transfer; used to detect
    # and trigger an escalation after the AI finishes its announcement sentence.
    AI_TRANSFER_PHRASES = [
        "transfer you to", "transferring you to", "connect you to",
        "put you through to", "one of our agents", "speak to an agent",
    ]

    def __init__(self, *, channel, ari_client, ari_app, azure_ws_url,
                 azure_api_key, voice_name, voice_type, system_prompt,
                 rtp_port, flask_app, caller_number=None):

        self.channel       = channel
        self.channel_id    = channel.id
        self.ari_client    = ari_client
        self.ari_app       = ari_app
        self.azure_ws_url  = azure_ws_url
        self.azure_api_key = azure_api_key
        self.voice_name    = voice_name
        self.voice_type    = voice_type
        self.system_prompt = system_prompt
        self.rtp_port      = rtp_port
        self.flask_app     = flask_app
        self.caller_number = caller_number

        self.caller_hung_up = False   # Set by StasisEnd/HangupRequest so close() skips re-hangup
        self.escalated      = False   # True after transfer is initiated
        self._closed        = False   # Master shutdown flag; all loops exit when True
        self._greeting_sent = False   # Ensures the opening greeting fires exactly once
        self._ai_buf        = ""      # Accumulates streaming AI transcript deltas

        # UDP socket used to send RTP back to Asterisk
        self._udp_sock       = None
        self._asterisk_addr  = None   # (host, port) discovered from first inbound RTP packet
        # RTP sequence number and timestamp for outbound packets (per RFC 3550)
        self._rtp_seq        = 0
        self._rtp_ts         = 0
        self._rtp_ssrc       = 0xDEADBEEF  # Fixed SSRC; arbitrary for a single-session stream

        self._azure_ws       = None   # Active WebSocket connection to Azure Voice Live
        self._bridge_id      = None   # ARI bridge ID (mixing bridge linking caller ↔ ExternalMedia)
        self._ext_channel_id = None   # ARI channel ID for the ExternalMedia endpoint

        # Async queues bridging the three concurrent loops. maxsize=500 provides back-pressure
        # without unbounded memory growth if one side runs faster than the other.
        self._to_azure    = asyncio.Queue(maxsize=500)   # PCM16 16kHz chunks → Azure
        self._to_asterisk = asyncio.Queue(maxsize=500)   # μ-law 8kHz frames → Asterisk
        self._ratecv_down = None   # audioop.ratecv state for 24 kHz → 8 kHz downsampling

        # When True, inbound caller audio is silently dropped (e.g. during AI playback overlap)
        self._suppress_caller_audio = False
        self._rtp_protocol: "_RTPProtocol | None" = None

    # ── Main run ───────────────────────────────────────────────────────────────

    async def run(self):
        """
        Full call lifecycle: answer → bridge → RTP socket → ExternalMedia →
        Azure WebSocket → concurrent audio loops → cleanup.
        """
        try:
            await self.channel.answer()
            logger.info(f"✅ [{self.channel_id[:12]}] Answered")
            # Brief pause to let Asterisk stabilise the channel before adding to bridge.
            await asyncio.sleep(0.1)

            # Create a mixing bridge so the caller channel and the ExternalMedia channel
            # share audio — the bridge mixes and forwards audio between both legs.
            bridge = await self.ari_client.bridges.create(type="mixing")
            self._bridge_id = bridge.id
            await bridge.addChannel(channel=self.channel_id)
            logger.info(f"🌉 [{self.channel_id[:12]}] Bridge: {self._bridge_id}")

            # Open a non-blocking UDP socket on the pre-allocated port.
            # We bind it before telling Asterisk the address so no packets are missed.
            loop = asyncio.get_running_loop()
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_sock.bind((RTP_HOST, self.rtp_port))
            self._udp_sock.setblocking(False)
            logger.info(f"🔌 [{self.channel_id[:12]}] RTP port {self.rtp_port}")

            # Ask Asterisk to open an ExternalMedia channel that forwards the caller's
            # μ-law 8 kHz audio as RTP UDP packets to our local socket.
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

            # Add the ExternalMedia channel into the bridge. Asterisk occasionally takes
            # a moment to prepare the channel, so we retry up to 5 times on 422 errors.
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

            # Hand the existing UDP socket to an asyncio DatagramProtocol for non-blocking
            # receives. _RTPProtocol decodes and upsamples each packet inline before
            # pushing PCM16 16 kHz frames onto the _to_azure queue.
            transport, self._rtp_protocol = await loop.create_datagram_endpoint(
                lambda: _RTPProtocol(
                    queue              = self._to_azure,
                    on_addr_discovered = self._on_asterisk_addr,
                ),
                sock=self._udp_sock,
            )

            # Open the Azure Voice Live WebSocket and send the session config.
            await self._connect_azure()

            # Run all three I/O loops concurrently. They exit when self._closed is True.
            await asyncio.gather(
                self._rtp_send_loop(),    # queue → UDP → Asterisk
                self._azure_recv_loop(),  # Azure WebSocket → queue
                self._azure_send_loop(),  # queue → Azure WebSocket
                return_exceptions=True,
            )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"❌ [{self.channel_id[:12]}] run() error: {e}", exc_info=True)
        finally:
            await self.close()

    # ── RTP addr discovery ─────────────────────────────────────────────────────

    def _on_asterisk_addr(self, addr):
        self._asterisk_addr = addr
        logger.info(f"📻 [{self.channel_id[:12]}] Asterisk RTP: {addr[0]}:{addr[1]}")

    async def _connect_azure(self):
        """
        Open the Azure Voice Live WebSocket and send the session.update configuration.
        This sets the AI's personality, audio format, VAD parameters, enabled tools,
        and the voice to use for TTS output. The session.updated event fired by Azure
        in response is used to trigger the opening greeting.

        Key settings:
          - input_audio_sampling_rate: MUST be 16000 or 24000 — NOT 8000.
            We send upsampled 16 kHz audio and declare that rate here.
          - turn_detection: azure_semantic_vad understands sentence boundaries
            so responses fire naturally at the end of a complete thought.
          - tools: exposes create_ticket so Azure can call it as a function when
            the caller wants something tracked without being transferred.
        """
        self._azure_ws = await websockets.connect(
            self.azure_ws_url,
            ping_interval = 20,
            ping_timeout  = 60,
            close_timeout = 5,
            **_ws_connect_kwargs(self.azure_api_key),
        )
        logger.info(f"🔗 [{self.channel_id[:12]}] Azure WS connected")
        logger.info(f"   {self.azure_ws_url}")

        await self._azure_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions":  self.system_prompt,
                "modalities":    ["text", "audio"],
                "voice": {
                    "name": self.voice_name,
                    "type": self.voice_type,
                },
                "input_audio_format":  "pcm16",
                "output_audio_format": "pcm16",
                # Declare 16000 Hz — the rate of our upsampled audio. Azure only supports
                # 16000 or 24000 Hz; 8000 Hz causes VAD to malfunction silently.
                "input_audio_sampling_rate": AZURE_IN_RATE,
                "input_audio_noise_reduction": {
                    "type": "azure_deep_noise_suppression"
                },
                "input_audio_echo_cancellation": {
                    "type": "server_echo_cancellation"
                },
                "input_audio_transcription": {
                    "model":    "azure-speech",
                    "language": "en-KE",
                },
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
                            "Log a support ticket for this caller. Use when they want "
                            "something tracked or followed up on but do NOT ask to be "
                            "transferred right now — e.g. callback request, billing dispute, "
                            "complaint, or asking someone to look into a delayed claim. "
                            "Confirm back in one short sentence that you've logged it."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "subject": {
                                    "type":        "string",
                                    "description": "Short one-line summary of the issue.",
                                },
                                "description": {
                                    "type":        "string",
                                    "description": "Fuller detail from the conversation.",
                                },
                                "category": {
                                    "type": "string",
                                    "enum": ["claims", "billing", "policy", "complaint",
                                             "callback_request", "general"],
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "normal", "high", "urgent"],
                                    "description": "urgent/high only for time-critical issues.",
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
        logger.info(
            f"   Voice: {self.voice_name} | "
            f"Audio in: PCM16 {AZURE_IN_RATE}Hz (up from 8kHz) | VAD: azure_semantic_vad"
        )

    # ── Azure send: queue → WebSocket ──────────────────────────────────────────

    async def _azure_send_loop(self):
        """
        Drain PCM16 16 kHz chunks from _to_azure and forward them to Azure as
        base64-encoded input_audio_buffer.append messages.
        When _suppress_caller_audio is True (e.g. during barge-in handling),
        frames are dequeued and discarded so the queue doesn't back up.
        """
        while not self._closed:
            try:
                chunk = await asyncio.wait_for(self._to_azure.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            if self._suppress_caller_audio:
                continue

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

    # ── Azure receive: events + audio → Asterisk ───────────────────────────────

    async def _azure_recv_loop(self):
        """
        Receive JSON events from Azure Voice Live and dispatch them to
        _handle_azure_event. Handles WebSocket disconnection by setting
        _closed=True and scheduling a call hangup.
        Uses a 1-second timeout on recv() so the loop can check _closed
        between Azure messages without blocking indefinitely.
        """
        while not self._closed:
            if not _ws_open(self._azure_ws):
                if not self._closed:
                    logger.warning(f"⚠️  [{self.channel_id[:12]}] Azure WS lost — ending session")
                    self._closed = True
                break
            try:
                raw   = await asyncio.wait_for(self._azure_ws.recv(), timeout=1.0)
                event = json.loads(raw)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed as e:
                if not self._closed:
                    logger.warning(
                        f"⚠️  [{self.channel_id[:12]}] Azure WS closed "
                        f"(code={e.code} reason={e.reason!r}) — hanging up call"
                    )
                    self._closed = True
                    asyncio.create_task(self._hangup_on_azure_drop())
                break
            except Exception as e:
                if not self._closed:
                    logger.warning(f"⚠️  [{self.channel_id[:12]}] Azure recv error: {e} — hanging up call")
                    self._closed = True
                    asyncio.create_task(self._hangup_on_azure_drop())
                break

            await self._handle_azure_event(event)

    async def _handle_azure_event(self, event: dict):
        """
        Route each Azure Voice Live event type to the appropriate handler.

        Key event types:
          response.audio.delta        — streamed PCM16 24kHz TTS audio chunks
          session.created             — Azure confirmed the session opened
          session.updated             — Azure applied our session.update config;
                                        we fire the greeting here, exactly once
          input_audio_buffer.*        — VAD speech start/stop signals (logged only)
          conversation.item.input_audio_transcription.completed
                                      — caller's utterance fully transcribed;
                                        saved to DB and checked for transfer intent
          response.audio_transcript.* — streaming AI transcript deltas and final text
          error                       — Azure-side error; closes the session
          response.function_call_arguments.done — AI wants to call a tool (e.g. create_ticket)
        """
        etype = event.get("type", "")

        if etype == "response.audio.delta":
            b64 = event.get("delta", "")
            if not b64:
                return
            pcm24 = base64.b64decode(b64)
            # Downsample the 24 kHz TTS audio to 8 kHz for Asterisk's μ-law pipeline.
            pcm8, self._ratecv_down = audioop.ratecv(
                pcm24, 2, 1, AZURE_OUT_RATE, ASTERISK_RATE, self._ratecv_down
            )
            ulaw = audioop.lin2ulaw(pcm8, 2)

            if not self._to_asterisk.full():
                self._to_asterisk.put_nowait(ulaw)

        elif etype == "session.created":
            sess_id = event.get("session", {}).get("id", "")
            logger.info(f"✅ [{self.channel_id[:12]}] Azure session: {sess_id}")

        elif etype == "session.updated":
            logger.info(f"⚙️  [{self.channel_id[:12]}] Session updated")
            if not self._greeting_sent:
                self._greeting_sent = True
                await self._azure_ws.send(json.dumps({
                    "type":     "response.create",
                    "response": {"instructions": GREETING_INSTRUCTIONS},
                }))
                logger.info(f"👋 [{self.channel_id[:12]}] Live greeting triggered via Azure")

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
            err = event.get("error", event)
            logger.error(f"❌ [{self.channel_id[:12]}] Azure error: {err}")
            if not self._closed:
                self._closed = True

        elif etype == "response.function_call_arguments.done":
            await self._handle_function_call(event)

    # ── RTP send: queue → Asterisk ─────────────────────────────────────────────

    async def _rtp_send_loop(self):
        """
        Pull μ-law frames from _to_asterisk and send them back to Asterisk as
        properly formatted RTP packets.

        Pacing: we target 20 ms per frame (FRAME_DURATION). If the system falls
        more than 200 ms behind real time (e.g. after a burst), we reset next_tick
        to now rather than trying to catch up, which would cause audible glitches.

        RTP packet format (RFC 3550):
          0x80       = version 2, no padding, no extension, no CSRC
          0x00       = PT=0 (μ-law/PCMU), marker bit off
          seq (16b)  = monotonically incrementing sequence number (wraps at 65535)
          ts  (32b)  = timestamp incremented by 160 per frame (160 samples @ 8 kHz = 20 ms)
          ssrc(32b)  = fixed arbitrary synchronisation source identifier
          payload    = 160 bytes of μ-law audio
        """
        loop      = asyncio.get_running_loop()
        buf       = bytearray()
        next_tick = loop.time()

        while not self._closed:
            try:
                chunk = await asyncio.wait_for(self._to_asterisk.get(), timeout=0.1)
                buf.extend(chunk)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

            # Drain without waiting — pull any additional frames already queued
            # to reduce latency when Azure is producing audio faster than 20ms/frame.
            while not self._to_asterisk.empty():
                try:
                    buf.extend(self._to_asterisk.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # Send complete 20ms frames only; partial frames stay buffered.
            while len(buf) >= ULAW_FRAME_BYTES:
                if not self._asterisk_addr or not self._udp_sock:
                    del buf[:ULAW_FRAME_BYTES]
                    continue

                payload = bytes(buf[:ULAW_FRAME_BYTES])
                del buf[:ULAW_FRAME_BYTES]

                pkt = struct.pack(
                    "!BBHII",
                    0x80, 0x00,
                    self._rtp_seq,
                    self._rtp_ts,
                    self._rtp_ssrc,
                ) + payload
                self._rtp_seq  = (self._rtp_seq + 1) & 0xFFFF
                self._rtp_ts  += ULAW_FRAME_BYTES

                # Pace output to 20 ms intervals. If we've drifted more than
                # 200 ms behind (e.g. after a pause or burst), reset to now
                # so we don't try to send a backlog of frames all at once.
                now = loop.time()
                if next_tick < now - 0.2:
                    next_tick = now
                if next_tick > now:
                    await asyncio.sleep(next_tick - now)
                next_tick = max(loop.time(), next_tick) + FRAME_DURATION

                try:
                    self._udp_sock.sendto(pkt, self._asterisk_addr)
                except OSError as e:
                    if not self._closed:
                        logger.debug(f"RTP sendto: {e}")
                    return

    # ── Transfer / escalation ──────────────────────────────────────────────────

    def _wants_transfer(self, text: str) -> bool:
        """Return True if the caller's transcript contains an explicit transfer keyword."""
        lower = text.lower()
        return any(kw in lower for kw in self.TRANSFER_KEYWORDS)

    async def _delayed_escalate(self, text: str, delay: float = 2.5):
        """
        Wait briefly after the AI announces a transfer before actually executing it,
        so the caller hears the AI finish its sentence before the call is moved.
        """
        await asyncio.sleep(delay)
        if not self._closed and not self.escalated:
            await self._escalate(text)

    async def _escalate(self, text: str):
        """
        Transfer the caller to a human agent in the appropriate department.

        Steps:
          1. Classify the caller's intent from the transcript to pick a department.
          2. Look up the department's Asterisk extension in the DB.
          3. Remove the ExternalMedia channel from the bridge (stops AI audio).
          4. Close the Azure WebSocket and UDP socket.
          5. Use continueInDialplan to hand the caller's channel to the target extension.
             Tries 'from-internal' first, falls back to 'default' context.
          6. If all transfer attempts fail, hangs up rather than leaving the caller stuck.
        """
        if self.escalated:
            return
        self.escalated = True
        self._closed   = True

        intent    = self._classify_intent(text)
        dept      = self._get_dept(intent)
        dept_name = dept.name      if dept else "Support"
        dept_ext  = dept.extension if dept else "1005"

        logger.info(f"🔀 [{self.channel_id[:12]}] → {dept_name} ext {dept_ext} (intent: {intent})")

        if self._bridge_id and self._ext_channel_id:
            try:
                bridge = await self.ari_client.bridges.get(bridgeId=self._bridge_id)
                await bridge.removeChannel(channel=self._ext_channel_id)
            except Exception:
                pass

        await self._close_media()

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
        """
        Simple keyword-based intent classifier. Maps the caller's words to one
        of four routing categories: sales, claims, billing, or support (default).
        Used by _escalate to pick which department extension to dial.
        """
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
        """
        Look up the best matching Department for the given intent type.

        Priority order:
          1. Active RoutingRule with the highest priority for this intent.
          2. Department whose name matches the intent (e.g. "Claims" for "claims").
          3. Any active Department with the highest priority (last-resort fallback).
        Returns None if no Flask app context is available or the DB query fails.
        """
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

    # ── Function calls ─────────────────────────────────────────────────────────

    async def _handle_function_call(self, event: dict):
        """
        Execute a tool call triggered by the AI and return the result to Azure.
        Azure sends response.function_call_arguments.done when it wants to invoke
        a tool. We run the function, then send a conversation.item.create message
        containing the result, followed by response.create to resume the AI's turn.
        Currently the only registered tool is 'create_ticket'.
        """
        name     = event.get("name", "")
        call_id  = event.get("call_id", "")
        raw_args = event.get("arguments", "{}")
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
                    "type":    "function_call_output",
                    "call_id": call_id,
                    "output":  json.dumps(result),
                },
            }))
            await self._azure_ws.send(json.dumps({"type": "response.create"}))
        except Exception as e:
            logger.error(f"Function call result send error: {e}")

    def _create_ai_ticket(self, args: dict) -> dict:
        """
        Create a support Ticket in the database on behalf of the AI.
        Looks up the caller's Customer record by phone number first; if no match
        is found, returns an error instructing the AI to ask for identifying info
        instead of guessing. On success returns the ticket number for the AI to
        read back to the caller.
        """
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
                    logger.warning(
                        f"🎫 [{self.channel_id[:12]}] create_ticket — no customer "
                        f"matched for {self.caller_number}"
                    )
                    return {
                        "error": "No customer record found for this caller number; "
                                 "ask for their full name and policy number, then "
                                 "let them know an agent will follow up."
                    }

                ticket = Ticket(
                    customer_id     = customer.id,
                    call_id         = call.id if call else None,
                    ticket_number   = Ticket.generate_ticket_number(),
                    subject         = subject,
                    description     = description,
                    category        = category,
                    priority        = priority,
                    status          = "open",
                    is_ai_generated = True,
                )
                db.session.add(ticket)
                db.session.commit()

                logger.info(
                    f"🎫 [{self.channel_id[:12]}] Ticket {ticket.ticket_number} "
                    f"for {customer.full_name} ({category}/{priority})"
                )
                return {
                    "success":       True,
                    "ticket_number": ticket.ticket_number,
                    "message":       f"Ticket {ticket.ticket_number} logged for {customer.full_name}.",
                }

        except Exception as e:
            logger.error(f"create_ticket error: {e}")
            return {"error": "Failed to log ticket"}

    # ── DB transcript ──────────────────────────────────────────────────────────

    def _db_transcript(self, speaker: str, text: str):
        """
        Append a transcript line to the CallTranscript table for this call.
        speaker is either 'caller' or 'agent'. Only called when there is actual
        text (empty strings are skipped). No-ops if Flask context is unavailable.
        """
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

    # ── Cleanup ────────────────────────────────────────────────────────────────

    async def _close_media(self):
        """
        Tear down the Azure WebSocket and the UDP/RTP socket.
        Called both on normal call end and before a transfer (so Asterisk can
        take the channel back without the AI still sending audio into the bridge).
        Silently ignores errors — cleanup must not raise.
        """
        if self._azure_ws:
            try:
                await self._azure_ws.close()
            except Exception:
                pass
            self._azure_ws = None

        if self._rtp_protocol and self._rtp_protocol.transport:
            try:
                self._rtp_protocol.transport.close()
            except Exception:
                pass
            self._rtp_protocol = None
        elif self._udp_sock:
            try:
                self._udp_sock.close()
            except Exception:
                pass
        self._udp_sock = None

    async def close(self):
        """
        Full session teardown. Closes the Azure WS + RTP socket, destroys the
        Asterisk bridge, and hangs up the caller channel if the caller didn't
        already hang up and the call wasn't transferred.
        Guards against double-close with _closed and escalated flags.
        """
        if self._closed and self.escalated:
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