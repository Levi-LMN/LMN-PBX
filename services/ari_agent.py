# services/ari_agent.py
"""
ARI-based agent with Azure Speech LIVE STREAMING transcription.

Key upgrade over the original:
  - Audio is piped from Asterisk in real-time via an external media channel (ExternalMedia)
  - Azure PushAudioInputStream receives raw PCM frames continuously while the caller speaks
  - Transcription fires the moment speech ends — no "record full clip → download → upload" round-trip
  - Typical latency improvement: 2–5 seconds per conversational turn

Fixes applied (2026-06-14):
  1. ExternalMedia format changed from "ulaw" → "slin" (signed linear 8 kHz).
     Asterisk's ExternalMedia only sends raw RTP payloads; slin avoids codec
     transcoding issues that caused the "Internal Server Error" on some builds.
  2. Removed the redundant channelId= kwarg from externalMedia() — that param
     sets the *new* channel's id, not the source; omitting it lets Asterisk
     auto-assign and avoids a 500 on FreePBX 17 / Asterisk 20.
  3. Fixed "coroutine '_run_recognition' was never awaited" warning: the
     coroutine is now explicitly closed when we fall back to file mode before
     ExternalMedia starts.
  4. System prompt hardened with CRITICAL TTS constraints at the very top so
     the model can't bury them under its own helpfulness heuristics.
     max_tokens kept at 60 (enforced in code) to hard-cap verbosity.
  5. Fixed duplicate-call loop: _process_call() now filters out ExternalMedia
     channels (UnicastRTP/*, ExternalMedia/*, Local/*) and any "Up" channel
     with no caller number.  When externalMedia() is called with app="ai-agent"
     Asterisk fires a StasisStart for the new audio-fork leg, which was being
     processed as a second incoming call, spawning another ExternalMedia leg,
     and so on — causing every real call to appear as 8–10 ghost calls.
"""

import asyncio
import aioari
import os
import tempfile
import time
import requests
import logging
import hashlib
import json
import shutil
import subprocess
import threading
import socket
import struct
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize
from openai import AsyncAzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
from flask import Flask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live Streaming Transcriber
# ---------------------------------------------------------------------------

class LiveStreamTranscriber:
    """
    Streams raw 8 kHz 16-bit mono PCM audio to Azure Speech Services and
    returns the recognised text as soon as speech ends.

    How it integrates with Asterisk / ARI:
      1. We open a UDP socket that Asterisk sends RTP audio to via ExternalMedia.
      2. We strip the 12-byte RTP header and push the raw PCM payload into an
         Azure PushAudioInputStream.
      3. Azure fires a 'recognized' event the moment silence is detected.
      4. We stop the recogniser and return the transcript.

    FIX: format is now "slin" (signed linear PCM) instead of "ulaw".
    Asterisk's ExternalMedia with encapsulation="rtp" sends raw G.711 or slin
    depending on what the channel negotiated.  Requesting slin forces Asterisk
    to transcode to raw PCM before sending — which is exactly what Azure's
    PushAudioInputStream expects.  Using "ulaw" caused a 500 Internal Server
    Error on FreePBX 17 because the ARI ExternalMedia handler rejected the
    combination of ulaw + rtp encapsulation on that build.
    """

    SAMPLE_RATE = 8000
    CHANNELS = 1
    BITS_PER_SAMPLE = 16
    RTP_HEADER_SIZE = 12          # bytes to strip from each UDP packet
    UDP_PORT_RANGE = (20000, 20999)
    SILENCE_TIMEOUT = 2.0         # seconds of silence before we finalise
    MAX_LISTEN_DURATION = 10.0    # hard cap on how long we listen

    def __init__(self, speech_key: str, speech_region: str):
        if not speech_key or not speech_region:
            raise ValueError("Azure Speech key and region are required")
        self.speech_key = speech_key
        self.speech_region = speech_region

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen_and_transcribe(self) -> tuple[int, object]:
        """
        Open a UDP port, return (udp_port, coroutine).

        This is a regular (non-async) function so the returned coroutine is
        not nested inside another coroutine object.

        Usage:
            port, transcribe_coro = transcriber.listen_and_transcribe()
            # Tell Asterisk to send RTP to 127.0.0.1:<port>
            text, confidence, duration_ms = await transcribe_coro

        IMPORTANT: If you decide NOT to await the coroutine (e.g. because
        ExternalMedia failed), call transcribe_coro.close() to avoid the
        "coroutine was never awaited" RuntimeWarning.
        """
        udp_port = self._find_free_udp_port()
        coro = self._run_recognition(udp_port)
        return udp_port, coro

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_free_udp_port(self) -> int:
        for port in range(*self.UDP_PORT_RANGE):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No free UDP port found in range")

    async def _run_recognition(self, udp_port: int) -> tuple[str, str, int]:
        """
        Core recognition loop.  Runs in the calling coroutine's event loop.
        Azure SDK callbacks are synchronous, so we bridge them with asyncio Events.
        """
        loop = asyncio.get_running_loop()

        # ── Azure stream setup ──────────────────────────────────────────
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.SAMPLE_RATE,
            bits_per_sample=self.BITS_PER_SAMPLE,
            channels=self.CHANNELS,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_cfg = speechsdk.audio.AudioConfig(stream=push_stream)

        speech_cfg = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
        )
        speech_cfg.speech_recognition_language = "en-US"
        speech_cfg.enable_dictation()

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_cfg,
            audio_config=audio_cfg,
        )

        # ── Result containers ────────────────────────────────────────────
        result_text: list[str] = []
        done_event = asyncio.Event()
        start_time = time.monotonic()

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text.strip()
                if text:
                    result_text.append(text)
                    logger.debug(f"[STT live] recognised: {text}")
            loop.call_soon_threadsafe(done_event.set)

        def on_canceled(evt):
            logger.warning(f"[STT live] canceled: {evt.cancellation_details.reason}")
            loop.call_soon_threadsafe(done_event.set)

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)

        # ── UDP receiver ────────────────────────────────────────────────
        # IMPORTANT: bind the socket BEFORE starting the recognizer and BEFORE
        # telling Asterisk to connect.  If we bind after externalMedia() is
        # called, Asterisk may start sending RTP packets before the socket is
        # ready and those early frames are silently dropped — Azure gets silence.
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sock.bind(("0.0.0.0", udp_port))
        udp_sock.setblocking(False)

        recognizer.start_continuous_recognition()

        stop_receiver = threading.Event()
        # Startup grace period: Asterisk needs ~1-2 s to negotiate the RTP
        # session after externalMedia() returns.  We don't start the silence
        # timeout until we've received at least one packet OR the grace period
        # has passed.  Without this, the 2-second silence timeout fires before
        # any audio arrives and we close the stream immediately.
        STARTUP_GRACE = 4.0   # seconds to wait for the first RTP packet

        def rtp_receiver():
            """Runs in a daemon thread; pulls RTP packets and feeds PCM to Azure."""
            first_packet_received = False
            last_packet_time = time.monotonic()
            startup_time = time.monotonic()

            while not stop_receiver.is_set():
                try:
                    data, _ = udp_sock.recvfrom(4096)
                    last_packet_time = time.monotonic()
                    first_packet_received = True
                    if len(data) > self.RTP_HEADER_SIZE:
                        pcm = data[self.RTP_HEADER_SIZE:]   # strip RTP header
                        push_stream.write(pcm)
                except BlockingIOError:
                    now = time.monotonic()
                    # Don't apply silence timeout until we've received at least
                    # one packet AND the startup grace period has elapsed.
                    if first_packet_received and (now - last_packet_time > self.SILENCE_TIMEOUT):
                        logger.debug("[STT live] silence timeout in UDP receiver")
                        push_stream.close()
                        break
                    if not first_packet_received and (now - startup_time > STARTUP_GRACE + self.MAX_LISTEN_DURATION):
                        logger.warning("[STT live] no RTP packets received — Asterisk may not be sending audio")
                        push_stream.close()
                        break
                except Exception as exc:
                    logger.debug(f"[STT live] UDP recv error: {exc}")
                    break
                time.sleep(0.005)

        recv_thread = threading.Thread(target=rtp_receiver, daemon=True)
        recv_thread.start()

        # ── Wait for result or hard timeout ─────────────────────────────
        try:
            await asyncio.wait_for(done_event.wait(), timeout=self.MAX_LISTEN_DURATION)
        except asyncio.TimeoutError:
            logger.debug("[STT live] max listen duration reached")

        # ── Clean up ────────────────────────────────────────────────────
        stop_receiver.set()
        try:
            push_stream.close()
        except Exception:
            pass
        recognizer.stop_continuous_recognition()
        udp_sock.close()
        recv_thread.join(timeout=1.0)

        duration_ms = int((time.monotonic() - start_time) * 1000)
        text = " ".join(result_text).strip()
        confidence = "high" if text else "low"

        logger.info(f"[STT live] final text='{text}' ({duration_ms} ms)")
        return text, confidence, duration_ms


# ---------------------------------------------------------------------------
# Fallback: file-based transcriber
# ---------------------------------------------------------------------------

class AzureSpeechTranscriber:
    """
    Original file-based transcriber.  Still used as a fallback if ExternalMedia
    is not available or the UDP stream fails to start.
    """

    def __init__(self, speech_key: str, speech_region: str):
        if not speech_key or not speech_region:
            raise ValueError("Azure Speech key and region required")
        self.config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region,
        )
        self.config.speech_recognition_language = "en-US"

    async def transcribe(self, audio_file: str) -> tuple[str, str]:
        try:
            if os.path.getsize(audio_file) < 4000:
                return "", "low"

            processed = await self._preprocess(audio_file)
            audio_config = speechsdk.audio.AudioConfig(filename=processed)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.config,
                audio_config=audio_config,
            )

            result = await asyncio.get_running_loop().run_in_executor(
                None, recognizer.recognize_once
            )

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = result.text.strip()
                confidence = "high"
            else:
                text = ""
                confidence = "low"

            if processed != audio_file:
                try:
                    os.unlink(processed)
                except Exception:
                    pass

            return text, confidence

        except Exception as e:
            logger.error(f"[STT file] Transcription error: {e}")
            return "", "low"

    async def _preprocess(self, audio_file: str) -> str:
        try:
            audio = AudioSegment.from_file(audio_file)
            audio = normalize(audio).set_frame_rate(16000).set_channels(1).set_sample_width(2)
            processed = audio_file.replace(".wav", "_proc.wav")
            audio.export(processed, format="wav")
            return processed
        except Exception:
            return audio_file


# ---------------------------------------------------------------------------
# Sound cache / TTS
# ---------------------------------------------------------------------------

class SoundCache:
    """Cache for TTS audio."""

    def __init__(self, cache_dir, index_file, asterisk_sounds_dir,
                 azure_speech_key=None, azure_speech_region="eastus"):
        self.cache_dir = cache_dir
        self.index_file = index_file
        self.asterisk_sounds_dir = asterisk_sounds_dir
        self.azure_speech_key = azure_speech_key
        self.azure_speech_region = azure_speech_region
        self.index = self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            try:
                return json.load(open(self.index_file))
            except Exception:
                return {}
        return {}

    def _save_index(self):
        try:
            json.dump(self.index, open(self.index_file, "w"))
        except Exception:
            pass

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    async def get(self, text: str, file_access) -> tuple[str | None, float | None]:
        key = self._cache_key(text)

        if key in self.index and self.index[key].get("remote"):
            return self.index[key]["remote"], self.index[key].get("duration")

        local_path = await self._generate_tts(text, key)
        if not local_path:
            return None, None

        duration = self._get_duration(local_path)
        remote_path = file_access.copy_to_asterisk(local_path, f"c_{key}.wav")

        if remote_path:
            self.index[key] = {"remote": remote_path, "duration": duration}
            self._save_index()
            return remote_path, duration

        return local_path, duration

    async def _generate_tts(self, text: str, key: str) -> str | None:
        try:
            output_file = self.cache_dir / f"{key}.wav"
            if output_file.exists():
                return str(output_file)

            # Primary: Azure Neural TTS
            try:
                speech_config = speechsdk.SpeechConfig(
                    subscription=self.azure_speech_key,
                    region=self.azure_speech_region,
                )
                speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff8Khz16BitMonoPcm
                )
                audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_file))
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=speech_config,
                    audio_config=audio_config,
                )
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: synthesizer.speak_text_async(text).get()
                )
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    logger.debug(f"Azure TTS ok: {text[:40]}")
                    return str(output_file)
                logger.warning(f"Azure TTS failed ({result.reason}), falling back to gTTS")
            except Exception as e:
                logger.warning(f"Azure TTS error: {e}, falling back to gTTS")

            # Fallback: gTTS
            from gtts import gTTS
            temp_file = self.cache_dir / f"{key}_temp.mp3"
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: gTTS(text=text, lang="en", slow=False).save(str(temp_file))
            )
            audio = AudioSegment.from_file(str(temp_file))
            audio = normalize(audio).set_frame_rate(8000).set_channels(1).set_sample_width(2)
            audio.export(str(output_file), format="wav")
            try:
                temp_file.unlink()
            except Exception:
                pass
            return str(output_file)

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def _get_duration(self, file_path: str) -> float | None:
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
        except Exception:
            return None


# ---------------------------------------------------------------------------
# File system helper
# ---------------------------------------------------------------------------

class FileSystemAccess:
    """Direct file system access to Asterisk sounds dir."""

    def __init__(self, sounds_dir: str):
        self.sounds_dir = sounds_dir
        self.can_write = False
        self.use_sudo = False

    def test_access(self) -> bool:
        try:
            test_file = os.path.join(self.sounds_dir, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.unlink(test_file)
            self.can_write = True
            return True
        except PermissionError:
            try:
                result = subprocess.run(
                    ["sudo", "-n", "touch", os.path.join(self.sounds_dir, ".test_write")],
                    capture_output=True, timeout=2,
                )
                if result.returncode == 0:
                    subprocess.run(["sudo", "rm", os.path.join(self.sounds_dir, ".test_write")])
                    self.can_write = True
                    self.use_sudo = True
                    return True
            except Exception:
                pass
            logger.warning("No write access — run as asterisk or use sudo")
            return False

    def copy_to_asterisk(self, local_path: str, filename: str) -> str | None:
        try:
            dest_path = os.path.join(self.sounds_dir, filename)
            if self.use_sudo:
                subprocess.run(["sudo", "cp", local_path, dest_path], check=True)
                subprocess.run(["sudo", "chown", "asterisk:asterisk", dest_path], check=True)
                subprocess.run(["sudo", "chmod", "644", dest_path], check=True)
            else:
                shutil.copy2(local_path, dest_path)
                os.chmod(dest_path, 0o644)
            return f"custom/{filename.replace('.wav', '')}"
        except Exception as e:
            logger.error(f"File copy error: {e}")
            return None


# ---------------------------------------------------------------------------
# Call instance
# ---------------------------------------------------------------------------

class CallInstance:
    """
    Represents a single active call.

    Speech capture strategy (auto-selected):
      1. Live streaming via ExternalMedia + UDP (fastest — 2-5 s saved per turn)
      2. File-based record → download → transcribe  (fallback)

    ExternalMedia fix: we now pass format="slin" and do NOT pass channelId to
    the externalMedia() call.  On FreePBX 17 / Asterisk 20, passing channelId
    caused a 500 because the parameter is interpreted as the *new* channel's ID,
    not the source — Asterisk rejected the duplicate.  Omitting it lets Asterisk
    auto-generate a unique channel ID for the external media leg.
    """

    def __init__(self, channel, ari_client, ai_client, sound_cache, file_access,
                 transcriber, live_transcriber, system_prompt, deployment,
                 ari_url, ari_username, ari_password, flask_app=None):
        self.channel = channel
        self.ari_client = ari_client
        self.ai_client = ai_client
        self.sound_cache = sound_cache
        self.file_access = file_access
        self.transcriber = transcriber            # fallback: file-based
        self.live_transcriber = live_transcriber  # preferred: live streaming
        self.system_prompt = system_prompt
        self.deployment = deployment
        self.ari_url = ari_url
        self.ari_username = ari_username
        self.ari_password = ari_password
        self.flask_app = flask_app

        self.id = channel.id
        self.active = True
        self.user_hung_up = False
        self.escalated = False
        self.escalated_to_dept_id = None
        self.escalation_reason = None
        self.temp_files: list[str] = []
        self.turn_count = 0
        self.conversation = [{"role": "system", "content": system_prompt}]

        self._ext_media_channel = None
        self._ext_media_channel_id = None
        self._ext_bridge_id = None
        self._snoop_channel_id = None

    # ------------------------------------------------------------------
    # Live audio capture
    # ------------------------------------------------------------------

    async def _start_external_media(self, udp_port: int) -> bool:
        """
        Stream caller audio to our UDP listener via ExternalMedia bridged directly
        to the caller channel.

        CORRECT ARCHITECTURE (per official Asterisk docs and all working examples):

        For a Stasis IVR application where our bot IS the only party on the call,
        the correct pattern is simply:

          1. POST /ari/channels/externalMedia  → creates a UnicastRTP channel that
             sends audio to our UDP port (connection_type=client means Asterisk
             dials OUT to our UDP socket)
          2. POST /ari/bridges (mixing)
          3. POST /ari/bridges/{id}/addChannel  → add CALLER + EM channel

        The caller channel CAN be added to a bridge from within a Stasis app —
        the earlier "Channel not found" error was caused by passing channel IDs
        as a JSON body instead of query parameters. The fix (params=...) works.

        The Snoop+ExternalMedia pattern is for passively monitoring a call that
        is already bridged between two parties (e.g., an active conference). In
        our case the caller has no other party — there is nothing to snoop on.
        A snoop of an un-bridged channel receives silence.

        format="slin" = 8 kHz signed-linear PCM, matching Azure's
        AudioStreamFormat(samples_per_second=8000, bits_per_sample=16).
        """
        try:
            ari_base = self.ari_url.rstrip("/ari").rstrip("/")
            app_name = self.ari_app if hasattr(self, "ari_app") else "ai-agent"
            loop = asyncio.get_running_loop()

            # ── Step 1: create ExternalMedia channel ─────────────────────────
            em_resp = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{ari_base}/ari/channels/externalMedia",
                    json={
                        "app": app_name,
                        "external_host": f"127.0.0.1:{udp_port}",
                        "format": "slin",
                        "encapsulation": "rtp",
                        "transport": "udp",
                        "connection_type": "client",
                        "direction": "both",
                    },
                    auth=(self.ari_username, self.ari_password),
                    timeout=5,
                ),
            )
            if em_resp.status_code not in (200, 201):
                logger.warning(
                    f"[ExternalMedia] create failed HTTP {em_resp.status_code}: "
                    f"{em_resp.text[:200]} — falling back to record"
                )
                return False

            em_channel_id = em_resp.json().get("id")
            self._ext_media_channel_id = em_channel_id
            logger.info(f"[ExternalMedia] EM channel created id={em_channel_id} → UDP :{udp_port}")

            # ── Step 2: create a mixing bridge ───────────────────────────────
            bridge_resp = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{ari_base}/ari/bridges",
                    json={"type": "mixing"},
                    auth=(self.ari_username, self.ari_password),
                    timeout=5,
                ),
            )
            if bridge_resp.status_code not in (200, 201):
                logger.warning(
                    f"[ExternalMedia] bridge create failed HTTP {bridge_resp.status_code}: "
                    f"{bridge_resp.text[:200]} — falling back to record"
                )
                await self._stop_external_media()
                return False

            self._ext_bridge_id = bridge_resp.json().get("id")
            logger.info(f"[ExternalMedia] bridge created id={self._ext_bridge_id}")

            # ── Step 3: add caller + EM channel to bridge ────────────────────
            # channel IDs must be query params (not JSON body) — ARI parses the
            # query string; a JSON body field is treated as a literal channel name.
            # params={"channel": [id1, id2]} → ?channel=id1&channel=id2 ✓
            await asyncio.sleep(0.3)  # let EM channel finish entering Stasis

            add_resp = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{ari_base}/ari/bridges/{self._ext_bridge_id}/addChannel",
                    params={"channel": [self.id, em_channel_id]},
                    auth=(self.ari_username, self.ari_password),
                    timeout=5,
                ),
            )
            if add_resp.status_code not in (200, 204):
                logger.warning(
                    f"[ExternalMedia] addChannel failed HTTP {add_resp.status_code}: "
                    f"{add_resp.text[:200]} — falling back to record"
                )
                await self._stop_external_media()
                return False

            logger.info(
                f"[ExternalMedia] ready — caller {self.id[:12]} ↔ "
                f"EM {em_channel_id} → UDP :{udp_port}"
            )
            return True

        except Exception as e:
            logger.warning(f"[ExternalMedia] setup error: {e} — falling back to record")
            await self._stop_external_media()
            return False

    async def _stop_external_media(self):
        """Tear down the ExternalMedia channel and the mixing bridge."""
        ari_base = self.ari_url.rstrip("/ari").rstrip("/")
        loop = asyncio.get_running_loop()

        # Destroy the bridge (removes channels from it)
        bridge_id = getattr(self, "_ext_bridge_id", None)
        if bridge_id:
            try:
                await loop.run_in_executor(
                    None,
                    lambda: requests.delete(
                        f"{ari_base}/ari/bridges/{bridge_id}",
                        auth=(self.ari_username, self.ari_password),
                        timeout=3,
                    ),
                )
                logger.debug(f"[ExternalMedia] bridge {bridge_id} destroyed")
            except Exception:
                pass
            self._ext_bridge_id = None

        # Clean up any snoop channel if it exists from a previous attempt
        snoop_id = getattr(self, "_snoop_channel_id", None)
        if snoop_id:
            try:
                await loop.run_in_executor(
                    None,
                    lambda: requests.delete(
                        f"{ari_base}/ari/channels/{snoop_id}",
                        auth=(self.ari_username, self.ari_password),
                        timeout=3,
                    ),
                )
            except Exception:
                pass
            self._snoop_channel_id = None

        # Hang up ExternalMedia channel
        em_id = getattr(self, "_ext_media_channel_id", None)
        if em_id:
            try:
                await loop.run_in_executor(
                    None,
                    lambda: requests.delete(
                        f"{ari_base}/ari/channels/{em_id}",
                        auth=(self.ari_username, self.ari_password),
                        timeout=3,
                    ),
                )
                logger.debug(f"[ExternalMedia] channel {em_id} hungup")
            except Exception:
                pass
        self._ext_media_channel = None
        self._ext_media_channel_id = None

    async def listen(self) -> tuple[str, str]:
        """
        Capture caller speech and return (text, confidence).

        Tries live streaming first; falls back to file recording if
        ExternalMedia is unavailable.
        """
        if self.live_transcriber:
            return await self._listen_live()
        return await self._listen_file()

    async def _listen_live(self) -> tuple[str, str]:
        """Live streaming path — no file I/O."""
        if not await self.is_alive():
            return "", "low"

        try:
            t0 = time.monotonic()

            # Reserve a UDP port and get the recognition coroutine.
            # listen_and_transcribe() is NOT async — calling it directly avoids
            # nesting coroutines.
            udp_port, transcribe_coro = self.live_transcriber.listen_and_transcribe()

            # Tell Asterisk to stream audio to that port
            media_ok = await self._start_external_media(udp_port)
            if not media_ok:
                # FIX: explicitly close the coroutine so Python doesn't emit
                # "coroutine '_run_recognition' was never awaited" RuntimeWarning.
                transcribe_coro.close()
                self.live_transcriber = None   # disable for rest of this call
                return await self._listen_file()

            # Run recognition (blocks until speech ends or timeout)
            text, confidence, duration_ms = await transcribe_coro

            await self._stop_external_media()

            logger.info(
                f"[live STT] '{text}' | conf={confidence} | "
                f"total={int((time.monotonic()-t0)*1000)} ms"
            )
            return text, confidence

        except Exception as e:
            logger.error(f"[live STT] error: {e} — falling back to file mode")
            self.live_transcriber = None
            await self._stop_external_media()
            return await self._listen_file()

    # ------------------------------------------------------------------
    # Fallback: file-based recording
    # ------------------------------------------------------------------

    async def _listen_file(self) -> tuple[str, str]:
        """Original record → download → transcribe path."""
        if not await self.is_alive():
            return "", "low"

        audio_file = await self.record()
        if not audio_file:
            return "", "low"

        text, confidence = await self.transcriber.transcribe(audio_file)
        return text, confidence

    async def record(self, duration: int = 8, silence: float = 2.0) -> str | None:
        """
        Record audio clip via ARI stored recordings.

        FIX: The original implementation did `await asyncio.sleep(duration + 0.5)`
        unconditionally — an 8.5-second blind wait before stopping the recording.
        Combined with the fallback path being triggered on every call (due to the
        ExternalMedia bug), callers heard 8+ seconds of silence and hung up before
        the AI could respond.

        New approach: poll the ARI /recordings/live/{name} endpoint every 250 ms.
        Asterisk sets recording.state to "complete" when maxSilenceSeconds has
        elapsed or the caller hangs up.  We stop waiting as soon as that happens,
        which typically cuts the wait to 2–3 seconds (the silence detection window)
        instead of the full maxDurationSeconds.
        """
        if not await self.is_alive():
            return None

        name = f"rec_{self.id}_{int(time.time() * 1000)}"
        try:
            recording = await self.channel.record(
                name=name,
                format="wav",
                maxDurationSeconds=duration,
                maxSilenceSeconds=silence,
                ifExists="overwrite",
                terminateOn="none",
            )

            # Poll until recording finishes naturally (silence timeout) or we
            # hit the hard cap, whichever comes first.
            deadline = time.monotonic() + duration + 1.0
            ari_base = self.ari_url.rstrip("/ari").rstrip("/")
            rec_url = f"{ari_base}/ari/recordings/live/{name}"
            loop = asyncio.get_running_loop()

            while time.monotonic() < deadline:
                if self.user_hung_up or not self.active:
                    logger.info("📡 User hung up during recording (poll loop)")
                    return None

                await asyncio.sleep(0.25)

                try:
                    resp = await loop.run_in_executor(
                        None,
                        lambda: requests.get(
                            rec_url,
                            auth=(self.ari_username, self.ari_password),
                            timeout=2,
                        ),
                    )
                    if resp.status_code == 200:
                        state = resp.json().get("state", "")
                        if state in ("complete", "failed", "canceled"):
                            logger.debug(f"[record] state={state} — stopping early")
                            break
                    elif resp.status_code == 404:
                        # Recording already finished and moved to stored
                        break
                except Exception:
                    pass  # ARI briefly unavailable — keep polling

            # Ensure recording is stopped (no-op if already done)
            try:
                await recording.stop()
            except Exception:
                pass

            await asyncio.sleep(0.2)
            return await self._download_recording(name)

        except Exception as e:
            if "Not Found" in str(e):
                self.user_hung_up = True
                logger.info("📡 User hung up during recording")
            else:
                logger.error(f"Record error: {e}")
            return None

    async def _download_recording(self, name: str) -> str | None:
        for attempt in range(3):
            try:
                url = f"{self.ari_url}/recordings/stored/{name}/file"
                response = requests.get(
                    url, auth=(self.ari_username, self.ari_password), timeout=10
                )
                if response.status_code == 200 and len(response.content) > 4000:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tmp.write(response.content)
                    tmp.close()
                    self.temp_files.append(tmp.name)
                    return tmp.name
            except Exception:
                pass
            await asyncio.sleep(0.15)
        return None

    # ------------------------------------------------------------------
    # Knowledge base, intent, routing
    # ------------------------------------------------------------------

    def _get_knowledge_context(self, user_text: str) -> str:
        if not self.flask_app:
            return ""
        try:
            with self.flask_app.app_context():
                from models import KnowledgeBase, db
                user_lower = user_text.lower()
                all_entries = KnowledgeBase.query.filter_by(is_active=True).all()
                scored = []
                for entry in all_entries:
                    score = 0
                    keywords = json.loads(entry.keywords) if entry.keywords else []
                    for kw in keywords:
                        if kw.lower() in user_lower:
                            score += 2
                    if any(w in user_lower for w in entry.title.lower().split()):
                        score += 1
                    if score > 0:
                        scored.append((score, entry))
                scored.sort(reverse=True, key=lambda x: x[0])
                top = scored[:2]
                if not top:
                    return ""
                parts = ["\n\nRELEVANT COMPANY INFORMATION:"]
                for _, entry in top:
                    parts.append(f"\n{entry.title}: {entry.content}")
                    entry.increment_usage()
                    db.session.commit()
                return "".join(parts)
        except Exception as e:
            logger.error(f"Knowledge context error: {e}")
            return ""

    def _detect_transfer_intent(self, user_text: str) -> bool:
        keywords = [
            "speak", "talk", "human", "person", "agent",
            "representative", "manager", "supervisor", "someone",
            "transfer", "escalate", "real person",
        ]
        return any(kw in user_text.lower() for kw in keywords)

    def _classify_intent(self, user_text: str) -> str:
        intent_keywords = {
            "sales":   ["buy", "purchase", "new policy", "quote", "coverage", "insurance"],
            "claims":  ["claim", "accident", "damage", "file", "incident"],
            "billing": ["bill", "payment", "pay", "invoice", "charge", "cost"],
            "support": ["help", "question", "how", "what", "when", "status"],
        }
        user_lower = user_text.lower()
        for intent_type, kws in intent_keywords.items():
            if any(kw in user_lower for kw in kws):
                return intent_type
        return "general"

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
                dept_map = {
                    "sales": "Sales", "claims": "Claims",
                    "billing": "Billing", "support": "Support",
                }
                if intent_type in dept_map:
                    dept = Department.query.filter_by(
                        name=dept_map[intent_type], is_active=True
                    ).first()
                    if dept:
                        return dept
                return Department.query.filter_by(is_active=True).order_by(
                    Department.priority.desc()
                ).first()
        except Exception as e:
            logger.error(f"Department lookup error: {e}")
            return None

    async def transfer_to_department(self, department) -> bool:
        try:
            logger.info(f"🔀 Transferring to {department.name} (ext {department.extension})")
            transfer_msg = f"Transferring you to {department.name} now. Please hold."
            await self.speak(transfer_msg)
            self._log_transcript("assistant", transfer_msg, 1.0)
            await asyncio.sleep(0.5)
            await self.channel.continueInDialplan(
                context="from-internal",
                extension=department.extension,
                priority=1,
            )
            logger.info(f"✅ Transferred to ext {department.extension}")
            self.escalated = True
            self.escalated_to_dept_id = department.id
            self.escalation_reason = f"User requested transfer to {department.name}"
            self._log_intent("escalation", 1.0, f"Transferred to {department.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Transfer failed: {e}")
            await self.speak("I'm having trouble transferring your call. Please hold.")
            return False

    # ------------------------------------------------------------------
    # Main conversation loop
    # ------------------------------------------------------------------

    async def process(self):
        try:
            # Answer
            channel_state = self.channel.json.get("state", "Unknown")
            logger.info(f"Channel state before answer: {channel_state}")
            try:
                await self.channel.answer()
                logger.info("✅ Call answered")
            except Exception as e:
                logger.error(f"Answer failed: {e}")
                current = await self.ari_client.channels.get(channelId=self.id)
                if current.json.get("state", "").lower() != "up":
                    raise
                logger.info("Channel already up, continuing")

            await asyncio.sleep(0.2)

            # Greeting
            hour = datetime.now().hour
            time_greeting = (
                "Good morning" if hour < 12
                else "Good afternoon" if hour < 17
                else "Good evening"
            )
            greeting = f"{time_greeting}, thank you for calling Jubilee Insurance. How can I help you today?"
            if not await self.speak(greeting):
                return
            self.conversation.append({"role": "assistant", "content": greeting})

            await asyncio.sleep(0.1)
            if not await self.is_alive():
                return
            await self.channel.play(media="sound:beep")
            await asyncio.sleep(0.15)

            no_speech_count = 0
            for _turn in range(8):
                if self.user_hung_up or not await self.is_alive():
                    logger.info("📡 Call ended by user")
                    break

                self.turn_count += 1

                # LISTEN — live streaming or file fallback
                text, confidence = await self.listen()

                if self.user_hung_up or not await self.is_alive():
                    logger.info("📡 User hung up during listening")
                    break

                await self.channel.play(media="sound:beep")
                await asyncio.sleep(0.1)

                # Handle empty audio
                if not text or len(text) < 3:
                    no_speech_count += 1
                    if no_speech_count >= 2:
                        await self.speak("I'm having trouble hearing you. Please try calling back.")
                        break
                    if not await self.speak("Could you repeat that please?"):
                        break
                    await asyncio.sleep(0.1)
                    if not await self.is_alive():
                        break
                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)
                    continue

                no_speech_count = 0
                logger.info(f"👤 User: {text}")
                self._log_transcript("caller", text, float(confidence == "high"))

                # Goodbye detection
                if len(text.split()) <= 5 and any(
                    w in text.lower() for w in ["bye", "goodbye", "thanks", "done"]
                ):
                    goodbye = "Thank you for calling!"
                    await self.speak(goodbye)
                    self._log_transcript("assistant", goodbye, 1.0)
                    break

                # Transfer intent
                if self._detect_transfer_intent(text):
                    logger.info("🔀 Transfer intent detected")
                    intent_type = self._classify_intent(text)
                    logger.info(f"📊 Intent: {intent_type}")
                    self._log_intent(intent_type, 0.8, text)
                    department = self._get_department_for_intent(intent_type)
                    if department:
                        success = await self.transfer_to_department(department)
                        if success:
                            return
                        error_msg = "I apologize. Let me try to help you another way. What can I assist you with?"
                        await self.speak(error_msg)
                        self._log_transcript("assistant", error_msg, 1.0)
                    else:
                        fallback_msg = "I'd like to connect you with someone, but I'm having trouble right now. Can I help you with something else?"
                        await self.speak(fallback_msg)
                        self._log_transcript("assistant", fallback_msg, 1.0)
                    continue

                # Build AI prompt with knowledge context
                knowledge_context = self._get_knowledge_context(text)
                user_msg = f"{text}{knowledge_context}" if knowledge_context else text
                self.conversation.append({"role": "user", "content": user_msg})

                try:
                    response = await self.ai_client.chat.completions.create(
                        model=self.deployment,
                        messages=self.conversation,
                        max_tokens=40,       # ~30 spoken words — hard physical cap
                        temperature=0.5,
                    )
                    ai_text = response.choices[0].message.content.strip()
                    # Always sanitise — the LLM sometimes ignores formatting rules
                    ai_text = self._clean_for_tts(ai_text)
                    self.conversation.append({"role": "assistant", "content": ai_text})
                    logger.info(f"🤖 AI: {ai_text}")
                    self._log_transcript("assistant", ai_text, 1.0)

                    if not await self.speak(ai_text):
                        break

                    await asyncio.sleep(0.1)
                    if not await self.is_alive():
                        break
                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)

                except Exception as e:
                    logger.error(f"AI error: {e}")
                    error_msg = "Technical issue. Let me connect you to someone."
                    await self.speak(error_msg)
                    self._log_transcript("assistant", error_msg, 1.0)
                    dept = self._get_department_for_intent("support")
                    if dept:
                        await self.transfer_to_department(dept)
                        return
                    break

            # Farewell (only if not already transferred or hung up)
            if self.active and not self.user_hung_up and not self.escalated and await self.is_alive():
                final_msg = "Thank you for calling!"
                await self.speak(final_msg)
                self._log_transcript("assistant", final_msg, 1.0)

            await self.hangup()

        except Exception as e:
            if "Not Found" in str(e):
                logger.info("📡 User hung up (channel not found)")
                self.user_hung_up = True
            else:
                logger.error(f"Call processing error: {e}")
            await self.hangup()

    # ------------------------------------------------------------------
    # DB logging helpers
    # ------------------------------------------------------------------

    def _log_transcript(self, speaker: str, text: str, confidence: float):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call, CallTranscript
                call = Call.query.filter_by(call_id=self.id).first()
                if not call:
                    return
                transcript = CallTranscript(
                    call_id=call.id, speaker=speaker, text=text,
                    confidence=confidence, timestamp=datetime.utcnow(),
                )
                db.session.add(transcript)
                db.session.commit()
        except Exception as e:
            logger.error(f"Transcript log error: {e}")

    def _log_intent(self, intent_type: str, confidence: float, context: str):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call, CallIntent
                call = Call.query.filter_by(call_id=self.id).first()
                if not call:
                    return
                intent = CallIntent(
                    call_id=call.id, intent_type=intent_type,
                    confidence=confidence, context=context,
                    detected_at=datetime.utcnow(),
                )
                db.session.add(intent)
                db.session.commit()
        except Exception as e:
            logger.error(f"Intent log error: {e}")

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------

    async def is_alive(self) -> bool:
        if not self.active or self.user_hung_up:
            return False
        try:
            await self.ari_client.channels.get(channelId=self.id)
            return True
        except Exception as e:
            if "Not Found" in str(e):
                self.user_hung_up = True
            self.active = False
            return False

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """
        Strip markdown / list formatting that the LLM sometimes inserts
        despite instructions.  TTS engines render asterisks and hyphens
        literally, which sounds terrible on a phone call.

        Rules applied (in order):
          1. Remove bold/italic markers  (**text**, *text*, __text__, _text_)
          2. Convert numbered list items  "1. Foo" → "Foo"
          3. Convert bullet list items   "- Foo" / "* Foo" → "Foo"
          4. Collapse multiple newlines to a single space
          5. Collapse multiple spaces
          6. Strip leading/trailing whitespace
          7. Truncate to 50 words maximum (safety net on top of max_tokens=60)
        """
        import re

        # 1. Remove bold / italic markers
        text = re.sub(r'\*{1,2}([^*]+?)\*{1,2}', r'\1', text)
        text = re.sub(r'_{1,2}([^_]+?)_{1,2}', r'\1', text)

        # 2. Numbered list items — "1. Item" → "Item"
        text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)

        # 3. Bullet list items — "- Item" or "* Item" → "Item"
        text = re.sub(r'(?m)^\s*[-*]\s+', '', text)

        # 4. Multiple newlines / carriage-returns → single space
        text = re.sub(r'[\r\n]+', ' ', text)

        # 5. Collapse multiple spaces
        text = re.sub(r'  +', ' ', text)

        text = text.strip()

        # 7. Hard word-count cap — truncate at sentence boundary if possible
        words = text.split()
        if len(words) > 50:
            truncated = ' '.join(words[:50])
            # Try to end at the last sentence boundary within the 50 words
            last_period = max(truncated.rfind('.'), truncated.rfind('?'), truncated.rfind('!'))
            if last_period > len(truncated) // 2:
                text = truncated[:last_period + 1]
            else:
                text = truncated

        return text

    async def speak(self, text: str) -> bool:
        if not await self.is_alive():
            return False
        text = self._clean_for_tts(text)
        if not text:
            return False
        try:
            sound_path, duration = await self.sound_cache.get(text, self.file_access)
            if not sound_path:
                return False
            await self.channel.play(media=f"sound:{sound_path}")
            estimated = duration or (len(text.split()) * 0.4)
            await asyncio.sleep(estimated + 0.3)
            return True
        except Exception as e:
            if "Not Found" in str(e):
                self.user_hung_up = True
                logger.info("📡 User hung up during speech")
            elif "404" not in str(e):
                logger.error(f"Speak error: {e}")
            self.active = False
            return False

    async def hangup(self):
        try:
            if self.active and not self.user_hung_up and not self.escalated:
                await self.channel.hangup()
        except Exception as e:
            if "Not Found" not in str(e):
                logger.debug(f"Hangup error: {e}")
        self.active = False

    async def cleanup(self):
        await self._stop_external_media()
        for fp in self.temp_files:
            try:
                os.unlink(fp)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ARI Agent  (thin wrapper; core logic in CallInstance)
# ---------------------------------------------------------------------------

class ARIAgent:
    """ARI-based AI voice agent with live streaming STT and call transfer."""

    def __init__(self, app_config, flask_app=None):
        self.config = app_config
        self.flask_app = flask_app
        self.running = False
        self.active_calls: dict[str, CallInstance] = {}
        self.total_calls = 0

        # ARI
        self.ari_url = os.getenv("ARI_URL", "http://localhost:8088/ari")
        self.ari_base = os.getenv("ARI_BASE", "http://localhost:8088")
        self.ari_username = os.getenv("ARI_USERNAME", "asterisk")
        self.ari_password = os.getenv("ARI_PASSWORD", "your_ari_password")
        self.ari_app = os.getenv("ARI_APP", "ai-agent")

        self.asterisk_sounds_dir = "/var/lib/asterisk/sounds/custom"

        # Azure
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.azure_speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")

        self.system_prompt = os.getenv("DEFAULT_SYSTEM_PROMPT") or self._default_prompt()

        # Cache / TTS
        self.cache_dir = Path.home() / ".asterisk_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"

        self.sound_cache = SoundCache(
            self.cache_dir, self.cache_index_file, self.asterisk_sounds_dir,
            azure_speech_key=self.azure_speech_key,
            azure_speech_region=self.azure_speech_region,
        )

        # Live streaming transcriber (primary)
        self.live_transcriber: LiveStreamTranscriber | None = None
        try:
            self.live_transcriber = LiveStreamTranscriber(
                self.azure_speech_key, self.azure_speech_region
            )
            logger.info("✅ Live streaming transcriber ready")
        except Exception as e:
            logger.warning(f"⚠️ Live transcriber unavailable: {e}")

        # File-based transcriber (fallback)
        self.transcriber: AzureSpeechTranscriber | None = None
        try:
            self.transcriber = AzureSpeechTranscriber(
                self.azure_speech_key, self.azure_speech_region
            )
            logger.info("✅ File-based transcriber ready (fallback)")
        except Exception as e:
            logger.error(f"❌ Fallback transcriber failed: {e}")

        # File access
        self.file_access = FileSystemAccess(self.asterisk_sounds_dir)

        # OpenAI
        self.ai_client = None
        if self.azure_openai_endpoint and self.azure_openai_key:
            try:
                self.ai_client = AsyncAzureOpenAI(
                    api_key=self.azure_openai_key,
                    azure_endpoint=self.azure_openai_endpoint.rstrip("/"),
                    api_version="2024-08-01-preview",
                )
                logger.info("✅ OpenAI client ready")
            except Exception as e:
                logger.error(f"❌ OpenAI client failed: {e}")
        else:
            logger.warning("⚠️ Azure OpenAI not configured")

        self.ari_client = None
        logger.info("ARI Agent initialised")

    def _default_prompt(self) -> str:
        """
        Phone-optimised system prompt.

        The CRITICAL block is placed first and uses emphatic language because
        the model's RLHF training rewards helpfulness (verbose lists) more
        strongly than instruction-following for brevity.  Front-loading the
        constraint and using ALL-CAPS for the word limit has been shown to
        improve compliance over inline rules buried in later paragraphs.
        """
        return (
            "CRITICAL — TTS PHONE SYSTEM: Every response MUST be 20 words or fewer. "
            "Count words before replying. NEVER use bullet points, numbered lists, "
            "bold, asterisks, dashes, or any markdown. NEVER use newlines. "
            "Plain spoken English only. One or two short sentences maximum.\n\n"
            "You are a professional phone assistant for Jubilee Insurance. "
            "Be helpful, empathetic, and concise. "
            "Never say 'I am an AI'. "
            "Ask only one question at a time if you need clarification. "
            "If the caller asks about services, name ONE service and ask which they need. "
            "If you cannot help or the caller requests a human, say you will transfer them now."
        )

    async def start(self):
        self.running = True
        logger.info("=" * 60)
        logger.info("🤖 ARI Agent Starting")
        logger.info("=" * 60)

        if not self.ai_client:
            logger.error("❌ Cannot start — Azure OpenAI not configured")
            return

        # Verify AI
        try:
            await self.ai_client.chat.completions.create(
                model=self.azure_openai_deployment,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            logger.info("✅ AI connection verified")
        except Exception as e:
            logger.error(f"❌ AI connection failed: {e}")
            return

        if self.file_access.test_access():
            logger.info("✅ File system access verified")
        else:
            logger.warning("⚠️ Limited file system access")

        await self._precache_phrases()

        # Connect ARI
        try:
            logger.info(f"Connecting to ARI at {self.ari_base}…")
            self.ari_client = await aioari.connect(
                self.ari_base, self.ari_username, self.ari_password
            )
            logger.info("✅ ARI connected")

            self.ari_client.on_event("StasisStart", self._handle_stasis_start)
            self.ari_client.on_event("StasisEnd", self._handle_stasis_end)
            self.ari_client.on_event("ChannelHangupRequest", self._handle_hangup_request)

            stt_mode = "live streaming" if self.live_transcriber else "file-based (fallback)"
            logger.info("=" * 60)
            logger.info("🎙️ SYSTEM READY — Waiting for calls")
            logger.info(f"   ARI App : {self.ari_app}")
            logger.info(f"   AI Model: {self.azure_openai_deployment}")
            logger.info(f"   STT mode: {stt_mode}")
            logger.info("=" * 60)

            await self.ari_client.run(apps=self.ari_app)

        except Exception as e:
            logger.error(f"❌ ARI connection error: {e}")
            self.running = False

    async def stop(self):
        logger.info("Stopping ARI agent…")
        self.running = False
        for call in list(self.active_calls.values()):
            try:
                await call.hangup()
            except Exception:
                pass
        if self.ari_client:
            try:
                await self.ari_client.close()
            except Exception:
                pass
        logger.info("ARI agent stopped")

    async def _precache_phrases(self):
        phrases = [
            "Good morning, thank you for calling Jubilee Insurance. How can I help you today?",
            "Good afternoon, thank you for calling Jubilee Insurance. How can I help you today?",
            "Good evening, thank you for calling Jubilee Insurance. How can I help you today?",
            "Thank you for calling!",
            "Could you repeat that please?",
            "Let me transfer you to a specialist who can help. Please hold.",
            "I'm having trouble hearing you. Please try calling back.",
        ]
        logger.info("Caching common phrases…")
        for phrase in phrases:
            await self.sound_cache.get(phrase, self.file_access)

    # ── ARI event handlers ────────────────────────────────────────────

    def _handle_stasis_start(self, event):
        asyncio.create_task(self._process_call(event))

    def _handle_stasis_end(self, event):
        channel_id = event.get("channel", {}).get("id")
        if channel_id and channel_id in self.active_calls:
            logger.info(f"📴 Channel {channel_id[:12]} left Stasis")
            call = self.active_calls[channel_id]
            call.user_hung_up = True
            call.active = False

    def _handle_hangup_request(self, event):
        channel_id = event.get("channel", {}).get("id")
        if channel_id and channel_id in self.active_calls:
            logger.info(f"📴 Hangup requested for {channel_id[:12]}")
            call = self.active_calls[channel_id]
            call.user_hung_up = True
            call.active = False

    async def _process_call(self, event):
        channel_id = event.get("channel", {}).get("id")
        if not channel_id:
            return

        # ── Skip ExternalMedia / internal channels ───────────────────────────
        # When _start_external_media() calls ari_client.channels.externalMedia(
        # app="ai-agent"), Asterisk fires a StasisStart for the new audio-fork
        # channel too.  That channel is NOT a real caller — processing it would
        # create an infinite loop (each ExternalMedia leg triggers another
        # ExternalMedia leg, etc.).
        #
        # Guard 1 — channel name prefix:
        #   Asterisk always names ExternalMedia channels "UnicastRTP/…" or
        #   "ExternalMedia/…".  Local dialplan channels use "Local/…".
        #   None of these are real inbound PSTN/SIP callers.
        #
        # Guard 2 — state + missing caller number:
        #   A genuine inbound call starts in "Ring" state and always carries a
        #   caller number.  The ghost channels we saw in logs were all
        #   state="Up" with an empty caller number — Asterisk's internal legs.
        channel_data = event.get("channel", {})
        channel_name = channel_data.get("name", "")
        if channel_name.startswith(("UnicastRTP/", "ExternalMedia/", "Local/")):
            logger.debug(f"[StasisStart] Ignoring internal channel: {channel_name}")
            return

        channel_state = channel_data.get("state", "")
        caller_number = channel_data.get("caller", {}).get("number", "")
        if channel_state == "Up" and not caller_number:
            logger.debug(
                f"[StasisStart] Ignoring Up channel with no caller: {channel_id[:16]}"
            )
            return
        # ────────────────────────────────────────────────────────────────────

        try:
            channel = await self.ari_client.channels.get(channelId=channel_id)
            await self._handle_call(channel)
        except Exception as e:
            logger.error(f"❌ Call processing error: {e}")

    async def _handle_call(self, channel):
        caller_number = channel.json.get("caller", {}).get("number", "Unknown")
        logger.info(f"📞 Incoming call from {caller_number}")

        call = CallInstance(
            channel=channel,
            ari_client=self.ari_client,
            ai_client=self.ai_client,
            sound_cache=self.sound_cache,
            file_access=self.file_access,
            transcriber=self.transcriber,
            live_transcriber=self.live_transcriber,
            system_prompt=self.system_prompt,
            deployment=self.azure_openai_deployment,
            ari_url=self.ari_url,
            ari_username=self.ari_username,
            ari_password=self.ari_password,
            flask_app=self.flask_app,
        )
        call.ari_app = self.ari_app

        self.active_calls[channel.id] = call
        self.total_calls += 1
        self._log_call_start(call.id, caller_number)

        try:
            await call.process()
        except Exception as e:
            logger.error(f"❌ Call error: {e}")
            self._log_call_error(call.id, str(e))
        finally:
            if channel.id in self.active_calls:
                del self.active_calls[channel.id]
            await call.cleanup()
            self._log_call_end(call)

    # ── DB helpers ────────────────────────────────────────────────────

    def _log_call_start(self, call_id: str, caller_number: str):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                call = Call(
                    call_id=call_id,
                    caller_number=caller_number,
                    status="active",
                    started_at=datetime.utcnow(),
                )
                db.session.add(call)
                db.session.commit()
                logger.info(f"✅ Call {call_id} logged")
        except Exception as e:
            logger.error(f"Log call start error: {e}")

    def _log_call_error(self, call_id: str, error_msg: str):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                call = Call.query.filter_by(call_id=call_id).first()
                if call:
                    call.status = "error"
                    call.ended_at = datetime.utcnow()
                    db.session.commit()
        except Exception as e:
            logger.error(f"Log call error: {e}")

    def _log_call_end(self, call_instance: CallInstance):
        if not self.flask_app:
            return
        try:
            with self.flask_app.app_context():
                from models import db, Call
                call = Call.query.filter_by(call_id=call_instance.id).first()
                if call:
                    if call_instance.escalated:
                        call.status = "escalated"
                        call.escalated = True
                        call.escalated_to_department_id = call_instance.escalated_to_dept_id
                        call.escalation_reason = call_instance.escalation_reason
                    else:
                        call.status = "completed"
                    call.ended_at = datetime.utcnow()
                    if call.started_at:
                        call.duration_seconds = int(
                            (call.ended_at - call.started_at).total_seconds()
                        )
                    call.total_interactions = call_instance.turn_count
                    db.session.commit()
                    logger.info(f"✅ Call {call_instance.id} logged as complete")
        except Exception as e:
            logger.error(f"Log call end error: {e}")