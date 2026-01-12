# services/ari_agent.py
"""
ARI-based agent service - FIXED AI Client Initialization
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
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize
from openai import AsyncAzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
from models import db, Call, CallTranscript, CallIntent

logger = logging.getLogger(__name__)


class ARIAgent:
    """ARI-based AI voice agent"""

    def __init__(self, app_config):
        self.config = app_config
        self.running = False
        self.active_calls = set()
        self.total_calls = 0

        # ARI Configuration
        self.ari_url = os.getenv('ARI_URL', 'http://localhost:8088/ari')
        self.ari_base = os.getenv('ARI_BASE', 'http://localhost:8088')
        self.ari_username = os.getenv('ARI_USERNAME', 'asterisk')
        self.ari_password = os.getenv('ARI_PASSWORD', 'your_ari_password')
        self.ari_app = os.getenv('ARI_APP', 'ai-agent')

        # File system
        self.asterisk_sounds_dir = '/var/lib/asterisk/sounds/custom'

        # Azure configuration - FIXED: Get from environment directly
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
        self.azure_openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
        self.azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
        self.azure_speech_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')

        logger.info(f"Azure OpenAI Endpoint: {self.azure_openai_endpoint}")
        logger.info(f"Azure OpenAI Key length: {len(self.azure_openai_key) if self.azure_openai_key else 0}")
        logger.info(f"Azure OpenAI Deployment: {self.azure_openai_deployment}")

        # System prompt
        self.system_prompt = os.getenv('DEFAULT_SYSTEM_PROMPT') or self._default_prompt()

        # Initialize components
        self.cache_dir = Path.home() / ".asterisk_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"

        self.sound_cache = SoundCache(self.cache_dir, self.cache_index_file, self.asterisk_sounds_dir)

        # Initialize transcriber
        try:
            self.transcriber = AzureSpeechTranscriber(self.azure_speech_key, self.azure_speech_region)
            logger.info("✅ Speech transcriber initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize transcriber: {e}")
            self.transcriber = None

        # File system access
        self.file_access = FileSystemAccess(self.asterisk_sounds_dir)

        # OpenAI client initialization - FIXED
        self.ai_client = None
        if self.azure_openai_endpoint and self.azure_openai_key:
            try:
                # Clean endpoint
                endpoint = self.azure_openai_endpoint.rstrip('/')

                logger.info(f"Initializing Azure OpenAI client...")
                logger.info(f"  Endpoint: {endpoint}")
                logger.info(f"  Deployment: {self.azure_openai_deployment}")

                self.ai_client = AsyncAzureOpenAI(
                    api_key=self.azure_openai_key,
                    azure_endpoint=endpoint,
                    api_version="2024-08-01-preview"
                )
                logger.info("✅ OpenAI client object created")

            except TypeError as e:
                logger.error(f"❌ TypeError initializing OpenAI client: {e}")
                logger.error("   This usually means incompatible openai package version")
                logger.error("   Try: pip install --upgrade openai")
                self.ai_client = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                logger.error(f"   Error type: {type(e).__name__}")
                self.ai_client = None
        else:
            logger.warning("⚠️ Azure OpenAI not configured:")
            if not self.azure_openai_endpoint:
                logger.warning("   - AZURE_OPENAI_ENDPOINT is missing")
            if not self.azure_openai_key:
                logger.warning("   - AZURE_OPENAI_KEY is missing")

        # ARI client
        self.ari_client = None

        logger.info("ARI Agent initialized")

    def _default_prompt(self):
        return """You are a professional phone assistant for an insurance company.

RULES:
- Keep responses between 15-35 words for phone conversations
- Be helpful, professional, and empathetic
- Never say "I'm an AI" or mention being artificial
- Use natural, conversational language

When you cannot help or caller seems frustrated, politely recommend speaking with a specialist."""

    async def start(self):
        """Start the ARI agent"""
        self.running = True

        logger.info("=" * 60)
        logger.info("🤖 ARI Agent Starting")
        logger.info("=" * 60)

        # Validate AI configuration
        if not self.ai_client:
            logger.error("❌ Cannot start - Azure OpenAI client failed to initialize")
            logger.error("   Check:")
            logger.error("   1. AZURE_OPENAI_KEY is set in .env")
            logger.error("   2. AZURE_OPENAI_ENDPOINT is set in .env")
            logger.error("   3. openai package is installed: pip install openai")
            return

        # Test AI connection
        try:
            logger.info("Testing AI connection...")
            test_response = await self.ai_client.chat.completions.create(
                model=self.azure_openai_deployment,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            logger.info("✅ AI connection verified")
        except Exception as e:
            logger.error(f"❌ AI connection test failed: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            if "404" in str(e):
                logger.error(f"   Deployment '{self.azure_openai_deployment}' not found")
                logger.error("   Check AZURE_OPENAI_DEPLOYMENT matches your deployment name")
            elif "401" in str(e) or "403" in str(e):
                logger.error("   Authentication failed - check AZURE_OPENAI_KEY")
            return

        # Test file system access
        if self.file_access.test_access():
            logger.info("✅ File system access verified")
        else:
            logger.warning("⚠️ Limited file system access")

        # Pre-cache phrases
        await self._precache_phrases()

        # Connect to ARI
        try:
            logger.info(f"Connecting to ARI at {self.ari_base}...")
            self.ari_client = await aioari.connect(
                self.ari_base,
                self.ari_username,
                self.ari_password
            )
            logger.info("✅ ARI connected")

            self.ari_client.on_event("StasisStart", self._handle_stasis_start)

            logger.info("=" * 60)
            logger.info("🎙️ SYSTEM READY - Waiting for calls")
            logger.info(f"   ARI App: {self.ari_app}")
            logger.info(f"   AI Model: {self.azure_openai_deployment}")
            logger.info("=" * 60)

            await self.ari_client.run(apps=self.ari_app)

        except Exception as e:
            logger.error(f"❌ ARI connection error: {e}")
            self.running = False

    async def stop(self):
        """Stop the ARI agent"""
        logger.info("Stopping ARI agent...")
        self.running = False

        for call in list(self.active_calls):
            try:
                await call.hangup()
            except:
                pass

        if self.ari_client:
            try:
                await self.ari_client.close()
            except:
                pass

        logger.info("ARI agent stopped")

    async def _precache_phrases(self):
        """Pre-cache common TTS phrases"""
        phrases = [
            "Good morning, thank you for calling. How can I help you today?",
            "Good afternoon, thank you for calling. How can I help you today?",
            "Good evening, thank you for calling. How can I help you today?",
            "Thank you for calling!",
            "Could you repeat that please?",
        ]

        logger.info("Caching common phrases...")
        for phrase in phrases:
            await self.sound_cache.get(phrase, self.file_access)

    def _handle_stasis_start(self, event):
        asyncio.create_task(self._process_call(event))

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
        caller_number = channel.json.get('caller', {}).get('number', 'Unknown')
        logger.info(f"📞 Incoming call from {caller_number}")

        call = CallInstance(
            channel=channel,
            ari_client=self.ari_client,
            ai_client=self.ai_client,
            sound_cache=self.sound_cache,
            file_access=self.file_access,
            transcriber=self.transcriber,
            system_prompt=self.system_prompt,
            deployment=self.azure_openai_deployment,
            ari_url=self.ari_url,
            ari_username=self.ari_username,
            ari_password=self.ari_password
        )

        self.active_calls.add(call)
        self.total_calls += 1

        self._log_call_start(call.id, caller_number)

        try:
            await call.process()
        except Exception as e:
            logger.error(f"❌ Call error: {e}")
            self._log_call_error(call.id, str(e))
        finally:
            self.active_calls.discard(call)
            await call.cleanup()
            self._log_call_end(call)

    def _log_call_start(self, call_id, caller_number):
        try:
            from flask import current_app
            with current_app.app_context():
                call = Call(
                    call_id=call_id,
                    caller_number=caller_number,
                    status='active',
                    started_at=datetime.utcnow()
                )
                db.session.add(call)
                db.session.commit()
        except Exception as e:
            logger.error(f"Failed to log call start: {e}")

    def _log_call_error(self, call_id, error_msg):
        try:
            from flask import current_app
            with current_app.app_context():
                call = Call.query.filter_by(call_id=call_id).first()
                if call:
                    call.status = 'error'
                    call.ended_at = datetime.utcnow()
                    db.session.commit()
        except Exception as e:
            logger.error(f"Failed to log error: {e}")

    def _log_call_end(self, call_instance):
        try:
            from flask import current_app
            with current_app.app_context():
                call = Call.query.filter_by(call_id=call_instance.id).first()
                if call:
                    call.status = 'completed'
                    call.ended_at = datetime.utcnow()
                    if call.started_at:
                        call.duration_seconds = int((call.ended_at - call.started_at).total_seconds())
                    call.total_interactions = call_instance.turn_count
                    db.session.commit()
        except Exception as e:
            logger.error(f"Failed to log call end: {e}")


class CallInstance:
    """Represents a single call"""

    def __init__(self, channel, ari_client, ai_client, sound_cache, file_access,
                 transcriber, system_prompt, deployment, ari_url, ari_username, ari_password):
        self.channel = channel
        self.ari_client = ari_client
        self.ai_client = ai_client
        self.sound_cache = sound_cache
        self.file_access = file_access
        self.transcriber = transcriber
        self.system_prompt = system_prompt
        self.deployment = deployment
        self.ari_url = ari_url
        self.ari_username = ari_username
        self.ari_password = ari_password

        self.id = channel.id
        self.active = True
        self.temp_files = []
        self.turn_count = 0
        self.conversation = [{"role": "system", "content": system_prompt}]

    async def process(self):
        try:
            await self.channel.answer()
            await asyncio.sleep(0.2)

            hour = datetime.now().hour
            time_greeting = 'Good morning' if hour < 12 else 'Good afternoon' if hour < 17 else 'Good evening'
            greeting = f"{time_greeting}, thank you for calling. How can I help you today?"

            await self.speak(greeting)
            self.conversation.append({"role": "assistant", "content": greeting})

            await asyncio.sleep(0.1)
            await self.channel.play(media="sound:beep")
            await asyncio.sleep(0.15)

            no_speech_count = 0
            for turn in range(8):
                if not await self.is_alive():
                    break

                self.turn_count += 1

                audio_file = await self.record()
                await self.channel.play(media="sound:beep")
                await asyncio.sleep(0.1)

                if not audio_file:
                    no_speech_count += 1
                    if no_speech_count >= 2:
                        await self.speak("I'm having trouble hearing you. Please try calling back.")
                        break
                    await self.speak("I didn't catch that. Please go ahead.")
                    await asyncio.sleep(0.1)
                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)
                    continue

                text, confidence = await self.transcriber.transcribe(audio_file)
                no_speech_count = 0

                if not text or len(text) < 3:
                    await self.speak("Could you repeat that please?")
                    await asyncio.sleep(0.1)
                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)
                    continue

                logger.info(f"👤 User: {text}")

                if len(text.split()) <= 5 and any(w in text.lower() for w in ["bye", "goodbye", "thanks", "done"]):
                    await self.speak("Thank you for calling!")
                    break

                self.conversation.append({"role": "user", "content": text})

                try:
                    response = await self.ai_client.chat.completions.create(
                        model=self.deployment,
                        messages=self.conversation,
                        max_tokens=200,
                        temperature=0.9
                    )

                    ai_text = response.choices[0].message.content.strip()
                    self.conversation.append({"role": "assistant", "content": ai_text})
                    logger.info(f"🤖 AI: {ai_text}")

                    if not await self.speak(ai_text):
                        break

                    await asyncio.sleep(0.1)
                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)

                except Exception as e:
                    logger.error(f"AI error: {e}")
                    await self.speak("Technical issue. Let me connect you to someone.")
                    break

            if self.active:
                await self.speak("Thank you for calling!")

            await self.hangup()

        except Exception as e:
            logger.error(f"Call processing error: {e}")
            await self.hangup()

    async def is_alive(self):
        if not self.active:
            return False
        try:
            await self.ari_client.channels.get(channelId=self.id)
            return True
        except:
            self.active = False
            return False

    async def speak(self, text):
        if not await self.is_alive():
            return False

        try:
            sound_path, duration = await self.sound_cache.get(text, self.file_access)
            if not sound_path:
                return False

            await self.channel.play(media=f"sound:{sound_path}")
            estimated_duration = duration or (len(text.split()) * 0.4)
            await asyncio.sleep(estimated_duration + 0.3)
            return True
        except Exception as e:
            if "404" not in str(e):
                logger.error(f"Speak error: {e}")
            self.active = False
            return False

    async def record(self, duration=8, silence=2.0):
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
                terminateOn="none"
            )

            await asyncio.sleep(duration + 0.5)

            try:
                await recording.stop()
            except:
                pass

            await asyncio.sleep(0.2)
            return await self._download_recording(name)
        except Exception as e:
            logger.error(f"Record error: {e}")
            return None

    async def _download_recording(self, name):
        for attempt in range(3):
            try:
                url = f"{self.ari_url}/recordings/stored/{name}/file"
                response = requests.get(
                    url,
                    auth=(self.ari_username, self.ari_password),
                    timeout=10
                )

                if response.status_code == 200 and len(response.content) > 4000:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_file.write(response.content)
                    temp_file.close()
                    self.temp_files.append(temp_file.name)
                    return temp_file.name
            except:
                pass
            await asyncio.sleep(0.15)

        return None

    async def hangup(self):
        try:
            if self.active:
                await self.channel.hangup()
        except:
            pass
        self.active = False

    async def cleanup(self):
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass


class SoundCache:
    """Cache for TTS audio"""

    def __init__(self, cache_dir, index_file, asterisk_sounds_dir):
        self.cache_dir = cache_dir
        self.index_file = index_file
        self.asterisk_sounds_dir = asterisk_sounds_dir
        self.index = self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            try:
                return json.load(open(self.index_file))
            except:
                return {}
        return {}

    def _save_index(self):
        try:
            json.dump(self.index, open(self.index_file, 'w'))
        except:
            pass

    def _cache_key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    async def get(self, text, file_access):
        key = self._cache_key(text)

        if key in self.index and self.index[key].get('remote'):
            return self.index[key]['remote'], self.index[key].get('duration')

        local_path = await self._generate_tts(text, key)
        if not local_path:
            return None, None

        duration = self._get_duration(local_path)

        remote_path = file_access.copy_to_asterisk(local_path, f"c_{key}.wav")

        if remote_path:
            self.index[key] = {'remote': remote_path, 'duration': duration}
            self._save_index()
            return remote_path, duration

        return local_path, duration

    async def _generate_tts(self, text, key):
        try:
            output_file = self.cache_dir / f"{key}.wav"
            if output_file.exists():
                return str(output_file)

            try:
                from gtts import gTTS
                temp_file = self.cache_dir / f"{key}_temp.mp3"
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: gTTS(text=text, lang='en', slow=False).save(str(temp_file))
                )

                audio = AudioSegment.from_file(str(temp_file))
                audio = normalize(audio).set_frame_rate(8000).set_channels(1).set_sample_width(2)
                audio.export(str(output_file), format="wav")

                try:
                    temp_file.unlink()
                except:
                    pass

                return str(output_file)

            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                return None

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def _get_duration(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
        except:
            return None


class FileSystemAccess:
    """Direct file system access"""

    def __init__(self, sounds_dir):
        self.sounds_dir = sounds_dir
        self.can_write = False
        self.use_sudo = False

    def test_access(self):
        try:
            test_file = os.path.join(self.sounds_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.unlink(test_file)
            self.can_write = True
            logger.info("Direct write access verified")
            return True
        except PermissionError:
            try:
                result = subprocess.run(
                    ['sudo', '-n', 'touch', os.path.join(self.sounds_dir, '.test_write')],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    subprocess.run(['sudo', 'rm', os.path.join(self.sounds_dir, '.test_write')])
                    self.can_write = True
                    self.use_sudo = True
                    logger.info("Sudo access verified")
                    return True
            except:
                pass

            logger.warning("No write access - run as asterisk or use sudo")
            return False

    def copy_to_asterisk(self, local_path, filename):
        try:
            dest_path = os.path.join(self.sounds_dir, filename)

            if self.use_sudo:
                subprocess.run(['sudo', 'cp', local_path, dest_path], check=True)
                subprocess.run(['sudo', 'chown', 'asterisk:asterisk', dest_path], check=True)
                subprocess.run(['sudo', 'chmod', '644', dest_path], check=True)
            else:
                shutil.copy2(local_path, dest_path)
                os.chmod(dest_path, 0o644)

            return f"custom/{filename.replace('.wav', '')}"

        except Exception as e:
            logger.error(f"File copy error: {e}")
            return None


class AzureSpeechTranscriber:
    """Azure Speech transcription"""

    def __init__(self, speech_key, speech_region):
        if not speech_key or not speech_region:
            raise ValueError("Azure Speech key and region required")

        self.config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region
        )
        self.config.speech_recognition_language = "en-US"

    async def transcribe(self, audio_file):
        try:
            if os.path.getsize(audio_file) < 4000:
                return "", "low"

            processed = await self._preprocess(audio_file)

            audio_config = speechsdk.audio.AudioConfig(filename=processed)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.config,
                audio_config=audio_config
            )

            result = await asyncio.get_running_loop().run_in_executor(
                None,
                recognizer.recognize_once
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
                except:
                    pass

            return text, confidence

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", "low"

    async def _preprocess(self, audio_file):
        try:
            audio = AudioSegment.from_file(audio_file)
            audio = normalize(audio).set_frame_rate(16000).set_channels(1).set_sample_width(2)
            processed = audio_file.replace('.wav', '_proc.wav')
            audio.export(processed, format="wav")
            return processed
        except:
            return audio_file