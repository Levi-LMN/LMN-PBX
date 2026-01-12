# services/ari_agent.py
"""
ARI-based agent service - Production version with proper connection validation
Runs on same machine as FreePBX
"""

import asyncio
import aioari
import os
import tempfile
import time
import requests
import logging
import paramiko
import hashlib
import json
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize
from openai import AsyncAzureOpenAI
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
from models import db, Call, CallTranscript, CallIntent

logger = logging.getLogger(__name__)


class ARIAgent:
    """
    ARI-based AI voice agent integrated with Flask
    """

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

        # SSH Configuration (localhost since running on same machine)
        self.ssh_host = os.getenv('SSH_HOST', 'localhost')
        self.ssh_port = int(os.getenv('SSH_PORT', '22'))
        self.ssh_user = os.getenv('SSH_USER', 'sangoma')
        self.ssh_password = os.getenv('SSH_PASSWORD', 'sangoma')
        self.asterisk_sounds_dir = '/var/lib/asterisk/sounds/custom'

        # Azure configuration
        self.azure_openai_endpoint = app_config.get('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_key = app_config.get('AZURE_OPENAI_KEY')
        self.azure_openai_deployment = app_config.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
        self.azure_speech_key = app_config.get('AZURE_SPEECH_KEY')
        self.azure_speech_region = app_config.get('AZURE_SPEECH_REGION', 'eastus')

        # System prompt
        self.system_prompt = app_config.get('DEFAULT_SYSTEM_PROMPT', self._default_prompt())

        # Initialize components
        self.cache_dir = Path.home() / ".asterisk_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"

        self.sound_cache = SoundCache(self.cache_dir, self.cache_index_file)
        self.ssh_client = SSHClient(
            self.ssh_host,
            self.ssh_port,
            self.ssh_user,
            self.ssh_password,
            self.asterisk_sounds_dir
        )
        self.transcriber = AzureSpeechTranscriber(self.azure_speech_key, self.azure_speech_region)

        # OpenAI client initialization
        self.ai_client = None
        if self.azure_openai_endpoint and self.azure_openai_key:
            try:
                self.ai_client = AsyncAzureOpenAI(
                    api_key=self.azure_openai_key,
                    azure_endpoint=self.azure_openai_endpoint.rstrip('/'),
                    api_version="2024-08-01-preview"
                )
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.ai_client = None
        else:
            logger.warning("⚠️ Azure OpenAI not configured")

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
- If you don't know something, be honest

CAPABILITIES:
- Answer questions about policies, claims, and billing
- Guide callers through common processes
- Identify when escalation to human agent is needed

When you cannot help or caller seems frustrated, politely recommend speaking with a specialist."""

    async def start(self):
        """Start the ARI agent"""
        self.running = True

        logger.info("=" * 60)
        logger.info("🤖 ARI Agent Starting")
        logger.info("=" * 60)

        # Validate AI configuration
        if not self.ai_client:
            logger.error("❌ Cannot start - Azure OpenAI not configured")
            logger.error("   Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in .env")
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
            logger.error(f"❌ AI connection failed: {e}")
            logger.error(f"   Endpoint: {self.azure_openai_endpoint}")
            logger.error(f"   Deployment: {self.azure_openai_deployment}")
            return

        # Test SSH connection (REAL validation now)
        ssh_connected = await self.ssh_client.connect()
        if ssh_connected:
            logger.info("✅ SSH connected - remote audio upload enabled")
        else:
            logger.warning("⚠️ SSH connection failed - using local audio only")
            logger.warning(f"   Tried: {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")

        # Pre-cache common phrases
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

            # Register event handler
            self.ari_client.on_event("StasisStart", self._handle_stasis_start)

            logger.info("=" * 60)
            logger.info("🎙️ SYSTEM READY - Waiting for calls")
            logger.info(f"   ARI App: {self.ari_app}")
            logger.info(f"   AI Model: {self.azure_openai_deployment}")
            logger.info(f"   SSH: {'Connected' if ssh_connected else 'Disconnected (local only)'}")
            logger.info("=" * 60)

            # Run ARI event loop
            await self.ari_client.run(apps=self.ari_app)

        except Exception as e:
            logger.error(f"❌ ARI connection error: {e}")
            logger.error(f"   URL: {self.ari_base}")
            logger.error(f"   Username: {self.ari_username}")
            logger.error("   Check FreePBX ARI configuration and credentials")
            self.running = False

    async def stop(self):
        """Stop the ARI agent"""
        logger.info("Stopping ARI agent...")
        self.running = False

        # Hangup all active calls
        for call in list(self.active_calls):
            try:
                await call.hangup()
            except:
                pass

        # Close ARI connection
        if self.ari_client:
            try:
                await self.ari_client.close()
            except:
                pass

        # Close SSH
        self.ssh_client.close()

        logger.info("ARI agent stopped")

    async def _precache_phrases(self):
        """Pre-cache common TTS phrases"""
        phrases = [
            "Good morning, thank you for calling. How can I help you today?",
            "Good afternoon, thank you for calling. How can I help you today?",
            "Good evening, thank you for calling. How can I help you today?",
            "Thank you for calling!",
            "Could you repeat that please?",
            "I'm having trouble hearing you. Please try calling back.",
            "Let me connect you with a specialist who can help.",
        ]

        logger.info("Caching common phrases...")
        cached_count = 0
        for phrase in phrases:
            success = await self.sound_cache.get(phrase, self.ssh_client)
            if success[0]:
                cached_count += 1

        logger.info(f"✅ Cached {cached_count}/{len(phrases)} phrases")

    def _handle_stasis_start(self, event):
        """Handle incoming call event"""
        asyncio.create_task(self._process_call(event))

    async def _process_call(self, event):
        """Process an incoming call"""
        channel_id = event.get("channel", {}).get("id")
        if not channel_id:
            return

        try:
            channel = await self.ari_client.channels.get(channelId=channel_id)
            await self._handle_call(channel)
        except Exception as e:
            logger.error(f"❌ Call processing error: {e}")

    async def _handle_call(self, channel):
        """Handle a single call"""
        caller_number = channel.json.get('caller', {}).get('number', 'Unknown')

        logger.info(f"📞 Incoming call from {caller_number}")

        # Create call instance
        call = CallInstance(
            channel=channel,
            ari_client=self.ari_client,
            ai_client=self.ai_client,
            sound_cache=self.sound_cache,
            ssh_client=self.ssh_client,
            transcriber=self.transcriber,
            system_prompt=self.system_prompt,
            deployment=self.azure_openai_deployment,
            ari_url=self.ari_url,
            ari_username=self.ari_username,
            ari_password=self.ari_password
        )

        self.active_calls.add(call)
        self.total_calls += 1

        # Log to database
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
            logger.info(f"✅ Call completed: {caller_number}")

    def _log_call_start(self, call_id, caller_number):
        """Log call start to database"""
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
        """Log call error to database"""
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
        """Log call end to database"""
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
    """Represents a single call with full conversation handling"""

    def __init__(self, channel, ari_client, ai_client, sound_cache, ssh_client,
                 transcriber, system_prompt, deployment, ari_url, ari_username, ari_password):
        self.channel = channel
        self.ari_client = ari_client
        self.ai_client = ai_client
        self.sound_cache = sound_cache
        self.ssh_client = ssh_client
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
        """Process the call with full conversation flow"""
        try:
            # Answer call
            await self.channel.answer()
            await asyncio.sleep(0.2)

            # Time-appropriate greeting
            hour = datetime.now().hour
            if hour < 12:
                time_greeting = 'Good morning'
            elif hour < 17:
                time_greeting = 'Good afternoon'
            else:
                time_greeting = 'Good evening'

            greeting = f"{time_greeting}, thank you for calling. How can I help you today?"
            await self.speak(greeting)
            self.conversation.append({"role": "assistant", "content": greeting})

            # Beep to signal user can speak
            await asyncio.sleep(0.1)
            await self.channel.play(media="sound:beep")
            await asyncio.sleep(0.15)

            # Conversation loop
            no_speech_count = 0
            max_turns = 8  # Maximum conversation turns

            for turn in range(max_turns):
                if not await self.is_alive():
                    break

                self.turn_count += 1

                # Record user speech
                audio_file = await self.record()

                # Play beep after recording
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

                # Transcribe
                text, confidence = await self.transcriber.transcribe(audio_file)
                no_speech_count = 0

                if not text or len(text) < 3:
                    await self.speak("Could you repeat that please?")
                    await asyncio.sleep(0.1)
                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)
                    continue

                logger.info(f"👤 User: {text}")

                # Check for goodbye
                goodbye_words = ["bye", "goodbye", "thanks", "thank you", "done", "that's all"]
                if len(text.split()) <= 5 and any(word in text.lower() for word in goodbye_words):
                    await self.speak("Thank you for calling!")
                    break

                # Add to conversation
                self.conversation.append({"role": "user", "content": text})

                # Get AI response
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

                    # Speak response
                    if not await self.speak(ai_text):
                        break

                    # Beep for next turn
                    await asyncio.sleep(0.1)
                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)

                except Exception as e:
                    logger.error(f"AI error: {e}")
                    await self.speak("I'm experiencing a technical issue. Let me connect you to someone who can help.")
                    break

            # Final goodbye
            if self.active:
                await self.speak("Thank you for calling!")

            await self.hangup()

        except Exception as e:
            logger.error(f"Call processing error: {e}")
            await self.hangup()

    async def is_alive(self):
        """Check if channel is still active"""
        if not self.active:
            return False
        try:
            await self.ari_client.channels.get(channelId=self.id)
            return True
        except:
            self.active = False
            return False

    async def speak(self, text):
        """Speak text to caller using cached TTS"""
        if not await self.is_alive():
            return False

        try:
            sound_path, duration = await self.sound_cache.get(text, self.ssh_client)
            if not sound_path:
                logger.error("Failed to generate/cache audio")
                return False

            await self.channel.play(media=f"sound:{sound_path}")

            # Wait for audio to finish
            estimated_duration = duration or (len(text.split()) * 0.4)
            await asyncio.sleep(estimated_duration + 0.3)

            return True
        except Exception as e:
            if "404" not in str(e):
                logger.error(f"Speak error: {e}")
            self.active = False
            return False

    async def record(self, duration=8, silence=2.0):
        """Record audio from caller"""
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
        """Download recording from Asterisk"""
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
            except Exception as e:
                logger.debug(f"Download attempt {attempt + 1} failed: {e}")

            await asyncio.sleep(0.15)

        return None

    async def hangup(self):
        """Hangup the call"""
        try:
            if self.active:
                await self.channel.hangup()
        except:
            pass
        self.active = False

    async def cleanup(self):
        """Cleanup temporary files"""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass


class SoundCache:
    """Cache for TTS audio files"""

    def __init__(self, cache_dir, index_file):
        self.cache_dir = cache_dir
        self.index_file = index_file
        self.index = self._load_index()

    def _load_index(self):
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                return json.load(open(self.index_file))
            except:
                return {}
        return {}

    def _save_index(self):
        """Save cache index to disk"""
        try:
            json.dump(self.index, open(self.index_file, 'w'))
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def _cache_key(self, text):
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    async def get(self, text, ssh_client):
        """Get cached audio or generate new"""
        key = self._cache_key(text)

        # Check if cached remotely on Asterisk
        if key in self.index and self.index[key].get('remote'):
            remote_path = self.index[key]['remote']
            duration = self.index[key].get('duration')
            return remote_path, duration

        # Generate locally
        local_path = await self._generate_tts(text, key)
        if not local_path:
            return None, None

        # Get duration
        duration = self._get_duration(local_path)

        # Try to upload to Asterisk if SSH is connected
        remote_path = await ssh_client.upload(local_path, f"c_{key}.wav")

        if remote_path:
            # Successfully uploaded - save to index
            self.index[key] = {'remote': remote_path, 'duration': duration}
            self._save_index()
            return remote_path, duration
        else:
            # SSH failed - return local path (won't work for actual calls but good for testing)
            logger.warning(f"Using local audio path (SSH upload failed): {local_path}")
            return local_path, duration

    async def _generate_tts(self, text, key):
        """Generate TTS audio using gTTS or pyttsx3"""
        try:
            output_file = self.cache_dir / f"{key}.wav"
            if output_file.exists():
                return str(output_file)

            # Try gTTS first (better quality)
            try:
                from gtts import gTTS
                temp_file = self.cache_dir / f"{key}_temp.mp3"

                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: gTTS(text=text, lang='en', slow=False).save(str(temp_file))
                )

                # Convert to WAV (8kHz mono for telephony)
                audio = AudioSegment.from_file(str(temp_file))
                audio = normalize(audio).set_frame_rate(8000).set_channels(1).set_sample_width(2)
                audio.export(str(output_file), format="wav")

                try:
                    temp_file.unlink()
                except:
                    pass

                logger.debug(f"Generated TTS with gTTS: {key}")
                return str(output_file)

            except Exception as e:
                logger.debug(f"gTTS failed: {e}, trying pyttsx3...")

                # Fallback to pyttsx3
                try:
                    import pyttsx3
                    temp_file = self.cache_dir / f"{key}_temp.wav"

                    engine = pyttsx3.init()
                    engine.setProperty('rate', 165)
                    engine.save_to_file(text, str(temp_file))
                    engine.runAndWait()
                    engine.stop()

                    # Normalize for telephony
                    audio = AudioSegment.from_file(str(temp_file))
                    audio = normalize(audio).set_frame_rate(8000).set_channels(1).set_sample_width(2)
                    audio.export(str(output_file), format="wav")

                    try:
                        temp_file.unlink()
                    except:
                        pass

                    logger.debug(f"Generated TTS with pyttsx3: {key}")
                    return str(output_file)

                except Exception as e2:
                    logger.error(f"pyttsx3 also failed: {e2}")
                    return None

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

    def _get_duration(self, file_path):
        """Get audio file duration in seconds"""
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
        except:
            return None


class SSHClient:
    """SSH client for uploading audio files to Asterisk sounds directory"""

    def __init__(self, host, port, user, password, sounds_dir):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.sounds_dir = sounds_dir
        self.client = None
        self.sftp = None
        self.lock = asyncio.Lock()
        self.connected = False

    async def connect(self):
        """Connect to SSH server - REAL validation"""
        try:
            logger.info(f"Attempting SSH connection to {self.user}@{self.host}:{self.port}...")

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Run connection in thread pool
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.connect(
                    self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password,
                    timeout=10,
                    look_for_keys=False,
                    allow_agent=False
                )
            )

            # Test SFTP
            self.sftp = self.client.open_sftp()

            # Test directory access
            try:
                self.sftp.stat(self.sounds_dir)
            except FileNotFoundError:
                logger.warning(f"Sounds directory doesn't exist: {self.sounds_dir}")
                # Try to create it
                await self._exec_cmd(f"sudo mkdir -p {self.sounds_dir}")
                await self._exec_cmd(f"sudo chown asterisk:asterisk {self.sounds_dir}")
                await self._exec_cmd(f"sudo chmod 755 {self.sounds_dir}")

            self.connected = True
            logger.info(f"✅ SSH connected to {self.host}")
            return True

        except paramiko.AuthenticationException:
            logger.error(f"❌ SSH authentication failed for {self.user}@{self.host}")
            logger.error("   Check SSH_USER and SSH_PASSWORD in .env")
            self.connected = False
            return False
        except paramiko.SSHException as e:
            logger.error(f"❌ SSH connection error: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"❌ SSH connection failed: {e}")
            self.connected = False
            return False

    async def upload(self, local_path, filename):
        """Upload file to Asterisk sounds directory"""
        if not self.connected:
            logger.debug("SSH not connected - cannot upload")
            return None

        async with self.lock:
            try:
                if not self.sftp:
                    logger.error("SFTP not initialized")
                    return None

                # Upload to /tmp first
                tmp_path = f"/tmp/{filename}"
                await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.sftp.put(local_path, tmp_path)
                )

                # Move to sounds directory with sudo
                final_path = f"{self.sounds_dir}/{filename}"
                await self._exec_cmd(f"sudo mv {tmp_path} {final_path}")
                await self._exec_cmd(f"sudo chown asterisk:asterisk {final_path}")
                await self._exec_cmd(f"sudo chmod 644 {final_path}")

                # Return path without extension for Asterisk
                return f"custom/{filename.replace('.wav', '')}"

            except Exception as e:
                logger.error(f"Upload error: {e}")
                return None

    async def _exec_cmd(self, command):
        """Execute SSH command"""
        try:
            _, stdout, stderr = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self.client.exec_command(command)
            )
            exit_status = stdout.channel.recv_exit_status()

            if exit_status != 0:
                error = stderr.read().decode()
                logger.debug(f"Command failed: {command} - {error}")
                return False

            return True
        except Exception as e:
            logger.debug(f"Command error: {e}")
            return False

    def close(self):
        """Close SSH connection"""
        try:
            if self.sftp:
                self.sftp.close()
            if self.client:
                self.client.close()
            self.connected = False
        except:
            pass


class AzureSpeechTranscriber:
    """Azure Speech Services transcription"""

    def __init__(self, speech_key, speech_region):
        if not speech_key or not speech_region:
            raise ValueError("Azure Speech key and region required")

        self.config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region
        )
        self.config.speech_recognition_language = "en-US"

    async def transcribe(self, audio_file):
        """Transcribe audio file to text"""
        try:
            # Check file size
            if os.path.getsize(audio_file) < 4000:
                return "", "low"

            # Preprocess audio for better recognition
            processed = await self._preprocess(audio_file)

            # Transcribe using Azure
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

            # Cleanup processed file
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
        """Preprocess audio for better recognition"""
        try:
            audio = AudioSegment.from_file(audio_file)
            audio = normalize(audio).set_frame_rate(16000).set_channels(1).set_sample_width(2)

            processed = audio_file.replace('.wav', '_proc.wav')
            audio.export(processed, format="wav")
            return processed
        except:
            return audio_file