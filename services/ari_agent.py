# services/ari_agent.py
"""
Enhanced ARI-based agent service with:
- Actual call transfers to human agents
- Knowledge base integration
- Intent-based routing
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
from flask import Flask

logger = logging.getLogger(__name__)


class ARIAgent:
    """ARI-based AI voice agent with call transfer and knowledge base"""

    def __init__(self, app_config, flask_app=None):
        self.config = app_config
        self.flask_app = flask_app
        self.running = False
        self.active_calls = {}
        self.total_calls = 0

        # ARI Configuration
        self.ari_url = os.getenv('ARI_URL', 'http://localhost:8088/ari')
        self.ari_base = os.getenv('ARI_BASE', 'http://localhost:8088')
        self.ari_username = os.getenv('ARI_USERNAME', 'asterisk')
        self.ari_password = os.getenv('ARI_PASSWORD', 'your_ari_password')
        self.ari_app = os.getenv('ARI_APP', 'ai-agent')

        # File system
        self.asterisk_sounds_dir = '/var/lib/asterisk/sounds/custom'

        # Azure configuration
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
        self.azure_openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
        self.azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
        self.azure_speech_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')

        # System prompt
        self.system_prompt = os.getenv('DEFAULT_SYSTEM_PROMPT') or self._default_prompt()

        # Initialize components
        self.cache_dir = Path.home() / ".asterisk_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"

        self.sound_cache = SoundCache(
            self.cache_dir, self.cache_index_file, self.asterisk_sounds_dir,
            azure_speech_key=self.azure_speech_key,
            azure_speech_region=self.azure_speech_region
        )

        # Initialize transcriber
        try:
            self.transcriber = AzureSpeechTranscriber(self.azure_speech_key, self.azure_speech_region)
            logger.info("✅ Speech transcriber initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize transcriber: {e}")
            self.transcriber = None

        # File system access
        self.file_access = FileSystemAccess(self.asterisk_sounds_dir)

        # OpenAI client initialization
        self.ai_client = None
        if self.azure_openai_endpoint and self.azure_openai_key:
            try:
                endpoint = self.azure_openai_endpoint.rstrip('/')
                self.ai_client = AsyncAzureOpenAI(
                    api_key=self.azure_openai_key,
                    azure_endpoint=endpoint,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
                )
                logger.info("✅ OpenAI client object created")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.ai_client = None
        else:
            logger.warning("⚠️ Azure OpenAI not configured")

        # ARI client
        self.ari_client = None

        logger.info("ARI Agent initialized")

    def _default_prompt(self):
        return """You are a professional phone assistant for Jubilee Insurance.

RULES:
- STRICT LIMIT: Respond in 20 words or fewer. This is a phone call — be brief.
- Never exceed 2 short sentences.
- Be helpful, professional, and empathetic.
- Never say "I'm an AI" or mention being artificial.
- Use natural, conversational language.
- If you need more information, ask only one question.

When you cannot help or the caller requests a human, say you will transfer them now."""

    async def start(self):
        """Start the ARI agent"""
        self.running = True

        logger.info("=" * 60)
        logger.info("🤖 ARI Agent Starting")
        logger.info("=" * 60)

        if not self.ai_client:
            logger.error("❌ Cannot start - Azure OpenAI client failed to initialize")
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

            # Register event handlers
            self.ari_client.on_event("StasisStart", self._handle_stasis_start)
            self.ari_client.on_event("StasisEnd", self._handle_stasis_end)
            self.ari_client.on_event("ChannelHangupRequest", self._handle_hangup_request)

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

        for call in list(self.active_calls.values()):
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
            "Let me transfer you to a specialist who can help. Please hold.",
        ]

        logger.info("Caching common phrases...")
        for phrase in phrases:
            await self.sound_cache.get(phrase, self.file_access)

    def _handle_stasis_start(self, event):
        asyncio.create_task(self._process_call(event))

    def _handle_stasis_end(self, event):
        """Handle when a channel leaves the Stasis application"""
        channel_id = event.get("channel", {}).get("id")
        if channel_id and channel_id in self.active_calls:
            logger.info(f"📴 Channel {channel_id[:12]} left Stasis (user hung up)")
            call = self.active_calls[channel_id]
            call.user_hung_up = True
            call.active = False

    def _handle_hangup_request(self, event):
        """Handle hangup request event"""
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

        try:
            channel = await self.ari_client.channels.get(channelId=channel_id)
            await self._handle_call(channel)
        except Exception as e:
            logger.error(f"❌ Call processing error: {e}")

    async def _handle_call(self, channel):
        caller_number = channel.json.get('caller', {}).get('number', 'Unknown')
        channel_state = channel.json.get('state', 'Unknown')
        logger.info(f"📞 Incoming call from {caller_number} (State: {channel_state})")

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
            ari_password=self.ari_password,
            flask_app=self.flask_app
        )

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

    def _log_call_start(self, call_id, caller_number):
        """Log call start to database with proper application context"""
        if not self.flask_app:
            logger.warning("Flask app not available - skipping database logging")
            return

        try:
            with self.flask_app.app_context():
                from models import db, Call
                call = Call(
                    call_id=call_id,
                    caller_number=caller_number,
                    status='active',
                    started_at=datetime.utcnow()
                )
                db.session.add(call)
                db.session.commit()
                logger.info(f"✅ Call {call_id} logged to database")
        except Exception as e:
            logger.error(f"Failed to log call start: {e}")

    def _log_call_error(self, call_id, error_msg):
        """Log call error to database with proper application context"""
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
            logger.error(f"Failed to log error: {e}")

    def _log_call_end(self, call_instance):
        """Log call end to database with proper application context"""
        if not self.flask_app:
            return

        try:
            with self.flask_app.app_context():
                from models import db, Call
                call = Call.query.filter_by(call_id=call_instance.id).first()
                if call:
                    if call_instance.escalated:
                        call.status = 'escalated'
                        call.escalated = True
                        call.escalated_to_department_id = call_instance.escalated_to_dept_id
                        call.escalation_reason = call_instance.escalation_reason
                    elif call_instance.user_hung_up:
                        call.status = 'completed'
                    else:
                        call.status = 'completed'

                    call.ended_at = datetime.utcnow()
                    if call.started_at:
                        call.duration_seconds = int((call.ended_at - call.started_at).total_seconds())
                    call.total_interactions = call_instance.turn_count
                    db.session.commit()
                    logger.info(f"✅ Call {call_instance.id} completed - logged to database")
        except Exception as e:
            logger.error(f"Failed to log call end: {e}")


class CallInstance:
    """Represents a single call with knowledge base and transfer capability"""

    def __init__(self, channel, ari_client, ai_client, sound_cache, file_access,
                 transcriber, system_prompt, deployment, ari_url, ari_username, ari_password,
                 flask_app=None):
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
        self.flask_app = flask_app

        self.id = channel.id
        self.active = True
        self.user_hung_up = False
        self.escalated = False
        self.escalated_to_dept_id = None
        self.escalation_reason = None
        self.temp_files = []
        self.turn_count = 0
        self.conversation = [{"role": "system", "content": system_prompt}]

    def _get_knowledge_context(self, user_text):
        """Get relevant knowledge base entries for the user's query"""
        if not self.flask_app:
            return ""

        try:
            with self.flask_app.app_context():
                from models import KnowledgeBase

                # Search for relevant knowledge entries
                user_lower = user_text.lower()
                all_entries = KnowledgeBase.query.filter_by(is_active=True).all()

                scored_entries = []
                for entry in all_entries:
                    score = 0
                    keywords = json.loads(entry.keywords) if entry.keywords else []

                    # Score based on keyword matches
                    for keyword in keywords:
                        if keyword.lower() in user_lower:
                            score += 2

                    # Score based on title match
                    if any(word in user_lower for word in entry.title.lower().split()):
                        score += 1

                    if score > 0:
                        scored_entries.append((score, entry))

                # Sort by score and take top 2
                scored_entries.sort(reverse=True, key=lambda x: x[0])
                top_entries = scored_entries[:2]

                if not top_entries:
                    return ""

                # Format knowledge context
                context_parts = ["\n\nRELEVANT COMPANY INFORMATION:"]
                for _, entry in top_entries:
                    context_parts.append(f"\n{entry.title}: {entry.content}")

                    # Track usage
                    entry.increment_usage()
                    from models import db
                    db.session.commit()

                return "".join(context_parts)

        except Exception as e:
            logger.error(f"Error getting knowledge context: {e}")
            return ""

    def _detect_transfer_intent(self, user_text):
        """Detect if user wants to speak with a human agent"""
        transfer_keywords = [
            'speak', 'talk', 'human', 'person', 'agent',
            'representative', 'manager', 'supervisor', 'someone',
            'transfer', 'escalate', 'real person'
        ]

        user_lower = user_text.lower()
        return any(keyword in user_lower for keyword in transfer_keywords)

    def _classify_intent(self, user_text):
        """Simple intent classification based on keywords"""
        user_lower = user_text.lower()

        intent_keywords = {
            'sales': ['buy', 'purchase', 'new policy', 'quote', 'coverage', 'insurance'],
            'claims': ['claim', 'accident', 'damage', 'file', 'incident'],
            'billing': ['bill', 'payment', 'pay', 'invoice', 'charge', 'cost'],
            'support': ['help', 'question', 'how', 'what', 'when', 'status']
        }

        for intent_type, keywords in intent_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                return intent_type

        return 'general'

    def _get_department_for_intent(self, intent_type):
        """Get the appropriate department based on intent"""
        if not self.flask_app:
            return None

        try:
            with self.flask_app.app_context():
                from models import Department, RoutingRule

                # Try to find a routing rule for this intent
                rule = RoutingRule.query.filter_by(
                    intent_type=intent_type,
                    is_active=True
                ).order_by(RoutingRule.priority.desc()).first()

                if rule and rule.department:
                    return rule.department

                # Fallback: find department by name matching intent
                dept_name_map = {
                    'sales': 'Sales',
                    'claims': 'Claims',
                    'billing': 'Billing',
                    'support': 'Support'
                }

                if intent_type in dept_name_map:
                    dept = Department.query.filter_by(
                        name=dept_name_map[intent_type],
                        is_active=True
                    ).first()
                    if dept:
                        return dept

                # Ultimate fallback: highest priority department
                return Department.query.filter_by(is_active=True).order_by(
                    Department.priority.desc()
                ).first()

        except Exception as e:
            logger.error(f"Error getting department: {e}")
            return None

    async def transfer_to_department(self, department):
        """Actually transfer the call to a department extension"""
        try:
            logger.info(f"🔀 Transferring call to {department.name} (ext {department.extension})")

            # Inform the caller
            transfer_msg = f"Transferring you to {department.name} now. Please hold."
            await self.speak(transfer_msg)
            self._log_transcript('assistant', transfer_msg, 1.0)

            await asyncio.sleep(0.5)

            # Perform the transfer using ARI
            # This continues the call to the specified extension in the dialplan
            await self.channel.continueInDialplan(
                context='from-internal',  # FreePBX default context
                extension=department.extension,
                priority=1
            )

            logger.info(f"✅ Call transferred to extension {department.extension}")
            self.escalated = True
            self.escalated_to_dept_id = department.id
            self.escalation_reason = f"User requested transfer to {department.name}"

            # Log the intent
            self._log_intent('escalation', 1.0, f"Transferred to {department.name}")

            return True

        except Exception as e:
            logger.error(f"❌ Transfer failed: {e}")
            await self.speak(
                "I apologize, but I'm having trouble transferring your call. Please hold while I try again.")
            return False

    async def process(self):
        try:
            # Answer the call
            channel_state = self.channel.json.get('state', 'Unknown')
            logger.info(f"Channel state before answer: {channel_state}")

            try:
                await self.channel.answer()
                logger.info("✅ Call answered successfully")
            except Exception as e:
                logger.error(f"Failed to answer call: {e}")
                current_channel = await self.ari_client.channels.get(channelId=self.id)
                current_state = current_channel.json.get('state', 'Unknown')
                if current_state.lower() != 'up':
                    raise
                logger.info("Channel is already up, continuing...")

            await asyncio.sleep(0.2)

            # Greeting
            hour = datetime.now().hour
            time_greeting = 'Good morning' if hour < 12 else 'Good afternoon' if hour < 17 else 'Good evening'
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
            for turn in range(8):
                if self.user_hung_up or not await self.is_alive():
                    logger.info("📡 Call ended by user")
                    break

                self.turn_count += 1

                audio_file = await self.record()

                if self.user_hung_up or not await self.is_alive():
                    logger.info("📡 User hung up during recording")
                    break

                await self.channel.play(media="sound:beep")
                await asyncio.sleep(0.1)

                if not audio_file:
                    no_speech_count += 1
                    if no_speech_count >= 2:
                        await self.speak("I'm having trouble hearing you. Please try calling back.")
                        break

                    if not await self.speak("I didn't catch that. Please go ahead."):
                        break

                    await asyncio.sleep(0.1)
                    if not await self.is_alive():
                        break

                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)
                    continue

                text, confidence = await self.transcriber.transcribe(audio_file)
                no_speech_count = 0

                if not text or len(text) < 3:
                    if not await self.speak("Could you repeat that please?"):
                        break

                    await asyncio.sleep(0.1)
                    if not await self.is_alive():
                        break

                    await self.channel.play(media="sound:beep")
                    await asyncio.sleep(0.15)
                    continue

                logger.info(f"👤 User: {text}")
                self._log_transcript('caller', text, confidence)

                # Check for goodbye
                if len(text.split()) <= 5 and any(w in text.lower() for w in ["bye", "goodbye", "thanks", "done"]):
                    goodbye = "Thank you for calling!"
                    await self.speak(goodbye)
                    self._log_transcript('assistant', goodbye, 1.0)
                    break

                # Check if user wants to be transferred
                if self._detect_transfer_intent(text):
                    logger.info("🔀 Transfer intent detected")

                    # Classify the intent to determine department
                    intent_type = self._classify_intent(text)
                    logger.info(f"📊 Intent classified as: {intent_type}")
                    self._log_intent(intent_type, 0.8, text)

                    # Get the appropriate department
                    department = self._get_department_for_intent(intent_type)

                    if department:
                        # Perform the transfer
                        success = await self.transfer_to_department(department)
                        if success:
                            # Call will continue in dialplan, exit our loop
                            return
                        else:
                            # Transfer failed, continue conversation
                            error_msg = "I apologize for the difficulty. Let me try to help you another way. What can I assist you with?"
                            await self.speak(error_msg)
                            self._log_transcript('assistant', error_msg, 1.0)
                            continue
                    else:
                        # No department found
                        logger.warning("⚠️ No department found for transfer")
                        fallback_msg = "I'd like to connect you with someone, but I'm having trouble right now. Can I help you with something else?"
                        await self.speak(fallback_msg)
                        self._log_transcript('assistant', fallback_msg, 1.0)
                        continue

                # Get knowledge base context for this query
                knowledge_context = self._get_knowledge_context(text)

                # Build AI message with knowledge
                user_message_with_context = text
                if knowledge_context:
                    user_message_with_context = f"{text}{knowledge_context}"

                self.conversation.append({"role": "user", "content": user_message_with_context})

                try:
                    response = await self.ai_client.chat.completions.create(
                        model=self.deployment,
                        messages=self.conversation,
                        max_tokens=60,
                        temperature=0.5
                    )

                    ai_text = response.choices[0].message.content.strip()

                    # Store only the response without knowledge context
                    self.conversation.append({"role": "assistant", "content": ai_text})
                    logger.info(f"🤖 AI: {ai_text}")

                    self._log_transcript('assistant', ai_text, 1.0)

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
                    self._log_transcript('assistant', error_msg, 1.0)

                    # Try to transfer to support
                    dept = self._get_department_for_intent('support')
                    if dept:
                        await self.transfer_to_department(dept)
                        return
                    break

            # Only say goodbye if user didn't hang up and we're not transferring
            if self.active and not self.user_hung_up and not self.escalated and await self.is_alive():
                final_msg = "Thank you for calling!"
                await self.speak(final_msg)
                self._log_transcript('assistant', final_msg, 1.0)

            await self.hangup()

        except Exception as e:
            if "Not Found" in str(e):
                logger.info("📡 User hung up (channel not found)")
                self.user_hung_up = True
            else:
                logger.error(f"Call processing error: {e}")
            await self.hangup()

    def _log_transcript(self, speaker, text, confidence):
        """Log transcript to database with proper application context"""
        if not self.flask_app:
            return

        try:
            with self.flask_app.app_context():
                from models import db, Call, CallTranscript

                call = Call.query.filter_by(call_id=self.id).first()
                if not call:
                    return

                transcript = CallTranscript(
                    call_id=call.id,
                    speaker=speaker,
                    text=text,
                    confidence=confidence if isinstance(confidence, float) else 0.0,
                    timestamp=datetime.utcnow()
                )
                db.session.add(transcript)
                db.session.commit()

        except Exception as e:
            logger.error(f"Failed to log transcript: {e}")

    def _log_intent(self, intent_type, confidence, context):
        """Log detected intent to database"""
        if not self.flask_app:
            return

        try:
            with self.flask_app.app_context():
                from models import db, Call, CallIntent

                call = Call.query.filter_by(call_id=self.id).first()
                if not call:
                    return

                intent = CallIntent(
                    call_id=call.id,
                    intent_type=intent_type,
                    confidence=confidence,
                    context=context,
                    detected_at=datetime.utcnow()
                )
                db.session.add(intent)
                db.session.commit()

        except Exception as e:
            logger.error(f"Failed to log intent: {e}")

    async def is_alive(self):
        """Check if channel is still active"""
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

    async def speak(self, text):
        """Speak text to user, return False if call ended"""
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
            if "Not Found" in str(e):
                self.user_hung_up = True
                logger.info("📡 User hung up during speech")
            elif "404" not in str(e):
                logger.error(f"Speak error: {e}")
            self.active = False
            return False

    async def record(self, duration=8, silence=2.0):
        """Record audio from user"""
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
            if "Not Found" in str(e):
                self.user_hung_up = True
                logger.info("📡 User hung up during recording")
            else:
                logger.error(f"Record error: {e}")
            return None

    async def _download_recording(self, name):
        """Download recorded audio file"""
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
        """Hang up the call gracefully"""
        try:
            if self.active and not self.user_hung_up and not self.escalated:
                await self.channel.hangup()
        except Exception as e:
            if "Not Found" not in str(e):
                logger.debug(f"Hangup error: {e}")
        self.active = False

    async def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except:
                pass


class SoundCache:
    """Cache for TTS audio"""

    def __init__(self, cache_dir, index_file, asterisk_sounds_dir,
                 azure_speech_key=None, azure_speech_region='eastus'):
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

            # Use Azure Speech TTS (faster, no external HTTP round-trip, same region as STT)
            try:
                import azure.cognitiveservices.speech as speechsdk

                speech_config = speechsdk.SpeechConfig(
                    subscription=self.azure_speech_key,
                    region=self.azure_speech_region
                )
                speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff8Khz16BitMonoPcm
                )

                audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_file))
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=speech_config,
                    audio_config=audio_config
                )

                result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: synthesizer.speak_text_async(text).get()
                )

                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    logger.debug(f"Azure TTS generated: {text[:40]}...")
                    return str(output_file)
                else:
                    logger.error(f"Azure TTS failed: {result.reason}")
                    # Fall through to gTTS fallback

            except Exception as e:
                logger.warning(f"Azure TTS error, falling back to gTTS: {e}")

            # Fallback: gTTS
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