# services/ai_service.py
"""
Azure OpenAI integration for conversational AI responses.
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    from openai import AzureOpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai>=1.0.0")
    AzureOpenAI = None

logger = logging.getLogger(__name__)


class AIService:
    """
    Wrapper for Azure OpenAI API.
    Manages conversation context and generates responses.
    """

    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str,
                 system_prompt: str):
        if not AzureOpenAI:
            raise ImportError("OpenAI package not available")

        if not api_key:
            raise ValueError("Azure OpenAI API key is required")

        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        # Clean up endpoint if needed
        endpoint = endpoint.rstrip('/')

        # Initialize client - try different methods
        client_created = False
        last_error = None

        # Method 1: Standard initialization (newest SDK version >= 1.0.0)
        if not client_created:
            try:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                client_created = True
                logger.info("AI client initialized (method 1: openai >= 1.0.0)")
            except TypeError as e:
                last_error = e
                logger.debug(f"Method 1 failed (likely old SDK version): {e}")
            except Exception as e:
                last_error = e
                logger.debug(f"Method 1 failed: {e}")

        # Method 2: Legacy initialization (openai < 1.0.0)
        if not client_created:
            try:
                import openai
                # Check if this is the old API
                if hasattr(openai, 'api_type'):
                    # Very old version
                    openai.api_type = "azure"
                    openai.api_key = api_key
                    openai.api_base = endpoint
                    openai.api_version = api_version
                    self.client = openai
                    client_created = True
                    logger.info("AI client initialized (method 2: legacy openai < 0.28.0)")
                else:
                    # Try older but not ancient version
                    self.client = AzureOpenAI(
                        api_key=api_key,
                        azure_endpoint=endpoint,
                        api_version=api_version,
                        # Don't pass any extra parameters that might not be supported
                    )
                    client_created = True
                    logger.info("AI client initialized (method 2: openai 0.28.x)")
            except TypeError as e:
                last_error = e
                logger.debug(f"Method 2 failed: {e}")
            except Exception as e:
                last_error = e
                logger.debug(f"Method 2 failed: {e}")

        # Method 3: Using environment variables
        if not client_created:
            try:
                import os
                os.environ['AZURE_OPENAI_API_KEY'] = api_key
                os.environ['AZURE_OPENAI_ENDPOINT'] = endpoint

                self.client = AzureOpenAI(
                    api_version=api_version
                )
                client_created = True
                logger.info("AI client initialized (method 3: environment variables)")
            except Exception as e:
                last_error = e
                logger.debug(f"Method 3 failed: {e}")

        if not client_created:
            error_msg = f"Could not initialize Azure OpenAI client. Last error: {last_error}"
            logger.error(error_msg)
            logger.error(
                f"Configuration - Endpoint: {endpoint}, API Version: {api_version}, Key length: {len(api_key)}")
            logger.error("This may be due to an incompatible version of the 'openai' package.")
            logger.error("Try: pip install --upgrade openai>=1.0.0")
            raise RuntimeError(error_msg)

        self.deployment = deployment
        self.system_prompt = system_prompt

        logger.info(f"AI service initialized with deployment '{deployment}'")

    def generate_response(self,
                          conversation_history: List[Dict[str, str]],
                          knowledge_context: Optional[str] = None,
                          caller_info: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Generate AI response based on conversation history.

        Args:
            conversation_history: List of message dicts with 'role' and 'content'
            knowledge_context: Optional additional context from knowledge base
            caller_info: Optional caller metadata for personalization

        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            start_time = datetime.utcnow()

            # Build messages for API
            messages = [{"role": "system", "content": self._build_system_message(
                knowledge_context, caller_info
            )}]

            # Add conversation history
            messages.extend(conversation_history)

            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=0.95,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )

            # Extract response
            assistant_message = response.choices[0].message.content

            # Calculate metadata
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)

            metadata = {
                'model': self.deployment,
                'tokens_used': response.usage.total_tokens,
                'response_time_ms': response_time_ms,
                'finish_reason': response.choices[0].finish_reason
            }

            logger.info(f"Generated response ({response_time_ms}ms, "
                        f"{metadata['tokens_used']} tokens)")

            return assistant_message, metadata

        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return "I apologize, but I'm having trouble processing that right now. " \
                   "Let me connect you with a specialist who can help.", {}

    def _build_system_message(self, knowledge_context: Optional[str],
                              caller_info: Optional[Dict]) -> str:
        """Build comprehensive system message with context."""
        message_parts = [self.system_prompt]

        if knowledge_context:
            message_parts.append(f"\nRelevant Information:\n{knowledge_context}")

        if caller_info:
            caller_context = f"\nCaller Information:\n"
            if 'number' in caller_info:
                caller_context += f"- Phone: {caller_info['number']}\n"
            if 'previous_calls' in caller_info:
                caller_context += f"- Previous calls: {caller_info['previous_calls']}\n"
            message_parts.append(caller_context)

        return "\n".join(message_parts)

    def classify_intent(self, user_message: str,
                        conversation_history: List[Dict[str, str]]) -> Tuple[str, float, List[str]]:
        """
        Classify the intent of a user message.

        Args:
            user_message: The user's latest message
            conversation_history: Previous conversation context

        Returns:
            Tuple of (intent_type, confidence, keywords)
        """
        try:
            # Build intent classification prompt
            intent_prompt = """Analyze the user's message and classify their intent.

Available intent categories:
- sales: Inquiring about new policies or products
- support: General questions or assistance
- claims: Filing or checking claim status
- billing: Payment or billing questions
- escalation: Requesting to speak with a person
- general: Other inquiries

Respond with a JSON object:
{
    "intent": "category_name",
    "confidence": 0.0-1.0,
    "keywords": ["keyword1", "keyword2"],
    "reasoning": "brief explanation"
}"""

            messages = [
                {"role": "system", "content": intent_prompt},
                *conversation_history[-3:],  # Last 3 messages for context
                {"role": "user", "content": user_message}
            ]

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            # Parse response
            result = json.loads(response.choices[0].message.content)

            intent = result.get('intent', 'general')
            confidence = result.get('confidence', 0.5)
            keywords = result.get('keywords', [])

            logger.info(f"Intent classified: {intent} (confidence: {confidence:.2f})")

            return intent, confidence, keywords

        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return 'general', 0.0, []

    def should_escalate(self, conversation_history: List[Dict[str, str]],
                        failed_interactions: int, max_threshold: int) -> Tuple[bool, str]:
        """
        Determine if the conversation should be escalated to a human.

        Args:
            conversation_history: Full conversation history
            failed_interactions: Count of unsuccessful interactions
            max_threshold: Maximum allowed failed interactions

        Returns:
            Tuple of (should_escalate, reason)
        """
        # Hard threshold on failed interactions
        if failed_interactions >= max_threshold:
            return True, f"Exceeded maximum failed interactions ({failed_interactions}/{max_threshold})"

        # Check for explicit escalation requests
        if conversation_history:
            last_user_message = next(
                (m['content'] for m in reversed(conversation_history)
                 if m['role'] == 'user'),
                ""
            ).lower()

            escalation_keywords = [
                'speak to', 'talk to', 'human', 'person', 'agent',
                'representative', 'manager', 'supervisor'
            ]

            if any(keyword in last_user_message for keyword in escalation_keywords):
                return True, "User requested human agent"

        # Use AI to detect frustration or complex issues
        try:
            escalation_prompt = """Analyze this conversation and determine if it should be 
escalated to a human agent. Consider:
- User frustration or dissatisfaction
- Complex issues beyond basic FAQ
- Sensitive topics requiring human judgment
- Repeated misunderstandings

Respond with JSON:
{
    "should_escalate": true/false,
    "reason": "brief explanation",
    "urgency": "low/medium/high"
}"""

            messages = [
                {"role": "system", "content": escalation_prompt},
                *conversation_history[-5:]  # Last 5 messages
            ]

            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            if result.get('should_escalate', False):
                reason = result.get('reason', 'AI-detected escalation trigger')
                logger.info(f"AI recommends escalation: {reason}")
                return True, reason

        except Exception as e:
            logger.error(f"Escalation check error: {e}")

        return False, ""

    def get_relevant_knowledge(self, query: str, knowledge_entries: List[Dict]) -> str:
        """
        Select and format relevant knowledge base entries for context.

        Args:
            query: User's query or conversation context
            knowledge_entries: List of knowledge base entry dicts

        Returns:
            Formatted context string
        """
        if not knowledge_entries:
            return ""

        # Simple relevance scoring based on keyword matching
        query_lower = query.lower()
        scored_entries = []

        for entry in knowledge_entries:
            score = 0
            keywords = json.loads(entry.get('keywords', '[]'))

            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1

            if score > 0:
                scored_entries.append((score, entry))

        # Sort by relevance and take top entries
        scored_entries.sort(reverse=True, key=lambda x: x[0])
        top_entries = scored_entries[:3]

        if not top_entries:
            return ""

        # Format context
        context_parts = ["Reference Information:"]
        for _, entry in top_entries:
            context_parts.append(f"\n{entry['title']}:")
            context_parts.append(entry['content'])

        return "\n".join(context_parts)


def create_ai_service(app_config) -> AIService:
    """Factory function to create AI service from app config."""
    api_key = app_config.get('AZURE_OPENAI_KEY', '')
    endpoint = app_config.get('AZURE_OPENAI_ENDPOINT', '')

    # Validate configuration
    if not api_key:
        raise ValueError("AZURE_OPENAI_KEY not configured in environment")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT not configured in environment")

    logger.info(f"Creating AI service with endpoint: {endpoint}")

    return AIService(
        api_key=api_key,
        endpoint=endpoint,
        deployment=app_config.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini'),
        api_version=app_config.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
        system_prompt=app_config.get('DEFAULT_SYSTEM_PROMPT', 'You are a helpful AI assistant.')
    )