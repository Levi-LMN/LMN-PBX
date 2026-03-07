# services/ai_service.py
"""
Local IBM Granite (via Ollama) integration for conversational AI responses.
Replaces Azure OpenAI - uses the same openai Python package but pointed at localhost.

SETUP:
  1. Install Ollama: https://ollama.com
  2. Pull a Granite model:
       ollama pull granite3.2:8b     # best quality (requires ~5GB RAM)
       ollama pull granite3.2:2b     # faster / lighter (requires ~2GB RAM)
       ollama pull granite4          # latest (requires ~5GB RAM)
  3. Ollama runs automatically at http://localhost:11434
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    from openai import OpenAI  # Same package, just pointed at Ollama
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai>=1.0.0")
    OpenAI = None

logger = logging.getLogger(__name__)


class AIService:
    """
    Wrapper for local IBM Granite model served via Ollama.
    Uses the OpenAI-compatible API that Ollama exposes at localhost:11434.
    Drop-in replacement for the AzureOpenAI version.
    """

    def __init__(self, ollama_base_url: str, model: str, system_prompt: str):
        if not OpenAI:
            raise ImportError("openai package not available. Run: pip install openai>=1.0.0")

        # Connect to local Ollama server using the OpenAI-compatible endpoint.
        # No real API key is needed - Ollama accepts any string.
        self.client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama",  # Required by the SDK but ignored by Ollama
        )

        self.model = model
        self.system_prompt = system_prompt

        logger.info(f"AI service initialised → Ollama at {ollama_base_url} / model '{model}'")

    # ------------------------------------------------------------------
    # Public API (same signatures as the old AzureOpenAI version)
    # ------------------------------------------------------------------

    def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        knowledge_context: Optional[str] = None,
        caller_info: Optional[Dict] = None,
    ) -> Tuple[str, Dict]:
        """
        Generate AI response based on conversation history.

        Args:
            conversation_history: List of message dicts with 'role' and 'content'
            knowledge_context: Optional additional context from knowledge base
            caller_info: Optional caller metadata for personalisation

        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            start_time = datetime.utcnow()

            messages = [
                {
                    "role": "system",
                    "content": self._build_system_message(knowledge_context, caller_info),
                }
            ]
            messages.extend(conversation_history)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )

            assistant_message = response.choices[0].message.content

            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)

            metadata = {
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "response_time_ms": response_time_ms,
                "finish_reason": response.choices[0].finish_reason,
            }

            logger.info(f"Generated response ({response_time_ms}ms)")
            return assistant_message, metadata

        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return (
                "I apologize, but I'm having trouble processing that right now. "
                "Let me connect you with a specialist who can help.",
                {},
            )

    def classify_intent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
    ) -> Tuple[str, float, List[str]]:
        """
        Classify the intent of a user message.

        Returns:
            Tuple of (intent_type, confidence, keywords)
        """
        try:
            intent_prompt = """Analyze the user's message and classify their intent.

Available intent categories:
- sales: Inquiring about new policies or products
- support: General questions or assistance
- claims: Filing or checking claim status
- billing: Payment or billing questions
- escalation: Requesting to speak with a person
- general: Other inquiries

You MUST respond with ONLY a valid JSON object and nothing else. No explanation, no markdown.
Example:
{"intent": "claims", "confidence": 0.9, "keywords": ["claim", "accident"], "reasoning": "User mentioned filing a claim"}"""

            messages = [
                {"role": "system", "content": intent_prompt},
                *conversation_history[-3:],
                {"role": "user", "content": user_message},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
            )

            raw = response.choices[0].message.content.strip()
            result = self._safe_parse_json(raw)

            intent = result.get("intent", "general")
            confidence = float(result.get("confidence", 0.5))
            keywords = result.get("keywords", [])

            logger.info(f"Intent classified: {intent} (confidence: {confidence:.2f})")
            return intent, confidence, keywords

        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return "general", 0.0, []

    def should_escalate(
        self,
        conversation_history: List[Dict[str, str]],
        failed_interactions: int,
        max_threshold: int,
    ) -> Tuple[bool, str]:
        """
        Determine if the conversation should be escalated to a human.

        Returns:
            Tuple of (should_escalate, reason)
        """
        # Hard threshold check
        if failed_interactions >= max_threshold:
            return True, f"Exceeded maximum failed interactions ({failed_interactions}/{max_threshold})"

        # Check for explicit escalation keywords
        if conversation_history:
            last_user_message = next(
                (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
                "",
            ).lower()

            escalation_keywords = [
                "speak to", "talk to", "human", "person", "agent",
                "representative", "manager", "supervisor",
            ]
            if any(kw in last_user_message for kw in escalation_keywords):
                return True, "User requested human agent"

        # Ask the model
        try:
            escalation_prompt = """Analyze this conversation and decide if it needs a human agent.
Consider: user frustration, complex issues, sensitive topics, repeated misunderstandings.

You MUST respond with ONLY a valid JSON object and nothing else. No explanation, no markdown.
Example:
{"should_escalate": false, "reason": "Simple FAQ question", "urgency": "low"}"""

            messages = [
                {"role": "system", "content": escalation_prompt},
                *conversation_history[-5:],
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=150,
            )

            raw = response.choices[0].message.content.strip()
            result = self._safe_parse_json(raw)

            if result.get("should_escalate", False):
                reason = result.get("reason", "AI-detected escalation trigger")
                logger.info(f"AI recommends escalation: {reason}")
                return True, reason

        except Exception as e:
            logger.error(f"Escalation check error: {e}")

        return False, ""

    def get_relevant_knowledge(
        self, query: str, knowledge_entries: List[Dict]
    ) -> str:
        """
        Select and format relevant knowledge base entries for context.
        (Same keyword-matching logic as before - no AI call needed here.)
        """
        if not knowledge_entries:
            return ""

        query_lower = query.lower()
        scored_entries = []

        for entry in knowledge_entries:
            score = 0
            try:
                keywords = json.loads(entry.get("keywords", "[]"))
            except (json.JSONDecodeError, TypeError):
                keywords = []

            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1

            if score > 0:
                scored_entries.append((score, entry))

        scored_entries.sort(reverse=True, key=lambda x: x[0])
        top_entries = scored_entries[:3]

        if not top_entries:
            return ""

        context_parts = ["Reference Information:"]
        for _, entry in top_entries:
            context_parts.append(f"\n{entry['title']}:")
            context_parts.append(entry["content"])

        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_message(
        self,
        knowledge_context: Optional[str],
        caller_info: Optional[Dict],
    ) -> str:
        """Build comprehensive system message with context."""
        parts = [self.system_prompt]

        if knowledge_context:
            parts.append(f"\nRelevant Information:\n{knowledge_context}")

        if caller_info:
            caller_context = "\nCaller Information:\n"
            if "number" in caller_info:
                caller_context += f"- Phone: {caller_info['number']}\n"
            if "previous_calls" in caller_info:
                caller_context += f"- Previous calls: {caller_info['previous_calls']}\n"
            parts.append(caller_context)

        return "\n".join(parts)

    @staticmethod
    def _safe_parse_json(text: str) -> Dict:
        """
        Parse JSON from model output.
        Granite (and most local models) sometimes wrap JSON in markdown fences
        or add a preamble - this strips those before parsing.
        """
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Drop first and last fence lines
            inner = [l for l in lines if not l.startswith("```")]
            cleaned = "\n".join(inner).strip()

        # Find the first '{' and last '}' in case there's surrounding text
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from model output: {text[:200]}")
            return {}


def create_ai_service(app_config) -> AIService:
    """Factory function to create AI service from app config."""
    base_url = app_config.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = app_config.get("OLLAMA_MODEL", "granite3.2:8b")
    system_prompt = app_config.get("DEFAULT_SYSTEM_PROMPT", "You are a helpful AI assistant.")

    logger.info(f"Creating AI service → model: {model} at {base_url}")

    return AIService(
        ollama_base_url=base_url,
        model=model,
        system_prompt=system_prompt,
    )