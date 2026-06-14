# services/ai_service.py
"""
Azure OpenAI REST integration — retained for the admin dashboard's
intent-classification and knowledge-base scoring features.

NOTE: Live call voice I/O now uses the OpenAI Realtime API (WebSocket)
via services/ari_agent.py → RealtimeCallSession.  This module is NOT
called on the hot path of a phone call anymore.
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
    Wrapper for Azure OpenAI REST API (chat completions).
    Used by the admin dashboard for intent classification,
    escalation analysis, and knowledge-base relevance scoring.

    For live call voice responses, see RealtimeCallSession in ari_agent.py.
    """

    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str,
                 system_prompt: str):
        if not AzureOpenAI:
            raise ImportError("OpenAI package not available")
        if not api_key:
            raise ValueError("Azure OpenAI API key is required")
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        endpoint = endpoint.rstrip("/")

        try:
            self.client = AzureOpenAI(
                api_key       = api_key,
                api_version   = api_version,
                azure_endpoint= endpoint,
            )
            logger.info("AIService (REST) client initialised")
        except Exception as e:
            raise RuntimeError(f"Could not initialise Azure OpenAI client: {e}") from e

        self.deployment  = deployment
        self.system_prompt = system_prompt

    # ── generate_response ─────────────────────────────────────────────────────

    def generate_response(
        self,
        conversation_history: List[Dict[str, str]],
        knowledge_context: Optional[str] = None,
        caller_info: Optional[Dict] = None,
    ) -> Tuple[str, Dict]:
        """
        Generate a text response via REST chat completions.
        Used by the admin test panel — not on live call hot path.
        """
        try:
            start = datetime.utcnow()
            messages = [{"role": "system",
                         "content": self._build_system_message(knowledge_context, caller_info)}]
            messages.extend(conversation_history)

            response = self.client.chat.completions.create(
                model             = self.deployment,
                messages          = messages,
                temperature       = 0.7,
                max_tokens        = 500,
                top_p             = 0.95,
                frequency_penalty = 0.3,
                presence_penalty  = 0.3,
            )

            text    = response.choices[0].message.content
            elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
            meta    = {
                "model":           self.deployment,
                "tokens_used":     response.usage.total_tokens,
                "response_time_ms": elapsed,
                "finish_reason":   response.choices[0].finish_reason,
            }
            logger.info(f"REST response ({elapsed} ms, {meta['tokens_used']} tokens)")
            return text, meta

        except Exception as e:
            logger.error(f"generate_response error: {e}")
            return (
                "I apologise, but I'm having trouble processing that right now. "
                "Let me connect you with a specialist who can help.",
                {},
            )

    def _build_system_message(self, knowledge_context, caller_info):
        parts = [self.system_prompt]
        if knowledge_context:
            parts.append(f"\nRelevant Information:\n{knowledge_context}")
        if caller_info:
            ctx = "\nCaller Information:\n"
            if "number" in caller_info:
                ctx += f"- Phone: {caller_info['number']}\n"
            if "previous_calls" in caller_info:
                ctx += f"- Previous calls: {caller_info['previous_calls']}\n"
            parts.append(ctx)
        return "\n".join(parts)

    # ── classify_intent ───────────────────────────────────────────────────────

    def classify_intent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
    ) -> Tuple[str, float, List[str]]:
        """Classify intent via REST — used by admin dashboard analytics."""
        try:
            intent_prompt = """Analyse the user's message and classify their intent.

Categories: sales, support, claims, billing, escalation, general

Respond ONLY with JSON:
{"intent":"category","confidence":0.0,"keywords":["k1"],"reasoning":"..."}"""

            messages = [
                {"role": "system", "content": intent_prompt},
                *conversation_history[-3:],
                {"role": "user", "content": user_message},
            ]
            response = self.client.chat.completions.create(
                model                = self.deployment,
                messages             = messages,
                temperature          = 0.3,
                max_tokens           = 200,
                response_format      = {"type": "json_object"},
            )
            result     = json.loads(response.choices[0].message.content)
            intent     = result.get("intent", "general")
            confidence = result.get("confidence", 0.5)
            keywords   = result.get("keywords", [])
            logger.info(f"Intent: {intent} ({confidence:.2f})")
            return intent, confidence, keywords

        except Exception as e:
            logger.error(f"classify_intent error: {e}")
            return "general", 0.0, []

    # ── should_escalate ───────────────────────────────────────────────────────

    def should_escalate(
        self,
        conversation_history: List[Dict[str, str]],
        failed_interactions: int,
        max_threshold: int,
    ) -> Tuple[bool, str]:
        if failed_interactions >= max_threshold:
            return True, f"Exceeded max failed interactions ({failed_interactions}/{max_threshold})"

        if conversation_history:
            last = next(
                (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
                "",
            ).lower()
            esc_kw = ["speak to", "talk to", "human", "person", "agent",
                      "representative", "manager", "supervisor"]
            if any(k in last for k in esc_kw):
                return True, "User requested human agent"

        try:
            prompt = """Analyse this conversation — should it be escalated to a human agent?
Consider: user frustration, complex issues, repeated misunderstandings.
Respond ONLY with JSON: {"should_escalate":true/false,"reason":"...","urgency":"low/medium/high"}"""
            messages = [{"role": "system", "content": prompt}, *conversation_history[-5:]]
            response = self.client.chat.completions.create(
                model           = self.deployment,
                messages        = messages,
                temperature     = 0.3,
                max_tokens      = 150,
                response_format = {"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            if result.get("should_escalate"):
                reason = result.get("reason", "AI-detected escalation trigger")
                return True, reason
        except Exception as e:
            logger.error(f"should_escalate error: {e}")

        return False, ""

    # ── get_relevant_knowledge ────────────────────────────────────────────────

    def get_relevant_knowledge(self, query: str, knowledge_entries: List[Dict]) -> str:
        if not knowledge_entries:
            return ""
        query_lower   = query.lower()
        scored_entries = []
        for entry in knowledge_entries:
            score    = 0
            keywords = json.loads(entry.get("keywords", "[]"))
            for kw in keywords:
                if kw.lower() in query_lower:
                    score += 1
            if score > 0:
                scored_entries.append((score, entry))
        scored_entries.sort(reverse=True, key=lambda x: x[0])
        top = scored_entries[:3]
        if not top:
            return ""
        parts = ["Reference Information:"]
        for _, e in top:
            parts.append(f"\n{e['title']}:")
            parts.append(e["content"])
        return "\n".join(parts)


# ── factory ───────────────────────────────────────────────────────────────────

def create_ai_service(app_config) -> AIService:
    """Create AIService from Flask app config (used by admin panel)."""
    api_key  = app_config.get("AZURE_OPENAI_KEY", "")
    endpoint = app_config.get("AZURE_OPENAI_ENDPOINT", "")

    if not api_key:
        raise ValueError("AZURE_OPENAI_KEY not configured")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT not configured")

    return AIService(
        api_key       = api_key,
        endpoint      = endpoint,
        deployment    = app_config.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        api_version   = app_config.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        system_prompt = app_config.get("DEFAULT_SYSTEM_PROMPT", "You are a helpful AI assistant."),
    )