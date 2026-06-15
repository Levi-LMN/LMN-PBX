# config.py
"""
Configuration management for the FreePBX AI Assistant application.
Updated for Azure Voice Live API voice integration.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration."""

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # Database
    SQLALCHEMY_DATABASE_URI        = os.getenv("DATABASE_URL", "sqlite:///freepbx_ai.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ARI (Asterisk REST Interface)
    ARI_URL      = os.getenv("ARI_URL",      "http://localhost:8088/ari")
    ARI_BASE     = os.getenv("ARI_BASE",     "http://localhost:8088")
    ARI_USERNAME = os.getenv("ARI_USERNAME", "asterisk")
    ARI_PASSWORD = os.getenv("ARI_PASSWORD", "your_ari_password")
    ARI_APP      = os.getenv("ARI_APP",      "ai-agent")

    # ── Azure Voice Live API (PRIMARY voice path) ─────────────────────────────
    # Single WebSocket connection handles STT (Azure Speech), LLM (GPT-4o Realtime),
    # and TTS (Azure Neural Voices) simultaneously.
    #
    # Required env vars:
    #   AZURE_VOICE_LIVE_RESOURCE  — your AI Foundry resource name
    #                                (the subdomain of .services.ai.azure.com)
    #   AZURE_SPEECH_KEY           — your API key for the resource
    #
    # Optional:
    #   AZURE_VOICE_NAME           — Azure neural voice (default: en-US-AvaNeural)
    #   AZURE_VOICE_TYPE           — "azure-standard" or "azure-custom"
    #   AZURE_VOICE_LIVE_MODEL     — model to use (default: gpt-realtime)
    AZURE_VOICE_LIVE_RESOURCE = os.getenv("AZURE_VOICE_LIVE_RESOURCE", "")
    AZURE_SPEECH_KEY          = os.getenv("AZURE_SPEECH_KEY",          "")
    AZURE_SPEECH_REGION       = os.getenv("AZURE_SPEECH_REGION",       "eastus")
    AZURE_VOICE_NAME          = os.getenv("AZURE_VOICE_NAME",          "en-US-AvaNeural")
    AZURE_VOICE_TYPE          = os.getenv("AZURE_VOICE_TYPE",          "azure-standard")
    AZURE_VOICE_LIVE_MODEL    = os.getenv("AZURE_VOICE_LIVE_MODEL",    "gpt-realtime")

    # RTP port range for ExternalMedia channels (one port-pair per concurrent call)
    RTP_PORT_START = int(os.getenv("RTP_PORT_START", "20000"))
    RTP_PORT_END   = int(os.getenv("RTP_PORT_END",   "20100"))

    # ── Azure OpenAI REST (used by admin dashboard only) ──────────────────────
    # Not on the call hot-path — used for intent classification & KB scoring.
    AZURE_OPENAI_KEY         = os.getenv("AZURE_OPENAI_KEY",         "")
    AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT",    "")
    AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT",  "gpt-4o-mini")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    # ── Legacy OpenAI (no longer used on call hot-path) ───────────────────────
    # Kept so existing dashboard pages that reference OPENAI_API_KEY don't break.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # SSH (for direct file ops, if needed)
    SSH_HOST            = os.getenv("SSH_HOST",   "localhost")
    SSH_PORT            = int(os.getenv("SSH_PORT", "22"))
    SSH_USER            = os.getenv("SSH_USER",   "sangoma")
    SSH_PASSWORD        = os.getenv("SSH_PASSWORD", "sangoma")
    ASTERISK_SOUNDS_DIR = os.getenv("ASTERISK_SOUNDS_DIR", "/var/lib/asterisk/sounds/custom")

    # Dataverse (optional)
    DATAVERSE_URL = os.getenv("DATAVERSE_URL", "")
    TENANT_ID     = os.getenv("TENANT_ID",     "")
    CLIENT_ID     = os.getenv("CLIENT_ID",     "")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

    # Application
    ADMIN_USERNAME       = os.getenv("ADMIN_USERNAME",       "admin")
    ADMIN_PASSWORD       = os.getenv("ADMIN_PASSWORD",       "changeme")
    ESCALATION_THRESHOLD = int(os.getenv("ESCALATION_THRESHOLD", "3"))
    MAX_CALL_DURATION    = int(os.getenv("MAX_CALL_DURATION",    "600"))

    # AI system prompt — passed to Azure Voice Live session.update as `instructions`
    # AI system prompt — passed to Azure Voice Live session.update as `instructions`
    DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", (
        "You are Ari, a friendly and knowledgeable phone assistant for Jubilee Insurance Kenya. "
        "You have access to detailed information about our products: motor, medical, life, last expense, "
        "home, travel insurance, claims processes, and payment methods. "
        "RULES: "
        "(1) Respond in 2-3 natural sentences — enough to be helpful, short enough for a phone call. "
        "(2) Never read out lists or bullet points — weave information conversationally. "
        "(3) Be warm but efficient, like a knowledgeable human agent. "
        "(4) If asked about products broadly, mention the relevant category and invite them to ask more. "
        "(5) Use specific details from what you know — KES amounts, timelines, M-PESA paybill numbers — when relevant. "
        "(6) Never say you are an AI. "
        "(7) Never use filler like Certainly, Of course, or Great question. "
        "(8) If the caller wants a human agent, say you will transfer them now. "
        "EXAMPLE — Caller asks about home insurance: We cover your home structure and contents against fire, "
        "theft, and natural disasters, up to KES 10 million. Premiums are based on your property value — "
        "do you own or rent?"
    ))


class DevelopmentConfig(Config):
    DEBUG   = True
    TESTING = False


class ProductionConfig(Config):
    DEBUG   = False
    TESTING = False

    if not os.getenv("SECRET_KEY"):
        raise ValueError("SECRET_KEY must be set in production environment")

    if not os.getenv("AZURE_SPEECH_KEY"):
        raise ValueError("AZURE_SPEECH_KEY must be set in production environment")

    if not os.getenv("AZURE_VOICE_LIVE_RESOURCE"):
        raise ValueError("AZURE_VOICE_LIVE_RESOURCE must be set in production environment")


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///test.db"


config = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "testing":     TestingConfig,
    "default":     DevelopmentConfig,
}