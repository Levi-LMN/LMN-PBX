# config.py
"""
Configuration management for the FreePBX AI Assistant application.
Updated for localhost deployment with local IBM Granite model via Ollama.

AI BACKEND: IBM Granite served by Ollama (local, no cloud dependency, no API cost)

SETUP:
  1. Install Ollama from https://ollama.com
  2. Pull your chosen Granite model:
       ollama pull granite3.2:8b    ← recommended (best quality, ~5 GB RAM)
       ollama pull granite3.2:2b    ← lighter option (~2 GB RAM)
       ollama pull granite4         ← latest IBM Granite 4
  3. Ollama starts automatically; it listens on http://localhost:11434
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration class with common settings."""

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///freepbx_ai.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ARI Configuration (localhost since running on same machine as FreePBX)
    ARI_URL = os.getenv("ARI_URL", "http://localhost:8088/ari")
    ARI_BASE = os.getenv("ARI_BASE", "http://localhost:8088")
    ARI_USERNAME = os.getenv("ARI_USERNAME", "asterisk")
    ARI_PASSWORD = os.getenv("ARI_PASSWORD", "your_ari_password")
    ARI_APP = os.getenv("ARI_APP", "ai-agent")

    # SSH Configuration
    SSH_HOST = os.getenv("SSH_HOST", "localhost")
    SSH_PORT = int(os.getenv("SSH_PORT", "22"))
    SSH_USER = os.getenv("SSH_USER", "sangoma")
    SSH_PASSWORD = os.getenv("SSH_PASSWORD", "sangoma")
    ASTERISK_SOUNDS_DIR = os.getenv("ASTERISK_SOUNDS_DIR", "/var/lib/asterisk/sounds/custom")

    # Azure Speech Services (still used for STT/TTS - unaffected by the model swap)
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")

    # ----------------------------------------------------------------
    # LOCAL AI - IBM Granite via Ollama
    # ----------------------------------------------------------------
    # Ollama exposes an OpenAI-compatible REST API at /v1 on port 11434.
    # No API key, no cloud, no cost per token.
    #
    # Model options (set OLLAMA_MODEL in your .env file):
    #   granite3.2:8b  → best quality, needs ~5 GB free RAM
    #   granite3.2:2b  → fast & light, needs ~2 GB free RAM
    #   granite4       → newest IBM Granite 4 model
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite3.2:8b")

    # ----------------------------------------------------------------
    # Azure OpenAI - kept for easy rollback, but no longer the default.
    # To switch BACK to Azure, update ai_service.py to use create_azure_service().
    # ----------------------------------------------------------------
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    # Dataverse (Optional - for customer data)
    DATAVERSE_URL = os.getenv("DATAVERSE_URL", "")
    TENANT_ID = os.getenv("TENANT_ID", "")
    CLIENT_ID = os.getenv("CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

    # Application Settings
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
    ESCALATION_THRESHOLD = int(os.getenv("ESCALATION_THRESHOLD", "3"))
    MAX_CALL_DURATION = int(os.getenv("MAX_CALL_DURATION", "600"))

    # AI System Prompt (unchanged - Granite respects the same instruction format)
    DEFAULT_SYSTEM_PROMPT = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        """You are a professional AI assistant for an insurance company.

Your role is to:
- Answer questions about policies, claims, and billing professionally
- Guide callers through common workflows and procedures
- Be empathetic and helpful with customer concerns
- Identify when a caller needs to speak with a human agent

RULES:
- Keep responses between 15-35 words for phone conversations
- Never say "I'm an AI" or mention being artificial
- Use natural, conversational language
- Be honest when you don't know something

If you cannot help with a request or if the caller seems frustrated,
politely recommend speaking with a specialist.""",
    )


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

    if not os.getenv("SECRET_KEY"):
        raise ValueError("SECRET_KEY must be set in the production environment")


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///test.db"


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}