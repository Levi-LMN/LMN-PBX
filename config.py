# config.py
"""
Configuration management for the FreePBX AI Assistant application.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class with common settings."""

    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///freepbx_ai.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ARI Configuration (replaces SIP)
    ARI_URL = os.getenv('ARI_URL', 'http://10.200.200.2:8088/ari')
    ARI_BASE = os.getenv('ARI_BASE', 'http://10.200.200.2:8088')
    ARI_USERNAME = os.getenv('ARI_USERNAME', 'asterisk')
    ARI_PASSWORD = os.getenv('ARI_PASSWORD', 'your_ari_password')
    ARI_APP = os.getenv('ARI_APP', 'ai-agent')

    # SSH Configuration (for audio file upload)
    SSH_HOST = os.getenv('SSH_HOST', '10.200.200.2')
    SSH_PORT = int(os.getenv('SSH_PORT', '22'))
    SSH_USER = os.getenv('SSH_USER', 'sangoma')
    SSH_PASSWORD = os.getenv('SSH_PASSWORD', 'sangoma')
    ASTERISK_SOUNDS_DIR = os.getenv('ASTERISK_SOUNDS_DIR', '/var/lib/asterisk/sounds/custom')

    # Azure Speech Services
    AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY', '')
    AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION', 'eastus')

    # Azure OpenAI
    AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY', '')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')

    # Dataverse (Optional - for customer data)
    DATAVERSE_URL = os.getenv('DATAVERSE_URL', '')
    TENANT_ID = os.getenv('TENANT_ID', '')
    CLIENT_ID = os.getenv('CLIENT_ID', '')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET', '')

    # Application Settings
    ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'changeme')
    ESCALATION_THRESHOLD = int(os.getenv('ESCALATION_THRESHOLD', '3'))
    MAX_CALL_DURATION = int(os.getenv('MAX_CALL_DURATION', '600'))

    # AI System Prompt
    DEFAULT_SYSTEM_PROMPT = os.getenv('DEFAULT_SYSTEM_PROMPT', """You are a professional AI assistant.

Your role is to:
- Answer questions professionally and clearly
- Guide callers through common workflows
- Be empathetic and helpful
- Identify when a caller needs to speak with a human agent

RULES:
- Keep responses between 15-35 words for phone conversations
- Never say "I'm an AI" or similar phrases
- Use natural, conversational language

If you cannot help with a request or if the caller seems frustrated, 
recommend speaking with a specialist.""")


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}