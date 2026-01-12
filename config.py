# config.py
"""
Configuration management for the FreePBX AI Assistant application.
Updated for localhost deployment (running on same machine as FreePBX)
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

    # ARI Configuration (localhost since running on same machine)
    ARI_URL = os.getenv('ARI_URL', 'http://localhost:8088/ari')
    ARI_BASE = os.getenv('ARI_BASE', 'http://localhost:8088')
    ARI_USERNAME = os.getenv('ARI_USERNAME', 'asterisk')
    ARI_PASSWORD = os.getenv('ARI_PASSWORD', 'your_ari_password')
    ARI_APP = os.getenv('ARI_APP', 'ai-agent')

    # SSH Configuration (localhost since running on same machine)
    SSH_HOST = os.getenv('SSH_HOST', 'localhost')
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
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')

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
    DEFAULT_SYSTEM_PROMPT = os.getenv('DEFAULT_SYSTEM_PROMPT', """You are a professional AI assistant for an insurance company.

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
politely recommend speaking with a specialist.""")


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False

    # Override with secure settings in production
    # Make sure SECRET_KEY is set in environment
    if not os.getenv('SECRET_KEY'):
        raise ValueError("SECRET_KEY must be set in production environment")


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