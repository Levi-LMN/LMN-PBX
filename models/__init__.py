# models/__init__.py
"""
Database models for the FreePBX AI Assistant application.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Import models to make them available when importing from models package
from .user import User
from .call import Call, CallTranscript, CallIntent
from .department import Department, RoutingRule
from .knowledge import KnowledgeBase

__all__ = [
    'db',
    'User',
    'Call',
    'CallTranscript',
    'CallIntent',
    'Department',
    'RoutingRule',
    'KnowledgeBase'
]