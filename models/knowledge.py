# models/knowledge.py
"""
Knowledge base model for storing information the AI can reference.
"""

from datetime import datetime
from . import db


class KnowledgeBase(db.Model):
    """Knowledge base entries for AI assistant responses."""

    __tablename__ = 'knowledge_base'

    id = db.Column(db.Integer, primary_key=True)

    # Content
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False, index=True)
    # Categories: policies, claims, billing, coverage, faq, procedures

    content = db.Column(db.Text, nullable=False)
    # Main content that will be used by the AI

    keywords = db.Column(db.Text)
    # JSON array of keywords for searching
    # Example: ["deductible", "out-of-pocket", "maximum"]

    # Metadata
    priority = db.Column(db.Integer, default=0)
    # Higher priority content is more likely to be included in AI context

    is_active = db.Column(db.Boolean, default=True, nullable=False)

    # Usage tracking
    usage_count = db.Column(db.Integer, default=0)
    last_used = db.Column(db.DateTime)

    # Versioning
    version = db.Column(db.Integer, default=1)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.String(80))

    def increment_usage(self):
        """Track when this knowledge entry is used."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()

    def __repr__(self):
        return f'<KnowledgeBase {self.title}>'