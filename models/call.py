# models/call.py
"""
Call-related database models for tracking calls, transcripts, and intents.
"""

from datetime import datetime
from . import db


class Call(db.Model):
    """Main call record tracking the entire call session."""

    __tablename__ = 'calls'

    id = db.Column(db.Integer, primary_key=True)

    # Call identification
    call_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    caller_number = db.Column(db.String(20), nullable=False)

    # Timing
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ended_at = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Integer)

    # Status tracking
    status = db.Column(db.String(20), nullable=False, default='active')
    # Status values: active, completed, escalated, error, abandoned

    # Escalation
    escalated = db.Column(db.Boolean, default=False, nullable=False)
    escalated_to_department_id = db.Column(db.Integer, db.ForeignKey('departments.id'))
    escalation_reason = db.Column(db.Text)

    # AI interaction metrics
    total_interactions = db.Column(db.Integer, default=0)
    failed_interactions = db.Column(db.Integer, default=0)

    # Relationships
    transcripts = db.relationship('CallTranscript', backref='call', lazy='dynamic',
                                  cascade='all, delete-orphan')
    intents = db.relationship('CallIntent', backref='call', lazy='dynamic',
                              cascade='all, delete-orphan')
    escalated_to_department = db.relationship('Department', backref='escalated_calls')

    def __repr__(self):
        return f'<Call {self.call_id} from {self.caller_number}>'


class CallTranscript(db.Model):
    """Individual transcript segments within a call."""

    __tablename__ = 'call_transcripts'

    id = db.Column(db.Integer, primary_key=True)
    call_id = db.Column(db.Integer, db.ForeignKey('calls.id'), nullable=False)

    # Transcript data
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    speaker = db.Column(db.String(20), nullable=False)  # 'caller' or 'assistant'
    text = db.Column(db.Text, nullable=False)

    # Confidence scores from speech-to-text
    confidence = db.Column(db.Float)

    # AI response metadata (for assistant messages)
    ai_model = db.Column(db.String(50))
    ai_tokens_used = db.Column(db.Integer)
    ai_response_time_ms = db.Column(db.Integer)

    def __repr__(self):
        return f'<CallTranscript {self.id} - {self.speaker}>'


class CallIntent(db.Model):
    """Intent classification for calls and conversations."""

    __tablename__ = 'call_intents'

    id = db.Column(db.Integer, primary_key=True)
    call_id = db.Column(db.Integer, db.ForeignKey('calls.id'), nullable=False)

    # Intent classification
    detected_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    intent_type = db.Column(db.String(50), nullable=False)
    # Intent types: sales, support, claims, billing, general, escalation

    confidence = db.Column(db.Float)

    # Supporting data
    keywords = db.Column(db.Text)  # JSON array of detected keywords
    context = db.Column(db.Text)  # Additional context from AI

    def __repr__(self):
        return f'<CallIntent {self.intent_type} for Call {self.call_id}>'