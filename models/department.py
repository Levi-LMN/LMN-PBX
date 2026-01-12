# models/department.py
"""
Department and routing rule models for call escalation.
"""

from datetime import datetime
from . import db


class Department(db.Model):
    """Department information for call routing."""

    __tablename__ = 'departments'

    id = db.Column(db.Integer, primary_key=True)

    # Basic information
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)

    # FreePBX integration
    extension = db.Column(db.String(20), nullable=False)
    # Extension number in FreePBX to transfer calls to

    priority = db.Column(db.Integer, default=0)
    # Higher priority departments are preferred for ambiguous cases

    is_active = db.Column(db.Boolean, default=True, nullable=False)

    # Business hours (stored as JSON string)
    business_hours = db.Column(db.Text)
    # Example: {"monday": {"start": "09:00", "end": "17:00"}, ...}

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    routing_rules = db.relationship('RoutingRule', backref='department', lazy='dynamic',
                                    cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Department {self.name} - Ext {self.extension}>'


class RoutingRule(db.Model):
    """Rules for routing calls to departments based on intent and keywords."""

    __tablename__ = 'routing_rules'

    id = db.Column(db.Integer, primary_key=True)
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)

    # Rule configuration
    intent_type = db.Column(db.String(50), nullable=False)
    # Must match one of the intent types from CallIntent

    keywords = db.Column(db.Text)
    # JSON array of keywords that trigger this rule
    # Example: ["claim", "accident", "damage"]

    priority = db.Column(db.Integer, default=0)
    # Higher priority rules are checked first

    conditions = db.Column(db.Text)
    # JSON object with additional conditions
    # Example: {"min_confidence": 0.7, "time_of_day": "business_hours"}

    is_active = db.Column(db.Boolean, default=True, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<RoutingRule {self.intent_type} -> Dept {self.department_id}>'