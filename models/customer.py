# models/customer.py
"""
Customer, policy, claim, and support ticket models.
Lets the AI look up a caller by phone number and see their claims/ticket
history, and lets the AI (or dashboard) create new tickets.
"""

from datetime import datetime
from . import db


class Customer(db.Model):
    """A known policyholder, keyed by phone number for caller lookup."""

    __tablename__ = 'customers'

    id = db.Column(db.Integer, primary_key=True)

    full_name = db.Column(db.String(150), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False, unique=True, index=True)
    email = db.Column(db.String(120))
    national_id = db.Column(db.String(20))

    customer_since = db.Column(db.Date)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    policies = db.relationship('Policy', backref='customer', lazy='dynamic',
                                cascade='all, delete-orphan')
    claims = db.relationship('Claim', backref='customer', lazy='dynamic',
                              cascade='all, delete-orphan')
    tickets = db.relationship('Ticket', backref='customer', lazy='dynamic',
                               cascade='all, delete-orphan')

    @staticmethod
    def _normalize_phone(number: str) -> str:
        """Normalize to a comparable form: digits only, last 9 (Kenyan local format)."""
        if not number:
            return ""
        digits = "".join(c for c in number if c.isdigit())
        return digits[-9:] if len(digits) >= 9 else digits

    @classmethod
    def find_by_phone(cls, number: str):
        """Lookup tolerant of +254/0/254 prefixes, e.g. caller ID formats."""
        if not number:
            return None
        target = cls._normalize_phone(number)
        if not target:
            return None
        for cust in cls.query.filter_by(is_active=True).all():
            if cls._normalize_phone(cust.phone_number) == target:
                return cust
        return None

    def to_context_dict(self) -> dict:
        """Compact dict for AI prompt injection — keep it short, phone-call sized."""
        open_claims = self.claims.filter(Claim.status != 'closed').all()
        open_tickets = self.tickets.filter(Ticket.status != 'closed').all()
        active_policies = self.policies.filter_by(is_active=True).all()

        return {
            "name": self.full_name,
            "policies": [
                {"type": p.policy_type, "number": p.policy_number, "status": p.status}
                for p in active_policies
            ],
            "open_claims": [
                {
                    "claim_number": c.claim_number,
                    "type": c.claim_type,
                    "status": c.status,
                    "filed": c.filed_at.strftime("%Y-%m-%d") if c.filed_at else None,
                    "summary": c.description[:120] if c.description else "",
                }
                for c in open_claims
            ],
            "open_tickets": [
                {
                    "ticket_number": t.ticket_number,
                    "subject": t.subject,
                    "status": t.status,
                    "created": t.created_at.strftime("%Y-%m-%d") if t.created_at else None,
                }
                for t in open_tickets
            ],
        }

    def __repr__(self):
        return f'<Customer {self.full_name} {self.phone_number}>'


class Policy(db.Model):
    """An insurance policy held by a customer."""

    __tablename__ = 'policies'

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)

    policy_number = db.Column(db.String(40), nullable=False, unique=True, index=True)
    policy_type = db.Column(db.String(30), nullable=False)
    # motor, medical, life, last_expense, home, travel

    status = db.Column(db.String(20), nullable=False, default='active')
    # active, lapsed, cancelled, pending

    premium_amount = db.Column(db.Float)
    premium_frequency = db.Column(db.String(20))  # monthly, quarterly, annual

    start_date = db.Column(db.Date)
    renewal_date = db.Column(db.Date)

    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    claims = db.relationship('Claim', backref='policy', lazy='dynamic')

    def __repr__(self):
        return f'<Policy {self.policy_number} ({self.policy_type})>'


class Claim(db.Model):
    """A claim filed against a policy."""

    __tablename__ = 'claims'

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    policy_id = db.Column(db.Integer, db.ForeignKey('policies.id'))

    claim_number = db.Column(db.String(40), nullable=False, unique=True, index=True)
    claim_type = db.Column(db.String(30), nullable=False)
    # motor_accident, theft, medical, life, fire, travel, other

    description = db.Column(db.Text)

    status = db.Column(db.String(20), nullable=False, default='submitted')
    # submitted, under_review, awaiting_documents, approved, rejected, paid, closed

    amount_claimed = db.Column(db.Float)
    amount_approved = db.Column(db.Float)

    filed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = db.Column(db.DateTime)
    last_update_note = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Claim {self.claim_number} ({self.status})>'


class Ticket(db.Model):
    """
    A support ticket / issue log.
    Can be created by a human via the dashboard, or by the AI during a call
    (is_ai_generated=True) — e.g. "log a callback request" or "raise an issue
    about a delayed claim" without doing a full live transfer.
    """

    __tablename__ = 'tickets'

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    call_id = db.Column(db.Integer, db.ForeignKey('calls.id'))
    # Linked to the Call record if raised during a live AI call

    ticket_number = db.Column(db.String(40), nullable=False, unique=True, index=True)
    subject = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)

    category = db.Column(db.String(30), nullable=False, default='general')
    # claims, billing, policy, complaint, callback_request, general

    priority = db.Column(db.String(10), nullable=False, default='normal')
    # low, normal, high, urgent

    status = db.Column(db.String(20), nullable=False, default='open')
    # open, in_progress, resolved, closed

    # Flags this ticket as having been raised autonomously by the AI agent
    # during a call, rather than by a staff member — surfaced clearly in the
    # dashboard so agents know it needs a first human read-through.
    is_ai_generated = db.Column(db.Boolean, default=False, nullable=False)

    assigned_department_id = db.Column(db.Integer, db.ForeignKey('departments.id'))

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)

    call = db.relationship('Call', backref='tickets')
    assigned_department = db.relationship('Department', backref='tickets')

    @staticmethod
    def generate_ticket_number() -> str:
        """e.g. TCK-20260616-3F2A"""
        import uuid
        return f"TCK-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

    def __repr__(self):
        tag = " [AI]" if self.is_ai_generated else ""
        return f'<Ticket {self.ticket_number}{tag} ({self.status})>'