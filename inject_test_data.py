#!/usr/bin/env python
"""
Inject realistic Kenyan data into FreePBX AI Assistant database.
Populate the database with sample data for testing.

Usage: python inject_kenyan_data.py
"""

import random
import json
from datetime import datetime, timedelta
from app import create_app
from models import db, User, Call, CallTranscript, CallIntent, Department, RoutingRule, KnowledgeBase

# Kenyan phone number prefixes
KENYAN_PREFIXES = [
    '0710', '0711', '0712', '0713', '0714', '0715', '0716', '0717', '0718', '0719',  # Safaricom
    '0720', '0721', '0722', '0723', '0724', '0725', '0726', '0727', '0728', '0729',  # Safaricom
    '0740', '0741', '0742', '0743', '0745', '0746', '0748',  # Airtel
    '0750', '0751', '0752', '0753', '0754', '0755', '0756', '0757', '0758', '0759',  # Airtel
    '0760', '0761', '0762', '0763', '0764', '0765', '0766', '0767', '0768', '0769',  # Telkom
]

# Kenyan names
KENYAN_FIRST_NAMES = [
    'Wanjiru', 'Kamau', 'Njeri', 'Otieno', 'Akinyi', 'Mwangi', 'Wambui', 'Omondi',
    'Chebet', 'Kipchoge', 'Nyambura', 'Kariuki', 'Adhiambo', 'Kimani', 'Wairimu',
    'Odhiambo', 'Jeptoo', 'Mutua', 'Njoroge', 'Atieno', 'Kiprono', 'Wangari',
    'Onyango', 'Chepkoech', 'Karanja', 'Auma', 'Ruto', 'Mumbi', 'Okello', 'Jepchumba'
]

KENYAN_LAST_NAMES = [
    'Wanjiku', 'Kamau', 'Ochieng', 'Kiplagat', 'Muthoni', 'Omondi', 'Cheruiyot',
    'Kariuki', 'Otieno', 'Chepkwony', 'Wambugu', 'Onyango', 'Kiprotich', 'Nyambura',
    'Okoth', 'Chesang', 'Muriuki', 'Awuor', 'Kibet', 'Wanjiru', 'Odero', 'Jepkorir',
    'Ndungu', 'Atieno', 'Kiptoo', 'Njoroge', 'Achieng', 'Rotich', 'Wangui', 'Ouma'
]

# Kenyan locations/counties
KENYAN_LOCATIONS = [
    'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 'Malindi',
    'Kitale', 'Garissa', 'Kakamega', 'Nyeri', 'Meru', 'Machakos', 'Kilifi',
    'Kiambu', 'Embu', 'Kericho', 'Bungoma', 'Kisii', 'Naivasha'
]

# Insurance-related conversation starters
CONVERSATION_STARTERS = [
    "Habari, nataka kuuliza kuhusu insurance cover yangu",
    "Hello, I need to file a claim for my car accident",
    "Jambo, je, mnaweza kunisaidia na hospital bill?",
    "Good morning, I want to inquire about life insurance",
    "Niaje, nataka kujua about premiums za insurance yangu",
    "Hi, my house was damaged and I need to make a claim",
    "Sasa, nataka kuongeza beneficiary kwa policy yangu",
    "Hello, I haven't received my insurance card",
    "Niulize tu, je, insurance cover inashinda lini?",
    "Good afternoon, I want to upgrade my insurance package",
    "Habari ya leo, nataka kujua what is covered in my policy",
    "Hi there, I was involved in an accident last week",
    "Jambo sana, je, mnaweza cover dental services?",
    "Hello, I need to update my contact information",
    "Sasa buda, nataka kucancel insurance yangu"
]

# AI responses (insurance-related)
AI_RESPONSES = [
    "Thank you for calling. I can help you with that. Can you please provide your policy number?",
    "I understand you need assistance. Let me pull up your account details.",
    "Pole sana for the inconvenience. I'm here to help you resolve this.",
    "Based on your policy, you are covered for that service. Let me explain the process.",
    "I see that you're calling about a claim. I'll guide you through the steps.",
    "Your premium is due on the 15th of next month. Would you like to make a payment now?",
    "I can help you update your information. What would you like to change?",
    "Let me check your coverage details. One moment please.",
    "I understand this is urgent. Let me connect you with our claims department.",
    "Your policy covers hospital bills up to KES 500,000 per year."
]

# Intent types and their probabilities
INTENT_TYPES = {
    'claims': 0.25,
    'billing': 0.20,
    'support': 0.20,
    'sales': 0.15,
    'coverage': 0.12,
    'general': 0.08
}

# Escalation reasons
ESCALATION_REASONS = [
    "Complex claim requiring specialist review",
    "Customer requested human agent",
    "Payment processing issue",
    "Policy modification needed",
    "Exceeded failed interaction threshold",
    "Sensitive medical information disclosure",
    "Customer dissatisfaction with AI responses",
    "Technical issue with account access"
]


def generate_kenyan_phone():
    """Generate a realistic Kenyan phone number."""
    prefix = random.choice(KENYAN_PREFIXES)
    suffix = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return f"{prefix}{suffix}"


def generate_kenyan_name():
    """Generate a Kenyan name."""
    first = random.choice(KENYAN_FIRST_NAMES)
    last = random.choice(KENYAN_LAST_NAMES)
    return f"{first} {last}"


def weighted_choice(choices_dict):
    """Select random item based on weights."""
    items = list(choices_dict.keys())
    weights = list(choices_dict.values())
    return random.choices(items, weights=weights, k=1)[0]


def create_departments():
    """Create realistic Kenyan insurance departments."""
    print("Creating departments...")

    departments = [
        {
            'name': 'Claims Department',
            'description': 'Handle all insurance claims and settlements',
            'extension': '2001',
            'priority': 10
        },
        {
            'name': 'Sales & New Business',
            'description': 'New policy inquiries and sales',
            'extension': '2002',
            'priority': 8
        },
        {
            'name': 'Customer Support',
            'description': 'General customer service and support',
            'extension': '2003',
            'priority': 7
        },
        {
            'name': 'Billing & Payments',
            'description': 'Premium payments and billing inquiries',
            'extension': '2004',
            'priority': 9
        },
        {
            'name': 'Medical Underwriting',
            'description': 'Medical insurance and health-related claims',
            'extension': '2005',
            'priority': 10
        },
        {
            'name': 'Motor Insurance',
            'description': 'Vehicle insurance and accident claims',
            'extension': '2006',
            'priority': 9
        },
    ]

    created = []
    for dept_data in departments:
        dept = Department.query.filter_by(name=dept_data['name']).first()
        if not dept:
            dept = Department(**dept_data)
            db.session.add(dept)
            created.append(dept)

    db.session.commit()
    print(f"✓ Created {len(created)} departments")
    return Department.query.all()


def create_routing_rules(departments):
    """Create routing rules for departments."""
    print("Creating routing rules...")

    rules = [
        {
            'department': 'Claims Department',
            'intent_type': 'claims',
            'keywords': json.dumps(['claim', 'accident', 'damage', 'madharau', 'ajali']),
            'priority': 10
        },
        {
            'department': 'Billing & Payments',
            'intent_type': 'billing',
            'keywords': json.dumps(['payment', 'premium', 'malipo', 'bill', 'invoice']),
            'priority': 9
        },
        {
            'department': 'Sales & New Business',
            'intent_type': 'sales',
            'keywords': json.dumps(['buy', 'purchase', 'new policy', 'nunua', 'quote']),
            'priority': 8
        },
        {
            'department': 'Customer Support',
            'intent_type': 'support',
            'keywords': json.dumps(['help', 'assistance', 'msaada', 'question', 'issue']),
            'priority': 7
        },
        {
            'department': 'Medical Underwriting',
            'intent_type': 'coverage',
            'keywords': json.dumps(['hospital', 'medical', 'health', 'daktari', 'ugonjwa']),
            'priority': 9
        },
        {
            'department': 'Motor Insurance',
            'intent_type': 'claims',
            'keywords': json.dumps(['car', 'vehicle', 'gari', 'motor', 'accident']),
            'priority': 9
        },
    ]

    created = 0
    for rule_data in rules:
        dept = Department.query.filter_by(name=rule_data['department']).first()
        if dept:
            existing = RoutingRule.query.filter_by(
                department_id=dept.id,
                intent_type=rule_data['intent_type']
            ).first()

            if not existing:
                rule = RoutingRule(
                    department_id=dept.id,
                    intent_type=rule_data['intent_type'],
                    keywords=rule_data['keywords'],
                    priority=rule_data['priority']
                )
                db.session.add(rule)
                created += 1

    db.session.commit()
    print(f"✓ Created {created} routing rules")


def create_knowledge_base():
    """Create insurance knowledge base entries."""
    print("Creating knowledge base...")

    entries = [
        {
            'title': 'Comprehensive Motor Insurance Coverage',
            'category': 'coverage',
            'content': 'Our comprehensive motor insurance covers damage to your vehicle, third-party liability, theft, fire, and natural disasters. Coverage extends to all Kenyan roads including within East Africa. Maximum coverage is KES 5,000,000.',
            'keywords': json.dumps(['motor', 'car', 'gari', 'comprehensive', 'coverage', 'accident']),
            'priority': 10
        },
        {
            'title': 'Medical Insurance Hospital Coverage',
            'category': 'policies',
            'content': 'Medical cover includes inpatient and outpatient services at over 200 hospitals across Kenya. Annual limit is KES 2,000,000. Covers consultation, medication, surgery, and emergency services. Maternity covered after 10 months.',
            'keywords': json.dumps(['medical', 'hospital', 'health', 'daktari', 'treatment', 'inpatient']),
            'priority': 10
        },
        {
            'title': 'Claims Process and Timeline',
            'category': 'claims',
            'content': 'To file a claim: 1) Report incident within 48 hours, 2) Submit required documents, 3) Assessment by our team, 4) Approval and payment. Timeline: Motor claims 7-14 days, Medical claims 5-10 days, Property claims 14-21 days.',
            'keywords': json.dumps(['claim', 'process', 'timeline', 'file', 'madai', 'settlement']),
            'priority': 9
        },
        {
            'title': 'Premium Payment Methods',
            'category': 'billing',
            'content': 'Pay premiums via M-PESA (Paybill 400200), bank transfer, or at any of our branches. Annual, semi-annual, and monthly payment plans available. Grace period is 30 days. Late payment attracts 2% penalty per month.',
            'keywords': json.dumps(['payment', 'premium', 'malipo', 'mpesa', 'paybill', 'cost']),
            'priority': 8
        },
        {
            'title': 'Life Insurance Benefits',
            'category': 'policies',
            'content': 'Life insurance provides financial security for your family. Coverage from KES 500,000 to KES 10,000,000. Includes accidental death benefit, terminal illness cover, and funeral expenses. Premiums start from KES 2,500 per month.',
            'keywords': json.dumps(['life', 'death', 'beneficiary', 'family', 'protection', 'cover']),
            'priority': 9
        },
        {
            'title': 'Travel Insurance Coverage',
            'category': 'coverage',
            'content': 'Travel insurance covers medical emergencies abroad, trip cancellation, lost luggage, and flight delays. Valid for business and leisure travel worldwide. Coverage from KES 100,000 to KES 5,000,000. Premium starts at KES 1,500 per trip.',
            'keywords': json.dumps(['travel', 'trip', 'abroad', 'safari', 'luggage', 'flight']),
            'priority': 7
        },
        {
            'title': 'Home Insurance Protection',
            'category': 'policies',
            'content': 'Protects your home and contents against fire, theft, natural disasters, and vandalism. Covers building structure and household items. Optional: domestic worker liability. Coverage up to KES 10,000,000. Premium based on property value.',
            'keywords': json.dumps(['home', 'house', 'nyumba', 'property', 'fire', 'theft', 'burglary']),
            'priority': 8
        },
        {
            'title': 'Third Party Motor Insurance',
            'category': 'policies',
            'content': 'Minimum legal requirement in Kenya. Covers damage/injury to third parties only. Annual premium: Private cars KES 5,500-8,000, Matatus KES 25,000-50,000, Lorries KES 15,000-35,000. Certificate valid for 12 months.',
            'keywords': json.dumps(['third party', 'motor', 'minimum', 'legal', 'requirement', 'certificate']),
            'priority': 9
        },
    ]

    created = 0
    for entry_data in entries:
        existing = KnowledgeBase.query.filter_by(title=entry_data['title']).first()
        if not existing:
            entry = KnowledgeBase(**entry_data, created_by='system')
            db.session.add(entry)
            created += 1

    db.session.commit()
    print(f"✓ Created {created} knowledge base entries")


def create_calls(num_calls=100):
    """Create realistic call records."""
    print(f"Creating {num_calls} calls...")

    departments = Department.query.all()
    if not departments:
        print("⚠ No departments found. Creating departments first.")
        departments = create_departments()

    calls_created = 0

    # Generate calls over the past 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    for i in range(num_calls):
        # Random date within range
        random_days = random.uniform(0, 30)
        call_time = end_date - timedelta(days=random_days)

        # Generate call data with unique call_id
        caller_number = generate_kenyan_phone()
        call_id = f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}_{random.randint(1000, 9999)}"

        # Determine if call was escalated (30% chance)
        escalated = random.random() < 0.30
        status = 'escalated' if escalated else random.choice(['completed', 'completed', 'completed', 'active'])

        # Call duration (2-10 minutes)
        duration = random.randint(120, 600) if status == 'completed' else None

        # Number of interactions
        total_interactions = random.randint(3, 15)
        failed_interactions = random.randint(0, 3) if escalated else random.randint(0, 2)

        # Create call
        call = Call(
            call_id=call_id,
            caller_number=caller_number,
            started_at=call_time,
            ended_at=call_time + timedelta(seconds=duration) if duration else None,
            duration_seconds=duration,
            status=status,
            escalated=escalated,
            escalated_to_department_id=random.choice(departments).id if escalated else None,
            escalation_reason=random.choice(ESCALATION_REASONS) if escalated else None,
            total_interactions=total_interactions,
            failed_interactions=failed_interactions
        )

        db.session.add(call)
        db.session.flush()  # Get call.id

        # Create conversation transcripts
        create_transcripts_for_call(call, total_interactions)

        # Create intent classification
        intent_type = weighted_choice(INTENT_TYPES)
        create_intent_for_call(call, intent_type)

        calls_created += 1

        if (i + 1) % 20 == 0:
            db.session.commit()
            print(f"  ✓ Created {i + 1}/{num_calls} calls...")

    db.session.commit()
    print(f"✓ Created {calls_created} calls with transcripts and intents")


def create_transcripts_for_call(call, num_interactions):
    """Create realistic conversation transcripts for a call."""

    for i in range(num_interactions):
        timestamp = call.started_at + timedelta(seconds=i * 30)

        # Alternate between caller and assistant
        if i % 2 == 0:  # Caller
            text = random.choice(CONVERSATION_STARTERS)
            speaker = 'caller'
            confidence = random.uniform(0.85, 0.98)

            transcript = CallTranscript(
                call_id=call.id,
                speaker=speaker,
                text=text,
                confidence=confidence,
                timestamp=timestamp
            )
        else:  # Assistant
            text = random.choice(AI_RESPONSES)
            speaker = 'assistant'

            transcript = CallTranscript(
                call_id=call.id,
                speaker=speaker,
                text=text,
                timestamp=timestamp,
                ai_model='gpt-4o-mini',
                ai_tokens_used=random.randint(50, 200),
                ai_response_time_ms=random.randint(500, 2000)
            )

        db.session.add(transcript)


def create_intent_for_call(call, intent_type):
    """Create intent classification for a call."""

    keywords_map = {
        'claims': ['claim', 'accident', 'damage'],
        'billing': ['payment', 'premium', 'bill'],
        'support': ['help', 'question', 'issue'],
        'sales': ['buy', 'new', 'quote'],
        'coverage': ['cover', 'policy', 'benefits'],
        'general': ['inquiry', 'information', 'details']
    }

    intent = CallIntent(
        call_id=call.id,
        intent_type=intent_type,
        confidence=random.uniform(0.75, 0.95),
        keywords=json.dumps(keywords_map.get(intent_type, [])),
        detected_at=call.started_at + timedelta(seconds=30)
    )

    db.session.add(intent)


def create_users():
    """Create additional user accounts."""
    print("Creating additional users...")

    users_data = [
        {'username': 'manager', 'email': 'manager@insurance.co.ke', 'role': 'manager', 'password': 'manager123'},
        {'username': 'viewer', 'email': 'viewer@insurance.co.ke', 'role': 'viewer', 'password': 'viewer123'},
        {'username': generate_kenyan_name().lower().replace(' ', '.'), 'email': 'staff1@insurance.co.ke',
         'role': 'manager', 'password': 'password'},
        {'username': generate_kenyan_name().lower().replace(' ', '.'), 'email': 'staff2@insurance.co.ke',
         'role': 'viewer', 'password': 'password'},
    ]

    created = 0
    for user_data in users_data:
        existing = User.query.filter_by(username=user_data['username']).first()
        if not existing:
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                role=user_data['role']
            )
            user.set_password(user_data['password'])
            db.session.add(user)
            created += 1

    db.session.commit()
    print(f"✓ Created {created} users")


def main():
    """Main injection script."""
    print("=" * 60)
    print("FreePBX AI Assistant - Kenyan Data Injection Script")
    print("=" * 60)
    print()

    app = create_app('development')

    with app.app_context():
        print("Starting data injection...\n")

        # Create departments and routing
        departments = create_departments()
        create_routing_rules(departments)

        # Create knowledge base
        create_knowledge_base()

        # Create users
        create_users()

        # Create calls with transcripts and intents
        print()
        num_calls = int(input("How many calls to generate? (default: 100): ") or "100")
        create_calls(num_calls)

        print()
        print("=" * 60)
        print("✓ Data injection completed successfully!")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  • Departments: {Department.query.count()}")
        print(f"  • Routing Rules: {RoutingRule.query.count()}")
        print(f"  • Knowledge Entries: {KnowledgeBase.query.count()}")
        print(f"  • Users: {User.query.count()}")
        print(f"  • Calls: {Call.query.count()}")
        print(f"  • Transcripts: {CallTranscript.query.count()}")
        print(f"  • Intents: {CallIntent.query.count()}")
        print()
        print("You can now login and explore the dashboard with realistic data!")
        print()


if __name__ == '__main__':
    main()