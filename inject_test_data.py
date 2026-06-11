#!/usr/bin/env python
"""
Inject realistic Kenyan data into FreePBX AI Assistant database.
Populate the database with sample data for testing.
Jubilee Insurance Kenya - Realistic call transcripts (Swahili/English/Sheng mix)

Usage: python inject_test_data.py
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

# ---------------------------------------------------------------------------
# REALISTIC MULTI-TURN TRANSCRIPT SCENARIOS
# Each scenario is a list of (speaker, text) tuples representing a full call.
# Jubilee Insurance Kenya — Swahili / English / Sheng mix
# ---------------------------------------------------------------------------

TRANSCRIPT_SCENARIOS = [
    # 1. Motor accident claim - Nairobi
    [
        ("caller",    "Hello, habari. Nataka kuripoti accident. Gari yangu iligongwa jana usiku huko Mombasa Road."),
        ("assistant", "Habari yako. Pole sana kusikia hivyo. You've reached Jubilee Insurance. I can help you file a motor accident claim. Can you please provide your policy number?"),
        ("caller",    "Policy number ni JB-MOT-2024-087342. Jina langu ni Peter Omondi."),
        ("assistant", "Thank you, Mr. Omondi. I can see your Jubilee Comprehensive Motor policy. It's currently active. Can you describe what happened?"),
        ("caller",    "Nilikuwa naenda home from work around 10pm. Lorry fulani ilikuja wrong side ya barabara na ikauma rear side ya gari yangu. Front ya lorry ilikuwa imevunjika sana."),
        ("assistant", "I understand. Was anyone injured in the accident? And did you get the lorry driver's details and a police abstract?"),
        ("caller",    "Hapana, hakuna aliyeumia. Nilichukua number plate ya lorry — KBZ 445T. Nilikwenda Mlolongo Police Station na wakanipa abstract number — PS/MLG/2024/0341."),
        ("assistant", "Very good. You've done the right things. Under your comprehensive cover, vehicle damage is fully covered subject to the KES 15,000 excess. I'm raising claim reference JB-CLM-2024-19847 for you now. You need to take the vehicle to any of our approved garages — I recommend Jubilee Auto Centre on Mombasa Road or Highfield Motors in Industrial Area. Can I send those addresses to this number?"),
        ("caller",    "Ndio, tuma please. Na ni lini gari itakuwa tayari? Nina job ya delivery — gari ndio livelihood yangu."),
        ("assistant", "I understand how critical that is. Our assessor will inspect the vehicle within 24 hours of it reaching the garage. Repair timeline depends on the damage scope, but typically 5 to 10 working days. I'm also flagging this as urgent given your livelihood depends on it. Is there anything else I can help with?"),
        ("caller",    "Asante sana. Hiyo inatosha kwa sasa."),
        ("assistant", "Karibu sana Mr. Omondi. Your claim reference is JB-CLM-2024-19847. Keep that for follow-ups. We'll send updates via SMS. Have a good evening."),
    ],

    # 2. Medical claim reimbursement - outpatient
    [
        ("caller",    "Good morning. I was treated at Aga Khan Hospital last week and I want to know how to get reimbursed. My policy is JB-MED-2023-054219."),
        ("assistant", "Good morning! Welcome to Jubilee Insurance. I can help with your medical reimbursement. I can see your Jubilee Health Bronze plan — it does include outpatient cover up to KES 60,000 per year. What was the nature of the treatment?"),
        ("caller",    "Nilikuwa na chest infection. Nilipiewa antibiotics na X-ray. Total bill ilikuwa KES 12,400."),
        ("assistant", "That falls well within your outpatient limit. For reimbursement you'll need to submit: the original receipts, a filled Jubilee claim form — available on our website or any branch, the doctor's notes or prescription, and the lab or X-ray results. Have you already collected those documents from Aga Khan?"),
        ("caller",    "Niko na receipts na prescription. Lab results, sijui kama nilipewa."),
        ("assistant", "No problem — you can request a copy from Aga Khan's medical records department. It usually takes one to two days. Once you have everything, you can submit via our portal at my.jubileeinsurance.com, email to medicalclaims@jubilee.co.ke, or drop the originals at any Jubilee branch. Reimbursement is processed within 7 to 10 working days after complete document submission."),
        ("caller",    "Sawa. Portal inawork vizuri? Nilitry last time na ilikuwa ina issues."),
        ("assistant", "I'm sorry to hear you had trouble. The portal was upgraded last month and should be more stable now. If you still face issues, the email option is equally fast. Would you like me to send you the claim form link via SMS right now?"),
        ("caller",    "Yes please, send it to this number."),
        ("assistant", "Done — you'll receive it in a moment. Is there anything else I can assist you with today?"),
        ("caller",    "That's all for now. Thank you."),
        ("assistant", "You're welcome. Get well soon, and don't hesitate to call us if you need anything. Have a lovely day."),
    ],

    # 3. Premium payment inquiry - M-PESA
    [
        ("caller",    "Niaje. Nataka kujua — nililipa premium yangu via M-PESA wiki iliyopita but sijapata confirmation. Policy yangu ni JB-LIFE-2024-112045."),
        ("assistant", "Mambo vipi! Thank you for calling Jubilee Insurance. Let me check that for you. I can see your Jubilee Maisha Life policy. Can you confirm the M-PESA transaction code?"),
        ("caller",    "Transaction code ni QGH72XKPL9. Nililipa KES 4,200 on Monday the 3rd."),
        ("assistant", "Thank you. I can see the payment of KES 4,200 was received on Monday 3rd via our M-PESA Paybill 222000. It has been posted to your account successfully. Your policy is now current up to end of next month. The SMS confirmation may have been delayed — I'm triggering a manual resend now."),
        ("caller",    "Oh, poa. So policy yangu haijafungwa? Nilikuwa naworry sana."),
        ("assistant", "Your policy is fully active — no worries at all. You're within the 30-day grace period in any case, so even a slight delay in payment would not lapse your cover. You should receive the confirmation SMS within the next 10 minutes."),
        ("caller",    "Sawa kabisa. Na next time, naweza kulipa monthly badala ya lump sum?"),
        ("assistant", "Yes, absolutely. We offer monthly, quarterly, semi-annual, and annual payment options. Monthly is very popular. Would you like me to switch your policy to monthly billing from next cycle? The monthly amount for your plan would be KES 4,200 — same amount you're already paying."),
        ("caller",    "Basi hiyo ni easy. Weka monthly tafadhali."),
        ("assistant", "Done! I've updated your billing preference to monthly. Your next payment will be due on the 3rd of next month. You'll get an SMS reminder 5 days before. Is there anything else?"),
        ("caller",    "Hiyo tu. Asante sana."),
        ("assistant", "Asante na karibu tena. Have a wonderful day!"),
    ],

    # 4. Life insurance policy — new inquiry
    [
        ("caller",    "Hello, good afternoon. I'm interested in taking a life insurance policy for myself and my wife. We are both in our mid-thirties."),
        ("assistant", "Good afternoon! Welcome to Jubilee Insurance. I'd be happy to tell you about our life insurance options. We have several plans ranging from term life to whole life with investment components. May I ask — are you looking for pure life cover, or something that also builds savings over time?"),
        ("caller",    "Nataka kitu ambacho kinashield familia yangu in case something happens, but also kitu kinachokua kama savings. We have two kids."),
        ("assistant", "That sounds like our Jubilee Afya Timiza Whole Life plan — it provides a death benefit and builds a cash value you can access after 10 years. For two adults in their mid-thirties, a combined cover of KES 4,000,000 would cost roughly KES 8,500 to KES 11,000 per month depending on the medical underwriting results. It also includes a critical illness rider for cancer, stroke, and heart attack at no extra cost."),
        ("caller",    "That sounds good. Hiyo critical illness cover ni muhimu sana. Je, kuna waiting period?"),
        ("assistant", "There is a 90-day waiting period for critical illness claims from the policy start date. Life cover kicks in immediately from day one. For the application, we need both of you to fill a proposal form and do a basic medical check — we have partner clinics across Kenya including The Nairobi Hospital, Kenyatta National, and county hospitals. It's free of charge."),
        ("caller",    "Tuko Nakuru. Kuna clinic huko?"),
        ("assistant", "Yes — our partner clinic in Nakuru is Nakuru War Memorial Hospital and also Rift Valley Provincial General. The medical takes about an hour. Once done, we usually give a decision within 5 to 7 working days. Would you like me to send you the proposal forms and a full product brochure via email or WhatsApp?"),
        ("caller",    "WhatsApp inawork. Number hii hii."),
        ("assistant", "Perfect. I'll send the Jubilee Afya Timiza brochure and proposal forms to this number right away. One of our Nakuru branch advisors will also follow up with you this week to answer any questions. Can I share your contact with them?"),
        ("caller",    "That's fine. Jina langu ni Samuel Kiplagat."),
        ("assistant", "Thank you Mr. Kiplagat. You'll receive the documents shortly, and our advisor will reach out. Is there anything else I can help you with today?"),
        ("caller",    "No, that's fine. Thank you very much."),
        ("assistant", "Thank you for your interest in Jubilee Insurance, Mr. Kiplagat. We look forward to protecting your family. Have a great afternoon."),
    ],

    # 5. Hospital pre-authorization — inpatient
    [
        ("caller",    "Haraka haraka — daktari wangu amesema I need to be admitted to Nairobi Hospital leo. Nataka pre-authorization. Policy number JB-MED-2024-078901."),
        ("assistant", "I can see you're in an urgent situation — let me help you immediately. I can see your Jubilee Health Gold policy which includes inpatient cover up to KES 1,500,000 per year. Which hospital and what is the diagnosis or procedure?"),
        ("caller",    "Nairobi Hospital, Upper Hill. Daktari anasema nina appendicitis na inaweza kuwa emergency surgery."),
        ("assistant", "Understood — this is a medical emergency. Nairobi Hospital is in our network. I'm issuing an emergency pre-authorization right now. Authorization number is JB-AUTH-2024-00934. Please give this number to the hospital admissions desk immediately. They already have our direct billing agreement so you will not pay out of pocket for the main procedure."),
        ("caller",    "Sawa. Na family members wangu wanaweza kulipa nini kama deposit?"),
        ("assistant", "With this pre-authorization, Nairobi Hospital should NOT require a deposit for the procedure itself. If they ask for one, please call us immediately on this same line and we will speak directly with the hospital. Your cover includes surgeon fees, anaesthetist, theatre, and up to 7 days ward — all under direct billing."),
        ("caller",    "God bless you. Nitakwenda sasa hivi."),
        ("assistant", "Please go now — your health comes first. Authorization JB-AUTH-2024-00934 is live. We will monitor your admission and a case manager will check in with you or your family tomorrow morning. Get well soon."),
    ],

    # 6. Property claim — fire damage
    [
        ("caller",    "Habari za asubuhi. Nyumba yangu ilichomwa moto usiku wa kuamkia. Niko na home insurance na Jubilee — policy JB-HOME-2023-033210."),
        ("assistant", "Habari njema. Pole sana sana kusikia hilo — I'm very sorry about your home. Are you and your family safe?"),
        ("caller",    "Tuko salama, asante Mungu. Tulikimbia nje mapema. Lakini nyumba na vitu vingi vimechomeka."),
        ("assistant", "I'm so relieved to hear you are all safe. That is the most important thing. I can see your Jubilee Home Comprehensive policy is active with building cover up to KES 8,000,000 and contents up to KES 1,200,000. I'm opening a claim now — reference JB-CLM-2024-21103. A few things you must do today if you haven't already: get a police abstract from your nearest station, and do NOT remove or clear any debris yet as our assessor needs to survey the scene first."),
        ("caller",    "Sawa. Polisi walikuwa hapa usiku — walisema watanipa abstract kesho. Na hatuwezi kuingia nyumba — imesimamishwa na fire brigade."),
        ("assistant", "That's fine — please don't enter until it's declared safe. Our senior assessor will contact you within 4 working hours to arrange the site visit. In the meantime, we are covering emergency accommodation costs up to KES 50,000 under your policy — can I arrange an emergency hotel voucher for your family tonight?"),
        ("caller",    "Serious? Hiyo itasaidia sana. Tuko na watoto wadogo."),
        ("assistant", "Absolutely serious — it's part of your cover. I'm arranging a 3-night voucher at a guest house near you. Can you tell me which area you're in so I find the nearest option?"),
        ("caller",    "Tuko Kitengela, Kajiado County."),
        ("assistant", "I'm locating approved accommodation in Kitengela now. I'll send a voucher code and address via SMS within 15 minutes. Your claim reference is JB-CLM-2024-21103 — keep it for all follow-up. Is there a family member or friend I can also copy on updates?"),
        ("caller",    "Ndio — mke wangu anaitwa Grace. Number yake ni 0725 334 789."),
        ("assistant", "I've noted Grace's number and she'll receive updates too. Mr...?"),
        ("caller",    "Njoroge. David Njoroge."),
        ("assistant", "Mr. Njoroge, please look after your family. We will handle the claim end-to-end. You'll hear from our assessor very soon. God bless you and your family."),
    ],

    # 7. Policy cancellation request
    [
        ("caller",    "Hi, nataka kufunga insurance yangu. Sina pesa ya kulipa premiums. Policy number JB-MOT-2024-056712."),
        ("assistant", "Habari. I understand you're facing financial pressure — I'm sorry to hear that. Before I process a cancellation, may I ask — is this a temporary cash flow issue or a permanent decision? I ask because we have options that may help you keep your cover without losing what you've already paid."),
        ("caller",    "Nimepitiwa na biashara yangu imeshuka. Sijui kama ni ya muda au permanent."),
        ("assistant", "Thank you for sharing that. I want to make sure you're aware of two options. First, we can suspend your policy for up to 3 months — your cover pauses, no premiums are due, and you restart where you left off when you're ready. Second, we can reduce your cover to Third Party Only, which brings your premium from KES 18,000 per year down to KES 5,500 — saving you over KES 12,000 while keeping you legally covered on the road. Would either of those work for you?"),
        ("caller",    "Sijawahi skia ya ku-suspend policy. Hiyo inawezekana kweli?"),
        ("assistant", "Yes, absolutely. It's called a Policy Holiday and you can take it once per policy year for up to 90 days. During that time your no-claims discount is protected too. Would you like to activate a 3-month holiday now?"),
        ("caller",    "Hiyo inaonekana smart zaidi. Basi fanya hivyo. Nitawasiliana nawe baadaye nikiwa ready kuanza tena."),
        ("assistant", "Noted. I'm activating a 90-day policy holiday on JB-MOT-2024-056712 effective today. Your policy resumes automatically on that date or earlier if you call us. You'll receive an SMS confirmation. Is there anything else I can help with?"),
        ("caller",    "Hiyo tu. Asante kwa uelewa wako."),
        ("assistant", "Karibu sana. We're here for you — biashara itarudi tena. Take care."),
    ],

    # 8. Claim status follow-up — frustrated customer
    [
        ("caller",    "Hello, nataka kujua status ya claim yangu. Nimekuwa nikingoja for THREE WEEKS and nobody calls me back. This is unacceptable."),
        ("assistant", "I completely understand your frustration and I sincerely apologise for the delay and lack of communication. Three weeks is too long and you deserve better service than that. Can I have your claim reference number?"),
        ("caller",    "JB-CLM-2024-18450. Ilikuwa motor claim — rear bumper replacement."),
        ("assistant", "Thank you. I can see your claim was lodged on the 14th. I can see it was assessed on the 17th and approved for KES 34,500 on the 21st. It then went to our finance team for payment processing. I can see there was a delay because your bank account details on file were flagged for verification. This is something we should have communicated to you immediately — I'm very sorry we didn't."),
        ("caller",    "Okay, so what do I need to do now?"),
        ("assistant", "I need to confirm your bank details with you now — it takes two minutes — and then I will manually escalate this to our finance supervisor to process the payment by end of business today or first thing tomorrow. Can you confirm your bank name, account name, and account number?"),
        ("caller",    "Equity Bank. Account name John Otieno Ouma. Account 0260293040087."),
        ("assistant", "Thank you Mr. Ouma. I've updated the details and flagged this as urgent with the finance supervisor — reference escalation ESC-2024-0847. You will receive KES 34,500 to that Equity account by tomorrow 5 PM. I'll personally ensure you get an SMS once it's dispatched. Again, I am truly sorry for this experience."),
        ("caller",    "Okay. At least something is happening now. Thank you."),
        ("assistant", "Thank you for your patience, Mr. Ouma. This should not have taken this long. If you don't receive the payment by tomorrow evening, please call us and quote ESC-2024-0847 for immediate action. Have a good day."),
    ],

    # 9. Travel insurance inquiry — business trip to UAE
    [
        ("caller",    "Good morning. I'm travelling to Dubai next week for a business expo. Naomba travel insurance — ni ya kwanza kufanya. Policy yangu ni JB-MED-2024-091230."),
        ("assistant", "Good morning! How exciting — the Dubai Expo is a great event. We can add a travel insurance extension to your existing Jubilee Health policy or we can issue a standalone travel policy. For a business trip to UAE, our Business Travel cover is the most popular. How long will you be there?"),
        ("caller",    "Nitakuwa huko for 8 days — from Monday to Tuesday next week."),
        ("assistant", "Perfect. Our Jubilee Business Travel plan for 8 days to UAE would cost KES 3,200. It covers medical emergencies up to USD 50,000, trip cancellation up to KES 80,000, lost luggage up to KES 50,000, travel delay after 6 hours, and personal liability. The UAE requires travel insurance for a visa on arrival — this policy meets that requirement."),
        ("caller",    "Does it cover COVID-related things? Kama nikifeel sick over there?"),
        ("assistant", "Yes — COVID-19 is treated the same as any other illness under this plan. If you fall ill including with COVID, medical costs are fully covered up to the USD 50,000 limit. Quarantine costs abroad are also covered up to 7 days if mandated by local authorities."),
        ("caller",    "That's exactly what I needed to know. How do I pay and get the documents?"),
        ("assistant", "You can pay via M-PESA Paybill 222000, account number TRAVEL. Or I can send you a payment link via SMS right now — once paid the policy document and certificate arrive by email within 30 minutes. What's your email address?"),
        ("caller",    "Email ni kariuki.business@gmail.com."),
        ("assistant", "Got it. Sending the payment link to this number now — total KES 3,200. Once paid, your travel documents will be emailed to kariuki.business@gmail.com. Is there anything else before your trip?"),
        ("caller",    "That's it. Very helpful, asante."),
        ("assistant", "Asante! Safe travels and have a successful expo. Call us anytime if you need anything while abroad — our emergency line operates 24/7."),
    ],

    # 10. Group medical insurance — HR manager inquiry
    [
        ("caller",    "Hello, I'm calling from Bidco Africa. I'm the HR Manager — my name is Esther Chebet. We want to review our group medical scheme. Currently tuko na insurance nyingine lakini tunafikiria ku-switch to Jubilee."),
        ("assistant", "Good afternoon Ms. Chebet, and thank you for your interest in Jubilee Insurance. We'd love to have Bidco Africa on board. Group medical schemes are our strength — we cover over 200 corporates in Kenya. Can I ask how many employees would be in the scheme and whether you require inpatient only or both inpatient and outpatient?"),
        ("caller",    "Tuna approximately 850 staff including dependants — karibu 2,200 lives. Tunataka comprehensive cover — inpatient, outpatient, maternity, dental, and optical."),
        ("assistant", "That's a significant scheme and we can definitely customise a competitive package for Bidco. For 2,200 lives with full cover, we typically design a tiered structure — different limits for management, supervisory staff, and general workers. Our corporate team will prepare a detailed proposal with specific limits and pricing. The proposal would normally take 3 to 5 business days after we receive your current claims experience — that's a summary of claims over the past 12 months from your current insurer. Do you have that?"),
        ("caller",    "Ndio, tuna. Naweza send it to which email?"),
        ("assistant", "Please send it to corporate@jubilee.co.ke with subject line: Bidco Africa Group Medical Proposal. I'm also assigning you a dedicated Corporate Relationship Manager — his name is Mr. Brian Mwangi, and he'll contact you within 24 hours to personally walk you through the process and arrange a site visit. Is your direct line a good contact for him?"),
        ("caller",    "Yes, this number and my email — esther.chebet@bidco.co.ke."),
        ("assistant", "Noted. Ms. Chebet, you've made a great decision exploring Jubilee — our group schemes have a 97% renewal rate because we deliver on service especially during claims. We look forward to partnering with Bidco Africa."),
        ("caller",    "Sounds good. We'll send the claims data today."),
        ("assistant", "Excellent. Brian will be in touch shortly. Thank you for calling Jubilee and have a productive afternoon."),
    ],

    # 11. Maternity claim — first-time mother
    [
        ("caller",    "Hi habari. Nimejifungua wiki mbili zilizopita — baby boy! Lakini sijui jinsi ya ku-claim maternity costs. Policy number JB-MED-2023-062100."),
        ("assistant", "Congratulations on your baby boy! Hongera sana! I can see your Jubilee Health Silver plan which includes maternity cover. Which hospital did you deliver at, and was it a normal delivery or C-section?"),
        ("caller",    "Nairobi Women's Hospital, Hurlingham. Ilikuwa C-section — bill ilikuwa KES 165,000."),
        ("assistant", "Okay. Your plan includes maternity cover up to KES 150,000 for C-sections. So KES 150,000 would be covered and the remaining KES 15,000 would be your co-payment. Did Nairobi Women's Hospital do direct billing, or did you pay and now need reimbursement?"),
        ("caller",    "Nililipa cash because emergency and they said they don't do direct billing at night."),
        ("assistant", "I understand — emergency situations can be chaotic. For reimbursement you'll need: the original hospital receipt, the discharge summary, the doctor's operative report since it was a C-section, and a completed Jubilee claim form. Your baby's birth notification from the hospital is also helpful. You can submit via email to medicalclaims@jubilee.co.ke or at any branch. Once complete documents are received, reimbursement is within 10 working days."),
        ("caller",    "Na mtoto wangu — anaweza ku-join my policy pia?"),
        ("assistant", "Absolutely! You can add your newborn within 30 days of birth at no extra premium for the remainder of your policy year. After that date there's a normal top-up charge. I can add the baby now — what's his name?"),
        ("caller",    "Ethan Mwangi Karanja."),
        ("assistant", "Baby Ethan is now added as a dependant on your policy. Hongera tena mama! Is there anything else I can help you with?"),
        ("caller",    "Hiyo inatosha. Thank you so much."),
        ("assistant", "Karibu sana. Enjoy every moment with baby Ethan. Take care of yourself too!"),
    ],

    # 12. Renewal reminder — policy lapsing soon
    [
        ("caller",    "Nimepokea SMS kwamba insurance yangu inaisha mwisho wa mwezi. Nataka ku-renew. Policy JB-MOT-2023-041887."),
        ("assistant", "Habari! Yes, your Jubilee Comprehensive Motor policy for your Toyota Fielder KCB 673Z expires on the 30th. Great that you're calling early — let me pull up your renewal quotation. You have 3 years no-claims discount so you're entitled to a 20% premium reduction. Your renewal premium is KES 29,400 compared to KES 36,750 last year."),
        ("caller",    "Poa sana. Na cover itabaki same — comprehensive?"),
        ("assistant", "Same comprehensive cover including windscreen, radio cassette, and political violence extension that you had last year. All unchanged. And since you've been with us for 4 years, you also qualify for our Jubilee Loyalty discount of an additional 5% — bringing the total to KES 27,930. I can apply that for you now if you'd like."),
        ("caller",    "Sawa kabisa! Nitatumia M-PESA lini na kiasi gani?"),
        ("assistant", "Pay KES 27,930 to M-PESA Paybill 222000, account number JB-MOT-2023-041887. Once payment comes through — usually within 2 hours — I'll generate and send your new insurance certificate and sticker by email and WhatsApp. Do you still have the same email as on file?"),
        ("caller",    "Ndio, bado same."),
        ("assistant", "Perfect. Your renewal is all set pending payment. Make sure to pay before the 30th to avoid any gap in cover. Is there anything you'd like to add or change on the policy this year?"),
        ("caller",    "Hapana, kila kitu ni sawa. Asante."),
        ("assistant", "Asante na karibu. We appreciate your loyalty, and we'll make sure your new certificate is ready as soon as payment comes through. Drive safe!"),
    ],

    # 13. Dental claim — child's braces
    [
        ("caller",    "Good afternoon. Mtoto wangu anahitaji braces — orthodontic treatment. Je, hii inashughulikiwa na health insurance yangu? Policy JB-MED-2024-103456."),
        ("assistant", "Good afternoon. I can see you have the Jubilee Health Platinum family plan which does include dental cover for dependants. Orthodontic treatment is a special category though — let me check the specifics of your plan."),
        ("caller",    "Okay please check. Mtoto ana miaka 12 na daktari wa meno amesema braces ni necessary for alignment."),
        ("assistant", "I've checked. Your Platinum plan includes orthodontic cover up to KES 80,000 per dependant, but it requires prior authorization and the treatment must be medically necessary — not cosmetic. Your dentist's letter saying it's necessary for alignment is exactly what we need. You'll also need an orthodontic treatment plan from the specialist."),
        ("caller",    "Tuna appointment na orthodontist next week at Avenue Healthcare. Wataweza kutoa hiyo plan."),
        ("assistant", "Excellent. Avenue Healthcare is in our network so billing can be direct. After the orthodontist prepares the treatment plan, ask them to submit it to us directly for pre-authorization via our provider portal or by email to dentalauth@jubilee.co.ke. We respond within 3 to 5 working days. Once authorized, they bill us directly and you won't pay out of pocket beyond the KES 80,000 limit."),
        ("caller",    "What if braces cost more than 80K?"),
        ("assistant", "If the total cost exceeds KES 80,000, the difference becomes your co-payment. Braces in Nairobi typically run between KES 70,000 and KES 120,000, so there may be a small balance to pay depending on the complexity. The orthodontist will give you an exact quote next week."),
        ("caller",    "Sawa, hiyo ni fair enough. Asante sana for the information."),
        ("assistant", "Karibu! Your child is lucky to have such an attentive parent. Good luck with the appointment next week — let us know if you need anything."),
    ],

    # 14. Fraudulent claim suspicion — caller disputing denied claim
    [
        ("caller",    "My claim was rejected and I want to know why! This is very unfair. JB-CLM-2024-17723."),
        ("assistant", "I understand you're frustrated about the claim decision, and I want to help clarify it. Let me pull up the details for JB-CLM-2024-17723. I can see this was a motor claim for accident damage filed on the 5th. The claim was reviewed and declined following the assessment. The decline reason noted is that the damage pattern was inconsistent with the reported accident circumstances."),
        ("caller",    "That is not true! My car was hit in a parking lot at Westgate."),
        ("assistant", "I hear you. You have every right to dispute this decision. Our claims team's assessment is based on the garage report, but that is not final if you believe it's incorrect. You can formally appeal by writing to appeals@jubilee.co.ke within 30 days of the decline letter. In the appeal include any additional evidence — CCTV footage from the parking lot, witness statements, or any photos you took at the scene. The appeal is reviewed by a senior independent assessor."),
        ("caller",    "Westgate has CCTV — can I get that?"),
        ("assistant", "Yes — contact Westgate Mall management or security directly and request CCTV footage for the date and approximate time of the incident. It can take a few days but it's strong evidence. Once you have it, include it with your appeal letter. If your appeal is upheld, the claim will be paid in full."),
        ("caller",    "Okay. I'll get that footage. And if the appeal fails?"),
        ("assistant", "If the appeal is not resolved in your favour, you have the right to escalate to the Insurance Regulatory Authority of Kenya — the IRA — at complaints@ira.go.ke. They independently review disputes between customers and insurers. But I'm hopeful the CCTV evidence will resolve this at the appeal stage."),
        ("caller",    "Fine. I'll do that. Thanks for at least explaining properly."),
        ("assistant", "Of course — you deserve a clear explanation. I hope the appeal goes well. Don't hesitate to call us if you need any assistance during the process."),
    ],

    # 15. Last expense / funeral cover inquiry
    [
        ("caller",    "Habari. Baba yangu amefariki leo asubuhi — God rest his soul. Alikuwa na policy ya last expense na Jubilee. Nataka kujua tunapata nini. Policy yake ni JB-LAST-2022-009841."),
        ("assistant", "Pole sana kwa msiba. May his soul rest in eternal peace. Please accept my deepest condolences on behalf of everyone at Jubilee Insurance. I can see the policy — it's a Jubilee Last Expense plan with a cover amount of KES 200,000. I want to make this process as smooth as possible for you during this difficult time."),
        ("caller",    "Asante. Tunajaribu kupanga mazishi — ninajua mnaweza kulipa haraka haraka."),
        ("assistant", "Yes — last expense claims are our fastest. We target payment within 48 hours of receiving complete documents. What you need to provide is: the original death certificate from the county government, a copy of the deceased's national ID, the claimant's ID — that's you — and a copy of the policy document if available. You do NOT need an original policy document if you have the policy number. Bank details for payment."),
        ("caller",    "Death certificate tutaipata kesho from hospital — marehemu alifariki Kenyatta National. ID yake tuna. Yangu pia niko nayo."),
        ("assistant", "That's good. Once you have the death certificate, you can submit everything to our offices at Jubilee Insurance House on Wabera Street, Nairobi, or scan and email to lastexpense@jubilee.co.ke. I'm opening the claim now with reference JB-CLM-2024-21890 so everything is ready when you submit."),
        ("caller",    "Na payment — lazima iingie bank au mnaweza fanya M-PESA?"),
        ("assistant", "We can do M-PESA up to KES 150,000. Since your cover is KES 200,000, we would recommend bank transfer for the full amount. If you have an M-PESA-linked account like M-Shwari or KCB M-PESA, that can also work. Please bring your bank account details when you submit."),
        ("caller",    "Sawa. Asante sana kwa huruma na msaada wenu."),
        ("assistant", "Pole tena sana. Mungu amfariji familia yote. Your claim reference is JB-CLM-2024-21890 — share it with any of our offices and they will prioritise you. We are here for you."),
    ],
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
        {'name': 'Claims Department', 'description': 'Handle all insurance claims and settlements', 'extension': '2001', 'priority': 10},
        {'name': 'Sales & New Business', 'description': 'New policy inquiries and sales', 'extension': '2002', 'priority': 8},
        {'name': 'Customer Support', 'description': 'General customer service and support', 'extension': '2003', 'priority': 7},
        {'name': 'Billing & Payments', 'description': 'Premium payments and billing inquiries', 'extension': '2004', 'priority': 9},
        {'name': 'Medical Underwriting', 'description': 'Medical insurance and health-related claims', 'extension': '2005', 'priority': 10},
        {'name': 'Motor Insurance', 'description': 'Vehicle insurance and accident claims', 'extension': '2006', 'priority': 9},
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
        {'department': 'Claims Department', 'intent_type': 'claims', 'keywords': json.dumps(['claim', 'accident', 'damage', 'madharau', 'ajali']), 'priority': 10},
        {'department': 'Billing & Payments', 'intent_type': 'billing', 'keywords': json.dumps(['payment', 'premium', 'malipo', 'bill', 'invoice']), 'priority': 9},
        {'department': 'Sales & New Business', 'intent_type': 'sales', 'keywords': json.dumps(['buy', 'purchase', 'new policy', 'nunua', 'quote']), 'priority': 8},
        {'department': 'Customer Support', 'intent_type': 'support', 'keywords': json.dumps(['help', 'assistance', 'msaada', 'question', 'issue']), 'priority': 7},
        {'department': 'Medical Underwriting', 'intent_type': 'coverage', 'keywords': json.dumps(['hospital', 'medical', 'health', 'daktari', 'ugonjwa']), 'priority': 9},
        {'department': 'Motor Insurance', 'intent_type': 'claims', 'keywords': json.dumps(['car', 'vehicle', 'gari', 'motor', 'accident']), 'priority': 9},
    ]

    created = 0
    for rule_data in rules:
        dept = Department.query.filter_by(name=rule_data['department']).first()
        if dept:
            existing = RoutingRule.query.filter_by(department_id=dept.id, intent_type=rule_data['intent_type']).first()
            if not existing:
                rule = RoutingRule(department_id=dept.id, intent_type=rule_data['intent_type'], keywords=rule_data['keywords'], priority=rule_data['priority'])
                db.session.add(rule)
                created += 1

    db.session.commit()
    print(f"✓ Created {created} routing rules")


def create_knowledge_base():
    """Create insurance knowledge base entries."""
    print("Creating knowledge base...")

    entries = [
        {'title': 'Comprehensive Motor Insurance Coverage', 'category': 'coverage', 'content': 'Comprehensive motor insurance covers damage to your vehicle, third-party liability, theft, fire, and natural disasters. Coverage extends to all Kenyan roads including within East Africa. Maximum coverage is KES 5,000,000.', 'keywords': json.dumps(['motor', 'car', 'gari', 'comprehensive', 'coverage', 'accident']), 'priority': 10},
        {'title': 'Medical Insurance Hospital Coverage', 'category': 'policies', 'content': 'Medical cover includes inpatient and outpatient services at over 200 hospitals across Kenya. Annual limit is KES 2,000,000. Covers consultation, medication, surgery, and emergency services. Maternity covered after 10 months.', 'keywords': json.dumps(['medical', 'hospital', 'health', 'daktari', 'treatment', 'inpatient']), 'priority': 10},
        {'title': 'Claims Process and Timeline', 'category': 'claims', 'content': 'To file a claim: 1) Report incident within 48 hours, 2) Submit required documents, 3) Assessment by our team, 4) Approval and payment. Timeline: Motor claims 7-14 days, Medical claims 5-10 days, Property claims 14-21 days.', 'keywords': json.dumps(['claim', 'process', 'timeline', 'file', 'madai', 'settlement']), 'priority': 9},
        {'title': 'Premium Payment Methods', 'category': 'billing', 'content': 'Pay premiums via M-PESA Paybill 222000, bank transfer, or at any of our branches. Annual, semi-annual, and monthly payment plans available. Grace period is 30 days. Late payment attracts 2% penalty per month.', 'keywords': json.dumps(['payment', 'premium', 'malipo', 'mpesa', 'paybill', 'cost']), 'priority': 8},
        {'title': 'Life Insurance Benefits', 'category': 'policies', 'content': 'Life insurance provides financial security for your family. Coverage from KES 500,000 to KES 10,000,000. Includes accidental death benefit, terminal illness cover, and funeral expenses. Premiums start from KES 2,500 per month.', 'keywords': json.dumps(['life', 'death', 'beneficiary', 'family', 'protection', 'cover']), 'priority': 9},
        {'title': 'Last Expense Policy', 'category': 'policies', 'content': 'Jubilee Last Expense covers funeral and burial costs from KES 50,000 to KES 500,000. Claims paid within 48 hours of document submission. Covers principal member, spouse, children, and parents.', 'keywords': json.dumps(['funeral', 'last expense', 'death', 'mazishi', 'burial']), 'priority': 9},
        {'title': 'Travel Insurance Coverage', 'category': 'coverage', 'content': 'Travel insurance covers medical emergencies abroad, trip cancellation, lost luggage, and flight delays. Valid for business and leisure travel worldwide. Coverage from KES 100,000 to KES 5,000,000. Premium starts at KES 1,500 per trip.', 'keywords': json.dumps(['travel', 'trip', 'abroad', 'safari', 'luggage', 'flight']), 'priority': 7},
        {'title': 'Home Insurance Protection', 'category': 'policies', 'content': 'Protects your home and contents against fire, theft, natural disasters, and vandalism. Covers building structure and household items. Coverage up to KES 10,000,000. Premium based on property value.', 'keywords': json.dumps(['home', 'house', 'nyumba', 'property', 'fire', 'theft', 'burglary']), 'priority': 8},
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


def create_calls(num_calls=15):
    """Create realistic call records using predefined transcript scenarios."""
    print(f"Creating {num_calls} calls with realistic Jubilee Insurance transcripts...")

    departments = Department.query.all()
    if not departments:
        print("⚠ No departments found. Creating departments first.")
        departments = create_departments()

    calls_created = 0
    end_date = datetime.now()

    # Map scenarios to intent types for realistic classification
    scenario_intents = [
        'claims',    # 1. Motor accident claim
        'claims',    # 2. Medical reimbursement
        'billing',   # 3. Premium payment inquiry
        'sales',     # 4. Life insurance new inquiry
        'claims',    # 5. Hospital pre-authorization
        'claims',    # 6. Property claim (fire)
        'support',   # 7. Policy cancellation
        'claims',    # 8. Claim status follow-up
        'sales',     # 9. Travel insurance inquiry
        'sales',     # 10. Group medical inquiry
        'claims',    # 11. Maternity claim
        'billing',   # 12. Renewal reminder
        'coverage',  # 13. Dental claim inquiry
        'support',   # 14. Denied claim dispute
        'claims',    # 15. Last expense claim
    ]

    # Generate calls spread across the past 30 days
    for i in range(min(num_calls, len(TRANSCRIPT_SCENARIOS))):
        scenario = TRANSCRIPT_SCENARIOS[i]
        intent_type = scenario_intents[i]

        random_days = random.uniform(0, 30)
        call_time = end_date - timedelta(days=random_days)

        caller_number = generate_kenyan_phone()
        call_id = f"call_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}_{random.randint(1000, 9999)}"

        # Some calls escalate (scenarios 5, 8, 14 are natural escalation candidates)
        escalated = i in [4, 7, 13]
        status = 'escalated' if escalated else 'completed'

        # Duration based on scenario length
        duration = len(scenario) * random.randint(25, 45)
        total_interactions = len(scenario)
        failed_interactions = random.randint(0, 1) if not escalated else random.randint(1, 2)

        call = Call(
            call_id=call_id,
            caller_number=caller_number,
            started_at=call_time,
            ended_at=call_time + timedelta(seconds=duration),
            duration_seconds=duration,
            status=status,
            escalated=escalated,
            escalated_to_department_id=random.choice(departments).id if escalated else None,
            escalation_reason=random.choice(ESCALATION_REASONS) if escalated else None,
            total_interactions=total_interactions,
            failed_interactions=failed_interactions
        )

        db.session.add(call)
        db.session.flush()

        # Insert full realistic transcript
        for turn_idx, (speaker, text) in enumerate(scenario):
            timestamp = call_time + timedelta(seconds=turn_idx * random.randint(20, 40))
            confidence = random.uniform(0.88, 0.99) if speaker == 'caller' else None

            transcript = CallTranscript(
                call_id=call.id,
                speaker=speaker,
                text=text,
                confidence=confidence,
                timestamp=timestamp,
                ai_model='gpt-4o-mini' if speaker == 'assistant' else None,
                ai_tokens_used=random.randint(80, 250) if speaker == 'assistant' else None,
                ai_response_time_ms=random.randint(400, 1800) if speaker == 'assistant' else None,
            )
            db.session.add(transcript)

        # Intent classification
        keywords_map = {
            'claims':   ['claim', 'accident', 'damage'],
            'billing':  ['payment', 'premium', 'bill'],
            'support':  ['help', 'question', 'issue'],
            'sales':    ['buy', 'new', 'quote'],
            'coverage': ['cover', 'policy', 'benefits'],
            'general':  ['inquiry', 'information', 'details']
        }
        intent = CallIntent(
            call_id=call.id,
            intent_type=intent_type,
            confidence=random.uniform(0.82, 0.97),
            keywords=json.dumps(keywords_map.get(intent_type, [])),
            detected_at=call_time + timedelta(seconds=35)
        )
        db.session.add(intent)

        calls_created += 1
        print(f"  ✓ Scenario {i + 1}/{num_calls}: {scenario[0][1][:60]}...")

    db.session.commit()
    print(f"✓ Created {calls_created} calls with full realistic transcripts")


def create_users():
    """Create additional user accounts."""
    print("Creating additional users...")

    users_data = [
        {'username': 'manager', 'email': 'manager@jubileeinsurance.co.ke', 'role': 'manager', 'password': 'manager123'},
        {'username': 'viewer', 'email': 'viewer@jubileeinsurance.co.ke', 'role': 'viewer', 'password': 'viewer123'},
        {'username': generate_kenyan_name().lower().replace(' ', '.'), 'email': 'staff1@jubileeinsurance.co.ke', 'role': 'manager', 'password': 'password'},
        {'username': generate_kenyan_name().lower().replace(' ', '.'), 'email': 'staff2@jubileeinsurance.co.ke', 'role': 'viewer', 'password': 'password'},
    ]

    created = 0
    for user_data in users_data:
        existing = User.query.filter_by(username=user_data['username']).first()
        if not existing:
            user = User(username=user_data['username'], email=user_data['email'], role=user_data['role'])
            user.set_password(user_data['password'])
            db.session.add(user)
            created += 1

    db.session.commit()
    print(f"✓ Created {created} users")


def main():
    """Main injection script."""
    print("=" * 60)
    print("Jubilee Insurance Kenya - AI Call Centre Data Injection")
    print("=" * 60)
    print()

    app = create_app('development')

    with app.app_context():
        print("Starting data injection...\n")

        departments = create_departments()
        create_routing_rules(departments)
        create_knowledge_base()
        create_users()

        print()
        num_calls = int(input(f"How many calls to generate? (max {len(TRANSCRIPT_SCENARIOS)}, default: {len(TRANSCRIPT_SCENARIOS)}): ") or str(len(TRANSCRIPT_SCENARIOS)))
        num_calls = min(num_calls, len(TRANSCRIPT_SCENARIOS))
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
        print("Login and explore the Jubilee Insurance dashboard with realistic Kenyan call data!")
        print()


if __name__ == '__main__':
    main()