#!/usr/bin/env python3



from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

LABEL2ID: Dict[str, int] = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12,
    "B-PERCENT": 13,
    "I-PERCENT": 14,
    "B-ORDINAL": 15,
    "I-MONEY": 17,
    "B-WORK_OF_ART": 18,
    "I-WORK_OF_ART": 19,
    "B-FAC": 20,
    "B-TIME": 21,
    "I-CARDINAL": 22,
    "B-MONEY": 16,
    "B-LOC": 23,
    "B-QUANTITY": 24,
    "I-QUANTITY": 25,
    "I-NORP": 26,
    "I-LOC": 27,
    "B-PRODUCT": 28,
    "I-TIME": 29,
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36,
    "B-CANADIAN_BANK_ACCOUNT_NUMBER": 37,
    "I-CANADIAN_BANK_ACCOUNT_NUMBER": 38,
    "B-ALBERTA_DRIVERS_LICENCE_NUMBER": 39,
    "I-ALBERTA_DRIVERS_LICENCE_NUMBER": 40,
    "B-CANADIAN_INDIVIDUAL_TAX_NUMBER": 41,
    "I-CANADIAN_INDIVIDUAL_TAX_NUMBER": 42,
    "B-ALBERTA_PERSONAL_HEALTH_NUMBER": 43,
    "I-ALBERTA_PERSONAL_HEALTH_NUMBER": 44,
    "B-CANADIAN_PROVIDER_IDENTIFIER": 45,
    "I-CANADIAN_PROVIDER_IDENTIFIER": 46,
    "B-CANADIAN_PASSPORT_NUMBER": 47,
    "I-CANADIAN_PASSPORT_NUMBER": 48,
    "B-CANADIAN_SOCIAL_INSURANCE_NUMBER": 49,
    "I-CANADIAN_SOCIAL_INSURANCE_NUMBER": 50,
}

NEW_ENTITY_TYPES: Tuple[str, ...] = (
    "CANADIAN_BANK_ACCOUNT_NUMBER",
    "ALBERTA_DRIVERS_LICENCE_NUMBER",
    "CANADIAN_INDIVIDUAL_TAX_NUMBER",
    "ALBERTA_PERSONAL_HEALTH_NUMBER",
    "CANADIAN_PROVIDER_IDENTIFIER",
    "CANADIAN_PASSPORT_NUMBER",
    "CANADIAN_SOCIAL_INSURANCE_NUMBER",
)

ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NEW_B_LABEL_IDS = {LABEL2ID[f"B-{entity_type}"] for entity_type in NEW_ENTITY_TYPES}

DEFAULT_TRAIN_SIZE = 12000
DEFAULT_VALIDATION_SIZE = 1600
DEFAULT_TEST_SIZE = 1600
MIN_TEMPLATE_PATTERNS = 430

Example = Dict[str, Union[List[str], List[int]]]
Segment = Tuple[str, Optional[str]]

ENTITY_SLOT = object()      # full realizer (keyword prefix + value)
BARE_VALUE_SLOT = object()  # value only — keyword lives in the template text


# ---------------------------------------------------------------------------
# Synthetic data pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Aaliyah", "Abigail", "Adrien", "Alejandro", "Alexis", "Amira",
    "André", "Anita", "Antoine", "Aryan", "Avery", "Bianca",
    "Caleb", "Chandra", "Clara", "Colton", "Daphne", "Deepa",
    "Elena", "Elias", "Emiko", "Evan", "Fiona", "François",
    "Geneviève", "Gurpreet", "Hamza", "Helena", "Ibrahim", "Jade",
    "Jasmine", "Kai", "Kayla", "Khalid", "Layla", "Lina",
    "Malcolm", "Manpreet", "Margot", "Maxime", "Naomi", "Navdeep",
    "Nikhil", "Noor", "Oscar", "Paloma", "Raj", "Renée",
    "Rhea", "Riley", "Ryder", "Salma", "Sanjay", "Simone",
    "Skylar", "Soren", "Taryn", "Tristan", "Valentina", "Vivian",
    "Xavier", "Yasmin", "Zahra", "Zain","Maya", "Liam", "Noah", "Olivia", "Emma", "Ava", "Sophia", "Lucas",
    "Ethan", "Amelia", "Charlotte", "Benjamin", "William", "James", "Henry",
    "Ella", "Grace", "Chloe", "Aria", "Nora", "Zoe", "Leo", "Mila", "Aiden",
    "Elijah", "Logan", "Sebastian", "Isabella", "Emily", "Jacob", "Mason",
    "Harper", "Evelyn", "Daniel", "Samuel", "Priya", "Anika", "Ravi",
    "Nadia", "Omar", "Fatima", "Aisha", "Sofia", "Mateo", "Luis", "Mei",
    "Jun", "Hana", "Yuki", "Noémie", "Étienne", "Camille", "Gabriel",
    "Élodie", "Saanvi", "Dev", "Iris", "Theo", "Sarah", "Adam", "Tara",
    "Jonah", "Leila", "Samira", "Dylan", "Morgan", "Casey", "Rowan",
    "Jaspreet", "Kiran", "Farah", "Amir", "Mina",
]

LAST_NAMES = [
    "Agarwal", "Beaulieu", "Bélanger", "Bergeron", "Bhatt", "Bolduc",
    "Boucher", "Brar", "Breton", "Chabot", "Chandra", "Cheng",
    "Choi", "Cloutier", "Côté", "Cyr", "Das", "Desrosiers",
    "Diaz", "Fortin", "Fournier", "Gauthier", "Girard", "Gupta",
    "Hébert", "Ho", "Huynh", "Iqbal", "Jain", "Kapoor",
    "Khan", "Khatri", "Lapointe", "Lau", "Leblanc", "Lemieux",
    "Lévesque", "Malhotra", "Martel", "Mehta", "Morin", "Nadeau",
    "Nair", "Ouellet", "Pelletier", "Perron", "Poirier", "Prasad",
    "Rao", "Raza", "Rousseau", "Sandhu", "Savard", "Shan",
    "Sidhu", "Simard", "Soni", "Thibault", "Thériault", "Verma",
    "Villeneuve", "Wu", "Yang", "Yao", "Singh", "Patel", "Chen", "Wong", "Martin", "Roy", "Tremblay", "Gagnon",
    "Wilson", "Brown", "Taylor", "Anderson", "Thomas", "Moore", "Jackson",
    "White", "Harris", "Clark", "Lewis", "Walker", "Hall", "Allen", "Young",
    "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Carter",
    "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell",
    "Parker", "Evans", "Edwards", "Collins", "Stewart", "Morris", "Rogers",
    "Reed", "Cook", "Morgan", "Bell", "Murphy", "Bailey", "Rivera", "Cooper",
    "Richardson", "Cox", "Howard", "Ward", "Torres", "Peterson", "Gray",
    "Ramirez", "Kaur", "Gill", "Dhillon", "Miller", "Bennett", "Foster",
    "Ahmed", "Hassan", "Nguyen", "Tran",
]

PERSON_TITLES = [
    "Prof.", "Sgt.", "Cpl.", "Inspector", "Coordinator", "Supervisor",
    "Director", "Technician", "Ms.", "Mr.", "Mx.", "Dr.", "Agent", "Officer", "Auditor", "Caseworker",
    "Analyst", "Registrar", "Nurse", "Pharmacist",
]

ORGS = [
    "Rideau Payroll Bureau", "Thunderbird Financial Group",
    "Westcoast Document Services", "Maple Leaf Credentialing",
    "Frontier Benefits Administration", "Canmore Tax Advisory",
    "Rocky View Employment Services", "Prairie Sunrise Pharmacy",
    "Elk Island Health Clinic", "Jasper Ridge Hospital",
    "Crowfoot Medical Centre", "Bow Valley Laboratory",
    "Chinook Dental Group", "Parkland Rehabilitation Centre",
    "Peace River Travel Agency", "Athabasca Pension Fund",
    "Lakeland Insurance Brokers", "Yellowhead Compliance Group",
    "Vermilion Credit Union", "Drumheller Accounting LLP",
    "Stettler Community College", "Wetaskiwin HR Solutions",
    "Ponoka Registry Services", "Camrose Document Intake",
    "Leduc Payroll Processing", "Spruce Grove Benefits Office",
    "St. Albert Medical Centre", "Fort Saskatchewan Clinic",
    "Sherwood Park Pharmacy", "Devon Travel Insurance",
    "Beaumont Employment Agency", "Morinville Tax Services",
    "Cold Lake Air Services", "Lloydminster Border Clinic",
    "Hinton Forest Industries", "Edson Railway Services",
    "Whitecourt Energy Corp", "Slave Lake Recovery Centre",
    "High Level Community Services", "Banff Hospitality Group", 
    "Maple Payroll Services", "Northstar Tax Services", "Prairie Payroll Solutions",
    "Cedar Coast Bank", "Harbourview Benefits", "Laurentian Student Services",
    "Fictional National Bank", "Acorn Document Intake", "Summit Payroll Services",
    "Blue Heron Compliance Group", "Evergreen Accounting LLP",
    "Dominion Filing Support", "Arctic Ledger Inc.", "Red Maple Benefits Office",
    "Silver Birch Capital", "Riverbend Tax Advisory", "Pacific KYC Services",
    "Great Lakes Payroll", "Pine Valley University", "Nova HR Support",
    "Metro Verification Bureau", "Starlight Document Services",
    "Northern Remittance Desk", "Cloudberry Finance", "Crescent Audit Partners",
    "Aurora Tax Filing Centre", "Birchwood Benefits", "Seaway Intake Vendors",
    "Lakeside University", "Cobalt Trust Company", "Prairie Case Management",
    "Clearwater Accounting Group", "Highland Student Aid Office",
    "Red River Payroll Bureau", "Sagebrush Compliance Office",
    "Bridgeport Pension Services", "Maple Ridge Community College",
    "Granite HR Systems", "Willow Creek Employment Screening",
    "Harbourview Pension Administration", "Prairie Health Registration",
    "Foothills Clinic Group", "North Calgary Medical Centre",
    "Edmonton Credentialing Office", "Rocky Mountain Pharmacy",
    "Alpine Laboratory Services", "Wheatland Health Claims",
    "Prairie Provider Enrollment", "College of Synthetic Practitioners",
    "Maple Air", "Prairie Wings", "Northern Horizon Airlines",
    "Cedar Travel Group", "Harbourview Travel", "Clearwater Travel Insurance",
    "Red River Hotel Group", "Voyage Identity Systems", "Skyline Check-In Services",
    "Crescent Health Administration", "Wildrose Service Desk",
    "Prairie Vendor Onboarding", "Capital Refund Desk", "Northshore Loan Intake",
    "Blue Sky Employer Services", "MediPortal Support Services",
]

PUBLIC_ORGS = [
    "Employment and Social Development Canada", "ESDC",
    "Public Health Agency of Canada", "PHAC",
    "Statistics Canada", "Treasury Board of Canada Secretariat",
    "Canadian Food Inspection Agency", "CFIA",
    "Natural Resources Canada", "Parks Canada",
    "Veterans Affairs Canada", "Royal Canadian Mounted Police", "Canada Revenue Agency", "CRA", "Service Canada", "Government of Canada",
    "Bank of Canada", "Alberta Health", "Alberta Health Services",
    "Government of Alberta", "Health Canada",
    "Immigration, Refugees and Citizenship Canada",
    "Canada Border Services Agency", "CBSA", "Global Affairs Canada",
]

DEPARTMENTS = [
    "immunization scheduling desk", "chronic disease registry team",
    "newborn registration unit", "organ donor registry office",
    "vital statistics desk", "marriage certificate office",
    "death registration unit", "change of name processing desk",
    "land titles office", "corporate registry desk",
    "personal property registry unit", "motor vehicle branch",
    "graduated licensing desk", "commercial vehicle inspection unit",
    "dangerous goods transport office", "occupational health team",
    "workplace safety compliance desk", "apprenticeship certification unit",
    "foreign credential assessment office", "language testing centre",
    "citizenship processing desk", "refugee resettlement unit",
    "consular services desk", "emergency travel document office",
    "diplomatic pouch processing unit", "payroll onboarding desk", "payroll processing unit",
    "payroll remittance office", "tax filing desk", "CRA correspondence unit",
    "Service Canada service counter", "banking KYC group",
    "AML-adjacent identity review desk", "benefits administration unit",
    "benefit payment review desk", "employment screening team",
    "background check unit", "HR records office", "HR identity verification team",
    "university employment office", "student worker onboarding desk",
    "government service counter", "credit review desk",
    "loan application intake desk", "pension administration unit",
    "pension enrollment desk", "employment insurance administration unit",
    "health-care registration desk", "AHCIP registration desk",
    "clinic intake unit", "hospital administration office",
    "pharmacy billing team", "laboratory requisition desk",
    "diagnostic imaging requisition desk", "provider enrollment desk",
    "Alberta health billing unit", "hospital credentialing team",
    "medical-staff office", "referral processing desk",
    "passport application desk", "passport renewal desk",
    "border services unit", "CBSA processing desk",
    "airport document check desk", "airline check-in desk",
    "travel booking team", "visa processing office", "consular assistance unit",
    "immigration support office", "insurance claims unit",
    "document intake desk", "document redaction queue",
    "document retention office", "audit team", "payroll audit team",
    "privacy review office", "compliance review group", "quality assurance desk",
    "call-centre support team", "customer-service callback team",
    "bilingual service desk", "accessibility support desk",
    "online portal support team", "mobile app support team",
    "system migration team", "document migration team", "case file cleanup unit",
    "vendor onboarding team", "refund processing unit",
    "balance-due collection desk", "penalty and interest review group",
]

GPE = [
    "Prince Edward Island", "Newfoundland and Labrador",
    "Northwest Territories", "Nunavut", "Yukon",
    "Kelowna", "Kamloops", "Nanaimo", "Prince George",
    "Fredericton", "Moncton", "Saint John", "Charlottetown",
    "Thunder Bay", "Sudbury", "Barrie", "Oshawa", "Guelph",
    "Sherbrooke", "Trois-Rivières", "Gatineau", "Laval",
    "Lévis", "Saguenay", "Airdrie", "Spruce Grove",
    "St. Albert", "Fort McMurray", "Lloydminster",
    "Drumheller", "Canmore", "Cochrane", "Okotoks", "Canada", "Alberta", "Ontario", "Quebec", "British Columbia", "Manitoba",
    "Saskatchewan", "Nova Scotia", "New Brunswick", "Toronto", "Calgary",
    "Edmonton", "Vancouver", "Montreal", "Ottawa", "Halifax", "Winnipeg",
    "Regina", "Saskatoon", "Victoria", "Mississauga", "Brampton", "Hamilton",
    "London", "Kitchener", "Windsor", "Markham", "Burnaby", "Surrey",
    "Quebec City", "Paris", "London", "New York",
]

NORP = [
    "Inuit", "First Nations", "Métis", "Acadian",
    "South Asian", "East Asian", "Latin American",
    "Caribbean", "African", "European", "Middle Eastern",
    "Pacific Islander", "Canadian", "Albertan", "Ontarian", "Quebecois", "British Columbian",
    "Indigenous", "French", "Indian", "Filipino", "Ukrainian", "American",
    "British", "Chinese", "Francophone", "Anglophone", "Manitoban",
    "Saskatchewanian", "Maritime",
]

LOC = [
   "the Bow River valley", "the North Saskatchewan River basin",
    "the Peace River region", "the Okanagan Valley",
    "the Annapolis Valley", "northern British Columbia",
    "the Niagara region", "the Laurentians",
    "the Gatineau Hills", "the Thousand Islands",
    "the Bay of Fundy", "the Cabot Trail",
    "the Columbia Icefield", "the badlands",
    "the boreal forest", "the tundra",
    "southern Manitoba", "eastern Ontario",
    "Western Canada", "Atlantic Canada", 
    "northern Alberta", "downtown Toronto",
    "the Prairies", "the Maritimes", 
    "the Ottawa Valley", "the Pacific coast",
    "the Great Lakes region", "the Fraser Valley", 
    "the St. Lawrence corridor","central Canada", 
    "the Calgary region", "downtown Vancouver",
    "northern Europe", "the Caribbean", 
    "rural Alberta", "downtown Calgary","the Edmonton region",
]

FAC = [
    "Misericordia Community Hospital", "Grey Nuns Community Hospital",
    "Sturgeon Community Hospital", "Westlock Healthcare Centre",
    "Peter Lougheed Centre", "Rockyview General Hospital",
    "South Health Campus", "Chinook Regional Hospital",
    "Queen Elizabeth II Hospital", "Northern Lights Regional Health Centre",
    "Cross Cancer Institute", "Glenrose Rehabilitation Hospital",
    "Alberta Children's Hospital", "Stollery Children's Hospital",
    "Mazankowski Alberta Heart Institute", "Kaye Edmonton Clinic",
    "Edmonton International Airport", "Montréal-Trudeau Airport",
    "Winnipeg James Armstrong Richardson Airport",
    "Halifax Stanfield International Airport",
    "Billy Bishop Toronto City Airport",
    "Service Canada Kiosk", "Registry Connect Office",
    "Provincial Courthouse", "Federal Building",
    "Service Canada Centre", "Sudbury Tax Centre", "Winnipeg Tax Centre",
    "Calgary Service Counter", "Toronto Document Intake Office",
    "Vancouver Student Employment Building", "Montreal HR Office",
    "Ottawa Benefits Branch", "Prairie Audit Centre", "Harbour Centre",
    "Union Station branch", "Commerce Tower", "Maple Plaza",
    "Riverside Operations Building", "North Terminal office",
    "Granville Service Centre", "Rideau branch", "Prairie Operations Hub",
    "Lakeside University Hall", "Cedar Records Archive", "Capital Courthouse",
    "Foothills Medical Centre", "Royal Alexandra Hospital",
    "Red Deer Regional Hospital", "Lethbridge Health Centre",
    "Medicine Hat Clinic", "Edmonton Laboratory", "Jasper Pharmacy Counter",
    "Medical Staff Office Tower", "Toronto Pearson Airport",
    "Vancouver International Airport", "Calgary International Airport",
    "Montréal Consulate Office", "Ottawa Passport Office",
    "Harbour Embassy Annex", "Skyline Check-In Counter", "Red River Hotel",
    "Capital Visa Centre", "Downtown Loan Centre", "Mobile Support Lab",
    "Online Portal Operations Centre",
]

LAWS = [
    "Canada Elections Act", "Canadian Human Rights Act",
    "Official Languages Act", "Controlled Drugs and Substances Act",
    "Food and Drugs Act", "Canada Evidence Act",
    "Criminal Code of Canada", "Youth Criminal Justice Act",
    "Bankruptcy and Insolvency Act", "Competition Act",
    "Copyright Act", "Trademarks Act",
    "Motor Vehicle Safety Act", "Railway Safety Act",
    "Quarantine Act", "Income Tax Act", "Employment Insurance Act", "Canada Pension Plan",
    "Health Information Act", "Alberta Health Care Insurance Act",
    "Canada Health Act", "Privacy Act", "Access to Information Act",
    "Personal Information Protection and Electronic Documents Act",
    "Immigration and Refugee Protection Act", "Customs Act", "Aeronautics Act",
    "Synthetic Records Retention Bylaw", "Internal Compliance Review Policy",
    "Payroll Identity Verification Rule", "Benefits Claim Reconciliation Policy",
    "Document Intake Validation Standard", "Pension Enrollment Verification Policy",
    "Provider Identifier Verification Rule", "Travel Document Verification Policy",
    "Bank Account Verification Policy", "Credentialing Review Standard",
]

WORKS_OF_ART = [
   "Cross-Reference Verification Guide", "Identity Bundle Checklist",
    "Multi-Factor Authentication Handbook", "Secure Fax Procedures Manual",
    "Call-Centre Scripting Guide", "Bilingual Service Protocol",
    "Accessibility Compliance Memo", "Portal User Guide",
    "Mobile App Intake Manual", "Data Migration Playbook",
    "Legacy System Cutover Checklist", "OCR Exception Guide",
    "Paper Digitization Protocol", "Archive Retrieval Manual",
    "Disposition Review Checklist", "Fraud Monitoring Playbook",
    "Overpayment Recovery Guide", "Garnishment Processing Memo",
    "Hardship Review Protocol", "Returned Document Handling Guide", "Payroll Onboarding Guide", "Quarterly Payroll Audit",
    "CRA Correspondence Memo", "Employment Screening Manual",
    "Benefits Administration Manual", "KYC Verification Handbook",
    "Student Worker Case File", "Pension Enrollment Checklist",
    "Annual Audit Brief", "Document Intake Playbook", "Privacy Notice",
    "The Remittance Ledger", "Employee Identity Review Memo",
    "Case File Quality Bulletin", "SIN Intake Checklist",
    "Personal Health Number Intake Guide", "Clinic Registration Manual",
    "Pharmacy Billing Handbook", "Laboratory Requisition Memo",
    "Diagnostic Imaging Requisition Guide", "Provider Enrollment Guide",
    "Clinic Credentialing Manual", "Referral Processing Protocol",
    "Provider Roster Playbook", "Passport Intake Guide", "Border Services Memo",
    "Visa Processing Manual", "Airline Check-In Checklist",
    "Traveller Identity Review Memo", "Bank Account Review Memo",
    "System Migration Runbook", "Document Migration Runbook",
    "Penalty and Interest Worksheet", "Loan Intake Quality Memo",
]

PRODUCTS = [
   "SecureSign 2.0", "DocuVerify", "ClaimFlow Pro",
    "PayrollGuard", "TaxTracker", "BenefitSync",
    "CredentialHub", "ProviderLink", "PassportTrack",
    "LicenceVerify", "HealthCardPlus", "IdentityCheck Pro",
    "AuditTrail Manager", "ComplianceBot", "RedactSafe",
    "MigrationPilot", "ArchiveVault", "IntakeExpress",
    "CallCentre Pro", "ServiceDesk Plus", "MapleTax Pro", "Payroll Direct", "KYC Checkpoint", "IdentityVault",
    "BenefitPay", "StudentAid Portal", "PensionLink", "Trust Ledger",
    "CRA My Account", "RemitTrack", "DocumentCloud Intake", "AuditWorks",
    "Maple Chequing Plus", "TaxReturn Express", "FilingDesk", "CaseFlow",
    "OnboardHR", "HRWorks", "Pension Portal", "LoanTrack", "Refund Express",
    "Mobile Payroll App", "Online Benefits Portal", "VendorCheck",
    "MediRecord EMR", "PharmaBill", "ClaimTrack Health", "LabOrder Pro",
    "Clinic Intake Portal", "ProviderEnroll", "ReferralDesk", "HealthVerify",
    "ProviderConnect", "InvoiceCare", "CoveragePlus", "Roster Manager",
    "Telehealth Gateway", "SkyPass Basic", "SkyPass Flex", "VisaTrack",
    "TravelDesk", "Passport Intake Portal", "BookingFlow", "HotelCheck",
    "BorderVerify", "ClaimSure Travel", "Treasury Connect", "EFT Batch Manager",
    "BillingHub",
]

EVENTS = [
    "Annual Benefits Enrollment", "Open Enrollment Period",
    "Year-End Payroll Closeout", "Tax Season Surge",
    "Spring Credential Renewal", "Fall Registration Drive",
    "Summer Student Intake", "Winter Flu Vaccination Drive",
    "National Immunization Week", "Data Centre Migration",
    "Platform Upgrade Sprint", "Security Audit Cycle",
    "Annual Privacy Review", "Quarterly Provider Audit",
    "Monthly Billing Reconciliation", "Weekly Claims Review",
    "Daily Intake Triage", "Biannual Roster Refresh",
    "Emergency Response Activation", "Pandemic Preparedness Drill", "Payroll Audit", "Filing Season", "Compliance Review",
    "Onboarding Campaign", "System Migration", "Benefits Review",
    "Pension Enrollment", "Document Intake Event", "Payroll Conversion",
    "KYC Remediation", "Annual Control Test", "Student Employment Intake",
    "Remittance Reconciliation", "Case File Cleanup", "Privacy Audit",
    "Vaccination Campaign", "Claims Review", "Provider Audit",
    "Credentialing Review", "Roster Cleanup", "Billing Reconciliation",
    "Laboratory Requisition Review", "Pharmacy Claims Review",
    "Passport Renewal Campaign", "Visa Interview", "Travel Disruption",
    "Document Migration", "Border Inspection", "Consular Assistance Review",
    "Quarterly Reconciliation", "Fraud Monitoring Sweep",
    "Penalty and Interest Review", "Refund Review", "Loan Intake Review",
]

LANGUAGES = [
    "Ojibwe", "Dene", "Blackfoot", "Mi'kmaq",
    "Portuguese", "German", "Italian", "Polish",
    "Somali", "Tigrinya", "Amharic", "Farsi",
    "Tamil", "Bengali", "Gujarati", "English", "French", "Cree", "Punjabi", "Mandarin", "Tagalog", "Arabic",
    "Spanish", "Ukrainian", "Inuktitut",
]

DATES = [
   "February 14, 2026", "March 1, 2025", "April 30, 2026",
    "May 15, 2025", "June 1, 2026", "July 4, 2025",
    "August 31, 2026", "September 1, 2025", "October 15, 2026",
    "November 30, 2025", "December 31, 2026", "January 15, 2025",
    "2026-01-01", "2025-06-15", "2026-12-31",
    "last Tuesday", "this Thursday", "two weeks ago",
    "the end of the month", "the start of the quarter",
    "Q4 2025", "Q1 2027", "fiscal year 2026-2027", "March 12, 2026", "April 5, 2025", "June 30, 2024", "January 2, 2027",
    "September 18, 2026", "Dec. 14, 2025", "2026-03-12", "2025/11/04",
    "Monday", "next Friday", "last quarter", "the first week of May",
    "Q2 2026", "fiscal 2025", "tax year 2024", "tax year 2025",
    "August 9", "Oct. 31", "May 1, 2026", "July 15, 2025",
    "the renewal date", "the issue date",
]
# added more dates template
DATES += [
    "yesterday", "today", "tomorrow",
    "last week", "next week", "this week",
    "last month", "next month", "last year", "next year",
    "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday",
    "last Wednesday", "last Friday", "next Monday", "next Tuesday", "next Thursday",
    "this morning", "this afternoon", "tonight", "earlier today",
    "a week ago", "a month ago", "the previous quarter", "the following week",
]
TIMES = [
    "6:00 a.m.", "6:30 AM", "7:00 a.m.", "9:45 AM",
    "12:30 PM", "1:00 p.m.", "2:00 PM", "3:30 p.m.",
    "4:00 PM", "6:00 p.m.", "8:00 PM", "10:00 p.m.", 
    "9:00 a.m.", "10:30 AM", "14:45", "5 p.m.", "noon", "midnight",
    "8:15 a.m.", "16:20", "7:05 PM", "11:40 a.m.",
]

MONEY = [
    "$50.00", "$175.00", "$325.50", "$475.00", "$625.00",
    "$1,500.00", "$2,750.00", "$5,200.00", "$8,400.00",
    "CAD 350.00", "CAD 750.00", "CAD 2,500.00", "CAD 10,000.00",
    "C$1,850.00", "C$4,200.00", "$22,500", "$25.00", "$75", "$150.00", "$250.00", "CAD 1,250.75", "$3,400",
    "C$89.99", "$12,000.00", "CAD 500", "$1.2 million", "$45,600",
    "C$7,300.10", "$999", "$2,850.50", "CAD 6,100",
]

PERCENTS = [
    "0.25%", "1%", "4.5 percent", "6%", "7 percent",
    "9.5%", "11 percent", "17.5%", "22 percent", "45%",
    "50 percent", "75%", "90 percent", "2%", "3.5 percent", "0.75%", "12 percent", "1.25%", "15%",
    "99.9 percent", "8 percent", "5%", "30 percent",
]

CARDINALS = [
   "five", "eight", "nine", "eleven", "thirteen",
    "16", "21", "30", "50", "75", "150", "300",
    "500", "750", "2,500", "10,000", "one", "two", "three", "4", "7", "12", "25", "100", "1,000", "42",
    "18", "64", "250", "AC742", "flight 618",
]

ORDINALS = [
    "eighth", "ninth", "eleventh", "twelfth",
    "13th", "15th", "20th", "25th", "50th", "100th", "first", "second", "third", "fourth", "fifth", "10th", "21st",
    "sixth", "seventh",
]

QUANTITIES = [
    "1 hour", "4 hours", "8 hours", "10 hours",
    "7 days", "21 days", "60 days", "120 days",
    "3 months", "9 months", "18 months", "36 months",
    "2 years", "5 years", "10 years",
    "15 kilometres", "100 kilometres",
    "5 business days", "15 business days", "12 months", "30 days", "5 kilometres", "two weeks", "3 hours",
    "40 hours", "20 kilograms", "2 hours", "10 pages", "6 years",
    "18 files", "3 business days", "45 minutes", "90 days", "24 months",
]


# ---------------------------------------------------------------------------
# False-positive decoy pools (tagged as CARDINAL or O, never as identifiers)
# ---------------------------------------------------------------------------

PHONE_NUMBERS = [
    "780-555-1234", "403-555-6789", "604-555-2345", "416-555-8901",
    "514-555-3456", "613-555-7890", "1-800-555-1234", "1-888-555-6789",
    "780-555-0198", "403-555-4477", "306-555-8832", "204-555-3310",
    "867-555-2201", "709-555-6654", "902-555-1198",
]

FAX_NUMBERS = [
    "780-555-4321", "403-555-9876", "604-555-5432", "416-555-1098",
    "514-555-6543", "613-555-0987",
]

POSTAL_CODES = [
    "T5J 2N9", "T2P 3G5", "M5V 2H1", "V6B 1A1", "K1A 0B1",
    "H3B 1S6", "R3C 0V8", "S4P 3Y2", "T6G 2R3", "T1Y 6J4",
    "E1A 3Z9", "A1B 4J6", "L5N 8R7",
]

REFERENCE_CODES = [
    "REF-283746", "REF-948271", "REF-384756", "REF-20260315",
    "CONF-849201", "CONF-293847", "CONF-573920",
    "TKT-384756", "TKT-293018", "TKT-482910",
    "CHK-849201", "CHK-374829", "REC-192837",
]

INVOICE_NUMBERS = [
    "INV-12345678", "INV-98765432", "INV-20260401-003",
    "INV-2026-0847", "INV-384756", "INV-00482910",
]

CASE_FILE_NUMBERS = [
    "2024-38291", "2025-84729", "2026-00384", "2026-19283",
    "CAS-384756", "CAS-2026-001", "FILE-20260315-001",
    "FILE-384729", "FILE-2025-482",
]

BADGE_NUMBERS = [
    "badge 4829", "badge 10384", "badge 293", "employee badge 38471",
    "ID badge 2938", "staff badge 49201",
]

PLATE_NUMBERS = [
    "ABC-1234", "BXR-5678", "CDE-9012", "FGH-3456",
    "plate ABC-1234", "plate BXR-5678",
]

ROOM_NUMBERS = [
    "room 204", "room 318", "bed 4A", "bed 12B", "unit 7C",
    "suite 402", "exam room 3", "bay 6",
]

DIN_NUMBERS = [
    "DIN 02345678", "DIN 01234567", "DIN 02468013",
    "drug identification number 02345678",
]

POLICY_NUMBERS = [
    "policy 384729-01", "policy GRP-482910", "group policy 38291",
    "certificate 482910-A", "plan number 293847",
]

CLAIM_NUMBERS = [
    "claim 2026-384729", "claim CLM-482910", "claim 38291",
    "WCB claim 2026-001", "insurance claim 482910",
]

BOOKING_CODES = [
    "XBRF4K", "MNTQ7P", "KCVW2R", "JHLX9N", "PQRS3T",
    "booking XBRF4K", "confirmation MNTQ7P", "PNR KCVW2R",
]

FLIGHT_CODES = [
    "flight 618", "flight AC742", "WS 3204", "AC 845",
    "flight WJ 514", "AC 127",
]

GENERIC_NUMBERS = [
    "384729", "482910", "293847", "102938", "574829",
    "847291", "938472", "192837", "483920", "573829",
]


# ---------------------------------------------------------------------------
# Tokenization and example construction
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(
    r"""
    [A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?  # words/numbers
    | [%$€£¥#]                                          # common symbols
    | [&]                                               # ampersand
    | [^\w\s]                                           # punctuation
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> List[str]:
    """Tokenize text deterministically while preserving punctuation tokens."""
    if not text:
        return []
    return TOKEN_RE.findall(text)


def realize_segments(items: Sequence, rng: random.Random) -> List[Segment]:
    """Expand entity-slot placeholders into (prefix, None) + (value, label) pairs."""
    out: List[Segment] = []
    for text, label in items:
        if text is ENTITY_SLOT:
            prefix, value = IDENTIFIER_REALIZERS[label](rng)
            if prefix:
                out.append((prefix, None))
            out.append((value, label))
        elif text is BARE_VALUE_SLOT:
            # keyword is already in the surrounding template text; emit value only
            value = BARE_VALUE_FUNCS[label](rng)
            out.append((value, label))
        else:
            out.append((text, label))
    return out


def make_example(items: Sequence, rng: random.Random) -> Example:
    """Build one token-classification example from explicit text/entity segments."""
    segments = realize_segments(items, rng)
    tokens: List[str] = []
    tags: List[int] = []

    for text, entity_type in segments:
        segment_tokens = tokenize(text)
        if not segment_tokens:
            continue

        if entity_type is None:
            segment_tags = [LABEL2ID["O"]] * len(segment_tokens)
        else:
            b_label = f"B-{entity_type}"
            i_label = f"I-{entity_type}"
            if b_label not in LABEL2ID or i_label not in LABEL2ID:
                raise ValueError(f"Missing BIO labels for entity type: {entity_type}")
            segment_tags = [LABEL2ID[b_label]] + [LABEL2ID[i_label]] * (
                len(segment_tokens) - 1
            )

        tokens.extend(segment_tokens)
        tags.extend(segment_tags)

    example: Example = {"tokens": tokens, "tags": tags}
    validate_example(example)
    return example


# ---------------------------------------------------------------------------
# Random helpers and synthetic identifier generation
# ---------------------------------------------------------------------------

def choice(rng: random.Random, values: Sequence[str]) -> str:
    return values[rng.randrange(len(values))]

def person_name(rng: random.Random, with_title_probability: float = 0.15) -> str:
    name = f"{choice(rng, FIRST_NAMES)} {choice(rng, LAST_NAMES)}"
    if rng.random() < with_title_probability:
        return f"{choice(rng, PERSON_TITLES)} {name}"
    return name



# ---------------------------------------------------------------------------
# Keyword pools (O-tagged context that precedes the identifier value)
# Entries have NO trailing space; make_prefix() appends one unless kw ends "-"
# ---------------------------------------------------------------------------

KNOWN_INSTITUTIONS: List[str] = ["001", "002", "003", "004", "006", "010"]

# Bank: empty-string leads mean "no keyword prefix" (pure value); the labelled
# span is the value token(s) only.
BANK_LEADS: List[str] = [
    "", "", "",
    "account", "bank account", "Direct deposit to", "EFT to",
    "Account",           # sentence-initial capitalised forms
    "Bank account",
    "Account number",
    "account number",
]

SIN_KEYWORDS: List[str] = [
    "SIN", "SIN:", "social insurance number", "Canadian SIN",
    "employee SIN", "payroll SIN no.", "Service Canada SIN", "SIN #",
    "tax identifier",
]

ITN_KEYWORDS: List[str] = [
    "ITN", "ITN:", "individual tax number", "Canadian ITN",
    "tax identifier", "individual tax no.", "CRA ITN", "non-resident ITN",
    "CRA individual tax number", "ITN number", "non-resident tax number",
    "individual tax identifier", "Canadian individual tax number",
]

PHN_KEYWORDS: List[str] = [
    "PHN", "PHN:", "Alberta PHN", "personal health number",
    "health care number", "healthcare number", "AHCIP number",
    "Alberta Health Care Insurance Plan number", "patient identifier",
]

PROVIDER_KEYWORDS: List[str] = [
    "PRAC-", "PRAC", "provider", "provider ID", "practitioner ID",
    "Alberta practitioner ID", "Alberta provider no.", "billing provider",
    "CPSA", "College ID", "AHS provider", "provider no.",
    "registration number", "prescriber ID", "clinician ID",
]

PASSPORT_KEYWORDS: List[str] = [
    "passport", "passport no.", "passport number", "Canadian passport",
    "Canadian passport number", "document no.", "travel document",
    "identity document", "passport #", "travel document no.",
]

DL_KEYWORDS: List[str] = [
    "AB DL", "Alberta DL", "driver's licence", "driver licence no.",
    "licence number", "DL#", "DL-", "operator licence",
    "AB licence", "driver license", "operator ID",
]


# ---------------------------------------------------------------------------
# Value generators (authoritative format per type)
# ---------------------------------------------------------------------------

def _nine_digit(rng: random.Random) -> str:
    d = f"{rng.randint(0, 999_999_999):09d}"
    return rng.choice([f"{d[:3]}-{d[3:6]}-{d[6:]}", f"{d[:3]} {d[3:6]} {d[6:]}", d])


def sin_value(rng: random.Random) -> str:
    first = rng.choice([1, 2, 3, 4, 5, 6, 7, 9])
    partial = [first] + [rng.randint(0, 9) for _ in range(7)]
    total = 0
    for i, d in enumerate(partial):
        n = d * 2 if i % 2 == 1 else d
        total += n - 9 if (i % 2 == 1 and n > 9) else n
    s = "".join(map(str, partial + [(10 - total % 10) % 10]))
    return rng.choice([f"{s[:3]}-{s[3:6]}-{s[6:]}", f"{s[:3]} {s[3:6]} {s[6:]}", s])


def itn_value(rng: random.Random) -> str:
    return _nine_digit(rng)


def phn_value(rng: random.Random) -> str:
    return _nine_digit(rng)


def provider_value(rng: random.Random) -> str:
    return f"{rng.randint(0, 999_999_999):09d}"


def bank_value(rng: random.Random) -> str:
    transit = f"{rng.randint(0, 99999):05d}"
    institution = rng.choice(KNOWN_INSTITUTIONS)
    account = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(7, 12)))
    fmt = rng.randint(0, 3)
    if fmt == 0:
        return f"{transit}-{institution}-{account}"
    if fmt == 1:
        return f"{transit} {institution} {account}"
    if fmt == 2:
        return f"transit {transit} account {account}"
    return f"acct. {account}"


def dl_value(rng: random.Random) -> str:
    # Focused on AB (primary) + two nearest neighbours; avoids spreading
    # probability too thin across rarely-seen alnum formats.
    gens = [
        lambda: "".join(rng.choices(string.digits, k=9)),  # AB — most common
        lambda: "".join(rng.choices(string.digits, k=9)),  # AB — doubled weight
        lambda: "".join(rng.choices(string.digits, k=9)),  # AB — tripled weight
        lambda: "".join(rng.choices(string.digits, k=7)),  # BC
        lambda: "".join(rng.choices(string.digits, k=8)),  # SK
    ]
    return rng.choice(gens)()


def passport_value(rng: random.Random) -> str:
    if rng.choice([True, False]):
        return (
            rng.choice(string.ascii_uppercase)
            + "".join(rng.choices(string.digits, k=6))
            + "".join(rng.choices(string.ascii_uppercase, k=2))
        )
    return "".join(rng.choices(string.ascii_uppercase, k=2)) + "".join(rng.choices(string.digits, k=6))


def make_prefix(kw: str) -> str:
    """Return kw with a trailing space, unless it already ends with '-'."""
    return kw if kw.endswith("-") else kw + " "


# ---------------------------------------------------------------------------
# Identifier realizers — return (prefix, value) where prefix is tagged O
# ---------------------------------------------------------------------------

def sin_realize(rng: random.Random) -> Tuple[str, str]:
    return make_prefix(choice(rng, SIN_KEYWORDS)), sin_value(rng)


def itn_realize(rng: random.Random) -> Tuple[str, str]:
    return make_prefix(choice(rng, ITN_KEYWORDS)), itn_value(rng)


def phn_realize(rng: random.Random) -> Tuple[str, str]:
    return make_prefix(choice(rng, PHN_KEYWORDS)), phn_value(rng)


def provider_realize(rng: random.Random) -> Tuple[str, str]:
    return make_prefix(choice(rng, PROVIDER_KEYWORDS)), provider_value(rng)


def dl_realize(rng: random.Random) -> Tuple[str, str]:
    return make_prefix(choice(rng, DL_KEYWORDS)), dl_value(rng)


def passport_realize(rng: random.Random) -> Tuple[str, str]:
    return make_prefix(choice(rng, PASSPORT_KEYWORDS)), passport_value(rng)


def bank_realize(rng: random.Random) -> Tuple[str, str]:
    lead = choice(rng, BANK_LEADS)
    prefix = (lead + " ") if lead else ""
    return prefix, bank_value(rng)


IDENTIFIER_REALIZERS: Dict[str, Callable[[random.Random], Tuple[str, str]]] = {
    "CANADIAN_SOCIAL_INSURANCE_NUMBER": sin_realize,
    "CANADIAN_INDIVIDUAL_TAX_NUMBER": itn_realize,
    "ALBERTA_PERSONAL_HEALTH_NUMBER": phn_realize,
    "CANADIAN_PROVIDER_IDENTIFIER": provider_realize,
    "ALBERTA_DRIVERS_LICENCE_NUMBER": dl_realize,
    "CANADIAN_PASSPORT_NUMBER": passport_realize,
    "CANADIAN_BANK_ACCOUNT_NUMBER": bank_realize,
}

# Value-only functions used by seg_bare() — no keyword prefix added.
BARE_VALUE_FUNCS: Dict[str, Callable[[random.Random], str]] = {
    "CANADIAN_SOCIAL_INSURANCE_NUMBER": sin_value,
    "CANADIAN_INDIVIDUAL_TAX_NUMBER": itn_value,
    "ALBERTA_PERSONAL_HEALTH_NUMBER": phn_value,
    "CANADIAN_PROVIDER_IDENTIFIER": provider_value,
    "ALBERTA_DRIVERS_LICENCE_NUMBER": dl_value,
    "CANADIAN_PASSPORT_NUMBER": passport_value,
    "CANADIAN_BANK_ACCOUNT_NUMBER": bank_value,
}


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

def values_for_templates(rng: random.Random) -> Dict[str, str]:
    return {
        "person": person_name(rng),
        "second_person": person_name(rng),
        "org": choice(rng, ORGS + PUBLIC_ORGS),
        "department": choice(rng, DEPARTMENTS),
        "gpe": choice(rng, GPE),
        "date": choice(rng, DATES),
        "time": choice(rng, TIMES),
        "money": choice(rng, MONEY),
        "percent": choice(rng, PERCENTS),
        "cardinal": choice(rng, CARDINALS),
        "ordinal": choice(rng, ORDINALS),
        "norp": choice(rng, NORP),
        "loc": choice(rng, LOC),
        "fac": choice(rng, FAC),
        "law": choice(rng, LAWS),
        "work": choice(rng, WORKS_OF_ART),
        "product": choice(rng, PRODUCTS),
        "event": choice(rng, EVENTS),
        "language": choice(rng, LANGUAGES),
        "quantity": choice(rng, QUANTITIES),
    }


def seg(entity_type: str, rng: random.Random = None) -> Segment:  # type: ignore[assignment]
    return (ENTITY_SLOT, entity_type)  # type: ignore[return-value]


def seg_bare(entity_type: str) -> Segment:  # type: ignore[return-value]
    """Value-only slot — the keyword is in the surrounding template O-text."""
    return (BARE_VALUE_SLOT, entity_type)  # type: ignore[return-value]


def optional_clause(rng: random.Random) -> List[Segment]:
    clauses: List[List[Segment]] = [
        [(" under ", None), (choice(rng, LAWS), "LAW")],
        [(" for ", None), (choice(rng, PRODUCTS), "PRODUCT")],
        [(" in ", None), (choice(rng, GPE), "GPE")],
        [(" near ", None), (choice(rng, FAC), "FAC")],
        [(" during ", None), (choice(rng, EVENTS), "EVENT")],
        [(" after ", None), (choice(rng, DATES), "DATE")],
        [(" by ", None), (choice(rng, TIMES), "TIME")],
        [(" with a rate of ", None), (choice(rng, PERCENTS), "PERCENT")],
        [(" for an amount of ", None), (choice(rng, MONEY), "MONEY")],
        [(" in ", None), (choice(rng, LANGUAGES), "LANGUAGE")],
        [(" across ", None), (choice(rng, LOC), "LOC")],
        [(" for the ", None), (choice(rng, ORDINALS), "ORDINAL"), (" review", None)],
        [(" covering ", None), (choice(rng, QUANTITIES), "QUANTITY")],
        [(" with ", None), (choice(rng, CARDINALS), "CARDINAL"), (" open records", None)],
        [(" for the ", None), (choice(rng, NORP), "NORP"), (" service group", None)],
        [(" in ", None), (choice(rng, WORKS_OF_ART), "WORK_OF_ART")],
    ]
    return choice(rng, clauses)


# ===================================================================
# 15 TEMPLATE GROUPS
# ===================================================================

def banking_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = "CANADIAN_BANK_ACCOUNT_NUMBER"
    return [
        [("Banking KYC verified account ", None), seg(e, rng), (" for ", None), (v["org"], "ORG"), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("Wire operations placed ", None), seg(e, rng), (" in the ", None), (v["ordinal"], "ORDINAL"), (" review queue.", None)],
        [("The branch at ", None), (v["fac"], "FAC"), (" read back ", None), seg(e, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("Treasury Connect reconciled ", None), seg(e, rng), (" with ", None), (v["money"], "MONEY"), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("PAD processing copied ", None), seg(e, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Refund routing note: ", None), seg(e, rng), (" before semicolon; payment review pending.", None)],
        [("The ", None), (v["norp"], "NORP"), (" banking queue confirmed ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Account intake in ", None), (v["gpe"], "GPE"), (" linked ", None), seg(e, rng), (" to ", None), (v["cardinal"], "CARDINAL"), (" open tickets.", None)],
        [("Privacy staff masked ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (" before export.", None)],
        [("The bank account field contains \"", None), seg(e, rng), ("\" near the verification note.", None)],
        [("Reconciliation across ", None), (v["loc"], "LOC"), (" sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("A synthetic banking record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("Mobile banking support stored ", None), seg(e, rng), (" in ", None), (v["product"], "PRODUCT"), (" after ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("The analyst ", None), (v["person"], "PERSON"), (" checked ", None), seg(e, rng), (" without adding customer history.", None)],
        [("Vendor payment file ends with ", None), seg(e, rng), (".", None)],
        [("Direct deposit setup matched ", None), seg(e, rng), (" to ", None), (v["money"], "MONEY"), (" in payroll notes.", None)],
        [("EFT batch review in ", None), (v["gpe"], "GPE"), (" retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Loan intake desk copied ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Account verification under ", None), (v["law"], "LAW"), (" selected ", None), seg(e, rng), (" for audit.", None)],
        [("Billing portal note (", None), seg(e, rng), (") was reviewed by ", None), (v["second_person"], "PERSON"), (".", None)],
        [("Treasury file ", None), (v["work"], "WORK_OF_ART"), (" references ", None), seg(e, rng), (" near the transit field.", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" bank queue item uses ", None), seg(e, rng), (" for vendor payment review.", None)],
    ]


def payroll_tax_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    sin = "CANADIAN_SOCIAL_INSURANCE_NUMBER"
    itn = "CANADIAN_INDIVIDUAL_TAX_NUMBER"
    return [
        [("Payroll onboarding verified ", None), seg(sin, rng), (" before ", None), (v["date"], "DATE"), (".", None)],
        [("The payroll processing unit routed ", None), seg(sin, rng), (" through ", None), (v["product"], "PRODUCT"), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("Payroll remittance matched ", None), seg(sin, rng), (" to ", None), (v["money"], "MONEY"), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Tax filing desk copied ", None), seg(itn, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("CRA correspondence references ", None), seg(itn, rng), (" for ", None), (v["date"], "DATE"), (".", None)],
        [("Service Canada support read back ", None), seg(sin, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Balance-due collection queued ", None), seg(sin, rng), (" beside ", None), (v["money"], "MONEY"), (".", None)],
        [("Penalty and interest review sampled ", None), seg(itn, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" filing batch selected ", None), seg(itn, rng), (" for QA.", None)],
        [("Payroll note (", None), seg(sin, rng), (") was reviewed by ", None), (v["person"], "PERSON"), (".", None)],
        [("Tax identifier field: ", None), seg(itn, rng), ("; refund review remains open.", None)],
        [("The ", None), (v["norp"], "NORP"), (" payroll group retained ", None), seg(sin, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Remittance Reconciliation in ", None), (v["gpe"], "GPE"), (" compared ", None), seg(sin, rng), (" with ", None), (v["cardinal"], "CARDINAL"), (" forms.", None)],
        [("The tax portal ", None), (v["product"], "PRODUCT"), (" validated ", None), seg(itn, rng), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("A synthetic payroll record contains ", None), seg(sin, rng), (" for NER training only.", None)],
        [("T4 support desk corrected ", None), seg(sin, rng), (" before ", None), (v["date"], "DATE"), (".", None)],
        [("T4A support note includes ", None), seg(itn, rng), (" and ", None), (v["cardinal"], "CARDINAL"), (" pages.", None)],
        [("Direct deposit setup copied ", None), seg(sin, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Payroll exception handling flagged ", None), seg(sin, rng), (" with rate ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Contractor onboarding reviewed ", None), seg(itn, rng), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Gig-worker platform onboarding stored ", None), seg(sin, rng), (" in ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Tax slip correction note: ", None), seg(sin, rng), (" after colon; slip review pending.", None)],
    ]


def health_pharmacy_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    phn = "ALBERTA_PERSONAL_HEALTH_NUMBER"
    return [
        [("Health-care registration verified ", None), seg(phn, rng), (" at ", None), (v["fac"], "FAC"), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("AHCIP registration copied ", None), seg(phn, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Clinic intake read back ", None), seg(phn, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Hospital administration placed ", None), seg(phn, rng), (" in ", None), (v["work"], "WORK_OF_ART"), (" for ", None), (v["event"], "EVENT"), (".", None)],
        [("Pharmacy billing routed ", None), seg(phn, rng), (" beside invoice amount ", None), (v["money"], "MONEY"), (".", None)],
        [("Laboratory requisition desk compared ", None), seg(phn, rng), (" with ", None), (v["cardinal"], "CARDINAL"), (" forms.", None)],
        [("Diagnostic imaging requisition note: ", None), seg(phn, rng), (" before semicolon; intake check pending.", None)],
        [("Benefits administration sampled ", None), seg(phn, rng), (" at ", None), (v["percent"], "PERCENT"), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" clinic intake batch included ", None), seg(phn, rng), (".", None)],
        [("Privacy review in ", None), (v["loc"], "LOC"), (" masked ", None), seg(phn, rng), (" before release.", None)],
        [("The ", None), (v["norp"], "NORP"), (" service counter checked ", None), seg(phn, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("AHCIP field contains \"", None), seg(phn, rng), ("\" near the billing note.", None)],
        [("A synthetic health registration record contains ", None), seg(phn, rng), (" for NER training only.", None)],
        [("The registrar ", None), (v["person"], "PERSON"), (" verified ", None), seg(phn, rng), (" without clinical details.", None)],
        [("Document retention kept ", None), seg(phn, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Immunization booking administration copied ", None), seg(phn, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Patient chart merge review compared ", None), seg(phn, rng), (" with ", None), (v["cardinal"], "CARDINAL"), (" candidate records.", None)],
        [("Duplicate record cleanup selected ", None), seg(phn, rng), (" for ", None), (v["event"], "EVENT"), (".", None)],
        [("Rural clinic administration in ", None), (v["loc"], "LOC"), (" retained ", None), seg(phn, rng), (".", None)],
        [("Mobile health portal validated ", None), seg(phn, rng), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("Pharmacy claim label ends with ", None), seg(phn, rng), (".", None)],
        [("Clinic administration note (", None), seg(phn, rng), (") was reviewed by ", None), (v["second_person"], "PERSON"), (".", None)],
    ]


def provider_billing_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = "CANADIAN_PROVIDER_IDENTIFIER"
    return [
        [("Provider enrollment verified ", None), seg(e, rng), (" for ", None), (v["org"], "ORG"), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("Alberta health billing routed ", None), seg(e, rng), (" through ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Hospital credentialing placed ", None), seg(e, rng), (" in ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Medical-staff office reviewed ", None), seg(e, rng), (" at ", None), (v["fac"], "FAC"), (" by ", None), (v["time"], "TIME"), (".", None)],
        [("Referral processing compared ", None), seg(e, rng), (" with ", None), (v["cardinal"], "CARDINAL"), (" requisitions.", None)],
        [("Pharmacy claims sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Lab requisition memo: ", None), seg(e, rng), (" after colon; provider review pending.", None)],
        [("Credentialing Review retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Roster cleanup in ", None), (v["gpe"], "GPE"), (" linked ", None), seg(e, rng), (" to ", None), (v["cardinal"], "CARDINAL"), (" records.", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" provider roster selected ", None), seg(e, rng), (".", None)],
        [("The ", None), (v["norp"], "NORP"), (" credentialing desk confirmed ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("A synthetic provider record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("Billing reconciliation matched ", None), seg(e, rng), (" to ", None), (v["money"], "MONEY"), (".", None)],
        [("The analyst ", None), (v["person"], "PERSON"), (" checked ", None), seg(e, rng), (" without clinical narrative.", None)],
        [("Provider portal entry ends with ", None), seg(e, rng), (".", None)],
        [("Locum provider onboarding added ", None), seg(e, rng), (" to ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Temporary privileges review retained ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("After-hours roster administration placed ", None), seg(e, rng), (" in the ", None), (v["ordinal"], "ORDINAL"), (" call block.", None)],
        [("Provider directory publishing masked ", None), seg(e, rng), (" before release.", None)],
        [("CPD tracking linked ", None), seg(e, rng), (" to ", None), (v["cardinal"], "CARDINAL"), (" administrative forms.", None)],
        [("Contract renewal review checked ", None), seg(e, rng), (" beside ", None), (v["money"], "MONEY"), (".", None)],
        [("Interprovincial provider registration reviewed ", None), seg(e, rng), (" in ", None), (v["gpe"], "GPE"), (".", None)],
    ]


def passport_travel_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = "CANADIAN_PASSPORT_NUMBER"
    return [
        [("Passport application verified ", None), seg(e, rng), (" before ", None), (v["date"], "DATE"), (".", None)],
        [("Passport renewal desk placed ", None), seg(e, rng), (" in the ", None), (v["ordinal"], "ORDINAL"), (" renewal batch.", None)],
        [("Border services checked ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (" in ", None), (v["gpe"], "GPE"), (".", None)],
        [("CBSA processing copied ", None), seg(e, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Airport document checks at ", None), (v["fac"], "FAC"), (" verified ", None), seg(e, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("Airline check-in routed ", None), seg(e, rng), (" through ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Travel booking linked ", None), seg(e, rng), (" to ", None), (v["cardinal"], "CARDINAL"), (" open tickets.", None)],
        [("Visa processing note: ", None), seg(e, rng), (" before semicolon; interview file open.", None)],
        [("Consular assistance reviewed ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Immigration support retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Travel insurance claim shows ", None), seg(e, rng), (" beside ", None), (v["money"], "MONEY"), (".", None)],
        [("Privacy Audit sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("The ", None), (v["norp"], "NORP"), (" travel desk confirmed ", None), seg(e, rng), (".", None)],
        [("A synthetic passport intake record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("The hotel counter asked ", None), (v["person"], "PERSON"), (" to confirm ", None), seg(e, rng), (" without travel history.", None)],
        [("Group travel administration copied ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Corporate travel desk verified ", None), seg(e, rng), (" before ", None), (v["date"], "DATE"), (".", None)],
        [("Cruise check-in administration reviewed ", None), seg(e, rng), (" at ", None), (v["fac"], "FAC"), (".", None)],
        [("Rail international check-in retained ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Bus international check-in support read back ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Travel disruption support routed ", None), seg(e, rng), (" to refund processing.", None)],
        [("Lost document report uses ", None), seg(e, rng), (" for replacement support.", None)],
    ]


def driver_registry_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = "ALBERTA_DRIVERS_LICENCE_NUMBER"
    return [
        [("Registry renewal verified ", None), seg(e, rng), (" at ", None), (v["fac"], "FAC"), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("Driver record correction copied ", None), seg(e, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Operator licence lookup returned ", None), seg(e, rng), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Traffic citation review compared ", None), seg(e, rng), (" with ", None), (v["cardinal"], "CARDINAL"), (" forms.", None)],
        [("Rental-car counter note: ", None), seg(e, rng), (" before semicolon; verification pending.", None)],
        [("Insurance underwriting sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Employment screening checked ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Court filing office attached ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" road-test booking file selected ", None), seg(e, rng), (".", None)],
        [("Mobile registry support logged ", None), seg(e, rng), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("The ", None), (v["norp"], "NORP"), (" registry desk verified ", None), seg(e, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("A synthetic driver registry record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("Fine reconciliation matched ", None), seg(e, rng), (" to ", None), (v["money"], "MONEY"), (".", None)],
        [("Document redaction masked ", None), seg(e, rng), (" across ", None), (v["loc"], "LOC"), (".", None)],
        [("Registry portal entry ends with ", None), seg(e, rng), (".", None)],
        [("Interprovincial licence exchange reviewed ", None), seg(e, rng), (" for ", None), (v["gpe"], "GPE"), (".", None)],
        [("Suspended licence review retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Photo radar dispute handling copied ", None), seg(e, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Dealership test-drive verification confirmed ", None), seg(e, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("Delivery driver onboarding routed ", None), seg(e, rng), (" to ", None), (v["department"], "ORG"), (".", None)],
        [("Fleet management note (", None), seg(e, rng), (") was reviewed by ", None), (v["second_person"], "PERSON"), (".", None)],
        [("Vehicle registration support marked ", None), seg(e, rng), (" as the ", None), (v["ordinal"], "ORDINAL"), (" identity document.", None)],
    ]


def benefits_credit_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    candidates = [
        "CANADIAN_SOCIAL_INSURANCE_NUMBER",
        "CANADIAN_BANK_ACCOUNT_NUMBER",
        "CANADIAN_INDIVIDUAL_TAX_NUMBER",
        "ALBERTA_PERSONAL_HEALTH_NUMBER",
    ]
    e = choice(rng, candidates)
    return [
        [("Benefit payment review verified ", None), seg(e, rng), (" for ", None), (v["money"], "MONEY"), (".", None)],
        [("Credit review placed ", None), seg(e, rng), (" in the ", None), (v["ordinal"], "ORDINAL"), (" intake batch.", None)],
        [("Loan application intake copied ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Pension administration retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Refund processing routed ", None), (v["money"], "MONEY"), (" after confirming ", None), seg(e, rng), (".", None)],
        [("Balance-due collection note: ", None), seg(e, rng), (" after colon; payment reminder pending.", None)],
        [("Penalty and interest review sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Benefits Review in ", None), (v["gpe"], "GPE"), (" linked ", None), seg(e, rng), (" to ", None), (v["cardinal"], "CARDINAL"), (" records.", None)],
        [("The benefits manual ", None), (v["work"], "WORK_OF_ART"), (" references ", None), seg(e, rng), (".", None)],
        [("The ", None), (v["norp"], "NORP"), (" credit queue confirmed ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Loan intake at ", None), (v["fac"], "FAC"), (" received ", None), seg(e, rng), (" before ", None), (v["time"], "TIME"), (".", None)],
        [("Identity verification under ", None), (v["law"], "LAW"), (" retained ", None), seg(e, rng), (".", None)],
        [("A synthetic benefits record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("The caseworker ", None), (v["person"], "PERSON"), (" checked ", None), seg(e, rng), (" without benefit history.", None)],
        [("Credit portal entry ends with ", None), seg(e, rng), (".", None)],
        [("Canada child benefit review retained ", None), seg(e, rng), (" for ", None), (v["date"], "DATE"), (".", None)],
        [("GST credit support matched ", None), seg(e, rng), (" with ", None), (v["money"], "MONEY"), (".", None)],
        [("Disability tax credit intake copied ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Collections hardship review placed ", None), seg(e, rng), (" in ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Overpayment recovery note: ", None), seg(e, rng), (" before semicolon; recovery review pending.", None)],
        [("Retirement savings administration verified ", None), seg(e, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("Garnishment administration under ", None), (v["law"], "LAW"), (" masked ", None), seg(e, rng), (".", None)],
    ]


def employment_university_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    candidates = [
        "CANADIAN_SOCIAL_INSURANCE_NUMBER",
        "ALBERTA_DRIVERS_LICENCE_NUMBER",
        "CANADIAN_PASSPORT_NUMBER",
        "CANADIAN_BANK_ACCOUNT_NUMBER",
    ]
    e = choice(rng, candidates)
    return [
        [("Employment screening verified ", None), seg(e, rng), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Background check unit copied ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("HR records office retained ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("HR identity verification read back ", None), seg(e, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("University services in ", None), (v["gpe"], "GPE"), (" linked ", None), seg(e, rng), (" to ", None), (v["cardinal"], "CARDINAL"), (" forms.", None)],
        [("Student worker onboarding placed ", None), seg(e, rng), (" in ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" HR batch selected ", None), seg(e, rng), (" for QA.", None)],
        [("OnboardHR displayed ", None), seg(e, rng), (" with completion ", None), (v["percent"], "PERCENT"), (".", None)],
        [("The ", None), (v["norp"], "NORP"), (" employment desk confirmed ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Employment onboarding note: ", None), seg(e, rng), (" before semicolon; access setup pending.", None)],
        [("University employment at ", None), (v["fac"], "FAC"), (" received ", None), seg(e, rng), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("Vendor onboarding for ", None), (v["org"], "ORG"), (" verified ", None), seg(e, rng), (".", None)],
        [("A synthetic HR record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("The analyst ", None), (v["person"], "PERSON"), (" checked ", None), seg(e, rng), (" without employment history.", None)],
        [("HR portal entry ends with ", None), seg(e, rng), (".", None)],
        [("Contractor onboarding desk copied ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Worker classification review sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Gig-worker onboarding retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Teaching assistant intake in ", None), (v["gpe"], "GPE"), (" verified ", None), seg(e, rng), (".", None)],
        [("University employment file (", None), seg(e, rng), (") was reviewed by ", None), (v["second_person"], "PERSON"), (".", None)],
        [("Background screening under ", None), (v["law"], "LAW"), (" masked ", None), seg(e, rng), (".", None)],
        [("HR records note after colon: ", None), seg(e, rng), ("; no employment history included.", None)],
    ]


def government_service_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("Government service counter verified ", None), seg(e, rng), (" at ", None), (v["fac"], "FAC"), (".", None)],
        [("Service desk in ", None), (v["gpe"], "GPE"), (" copied ", None), seg(e, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("The ", None), (v["department"], "ORG"), (" retained ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Counter note: ", None), seg(e, rng), (" after colon; identity check complete.", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" service queue selected ", None), seg(e, rng), (".", None)],
        [("Bilingual service at ", None), (v["fac"], "FAC"), (" read ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Accessibility support enlarged the form containing ", None), seg(e, rng), (".", None)],
        [("Online portal support copied ", None), seg(e, rng), (" from ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Mobile app support logged ", None), seg(e, rng), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("The service analyst ", None), (v["person"], "PERSON"), (" reviewed ", None), seg(e, rng), (" by ", None), (v["time"], "TIME"), (".", None)],
        [("Government intake across ", None), (v["loc"], "LOC"), (" sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("The case file includes ", None), seg(e, rng), (" and ", None), (v["cardinal"], "CARDINAL"), (" supporting forms.", None)],
        [("A synthetic service record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("Document retention kept ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Service portal entry ends with ", None), seg(e, rng), (".", None)],
        [("Service Canada callback verified ", None), seg(e, rng), (" before ", None), (v["date"], "DATE"), (".", None)],
        [("Government service counter at ", None), (v["time"], "TIME"), (" typed ", None), seg(e, rng), (" into the intake screen.", None)],
        [("The ", None), (v["norp"], "NORP"), (" service group retained ", None), seg(e, rng), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("Secure service intake note (", None), seg(e, rng), (") was routed to ", None), (v["department"], "ORG"), (".", None)],
        [("Counter staff counted ", None), (v["cardinal"], "CARDINAL"), (" attachments before masking ", None), seg(e, rng), (".", None)],
        [("Portal verification in ", None), (v["language"], "LANGUAGE"), (" confirmed ", None), seg(e, rng), (".", None)],
        [("The service review memo ", None), (v["work"], "WORK_OF_ART"), (" includes ", None), seg(e, rng), (".", None)],
    ]


def support_operations_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("Call-centre support read back ", None), seg(e, rng), (" for verification.", None)],
        [("Customer-service callback recorded ", None), seg(e, rng), (" after ", None), (v["person"], "PERSON"), (" called.", None)],
        [("Bilingual service desk repeated ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Accessibility support routed ", None), seg(e, rng), (" to ", None), (v["department"], "ORG"), (".", None)],
        [("Online portal support placed ", None), seg(e, rng), (" before semicolon; validation passed.", None)],
        [("Mobile app support noted \"", None), seg(e, rng), ("\" during ", None), (v["event"], "EVENT"), (".", None)],
        [("Support transcript ", None), (v["work"], "WORK_OF_ART"), (" shows ", None), seg(e, rng), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" callback attempt confirmed ", None), seg(e, rng), (".", None)],
        [("Support queue in ", None), (v["gpe"], "GPE"), (" linked ", None), seg(e, rng), (" to ", None), (v["cardinal"], "CARDINAL"), (" tickets.", None)],
        [("The agent at ", None), (v["time"], "TIME"), (" typed ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Support under ", None), (v["law"], "LAW"), (" masked ", None), seg(e, rng), (".", None)],
        [("Customer-service note from ", None), (v["loc"], "LOC"), (" retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("A synthetic support record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("Service support at ", None), (v["fac"], "FAC"), (" reviewed ", None), seg(e, rng), (".", None)],
        [("The support case ends with ", None), seg(e, rng), (".", None)],
        [("Mobile support in ", None), (v["gpe"], "GPE"), (" masked ", None), seg(e, rng), (" before screenshot review.", None)],
        [("Bilingual callback for ", None), (v["second_person"], "PERSON"), (" ended with ", None), seg(e, rng), (".", None)],
        [("Accessibility support under ", None), (v["law"], "LAW"), (" retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Online portal note: ", None), seg(e, rng), (" after colon; support case closed.", None)],
        [("The customer-service file at ", None), (v["fac"], "FAC"), (" includes ", None), seg(e, rng), (" and ", None), (v["cardinal"], "CARDINAL"), (" notes.", None)],
        [("Call-centre quality review sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Support operations near ", None), (v["loc"], "LOC"), (" archived ", None), seg(e, rng), (".", None)],
    ]


def privacy_audit_compliance_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("Privacy review masked ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Audit sampling selected ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("Compliance review found ", None), seg(e, rng), (" in ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Quality assurance approved ", None), seg(e, rng), (" after ", None), (v["quantity"], "QUANTITY"), (" of checks.", None)],
        [("The auditor ", None), (v["person"], "PERSON"), (" copied ", None), seg(e, rng), (" at ", None), (v["fac"], "FAC"), (".", None)],
        [("Annual control note: ", None), seg(e, rng), (" before semicolon; redaction confirmed.", None)],
        [("Compliance staff in ", None), (v["gpe"], "GPE"), (" reviewed ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Privacy Audit across ", None), (v["loc"], "LOC"), (" sampled ", None), seg(e, rng), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" QA batch selected ", None), seg(e, rng), (".", None)],
        [("AuditWorks flagged ", None), seg(e, rng), (" in ", None), (v["cardinal"], "CARDINAL"), (" records.", None)],
        [("Compliance note for ", None), (v["org"], "ORG"), (" references ", None), seg(e, rng), (".", None)],
        [("Privacy review retained ", None), seg(e, rng), (" for ", None), (v["date"], "DATE"), (".", None)],
        [("A synthetic audit record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("Quality review near ", None), (v["fac"], "FAC"), (" masked ", None), seg(e, rng), (".", None)],
        [("The compliance file ends with ", None), seg(e, rng), (".", None)],
        [("Document Redaction Standard requires ", None), seg(e, rng), (" to be masked from exports.", None)],
        [("The audit memo ", None), (v["work"], "WORK_OF_ART"), (" includes ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Quality assurance in ", None), (v["language"], "LANGUAGE"), (" repeated ", None), seg(e, rng), (" during training.", None)],
        [("Compliance Review counted ", None), (v["cardinal"], "CARDINAL"), (" files containing ", None), seg(e, rng), (".", None)],
        [("The privacy analyst ", None), (v["second_person"], "PERSON"), (" verified ", None), seg(e, rng), (" before ", None), (v["time"], "TIME"), (".", None)],
        [("Audit sample in ", None), (v["gpe"], "GPE"), (" retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Compliance portal entry ends with ", None), seg(e, rng), (".", None)],
    ]


def document_records_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("Document intake received ", None), seg(e, rng), (" before ", None), (v["time"], "TIME"), (".", None)],
        [("Document redaction queue masked ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Document retention archived ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Case file cleanup assigned ", None), seg(e, rng), (" to ", None), (v["department"], "ORG"), (".", None)],
        [("Records memo: ", None), seg(e, rng), (" after colon; retention review pending.", None)],
        [("The intake checklist ", None), (v["work"], "WORK_OF_ART"), (" includes ", None), seg(e, rng), (".", None)],
        [("Document packet from ", None), (v["gpe"], "GPE"), (" placed ", None), seg(e, rng), (" before notes.", None)],
        [("Archive staff at ", None), (v["fac"], "FAC"), (" indexed ", None), seg(e, rng), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" cleanup batch selected ", None), seg(e, rng), (" for review.", None)],
        [("DocumentCloud Intake stored ", None), seg(e, rng), (" in ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Redaction sample ", None), (v["cardinal"], "CARDINAL"), (" contains ", None), seg(e, rng), (".", None)],
        [("Records team in ", None), (v["loc"], "LOC"), (" retained ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (" coverage.", None)],
        [("A synthetic document record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("Document intake by ", None), (v["person"], "PERSON"), (" verified ", None), seg(e, rng), (" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("The document file ends with ", None), seg(e, rng), (".", None)],
        [("Secure mailroom processing received ", None), seg(e, rng), (" at ", None), (v["fac"], "FAC"), (".", None)],
        [("Paper form digitization scanned ", None), seg(e, rng), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("OCR exception handling reviewed ", None), seg(e, rng), (" with confidence ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Archive retrieval returned ", None), seg(e, rng), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Records disposition review retained ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Returned document handling flagged ", None), seg(e, rng), (" after ", None), (v["cardinal"], "CARDINAL"), (" attempts.", None)],
        [("OCR handling sheet ", None), (v["work"], "WORK_OF_ART"), (" labels ", None), seg(e, rng), (" as synthetic text.", None)],
    ]


def migration_qa_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("System migration mapped ", None), seg(e, rng), (" from legacy field ", None), (v["cardinal"], "CARDINAL"), (".", None)],
        [("Document migration copied ", None), seg(e, rng), (" into ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Case file cleanup moved ", None), seg(e, rng), (" to the ", None), (v["ordinal"], "ORDINAL"), (" archive batch.", None)],
        [("Migration QA compared ", None), seg(e, rng), (" with ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Software test note: ", None), seg(e, rng), (" after colon; no personal history included.", None)],
        [("The QA analyst ", None), (v["person"], "PERSON"), (" entered ", None), seg(e, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("System Migration in ", None), (v["gpe"], "GPE"), (" retained ", None), seg(e, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("Migration under ", None), (v["law"], "LAW"), (" masked ", None), seg(e, rng), (".", None)],
        [("Converted row sample ", None), (v["cardinal"], "CARDINAL"), (" includes ", None), seg(e, rng), (".", None)],
        [("The migration runbook ", None), (v["work"], "WORK_OF_ART"), (" references ", None), seg(e, rng), (".", None)],
        [("Quality Assurance Review sampled ", None), seg(e, rng), (" at ", None), (v["percent"], "PERCENT"), (".", None)],
        [("Migration queue across ", None), (v["loc"], "LOC"), (" selected ", None), seg(e, rng), (".", None)],
        [("A synthetic migration record contains ", None), seg(e, rng), (" for NER training only.", None)],
        [("The migration case for ", None), (v["org"], "ORG"), (" validated ", None), seg(e, rng), (".", None)],
        [("QA portal entry ends with ", None), seg(e, rng), (".", None)],
        [("Document migration in ", None), (v["language"], "LANGUAGE"), (" retained ", None), seg(e, rng), (" for validation.", None)],
        [("The migration batch ", None), (v["cardinal"], "CARDINAL"), (" ended with ", None), seg(e, rng), (" in the control row.", None)],
        [("Case cleanup report ", None), (v["work"], "WORK_OF_ART"), (" lists ", None), seg(e, rng), (" near the document field.", None)],
        [("Mobile app migration logged ", None), seg(e, rng), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("Online portal migration copied ", None), seg(e, rng), (" from ", None), (v["product"], "PRODUCT"), (".", None)],
        [("QA review at ", None), (v["fac"], "FAC"), (" read back ", None), seg(e, rng), (" before ", None), (v["time"], "TIME"), (".", None)],
        [("System migration file ends with ", None), seg(e, rng), (".", None)],
    ]


def multi_identifier_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    a = choice(rng, NEW_ENTITY_TYPES)
    b = choice(rng, NEW_ENTITY_TYPES)
    c = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("Please verify ", None), seg(a, rng), (" and ", None), seg(b, rng), (" before ", None), (v["date"], "DATE"), (".", None)],
        [("The ", None), (v["department"], "ORG"), (" compared ", None), seg(a, rng), (" with ", None), seg(b, rng), (" for a synthetic cross-check.", None)],
        [("Identity review stored ", None), seg(a, rng), (", ", None), seg(b, rng), (", and ", None), seg(c, rng), (" in ", None), (v["work"], "WORK_OF_ART"), (".", None)],
        [("Cross-check note: ", None), seg(a, rng), (" before semicolon; secondary value ", None), seg(b, rng), (" verified.", None)],
        [("The analyst ", None), (v["person"], "PERSON"), (" matched ", None), seg(a, rng), (" to ", None), seg(b, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("Service portal in ", None), (v["gpe"], "GPE"), (" validated ", None), seg(a, rng), (" and ", None), seg(b, rng), (" at ", None), (v["time"], "TIME"), (".", None)],
        [("A synthetic multi-identifier record contains ", None), seg(a, rng), (" and ", None), seg(b, rng), (" for NER training only.", None)],
        [("Audit sampled ", None), seg(a, rng), (" with ", None), seg(b, rng), (" at ", None), (v["percent"], "PERCENT"), (" coverage.", None)],
        [("Document cleanup retained ", None), seg(a, rng), (" plus ", None), seg(b, rng), (" for ", None), (v["quantity"], "QUANTITY"), (".", None)],
        [("The ", None), (v["ordinal"], "ORDINAL"), (" QA packet included ", None), seg(a, rng), (" and ", None), seg(b, rng), (".", None)],
        [("Call-centre support read back \"", None), seg(a, rng), ("\" and \"", None), seg(b, rng), ("\" in ", None), (v["language"], "LANGUAGE"), (".", None)],
        [("Compliance review for ", None), (v["org"], "ORG"), (" linked ", None), seg(a, rng), (" with ", None), seg(b, rng), (".", None)],
        [("Privacy staff at ", None), (v["fac"], "FAC"), (" masked ", None), seg(a, rng), (" and ", None), seg(b, rng), (".", None)],
        [("The reconciliation record shows ", None), seg(a, rng), (" beside ", None), (v["money"], "MONEY"), (" and ", None), seg(b, rng), (".", None)],
        [("Case file entry ends with ", None), seg(a, rng), (" and ", None), seg(b, rng), (".", None)],
        [("Before ", None), (v["event"], "EVENT"), (", compare ", None), seg(a, rng), (" with ", None), seg(b, rng), (" and notify ", None), (v["person"], "PERSON"), (".", None)],
        [("Multi-factor intake placed ", None), seg(a, rng), (" after colon and ", None), seg(b, rng), (" before semicolon; review open.", None)],
        [("The ", None), (v["norp"], "NORP"), (" service file included ", None), seg(a, rng), (" and ", None), seg(c, rng), (" for administrative verification.", None)],
        [("Cross-system migration linked ", None), seg(a, rng), (" to ", None), seg(b, rng), (" in ", None), (v["product"], "PRODUCT"), (".", None)],
        [("The verification memo ", None), (v["work"], "WORK_OF_ART"), (" contains ", None), seg(a, rng), (" plus ", None), seg(b, rng), (".", None)],
        [("Support staff in ", None), (v["loc"], "LOC"), (" checked ", None), seg(a, rng), (" and ", None), seg(b, rng), (" without personal history.", None)],
        [("The combined review file starts with ", None), seg(a, rng), (" and ends with ", None), seg(b, rng), (".", None)],
    ]


def false_positive_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    """
    Templates containing number-like tokens tagged as O or CARDINAL beside
    real Canadian identifiers. Trains the model to discriminate identifiers
    from phone numbers, postal codes, reference codes, invoice numbers,
    confirmation codes, badge numbers, plate numbers, room numbers, DINs,
    policy numbers, claim numbers, booking codes, fax numbers, etc.

    All decoy values are tagged as O (plain text) or CARDINAL where
    contextually appropriate. None are tagged as identifier entities.
    """
    e = choice(rng, NEW_ENTITY_TYPES)
    phone = choice(rng, PHONE_NUMBERS)
    fax = choice(rng, FAX_NUMBERS)
    postal = choice(rng, POSTAL_CODES)
    ref = choice(rng, REFERENCE_CODES)
    invoice = choice(rng, INVOICE_NUMBERS)
    case_file = choice(rng, CASE_FILE_NUMBERS)
    badge = choice(rng, BADGE_NUMBERS)
    plate = choice(rng, PLATE_NUMBERS)
    room = choice(rng, ROOM_NUMBERS)
    din = choice(rng, DIN_NUMBERS)
    policy_num = choice(rng, POLICY_NUMBERS)
    claim_num = choice(rng, CLAIM_NUMBERS)
    booking = choice(rng, BOOKING_CODES)
    flight = choice(rng, FLIGHT_CODES)
    generic = choice(rng, GENERIC_NUMBERS)

    sin = "CANADIAN_SOCIAL_INSURANCE_NUMBER"
    bank = "CANADIAN_BANK_ACCOUNT_NUMBER"
    phn = "ALBERTA_PERSONAL_HEALTH_NUMBER"
    dl = "ALBERTA_DRIVERS_LICENCE_NUMBER"
    itn = "CANADIAN_INDIVIDUAL_TAX_NUMBER"
    prov = "CANADIAN_PROVIDER_IDENTIFIER"
    passport = "CANADIAN_PASSPORT_NUMBER"

    return [
        # --- phone numbers beside real identifiers ---
        [("Call ", None), (phone, None), (" to verify ", None), seg(sin, rng), (" for ", None), (v["person"], "PERSON"), (".", None)],
        [("The contact number ", None), (phone, None), (" is not an identifier; the actual SIN is ", None), seg(sin, rng), (".", None)],
        [("Fax ", None), (fax, None), (" with a copy of ", None), seg(e, rng), (" to ", None), (v["department"], "ORG"), (".", None)],
        [("Phone ", None), (phone, None), (" reached ", None), (v["person"], "PERSON"), (" who confirmed ", None), seg(e, rng), (".", None)],
        [("Contact the client at ", None), (phone, None), (" to confirm ", None), seg(bank, rng), (" before ", None), (v["date"], "DATE"), (".", None)],

        # --- postal codes beside real identifiers ---
        [("Mail the form to ", None), (postal, None), (" after verifying ", None), seg(sin, rng), (".", None)],
        [("The mailing address ends with ", None), (postal, None), ("; the PHN on file is ", None), seg(phn, rng), (".", None)],
        [("Forward ", None), seg(e, rng), (" to the office at postal code ", None), (postal, None), (".", None)],

        # --- reference / confirmation codes beside real identifiers ---
        [("Ticket ", None), (ref, None), (" was opened to verify ", None), seg(sin, rng), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Reference ", None), (ref, None), (" links to ", None), seg(e, rng), (" in the intake log.", None)],
        [("Use confirmation code ", None), (booking, None), (" when presenting ", None), seg(passport, rng), (" at ", None), (v["fac"], "FAC"), (".", None)],
        [("The booking code ", None), (booking, None), (" is not a passport number; the passport is ", None), seg(passport, rng), (".", None)],

        # --- invoice numbers beside real identifiers ---
        [("Invoice ", None), (invoice, None), (" was paid from ", None), seg(bank, rng), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("Pay invoice ", None), (invoice, None), (" using ", None), seg(bank, rng), (" for ", None), (v["money"], "MONEY"), (".", None)],
        [("The invoice number ", None), (invoice, None), (" is not a bank account; the account is ", None), seg(bank, rng), (".", None)],

        # --- case / file numbers beside real identifiers ---
        [("Case ", None), (case_file, None), (" references ", None), seg(e, rng), (" under ", None), (v["law"], "LAW"), (".", None)],
        [("File ", None), (case_file, None), (" in ", None), (v["product"], "PRODUCT"), (" contains ", None), seg(e, rng), (".", None)],
        [("Open case ", None), (case_file, None), (" and verify ", None), seg(sin, rng), (" for ", None), (v["person"], "PERSON"), (".", None)],

        # --- badge / employee numbers beside real identifiers ---
        [("The officer with ", None), (badge, None), (" verified ", None), seg(e, rng), (" at ", None), (v["fac"], "FAC"), (".", None)],
        [("Staff ", None), (badge, None), (" processed ", None), seg(sin, rng), (" for ", None), (v["org"], "ORG"), (".", None)],

        # --- plate numbers beside real identifiers ---
        [("Vehicle ", None), (plate, None), (" is registered to the holder of ", None), seg(dl, rng), (".", None)],
        [("The licence plate ", None), (plate, None), (" is not a driver's licence; the DL is ", None), seg(dl, rng), (".", None)],

        # --- room / bed numbers beside real identifiers ---
        [("Patient in ", None), (room, None), (" has ", None), seg(phn, rng), (" on file.", None)],
        [("Admit to ", None), (room, None), (" and verify ", None), seg(phn, rng), (" with ", None), seg(prov, rng), (".", None)],

        # --- DIN numbers beside real identifiers ---
        [("Dispense ", None), (din, None), (" for patient ", None), seg(phn, rng), (" billed under ", None), seg(prov, rng), (".", None)],
        [("The ", None), (din, None), (" is a drug code, not a health number; the PHN is ", None), seg(phn, rng), (".", None)],

        # --- policy / claim numbers beside real identifiers ---
        [("Under ", None), (policy_num, None), (", verify ", None), seg(sin, rng), (" before ", None), (v["date"], "DATE"), (".", None)],
        [("Claim ", None), (claim_num, None), (" references ", None), seg(e, rng), (" for ", None), (v["money"], "MONEY"), (".", None)],
        [("The policy number ", None), (policy_num, None), (" is not a SIN; the SIN is ", None), seg(sin, rng), (".", None)],

        # --- flight / booking codes beside passport ---
        [("Check in for ", None), (flight, "CARDINAL"), (" using ", None), seg(passport, rng), (" at ", None), (v["fac"], "FAC"), (".", None)],
        [("The flight code ", None), (flight, "CARDINAL"), (" is not a passport number; the passport is ", None), seg(passport, rng), (".", None)],
        [("Boarding pass for ", None), (flight, "CARDINAL"), (" and ", None), (booking, None), (" linked to ", None), seg(passport, rng), (".", None)],

        # --- generic numbers that look like identifiers but are not ---
        [("Page ", None), (generic, "CARDINAL"), (" of the form contains ", None), seg(e, rng), (".", None)],
        [("Row ", None), (generic, "CARDINAL"), (" in the spreadsheet maps to ", None), seg(e, rng), (".", None)],
        [("Batch ", None), (generic, "CARDINAL"), (" imported ", None), seg(e, rng), (" into ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Sequence number ", None), (generic, "CARDINAL"), (" precedes ", None), seg(e, rng), (" in the file.", None)],
        [("The record count is ", None), (generic, "CARDINAL"), ("; the identifier on file is ", None), seg(e, rng), (".", None)],
        [("Transaction ", None), (generic, "CARDINAL"), (" posted ", None), (v["money"], "MONEY"), (" to ", None), seg(bank, rng), (".", None)],

        # --- explicit disambiguation sentences ---
        [("Note: ", None), (phone, None), (" is a phone number, not an identifier; see ", None), seg(e, rng), (" for the actual value.", None)],
        [("Do not confuse invoice ", None), (invoice, None), (" with ", None), seg(bank, rng), ("; they are different fields.", None)],
        [("The six-digit code ", None), (generic, "CARDINAL"), (" is a batch ID, not a licence; the DL is ", None), seg(dl, rng), (".", None)],
        [("Postal code ", None), (postal, None), (" and health number ", None), seg(phn, rng), (" both appear on the form.", None)],

        # --- mixed: multiple decoys + identifier ---
        [("Call ", None), (phone, None), (", reference ", None), (ref, None), (", and verify ", None), seg(sin, rng), (".", None)],
        [("File ", None), (case_file, None), (" at ", None), (postal, None), (" contains ", None), seg(e, rng), (" and invoice ", None), (invoice, None), (".", None)],
        [("Ticket ", None), (ref, None), (" for ", None), (v["person"], "PERSON"), (" in ", None), (room, None), (" references ", None), seg(phn, rng), (".", None)],
        [("The fax to ", None), (fax, None), (" included ", None), seg(e, rng), (" and claim ", None), (claim_num, None), (".", None)],
    ]

def plain_negative_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    """
    Negative templates — sentences with zero entities of any kind.
    All tokens will be tagged O.

    NOTE: NOT added to TEMPLATE_BUILDERS. Used directly by generate_negative_example.
    """
    return [
        # Workflow / status
        [("The system was updated successfully.", None)],
        [("Please review the document and resubmit.", None)],
        [("Our team completed the migration last week.", None)],
        [("The application was approved without issues.", None)],
        [("Update the documentation when convenient.", None)],
        [("Contact the support desk for further assistance.", None)],
        [("The meeting will be rescheduled to a later date.", None)],
        [("All requirements have been met for this case.", None)],
        [("Please confirm receipt of the message.", None)],
        [("The file has been uploaded successfully.", None)],
        [("The portal is currently undergoing maintenance.", None)],
        [("No further action is required at this time.", None)],
        [("Submit your feedback through the online form.", None)],
        [("The process completed without errors.", None)],
        [("Please verify the information before submitting.", None)],
        [("The audit team will review next quarter.", None)],
        [("Records will be retained per policy.", None)],
        [("Please escalate to the supervisor if needed.", None)],
        [("The training session is mandatory for all staff.", None)],
        [("Coverage decisions are reviewed annually.", None)],
        # Office / operations
        [("The intake desk closes at five.", None)],
        [("Please review the procedural updates.", None)],
        [("The shared mailbox is monitored daily.", None)],
        [("Operations resume after the long weekend.", None)],
        [("We will follow up next week.", None)],
        [("The form has been mailed out.", None)],
        [("Approvals are pending from the manager.", None)],
        [("The vendor confirmed the schedule change.", None)],
        [("Records have been archived to the vault.", None)],
        [("All systems are operating normally.", None)],
        [("The submission window has closed.", None)],
        [("Please refer to the attached guidance.", None)],
        [("The case has been transferred to another unit.", None)],
        [("Verification is in progress.", None)],
        [("Backups completed overnight without incident.", None)],
        [("Please reset your password if prompted.", None)],
        [("The mailing list has been updated.", None)],
        [("The newsletter goes out every Friday.", None)],
        [("The training room is reserved.", None)],
        [("Reminders will be sent automatically.", None)],
        # Communications
        [("Thank you for your patience.", None)],
        [("We appreciate your continued cooperation.", None)],
        [("Please disregard the previous message.", None)],
        [("An automated reply will follow shortly.", None)],
        [("Office hours are listed on the portal.", None)],
        [("The team is reviewing the request.", None)],
        [("Your inquiry has been forwarded.", None)],
        [("Confirmation will arrive by email.", None)],
        [("Please complete the online survey.", None)],
        [("Subscribe to receive updates.", None)],
        # Generic professional
        [("The procedure was completed successfully.", None)],
        [("Please complete the form attached.", None)],
        [("Our office hours are posted on the door.", None)],
        [("Documents are processed in the order received.", None)],
        [("Service availability may vary.", None)],
        [("Notifications are sent automatically.", None)],
        [("All requests follow the standard workflow.", None)],
        [("Please allow several business days for processing.", None)],
        [("Confirmation messages are sent upon completion.", None)],
        [("The internal handbook covers these scenarios.", None)],
        # Meetings / scheduling
        [("The meeting has been postponed.", None)],
        [("Please join the call when ready.", None)],
        [("The agenda will be circulated shortly.", None)],
        [("Minutes from the last meeting are available.", None)],
        [("The kickoff has been pushed back.", None)],
        [("Standups are held in the main conference room.", None)],
        [("Please RSVP at your earliest convenience.", None)],
        [("The retrospective is optional this sprint.", None)],
        [("Cancellation notices were distributed yesterday.", None)],
        [("Our weekly sync is moving online.", None)],
        # Project / IT
        [("The pipeline has been redeployed.", None)],
        [("Logs were rotated overnight.", None)],
        [("Tests are passing on the main branch.", None)],
        [("The release notes have been published.", None)],
        [("The dependency upgrade is scheduled.", None)],
        [("Maintenance windows are announced in advance.", None)],
        [("The bug has been triaged.", None)],
        [("Documentation lives in the shared wiki.", None)],
        [("Code reviews are required before merging.", None)],
        [("The deployment was rolled back.", None)],
        # Customer service
        [("Please hold while we transfer your call.", None)],
        [("The chat agent is currently unavailable.", None)],
        [("We are experiencing higher than usual volume.", None)],
        [("Your inquiry is being processed.", None)],
        [("Help articles are available on the support site.", None)],
        [("The status page reflects current incidents.", None)],
        [("Our team will respond within one business day.", None)],
        [("Please describe the issue in detail.", None)],
        [("Service updates appear on the dashboard.", None)],
        [("Thank you for reaching out.", None)],
        # Misc fillers
        [("The shipment will arrive shortly.", None)],
        [("Inventory levels have been restocked.", None)],
        [("Please refer to the user manual.", None)],
        [("Forms are available at the front desk.", None)],
        [("Submissions close on the posted date.", None)],
        [("Office furniture has been rearranged.", None)],
        [("The light bulbs in the hallway were replaced.", None)],
        [("Lunch will be catered tomorrow.", None)],
        [("The new printer is now operational.", None)],
        [("Coffee is available in the break room.", None)],
        [("The window blinds need adjustment.", None)],
        [("Parking is limited near the main entrance.", None)],
        [("Please dispose of recyclables properly.", None)],
        [("The fire drill is scheduled this morning.", None)],
        [("The lobby was repainted over the weekend.", None)],
        [("Carpets will be cleaned this evening.", None)],
        [("The water fountain is out of order.", None)],
        [("Wi-Fi credentials are posted on the board.", None)],
        [("The thermostat has been adjusted.", None)],
        [("Please keep the volume low in shared areas.", None)],
    ]


def hard_negative_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    """
    Hard negative templates — numbers that LOOK like Canadian identifiers
    but have no context keywords (SIN, PHN, ITN, transit, account, passport, etc.).
    Teaches the model that format alone is not enough — context matters.

    NOTE: NOT added to TEMPLATE_BUILDERS. Used directly by generate_negative_example.
    """
    return [
        # 3-3-3 dashed (looks like PHN / ITN / SIN)
        [("Reference 459-311-601 appears in the meeting notes.", None)],
        [("Invoice 257-574-243 was paid on Tuesday.", None)],
        [("Order number 130-692-544 has shipped from the warehouse.", None)],
        [("Section 459-311 of the document is missing.", None)],
        [("Pallet 257-574-243 was misrouted last week.", None)],
        [("Track number 459-311-601 was relabeled by the carrier.", None)],
        [("Asset tag 130-692-544 belongs to the lab.", None)],
        [("Postal route 257-574 covers the north end.", None)],
        [("Catalog number 459-311-601 is discontinued.", None)],
        [("Build identifier 130-692-544 failed the smoke test.", None)],
        # 3-3-3 spaced
        [("Pallet 459 311 601 arrived at the depot.", None)],
        [("Reference 257 574 243 was reassigned to another team.", None)],
        [("Run 130 692 544 completed without errors.", None)],
        [("Lot 459 311 601 is awaiting inspection.", None)],
        [("Container 257 574 243 will be unloaded next.", None)],
        # Plain 9-digit
        [("Job ID 459311601 has been scheduled.", None)],
        [("Catalog code 257574243 is on backorder.", None)],
        [("Manufacturing lot 130692544 requires inspection.", None)],
        [("Tracking number 459311601 was rerouted.", None)],
        [("Document hash 257574243 matches the archive copy.", None)],
        [("Lot 130692544 has been quarantined.", None)],
        [("Build number 459311601 was reverted.", None)],
        [("Order ID 257574243 is queued for shipping.", None)],
        # Bank-account-shaped (no transit/account keywords)
        [("Building 18016 office 86796296 is being renovated.", None)],
        [("Server rack 18016 hosts the legacy database.", None)],
        [("Asset 18016 86796296 was scanned during inventory.", None)],
        [("Sensor 18016-86796296 reported an error.", None)],
        [("Container 18016/86796296 will be unloaded next.", None)],
        [("Block 18016 unit 86796296 is currently unassigned.", None)],
        # 4-7 digit codes (passport-shaped)
        [("Conference room 459 is booked for the morning.", None)],
        [("Channel 18016 is reserved for emergency communications.", None)],
        [("Phone extension 4591 reached voicemail this morning.", None)],
        [("Aisle 18016 contains overflow inventory.", None)],
        [("Bus route 257 runs every fifteen minutes.", None)],
        [("Survey response 8294 has been recorded.", None)],
        [("Ticket 384729 was closed yesterday.", None)],
        [("Locker 482910 needs to be cleaned out.", None)],
        [("Cabinet 293847 has been relocated.", None)],
        [("Door code 102938 was reset overnight.", None)],
        # Letter + digits (looks like passport)
        [("Conference room A1234567 is being painted.", None)],
        [("Storage bay BX234567 is full.", None)],
        [("Server CD987654 was decommissioned.", None)],
        [("Asset code FG123456 is no longer in use.", None)],
        [("Locker number HJ987654 has been reassigned.", None)],
        # Mixed / odd formats
        [("Page 257 of 574 contains the executive summary.", None)],
        [("Last 4 digits of the confirmation are 8294.", None)],
        [("Chapter 130, section 692, paragraph 544.", None)],
        [("Latitude 459.311 longitude 601.243 is approximate.", None)],
        [("Score breakdown: 257 to 574.", None)],
        [("Distance: 459 km over 311 minutes.", None)],
        [("Population estimate: 130,692 residents.", None)],
        [("Sample size was 257, error margin 5.74%.", None)],
        [("Run time was 459 seconds for 311 records.", None)],
        [("Capacity: 18016 units across 8 warehouses.", None)],
        # Numeric-heavy decoys
        [("We processed 4591 records in 384 seconds.", None)],
        [("There were 25638 active users last quarter.", None)],
        [("Total throughput reached 86796296 transactions.", None)],
        [("We tracked 459311 events over the period.", None)],
        [("Sample contained 130692 unique observations.", None)],
        [("The dataset has 257574 rows.", None)],
        [("Storage usage hit 18016 gigabytes.", None)],
        [("We expected 4591 visitors per week.", None)],

        # --- clock / time values (digit-heavy, NOT identifiers) ---
        [("The office opens at 9:00 a.m. on weekdays.", None)],
        [("Please call before 10:30 AM.", None)],
        [("The session runs from 8:15 a.m. to 4:45 p.m.", None)],
        [("Doors close at 5:00 p.m. sharp.", None)],
        [("The deadline is 11:59 p.m. tonight.", None)],
        [("Shift starts at 7:30 AM and ends at 3:30 PM.", None)],
        [("The line opens at 9:00 and closes at 17:00.", None)],
        [("Call volume peaks between 10:00 and 14:00.", None)],
        [("The meeting was held at 14:45 on Tuesday.", None)],
        [("Service resumes at 8:00 a.m. tomorrow.", None)],
        # --- transit-institution-account bank-shaped numbers (no keyword) ---
        [("Building 12345-003-7891234 is under renovation.", None)],
        [("Sensor 09876-006-48291034 reported a fault.", None)],
        [("Asset tag 55123-002-9283746 was scanned.", None)],
        [("Container 34521-010-1029384 cleared the gate.", None)],
        [("Route 78901-004-3847291 runs overnight.", None)],
        [("Block 12345 003 7891234 is unassigned.", None)],
        [("Pallet 09876 001 48291034 arrived at dock 3.", None)],
        [("Survey code 55123 006 9283746 is on file.", None)],
        [("Row 34521 002 1029384 was archived.", None)],
        [("Segment 78901 010 3847291 failed QA.", None)],
        # --- long digit-only strings (DL-shaped, no keyword) ---
        [("Serial number 483920174 is stamped on the unit.", None)],
        [("Job ID 5839201748 was scheduled for Tuesday.", None)],
        [("Manifest entry 38472019483 was reconciled.", None)],
        [("Hash 483920174839201 matches the archive copy.", None)],
        [("The log records 48392017483920 events total.", None)],
        # --- alnum strings (DL MB/NS-shaped, no keyword) ---
        [("Storage bay BX9K3M4T7P2 is full.", None)],
        [("Asset code FG9K3M4T7P2QR is no longer in use.", None)],
        [("Server rack DK3M4T7P2QR48 was decommissioned.", None)],
        [("Locker MK3T7P2QR483N has been reassigned.", None)],
        # --- letter+digits+letters (passport-shaped, no keyword) ---
        [("Conference room A382947BC is being repainted.", None)],
        [("Storage unit B748291KX is reserved for archives.", None)],
        [("Equipment bay C920183MN is under maintenance.", None)],
        [("Lab bench D481920PQ was relabelled.", None)],
        # --- 2-letter+6-digit (passport LL6, no keyword) ---
        [("Cabinet AB382947 has been relocated.", None)],
        [("Slot CD748291 is available for booking.", None)],
        [("Bay XK920183 will be cleared tomorrow.", None)],
        [("Zone PQ481920 requires badge access.", None)],
        # --- bare 9-digit numbers (plain / hyphen / spaced) with NO keyword ---
        # These are the hardest negatives: same format as SIN/ITN/PHN, zero context.
        [("Serial number 748291035 was stamped on the unit.", None)],
        [("The batch code is 362948175.", None)],
        [("Equipment ID 591837264 needs recalibration.", None)],
        [("Asset 203948571 was transferred to the warehouse.", None)],
        [("The manifest lists 674920183 as the container identifier.", None)],
        [("Sequence 819274530 completed without errors.", None)],
        [("Run 405738291 was aborted at step four.", None)],
        [("Control value 927463018 matched the expected checksum.", None)],
        [("Catalog reference 546182730 is on back-order.", None)],
        [("Log entry 738920461 records the shutdown event.", None)],
        # hyphen-separated bare 9-digit
        [("Order 748-291-035 has been fulfilled.", None)],
        [("Lot 362-948-175 passed quality control.", None)],
        [("Track 591-837-264 was rerouted to depot B.", None)],
        [("Part 203-948-571 is discontinued.", None)],
        [("Invoice 674-920-183 was closed last Friday.", None)],
        [("Reference 819-274-530 appears in the shipping log.", None)],
        [("Record 405-738-291 was archived successfully.", None)],
        [("Entry 927-463-018 is flagged for review.", None)],
        [("Code 546-182-730 is listed in the parts manual.", None)],
        [("Job 738-920-461 was completed ahead of schedule.", None)],
        # space-separated bare 9-digit
        [("Pallet 748 291 035 arrived at the depot.", None)],
        [("Container 362 948 175 will be unloaded tomorrow.", None)],
        [("Package 591 837 264 requires a signature.", None)],
        [("Shipment 203 948 571 cleared customs.", None)],
        [("Manifest row 674 920 183 was reconciled.", None)],
        [("Run 819 274 530 finished in nominal time.", None)],
        [("Ticket 405 738 291 was reassigned to the overnight team.", None)],
        [("Sequence 927 463 018 is ready for dispatch.", None)],
        [("Batch 546 182 730 is awaiting inspection.", None)],
        [("Job 738 920 461 is on hold pending parts.", None)],
    ]

def casual_email_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("Hi, please update the file with ", None), seg(e, rng), (" at your earliest convenience.", None)],
        [("Just a heads-up: we need to verify ", None), seg(e, rng), (" before end of day.", None)],
        [("Can you confirm that ", None), seg(e, rng), (" is correct for ", None), (v["person"], "PERSON"), ("?", None)],
        [("FYI — the system flagged ", None), seg(e, rng), (" during the nightly batch.", None)],
        [("Per our call, the identifier is ", None), seg(e, rng), (" and should be updated in ", None), (v["product"], "PRODUCT"), (".", None)],
        [("Quick note: ", None), seg(e, rng), (" needs to be masked before the report goes to ", None), (v["org"], "ORG"), (".", None)],
        [("Forwarding this — ", None), (v["person"], "PERSON"), (" needs the value ", None), seg(e, rng), (" for the intake form.", None)],
        [("See below: the client provided ", None), seg(e, rng), (" as their identifier.", None)],
        [("Reminder: please redact ", None), seg(e, rng), (" from the shared document before ", None), (v["date"], "DATE"), (".", None)],
        [("Thanks for sending ", None), seg(e, rng), (" — I'll route it to ", None), (v["department"], "ORG"), (".", None)],
        [("Action required: validate ", None), seg(e, rng), (" by ", None), (v["time"], "TIME"), (".", None)],
        [("Could you double-check ", None), seg(e, rng), (" in the system? It may be outdated.", None)],
        [("Note from ", None), (v["person"], "PERSON"), (": the correct value is ", None), seg(e, rng), (".", None)],
        [("Attached is the form containing ", None), seg(e, rng), (" for your records.", None)],
        [("Follow-up: ", None), seg(e, rng), (" was confirmed by ", None), (v["org"], "ORG"), (" on ", None), (v["date"], "DATE"), (".", None)],
        [("Please forward ", None), seg(e, rng), (" to ", None), (v["department"], "ORG"), (" by ", None), (v["time"], "TIME"), (".", None)],
        [("Heads up — ", None), seg(e, rng), (" appears in the export and must be redacted.", None)],
        [("The value ", None), seg(e, rng), (" was provided by ", None), (v["person"], "PERSON"), (" over the phone.", None)],
        [("For your reference: ", None), seg(e, rng), (".", None)],
        [("Please confirm receipt of ", None), seg(e, rng), (" and update the intake log.", None)],
        [("Circling back on ", None), seg(e, rng), (" — has it been verified yet?", None)],
        [("The updated identifier is ", None), seg(e, rng), (", replacing the previous entry.", None)],
    ]


def fragment_label_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("Identifier: ", None), seg(e, rng), (".", None)],
        [("Value on file: ", None), seg(e, rng), (".", None)],
        [("Document field: ", None), seg(e, rng), (".", None)],
        [("Entry: ", None), seg(e, rng), ("; status: active.", None)],
        [("Verified: ", None), seg(e, rng), (".", None)],
        [("Flagged value: ", None), seg(e, rng), (".", None)],
        [("Masked: [REDACTED]; original: ", None), seg(e, rng), (".", None)],
        [("Label: ", None), seg(e, rng), ("; source: intake form.", None)],
        [("Field value: ", None), seg(e, rng), ("; verified by ", None), (v["person"], "PERSON"), (".", None)],
        [("Intake record: ", None), seg(e, rng), ("; date: ", None), (v["date"], "DATE"), (".", None)],
        [("Batch entry ", None), (v["ordinal"], "ORDINAL"), (": ", None), seg(e, rng), (".", None)],
        [("Row ", None), (v["cardinal"], "CARDINAL"), (" — identifier: ", None), seg(e, rng), (".", None)],
        [("On file: ", None), seg(e, rng), (" (", None), (v["org"], "ORG"), (").", None)],
        [("System record: ", None), seg(e, rng), ("; updated ", None), (v["date"], "DATE"), (".", None)],
        [("Submitted: ", None), seg(e, rng), ("; reviewed by ", None), (v["person"], "PERSON"), (".", None)],
        [("Key: ", None), seg(e, rng), ("; value type: identifier.", None)],
        [("Processed: ", None), seg(e, rng), (" during ", None), (v["event"], "EVENT"), (".", None)],
        [("Archived: ", None), seg(e, rng), (" per ", None), (v["law"], "LAW"), (".", None)],
        [("Redacted copy: [REDACTED]. Original: ", None), seg(e, rng), (".", None)],
        [("Form field: ", None), seg(e, rng), (" (", None), (v["gpe"], "GPE"), (").", None)],
        [("Control value: ", None), seg(e, rng), (".", None)],
        [("Reference field: ", None), seg(e, rng), (" for ", None), (v["department"], "ORG"), (".", None)],
    ]


def detached_keyword_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    """
    Templates where the keyword sits in O-tagged sentence text — separated from
    the value by a copula, punctuation, filler phrase, or placed AFTER the value.
    Trains the model to fire on keyword + gap + value and value + post-keyword.
    """
    sin  = "CANADIAN_SOCIAL_INSURANCE_NUMBER"
    itn  = "CANADIAN_INDIVIDUAL_TAX_NUMBER"
    phn  = "ALBERTA_PERSONAL_HEALTH_NUMBER"
    prov = "CANADIAN_PROVIDER_IDENTIFIER"
    dl   = "ALBERTA_DRIVERS_LICENCE_NUMBER"
    pp   = "CANADIAN_PASSPORT_NUMBER"
    bank = "CANADIAN_BANK_ACCOUNT_NUMBER"
    e    = choice(rng, NEW_ENTITY_TYPES)

    return [
        # --- SIN: copula / separator ---
        [("SIN is ", None), seg_bare(sin), (".", None)],
        [("SIN: ", None), seg_bare(sin), (".", None)],
        [("SIN — ", None), seg_bare(sin), (".", None)],
        [("SIN, ", None), seg_bare(sin), (".", None)],
        [("social insurance number is ", None), seg_bare(sin), (".", None)],
        [("The employee SIN is ", None), seg_bare(sin), (".", None)],
        [("Their social insurance number was ", None), seg_bare(sin), (".", None)],
        [("social insurance number: ", None), seg_bare(sin), (" — please verify.", None)],
        # --- SIN: post-positional (keyword after value) ---
        [("The number ", None), seg_bare(sin), (" is the SIN on file.", None)],
        [("Value ", None), seg_bare(sin), (" (SIN).", None)],
        [("The value ", None), seg_bare(sin), (" — SIN — was confirmed.", None)],
        # --- SIN: with gap + surrounding context ---
        [("Please verify the SIN ", None), seg_bare(sin), (" for ", None), (v["person"], "PERSON"), (".", None)],
        [("The payroll SIN on file reads ", None), seg_bare(sin), (" for ", None), (v["org"], "ORG"), (".", None)],
        [(v["person"], "PERSON"), ("'s SIN is ", None), seg_bare(sin), (".", None)],
        [(v["org"], "ORG"), (" recorded SIN ", None), seg_bare(sin), (" before ", None), (v["date"], "DATE"), (".", None)],

        # --- ITN: copula / separator ---
        [("ITN is ", None), seg_bare(itn), (".", None)],
        [("ITN: ", None), seg_bare(itn), (".", None)],
        [("ITN — ", None), seg_bare(itn), (".", None)],
        [("individual tax number is ", None), seg_bare(itn), (".", None)],
        [("CRA individual tax number: ", None), seg_bare(itn), (".", None)],
        [("The ITN on file is ", None), seg_bare(itn), (".", None)],
        [("The individual tax number provided is ", None), seg_bare(itn), (".", None)],
        [("non-resident ITN: ", None), seg_bare(itn), (".", None)],
        [("individual tax no. ", None), seg_bare(itn), (".", None)],
        # --- ITN: post-positional ---
        [("The number ", None), seg_bare(itn), (" is an individual tax number.", None)],
        [("Value ", None), seg_bare(itn), (" — ITN.", None)],
        # --- ITN: with context ---
        [("The ITN provided by ", None), (v["person"], "PERSON"), (" is ", None), seg_bare(itn), (".", None)],
        [("CRA file shows ITN ", None), seg_bare(itn), (" for ", None), (v["org"], "ORG"), (".", None)],

        # --- PHN: copula / separator ---
        [("PHN is ", None), seg_bare(phn), (".", None)],
        [("PHN: ", None), seg_bare(phn), (".", None)],
        [("PHN — ", None), seg_bare(phn), (".", None)],
        [("personal health number is ", None), seg_bare(phn), (".", None)],
        [("The PHN on file reads ", None), seg_bare(phn), (".", None)],
        [("personal health number on the form is ", None), seg_bare(phn), (".", None)],
        [("Alberta PHN: ", None), seg_bare(phn), (".", None)],
        [("patient PHN reads ", None), seg_bare(phn), (".", None)],
        # --- PHN: post-positional ---
        [("The number ", None), seg_bare(phn), (" (PHN on file).", None)],
        [("Value ", None), seg_bare(phn), (" — personal health number.", None)],
        # --- PHN: with context ---
        [("The PHN for ", None), (v["person"], "PERSON"), (" is ", None), seg_bare(phn), (".", None)],
        [(v["fac"], "FAC"), (" records PHN ", None), seg_bare(phn), (" on the intake form.", None)],

        # --- Passport: copula / separator ---
        [("passport is ", None), seg_bare(pp), (".", None)],
        [("Passport: ", None), seg_bare(pp), (".", None)],
        [("passport number is ", None), seg_bare(pp), (".", None)],
        [("The passport on file is ", None), seg_bare(pp), (".", None)],
        [("travel document: ", None), seg_bare(pp), (".", None)],
        [("document no. ", None), seg_bare(pp), (" was stamped.", None)],
        # --- Passport: post-positional ---
        [("The document ", None), seg_bare(pp), (" is a Canadian passport.", None)],
        [("Travel credential ", None), seg_bare(pp), (" (passport) was scanned.", None)],
        # --- Passport: with context ---
        [(v["person"], "PERSON"), ("'s passport is ", None), seg_bare(pp), (".", None)],
        [("Border check at ", None), (v["fac"], "FAC"), (" logged passport ", None), seg_bare(pp), (".", None)],

        # --- DL: copula / separator ---
        [("DL is ", None), seg_bare(dl), (".", None)],
        [("driver's licence is ", None), seg_bare(dl), (".", None)],
        [("licence number: ", None), seg_bare(dl), (".", None)],
        [("The DL on record is ", None), seg_bare(dl), (".", None)],
        [("AB DL: ", None), seg_bare(dl), (".", None)],
        [("Alberta DL reads ", None), seg_bare(dl), (".", None)],
        # --- DL: post-positional ---
        [("The number ", None), seg_bare(dl), (" is an Alberta driver's licence.", None)],
        # --- DL: with context ---
        [("Registry lookup returned DL ", None), seg_bare(dl), (" for ", None), (v["person"], "PERSON"), (".", None)],

        # --- Bank: sentence-initial / copula ---
        [("Account ", None), seg_bare(bank), (" is on file.", None)],
        [("Account: ", None), seg_bare(bank), (".", None)],
        [("account number is ", None), seg_bare(bank), (".", None)],
        [("Bank account: ", None), seg_bare(bank), (".", None)],
        [("The account on record is ", None), seg_bare(bank), (".", None)],
        [("Direct deposit account: ", None), seg_bare(bank), (".", None)],
        # --- Bank: post-positional ---
        [("The number ", None), seg_bare(bank), (" is the bank account.", None)],
        # --- Bank: with context ---
        [("Payroll routed to account ", None), seg_bare(bank), (" for ", None), (v["org"], "ORG"), (".", None)],
        [("Refund was sent to account ", None), seg_bare(bank), (" on ", None), (v["date"], "DATE"), (".", None)],

        # --- Provider: copula / separator ---
        [("provider ID is ", None), seg_bare(prov), (".", None)],
        [("provider: ", None), seg_bare(prov), (".", None)],
        [("The provider on file is ", None), seg_bare(prov), (".", None)],
        [("CPSA number: ", None), seg_bare(prov), (".", None)],
        [("prescriber ID reads ", None), seg_bare(prov), (".", None)],

        # --- Generic entity type ---
        [("The identifier on file is ", None), seg_bare(e), (".", None)],
        [("Value on record: ", None), seg_bare(e), (".", None)],
        [("The system shows ", None), seg_bare(e), (" as the current value.", None)],
    ]
def trailing_date_templates(rng, v):
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [seg(e, rng), (" was updated ", None), (v["date"], "DATE"), (".", None)],
        [seg(e, rng), (" was verified ", None), (v["date"], "DATE"), (".", None)],
        [seg(e, rng), (" expires ", None), (v["date"], "DATE"), (".", None)],
        [seg(e, rng), (" was renewed ", None), (v["date"], "DATE"), (".", None)],
        [seg(e, rng), (" needs review ", None), (v["date"], "DATE"), (".", None)],
        [("We processed ", None), seg(e, rng), (" ", None), (v["date"], "DATE"), (".", None)],
        [("They updated ", None), seg(e, rng), (" ", None), (v["date"], "DATE"), (".", None)],
        [("The record for ", None), seg(e, rng), (" was reviewed ", None), (v["date"], "DATE"), (".", None)],
    ]

def gov_org_templates(rng, v):
    gov = choice(rng, PUBLIC_ORGS)
    e = choice(rng, NEW_ENTITY_TYPES)
    return [
        [("The ", None), (gov, "ORG"), (" confirmed ", None), seg(e, rng), (".", None)],
        [(gov, "ORG"), (" processed ", None), seg(e, rng), (" ", None), (v["date"], "DATE"), (".", None)],
        [(gov, "ORG"), (" verified ", None), seg(e, rng), (" for the applicant.", None)],
        [("Contact ", None), (gov, "ORG"), (" about ", None), seg(e, rng), (".", None)],
        [("According to ", None), (gov, "ORG"), (", ", None), seg(e, rng), (" is on file.", None)],
        [(gov, "ORG"), (" flagged ", None), seg(e, rng), (" for review ", None), (v["date"], "DATE"), (".", None)],
    ]

TEMPLATE_BUILDERS: Tuple[Callable[[random.Random, Dict[str, str]], List[List[Segment]]], ...] = (
    banking_templates,
    payroll_tax_templates,
    health_pharmacy_templates,
    provider_billing_templates,
    passport_travel_templates,
    driver_registry_templates,
    benefits_credit_templates,
    employment_university_templates,
    government_service_templates,
    support_operations_templates,
    privacy_audit_compliance_templates,
    document_records_templates,
    migration_qa_templates,
    multi_identifier_templates,
    false_positive_templates,
    casual_email_templates,
    fragment_label_templates,
    detached_keyword_templates,
    trailing_date_templates,
    gov_org_templates,
)


def all_templates(rng: random.Random, v: Dict[str, str]) -> List[List[Segment]]:
    templates: List[List[Segment]] = []
    for builder in TEMPLATE_BUILDERS:
        templates.extend(builder(rng, v))
    return templates


def validate_template_inventory() -> None:
    """
    Validate the reusable template inventory:
      - at least MIN_TEMPLATE_PATTERNS patterns
      - every template contains at least one explicit new identifier segment
    """
    probe_rng = random.Random(0)
    templates = all_templates(probe_rng, values_for_templates(probe_rng))

    if len(templates) < MIN_TEMPLATE_PATTERNS:
        raise ValueError(
            f"Template inventory has {len(templates)} patterns; "
            f"expected at least {MIN_TEMPLATE_PATTERNS}."
        )

    for idx, template in enumerate(templates):
        if not any(entity_type in NEW_ENTITY_TYPES for _, entity_type in template):
            raise ValueError(
                f"Template {idx} is missing an explicit new Canadian identifier segment."
            )


def generate_segments_for_entity(entity_type: str, rng: random.Random) -> List[Segment]:
    """
    Generate one sentence as explicit tagged segments.

    The requested entity type is guaranteed to appear at least once, satisfying
    per-split coverage requirements deterministically.
    """
    v = values_for_templates(rng)
    forced_template: List[Segment] = [
        ("Please verify ", None),
        seg(entity_type, rng),
        (" for ", None),
        (v["department"], "ORG"),
        (" before ", None),
        (v["date"], "DATE"),
        (".", None),
    ]

    if rng.random() < 0.15:
        segments = forced_template
    else:
        templates = all_templates(rng, v)
        segments = list(choice(rng, templates))
        if not any(ent == entity_type for _, ent in segments) and rng.random() < 0.35:
            if segments and isinstance(segments[-1][0], str) and segments[-1][0].endswith("."):
                last_text, last_entity = segments[-1]
                segments[-1] = (last_text[:-1], last_entity)
            segments.extend([(" and reference ", None), seg(entity_type, rng), (".", None)])

    if rng.random() < 0.30:
        if segments and isinstance(segments[-1][0], str) and segments[-1][0].endswith("."):
            last_text, last_entity = segments[-1]
            segments[-1] = (last_text[:-1], last_entity)
        segments.extend(optional_clause(rng))
        segments.append((".", None))

    return segments

# ---------------------------------------------------------------------------
# Negative example generation
# ---------------------------------------------------------------------------

# Cache negative template pools at module load (templates are static text)
_NEG_PROBE_RNG = random.Random(0)
_PLAIN_NEGATIVES: Optional[List[List[Segment]]] = None
_HARD_NEGATIVES: Optional[List[List[Segment]]] = None


def _init_negative_pools() -> None:
    """Lazy-init negative template pools."""
    global _PLAIN_NEGATIVES, _HARD_NEGATIVES
    if _PLAIN_NEGATIVES is None:
        v = values_for_templates(_NEG_PROBE_RNG)
        _PLAIN_NEGATIVES = plain_negative_templates(_NEG_PROBE_RNG, v)
        _HARD_NEGATIVES = hard_negative_templates(_NEG_PROBE_RNG, v)


def generate_negative_example(rng: random.Random) -> Example:
    """Generate one negative example with no entities (all tags = O)."""
    _init_negative_pools()
    # 50% plain negatives, 50% hard negatives (decoy digit strings)
    if rng.random() < 0.5:
        segments = list(choice(rng, _PLAIN_NEGATIVES))
    else:
        segments = list(choice(rng, _HARD_NEGATIVES))

    # Build directly without calling validate_example
    # (which would require a Canadian B-tag)
    tokens: List[str] = []
    tags: List[int] = []
    for text, _ in segments:
        segment_tokens = tokenize(text)
        if not segment_tokens:
            continue
        tokens.extend(segment_tokens)
        tags.extend([LABEL2ID["O"]] * len(segment_tokens))

    return {"tokens": tokens, "tags": tags}
# ---------------------------------------------------------------------------
# OntoNotes rehearsal loader
# ---------------------------------------------------------------------------

def load_rehearsal_examples(rehearsal_dir: Path) -> List[Example]:
    """
    Load OntoNotes base-type rehearsal examples from all *.jsonl files in
    rehearsal_dir.  Examples must match the token/tags format but are NOT
    required to contain any Canadian identifier tags — they reinforce the 18
    base OntoNotes types to prevent catastrophic forgetting.
    """
    examples: List[Example] = []
    jsonl_files = sorted(rehearsal_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(
            f"No *.jsonl files found in rehearsal dir: {rehearsal_dir}"
        )
    for jsonl_path in jsonl_files:
        with jsonl_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    validate_example(example)
                    examples.append(example)
                except Exception as exc:
                    raise ValueError(
                        f"Bad rehearsal example in {jsonl_path}:{line_no}: {exc}"
                    ) from exc
    return examples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_required_label_map() -> None:
    expected = {
        "O": 0,
        "B-CARDINAL": 1,
        "B-DATE": 2,
        "I-DATE": 3,
        "B-PERSON": 4,
        "I-PERSON": 5,
        "B-NORP": 6,
        "B-GPE": 7,
        "I-GPE": 8,
        "B-LAW": 9,
        "I-LAW": 10,
        "B-ORG": 11,
        "I-ORG": 12,
        "B-PERCENT": 13,
        "I-PERCENT": 14,
        "B-ORDINAL": 15,
        "B-MONEY": 16,
        "I-MONEY": 17,
        "B-WORK_OF_ART": 18,
        "I-WORK_OF_ART": 19,
        "B-FAC": 20,
        "B-TIME": 21,
        "I-CARDINAL": 22,
        "B-LOC": 23,
        "B-QUANTITY": 24,
        "I-QUANTITY": 25,
        "I-NORP": 26,
        "I-LOC": 27,
        "B-PRODUCT": 28,
        "I-TIME": 29,
        "B-EVENT": 30,
        "I-EVENT": 31,
        "I-FAC": 32,
        "B-LANGUAGE": 33,
        "I-PRODUCT": 34,
        "I-ORDINAL": 35,
        "I-LANGUAGE": 36,
        "B-CANADIAN_BANK_ACCOUNT_NUMBER": 37,
        "I-CANADIAN_BANK_ACCOUNT_NUMBER": 38,
        "B-ALBERTA_DRIVERS_LICENCE_NUMBER": 39,
        "I-ALBERTA_DRIVERS_LICENCE_NUMBER": 40,
        "B-CANADIAN_INDIVIDUAL_TAX_NUMBER": 41,
        "I-CANADIAN_INDIVIDUAL_TAX_NUMBER": 42,
        "B-ALBERTA_PERSONAL_HEALTH_NUMBER": 43,
        "I-ALBERTA_PERSONAL_HEALTH_NUMBER": 44,
        "B-CANADIAN_PROVIDER_IDENTIFIER": 45,
        "I-CANADIAN_PROVIDER_IDENTIFIER": 46,
        "B-CANADIAN_PASSPORT_NUMBER": 47,
        "I-CANADIAN_PASSPORT_NUMBER": 48,
        "B-CANADIAN_SOCIAL_INSURANCE_NUMBER": 49,
        "I-CANADIAN_SOCIAL_INSURANCE_NUMBER": 50,
    }
    if LABEL2ID != expected:
        raise ValueError("LABEL2ID does not match the required mapping exactly.")

    required_ontonotes_entities = {
        "CARDINAL", "DATE", "PERSON", "NORP", "GPE", "LAW", "PERCENT",
        "ORDINAL", "MONEY", "WORK_OF_ART", "FAC", "TIME", "QUANTITY",
        "PRODUCT", "LANGUAGE", "ORG", "LOC", "EVENT",
    }
    for entity_type in required_ontonotes_entities:
        if f"B-{entity_type}" not in LABEL2ID:
            raise ValueError(f"Missing B label for OntoNotes entity type: {entity_type}")

    for entity_type in NEW_ENTITY_TYPES:
        if f"B-{entity_type}" not in LABEL2ID:
            raise ValueError(f"Missing B label for new entity type: {entity_type}")
        if f"I-{entity_type}" not in LABEL2ID:
            raise ValueError(f"Missing I label for new entity type: {entity_type}")

    if len(ID2LABEL) != len(LABEL2ID):
        raise ValueError("Duplicate label IDs detected in LABEL2ID.")


def validate_bio_sequence(tags: Sequence[int]) -> None:
    previous_entity: Optional[str] = None

    for idx, tag in enumerate(tags):
        label = ID2LABEL.get(tag)
        if label is None:
            raise ValueError(f"Unknown tag ID: {tag}")

        if label == "O":
            previous_entity = None
            continue

        if label.startswith("B-"):
            previous_entity = label[2:]
            continue

        if label.startswith("I-"):
            entity_type = label[2:]
            if previous_entity != entity_type:
                raise ValueError(
                    f"Invalid BIO sequence: {label} at position {idx} is not "
                    f"preceded by B-{entity_type} or I-{entity_type}."
                )
            previous_entity = entity_type
            continue

        raise ValueError(f"Invalid label format: {label}")


def validate_example(example: Example) -> None:
    """Validate one example structurally (Canadian entities NOT required per example)."""
    if "tokens" not in example:
        raise ValueError("Example is missing 'tokens'.")
    if "tags" not in example:
        raise ValueError("Example is missing 'tags'.")

    tokens = example["tokens"]
    tags = example["tags"]

    if not isinstance(tokens, list) or not tokens:
        raise ValueError("'tokens' must be a non-empty list.")
    if not all(isinstance(token, str) and token for token in tokens):
        raise ValueError("Every token must be a non-empty string.")

    if not isinstance(tags, list):
        raise ValueError("'tags' must be a list.")
    if not all(isinstance(tag, int) for tag in tags):
        raise ValueError("Every tag must be an integer.")

    if len(tokens) != len(tags):
        raise ValueError(
            f"Token/tag length mismatch: len(tokens)={len(tokens)}, len(tags)={len(tags)}"
        )

    valid_tag_ids = set(ID2LABEL.keys())
    unknown_tag_ids = [tag for tag in tags if tag not in valid_tag_ids]
    if unknown_tag_ids:
        raise ValueError(f"Unknown tag IDs found: {unknown_tag_ids}")

    # ✂ Removed: "must contain Canadian B label" — now enforced at split level
    validate_bio_sequence(tags)


def entity_types_present(example: Example) -> set[str]:
    tags = example["tags"]
    assert isinstance(tags, list)
    present: set[str] = set()
    for tag in tags:
        if isinstance(tag, int) and tag in NEW_B_LABEL_IDS:
            present.add(ID2LABEL[tag][2:])
    return present


def validate_split(
    name: str,
    examples: Sequence[Example],
    expected_size: int,
) -> None:
    if len(examples) != expected_size:
        raise ValueError(
            f"Split '{name}' has {len(examples)} examples; expected {expected_size}."
        )

    split_entity_coverage: set[str] = set()
    for idx, example in enumerate(examples):
        try:
            validate_example(example)
            split_entity_coverage.update(entity_types_present(example))
        except Exception as exc:
            raise ValueError(
                f"Validation failed for split '{name}' example {idx}: {exc}"
            ) from exc

    missing = set(NEW_ENTITY_TYPES) - split_entity_coverage
    if missing:
        raise ValueError(
            f"Split '{name}' is missing required new entity types: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Weighted entity sampling
# ---------------------------------------------------------------------------

# Keep 9-digit types (SIN/ITN/PHN) near-equal to avoid over-predicting them
# on bare digit strings. Passport/DL/BANK/PROVIDER at 1.0 base.
ENTITY_SAMPLING_WEIGHTS: Dict[str, float] = {
    "CANADIAN_SOCIAL_INSURANCE_NUMBER": 1.0,   # flatten — over-predicted before
    "CANADIAN_BANK_ACCOUNT_NUMBER": 1.0,
    "ALBERTA_PERSONAL_HEALTH_NUMBER": 1.0,     # flatten — over-predicted before
    "CANADIAN_PASSPORT_NUMBER": 1.0,
    "CANADIAN_INDIVIDUAL_TAX_NUMBER": 1.5,     # boost — weakest keyword set
    "ALBERTA_DRIVERS_LICENCE_NUMBER": 1.3,     # boost — recall issues in eval
    "CANADIAN_PROVIDER_IDENTIFIER": 1.2,
}

_ENTITY_TYPES_LIST: List[str] = list(ENTITY_SAMPLING_WEIGHTS.keys())
_ENTITY_WEIGHTS_LIST: List[float] = list(ENTITY_SAMPLING_WEIGHTS.values())


def weighted_entity(rng: random.Random) -> str:
    return rng.choices(_ENTITY_TYPES_LIST, weights=_ENTITY_WEIGHTS_LIST, k=1)[0]


# ---------------------------------------------------------------------------
# Generation and writing
# ---------------------------------------------------------------------------

def generate_split(
    size: int,
    rng: random.Random,
    negative_ratio: float = 0.0,
    rehearsal_pool: Optional[List[Example]] = None,
    rehearsal_count: int = 0,
) -> List[Example]:
    """Generate a split of Canadian synthetic examples, negatives, and optional
    OntoNotes rehearsal examples sampled from rehearsal_pool."""

    if not 0.0 <= negative_ratio < 1.0:
        raise ValueError(f"negative_ratio must be in [0.0, 1.0): got {negative_ratio}")

    n_negative = int(round(size * negative_ratio))
    n_rehearsal = min(rehearsal_count, max(0, size - n_negative - len(NEW_ENTITY_TYPES)))
    n_canadian = size - n_negative - n_rehearsal

    if n_canadian < len(NEW_ENTITY_TYPES):
        raise ValueError(
            f"Canadian bucket ({n_canadian}) too small to include all "
            f"{len(NEW_ENTITY_TYPES)} new entity types. "
            "Increase split size or reduce negative_ratio / rehearsal_ratio."
        )

    # --- Bucket 1: Canadian synthetic (guarantees every label appears) ---
    planned_entity_types: List[str] = list(NEW_ENTITY_TYPES)
    while len(planned_entity_types) < n_canadian:
        planned_entity_types.append(weighted_entity(rng))
    rng.shuffle(planned_entity_types)

    examples: List[Example] = []
    for entity_type in planned_entity_types:
        segments = generate_segments_for_entity(entity_type, rng)
        example = make_example(segments, rng)
        examples.append(example)

    # --- Bucket 2: Negatives ---
    for _ in range(n_negative):
        examples.append(generate_negative_example(rng))

    # --- Bucket 3: OntoNotes rehearsal (train split only, optional) ---
    if n_rehearsal > 0 and rehearsal_pool:
        sampled = rng.choices(rehearsal_pool, k=n_rehearsal)
        examples.extend(sampled)

    rng.shuffle(examples)

    print(
        f"  Buckets — Canadian: {n_canadian:,}  "
        f"Negative: {n_negative:,}  "
        f"Rehearsal: {n_rehearsal:,}"
    )

    return examples


def write_jsonl(path: Path, examples: Iterable[Example]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            validate_example(example)
            f.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_label_mapping(path: Path) -> None:
    validate_required_label_map()
    with path.open("w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_keyword_pools() -> None:
    pools = {
        "SIN_KEYWORDS": SIN_KEYWORDS,
        "BANK_LEADS": BANK_LEADS,
        "ITN_KEYWORDS": ITN_KEYWORDS,
        "PHN_KEYWORDS": PHN_KEYWORDS,
        "PROVIDER_KEYWORDS": PROVIDER_KEYWORDS,
        "PASSPORT_KEYWORDS": PASSPORT_KEYWORDS,
        "DL_KEYWORDS": DL_KEYWORDS,
    }
    for name, pool in pools.items():
        if not pool:
            raise ValueError(f"Keyword pool '{name}' is empty.")
        if not all(isinstance(p, str) for p in pool):
            raise ValueError(f"Keyword pool '{name}' contains non-string entries.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic OntoNotes5-style token-classification JSONL data "
            "with seven Canadian sensitive-identifier entity types."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="synthetic_ontonotes5_canadian_sensitive_identifiers",
        help=(
            "Directory where train.jsonl, validation.jsonl, test.jsonl, "
            "and label.json will be written."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help=f"Number of training examples to generate. Default: {DEFAULT_TRAIN_SIZE}.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=DEFAULT_VALIDATION_SIZE,
        help=(
            f"Number of validation examples to generate. "
            f"Default: {DEFAULT_VALIDATION_SIZE}."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=f"Number of test examples to generate. Default: {DEFAULT_TEST_SIZE}.",
    )
    
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.30,
        help=(
            "Fraction of negative examples (no entities) per split. "
            "Default: 0.30 (30%%)."
        ),
    )
    parser.add_argument(
        "--rehearsal-dir",
        type=str,
        default=None,
        help=(
            "Path to a directory containing *.jsonl files with OntoNotes base-type "
            "examples (no Canadian identifier tags). Mixed into the TRAIN split only "
            "to prevent catastrophic forgetting of the 18 base entity types."
        ),
    )
    parser.add_argument(
        "--rehearsal-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of training examples drawn from the rehearsal pool. "
            "Requires --rehearsal-dir. Default: 0.0 (disabled)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validate_required_label_map()
    validate_template_inventory()
    validate_keyword_pools()

    for split_name, split_size in (
        ("train", args.train_size),
        ("validation", args.validation_size),
        ("test", args.test_size),
    ):
        if split_size < len(NEW_ENTITY_TYPES):
            raise ValueError(
                f"--{split_name}-size must be at least {len(NEW_ENTITY_TYPES)} "
                "so each new entity type can appear at least once."
            )

    if args.rehearsal_ratio < 0.0 or args.rehearsal_ratio >= 1.0:
        raise ValueError(
            f"--rehearsal-ratio must be in [0.0, 1.0): got {args.rehearsal_ratio}"
        )
    if args.rehearsal_ratio > 0.0 and not args.rehearsal_dir:
        raise ValueError("--rehearsal-ratio requires --rehearsal-dir to be set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OntoNotes rehearsal pool (train split only)
    rehearsal_pool: Optional[List[Example]] = None
    rehearsal_count = 0
    if args.rehearsal_dir:
        rehearsal_dir = Path(args.rehearsal_dir)
        print(f"Loading rehearsal examples from {rehearsal_dir} ...")
        rehearsal_pool = load_rehearsal_examples(rehearsal_dir)
        rehearsal_count = int(round(args.train_size * args.rehearsal_ratio))
        print(
            f"  Loaded {len(rehearsal_pool):,} rehearsal examples; "
            f"will sample {rehearsal_count:,} into train split "
            f"({args.rehearsal_ratio:.0%})"
        )
        print()

    rng = random.Random(args.seed)
    total = args.train_size + args.validation_size + args.test_size

    print(f"Generating {total:,} synthetic NER examples (seed={args.seed})...")
    print(
        f"  Train: {args.train_size:,}  |  "
        f"Validation: {args.validation_size:,}  |  "
        f"Test: {args.test_size:,}"
    )
    print(f"  Negative ratio: {args.negative_ratio:.0%}")
    if rehearsal_count:
        print(f"  Rehearsal (train only): {rehearsal_count:,} ({args.rehearsal_ratio:.0%})")
    print()

    t0 = time.perf_counter()

    print("Generating training split...")
    train_examples = generate_split(
        args.train_size, rng, args.negative_ratio,
        rehearsal_pool=rehearsal_pool, rehearsal_count=rehearsal_count,
    )

    print("Generating validation split...")
    validation_examples = generate_split(args.validation_size, rng, args.negative_ratio)

    print("Generating test split...")
    test_examples = generate_split(args.test_size, rng, args.negative_ratio)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    rate = total / elapsed if elapsed > 0 else float("inf")

    print()
    print(f"Generation complete in {elapsed:.2f}s ({rate:,.1f} examples/sec)")

    if rate < 1.0:
        raise RuntimeError(
            f"Generation too slow: {rate:.2f} examples/sec (minimum: 1.0)"
        )

    print()
    print("Validating splits...")
    validate_split("train", train_examples, args.train_size)
    validate_split("validation", validation_examples, args.validation_size)
    validate_split("test", test_examples, args.test_size)

    print("Writing output files...")
    write_jsonl(output_dir / "train.jsonl", train_examples)
    write_jsonl(output_dir / "validation.jsonl", validation_examples)
    write_jsonl(output_dir / "test.jsonl", test_examples)
    write_label_mapping(output_dir / "label.json")

    print()
    print(f"Wrote {total:,} examples to {output_dir}/")
    print("All validations passed. Done.")


if __name__ == "__main__":
    main()