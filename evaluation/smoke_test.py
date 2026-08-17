#!/usr/bin/env python3
"""
Primary smoke test — 80 hand-written out-of-distribution cases, scored on the
7 Canadian PII labels. Value-only labelling: the trigger word is context (O);
only the identifier value should be tagged. `expect=None` = informational only.
"""

import spacy

MODEL_PATH = "./output_stage2/model-best"
nlp = spacy.load(MODEL_PATH)

SIN  = "CANADIAN_SOCIAL_INSURANCE_NUMBER"
ITN  = "CANADIAN_INDIVIDUAL_TAX_NUMBER"
PHN  = "ALBERTA_PERSONAL_HEALTH_NUMBER"
BANK = "CANADIAN_BANK_ACCOUNT_NUMBER"
DL   = "ALBERTA_DRIVERS_LICENCE_NUMBER"
PASS = "CANADIAN_PASSPORT_NUMBER"
PROV = "CANADIAN_PROVIDER_IDENTIFIER"
CANADIAN_LABELS = {SIN, ITN, PHN, BANK, DL, PASS, PROV}

SIN_H, SIN_S, SIN_P = "130-692-544", "130 692 544", "130692544"
PHN_H, PHN_S, PHN_P = "257-574-243", "257 574 243", "257574243"
ITN_H, ITN_S, ITN_P = "459-311-601", "459 311 601", "459311601"
PROV_A, PROV_B      = "748159263", "390215476"
BANK_A = "18016-003-86796296"
BANK_B = "12345 006 7891234"
BANK_C = "23456-001-98765432"
DL_AB  = "428913576"
DL_BC  = "1234567"
DL_ON  = "A12345678901234"
DL_SK  = "12345678"
PASS_A = "AB123456"
PASS_B = "CD234567"
PASS_C = "A123456BC"
PASS_D = "X987654YZ"

smoke_tests = [
    # SIN
    (f"Please update SIN {SIN_S} in the payroll system.",            {SIN}),
    (f"Employee SIN: {SIN_H} has been verified by HR.",              {SIN}),
    (f"Her social insurance number is {SIN_P}.",                     {SIN}),
    (f"Add {SIN_S} to the T4 batch for next week.",                  {SIN}),
    (f"Service Canada confirmed SIN {SIN_H} belongs to the applicant.", {SIN}),
    # PHN
    (f"Please update PHN {PHN_H} in the clinic system.",             {PHN}),
    (f"Patient AHCIP number {PHN_S} is on file.",                    {PHN}),
    (f"Verify Alberta health number {PHN_P} for the new registration.", {PHN}),
    (f"PHN: {PHN_H} \u2014 billing approved.",                       {PHN}),
    (f"The personal health number on the form reads {PHN_S}.",       {PHN}),
    # ITN
    (f"The CRA ITN is {ITN_H} for the non-resident filer.",          {ITN}),
    (f"Individual tax number {ITN_S} was registered last quarter.",  {ITN}),
    (f"Please enter ITN: {ITN_H} in the tax software.",              {ITN}),
    (f"Non-resident ITN {ITN_P} needs annual review.",               {ITN}),
    (f"Tax number {ITN_H} is linked to this account.",               {ITN}),
    # Bank
    (f"Wire transfer to {BANK_A}.",                                  {BANK}),
    (f"Bank details: {BANK_B}.",                                     {BANK}),
    (f"Direct deposit to {BANK_A} has been set up.",                 {BANK}),
    (f"Account {BANK_C} is the new payroll destination.",            {BANK}),
    (f"Please verify {BANK_B} for the refund.",                      {BANK}),
    # DL
    (f"Alberta DL {DL_AB} is on file for the new employee.",         {DL}),
    (f"Driver's licence number {DL_BC} has expired.",                {DL}),
    (f"Please update DL# {DL_SK} in the registry.",                  {DL}),
    (f"Operator licence {DL_ON} was renewed last week.",             {DL}),
    (f"AB licence {DL_AB} verified at the dealership.",              {DL}),
    # Passport
    (f"Canadian passport {PASS_A} expires next year.",               {PASS}),
    (f"Passport number {PASS_C} was confirmed by CBSA.",             {PASS}),
    (f"Travel document {PASS_D} needs verification before boarding.", {PASS}),
    (f"Please scan passport {PASS_B} at check-in.",                  {PASS}),
    (f"The passport on file is {PASS_A}.",                           {PASS}),
    # Provider
    (f"Billing provider {PROV_A} submitted the claim.",              {PROV}),
    (f"Practitioner ID {PROV_B} needs renewal.",                     {PROV}),
    (f"AHS provider {PROV_A} was added to the roster.",              {PROV}),
    (f"Please verify provider no. {PROV_B} for this referral.",      {PROV}),
    (f"CPSA {PROV_A} is registered for the new clinic.",             {PROV}),
    # Multiple
    (f"Patient SIN {SIN_H} and PHN {PHN_H} both updated.",           {SIN, PHN}),
    (f"Process refund to {BANK_C} for SIN {SIN_S}.",                 {BANK, SIN}),
    (f"ITN {ITN_H} and passport {PASS_A} are on file.",              {ITN, PASS}),
    (f"Provider {PROV_A} billed PHN {PHN_H} last Tuesday.",          {PROV, PHN}),
    (f"DL {DL_AB} and passport {PASS_C} verified at the border.",    {DL, PASS}),
    # OntoNotes only (no Canadian PII expected)
    ("Sarah Patel from CRA called on March 15, 2026.",               set()),
    ("Dr. Jonathan Smith updated the records yesterday.",            set()),
    ("The meeting at Foothills Medical Centre starts at 9:00 a.m.",  set()),
    ("Alberta Health Services processed 2,500 applications in Q1 2026.", set()),
    ("Please contact the Canada Revenue Agency in Edmonton by Friday.", set()),
    # Negatives
    ("Please update the system and restart the application.",        set()),
    ("The meeting has been postponed to next week.",                 set()),
    ("Submit your feedback through the online form.",                set()),
    ("All systems are operating normally today.",                    set()),
    ("Thank you for reaching out to our team.",                      set()),
    # Hard negatives
    (f"Reference {ITN_H} appears in the meeting notes.",             set()),
    (f"Invoice {PHN_H} was paid on Tuesday.",                        set()),
    ("Page 257 of 574 contains the executive summary.",              set()),
    ("Phone extension 4591 reached voicemail.",                      set()),
    (f"Order number {SIN_H} has shipped from the warehouse.",        set()),
    (f"Lot {BANK_A} shipped from the warehouse.",                    set()),
    (f"Asset tag {PROV_A} was scanned during inventory.",            set()),
    (f"Conference room {PASS_C} is booked for the morning.",         set()),
    # Disambiguation
    (f"PHN {PHN_H} is different from ITN {ITN_H} on the form.",      {PHN, ITN}),
    (f"Update SIN {SIN_S} but do not confuse it with PHN {PHN_S}.",  {SIN, PHN}),
    (f"The CRA ITN {ITN_H} is for tax purposes only, not health billing.", {ITN}),
    (f"Note: {SIN_H} is the SIN; {PHN_H} is the PHN.",               {SIN, PHN}),
    # Informal
    (f"Hey, can you double-check the SIN {SIN_S} for me?",           {SIN}),
    (f"FYI the PHN on the patient's file is {PHN_H}.",               {PHN}),
    (f"Quick note \u2014 Sarah's ITN is {ITN_H}, please update.",    {ITN}),
    (f"Heads up: the bank account {BANK_A} was flagged for review.", {BANK}),
    (f"Need to verify Alberta DL {DL_AB} before the test drive.",    {DL}),
    # Edge cases
    (SIN_H,                                                          set()),
    (f"{PHN_S}.",                                                    set()),
    (f"{ITN_H} \u2014 ITN on file.",                                 {ITN}),
    (f"Passport: {PASS_A}",                                          {PASS}),
    (f"SIN {SIN_H}",                                                 {SIN}),
    # Mixed
    (f"Maya Singh's SIN {SIN_S} was verified by HR on March 12, 2026.", {SIN}),
    (f"Send the refund of $1,500.00 to {BANK_A} by Friday.",         {BANK}),
    (f"Dr. Andr\u00e9 L\u00e9vesque updated PHN {PHN_H} at Misericordia Community Hospital.", {PHN}),
    (f"The Canada Revenue Agency processed ITN {ITN_H} last quarter.", {ITN}),
    (f"Officer Yang at Calgary International Airport verified passport {PASS_A}.", {PASS}),
    # Informational only (not scored)
    ("Wire transfer to transit 18016 account 86796296.",            None),
    ("Direct deposit to acct 86796296 has been set up.",            None),
    ("Account number 4567891234 is the new payroll destination.",   None),
]

print(f"\n{'='*80}")
print(f"Running {len(smoke_tests)} smoke tests   (model: {MODEL_PATH})")
print('='*80)

scored = passed = 0
failures = []
for i, (text, expect) in enumerate(smoke_tests, 1):
    doc = nlp(text)
    found_can = {e.label_ for e in doc.ents if e.label_ in CANADIAN_LABELS}
    if expect is None:
        status = "info"
    else:
        scored += 1
        ok = (found_can == expect)
        passed += ok
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append((i, text, expect, found_can))
    tag = "      " if status == "info" else f"[{status}]"
    print(f"\n{tag} [{i}] '{text}'")
    if doc.ents:
        for ent in doc.ents:
            mark = "*" if ent.label_ in CANADIAN_LABELS else " "
            print(f"      {mark} {ent.text:38s} \u2192 {ent.label_}")
    else:
        print("        (no entities)")
    if status == "FAIL":
        print(f"        expected Canadian: {sorted(expect) or '\u2205'}")
        print(f"        got      Canadian: {sorted(found_can) or '\u2205'}")

print(f"\n{'='*80}")
print((f"SCORED: {passed}/{scored} passed   ({(100*passed/scored):.1f}%)") if scored else "no scored tests")
print('='*80)
if failures:
    print("Failures:")
    for i, text, exp, got in failures:
        print(f"  [{i}] expected {sorted(exp) or '\u2205'} | got {sorted(got) or '\u2205'}  ::  {text[:60]}")
