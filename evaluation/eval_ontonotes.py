#!/usr/bin/env python3
"""
OntoNotes base-type evaluation over the ORIGINAL smoke-test sentences.

The standard smoke test only scores the 7 Canadian labels, so OntoNotes errors
(e.g. "Android 14" -> MONEY, "CRA" -> PRODUCT, boundary slips) pass silently.
This harness adds gold annotations for the 18 inherited OntoNotes types and
reports, per type, what the model gets right/wrong.

Scoring
-------
* OntoNotes: span-based, matched on (entity_text, label). TP / FP / FN per type.
* Canadian: sentence-level set match (as in the main smoke test), for context.

Gold conventions (so debatable calls are explicit)
-------------------------------------------------
* Only clearly-named / explicit entities are gold: proper-noun PERSON, named ORG,
  GPE, named FAC, and explicit DATE/TIME/MONEY/CARDINAL expressions.
* Adjectival NORP ("Canadian", "Alberta ...") and bare unit-less numbers used as
  IDs are NOT gold -> if the model emits them, they count as false positives.
* 10 of the 18 types have no gold positives in these sentences; for them only
  precision (false positives) is meaningful. Recall reads "-".

This is a small, partly-subjective diagnostic. For real per-type numbers on the
18 base types, run:  python -m spacy evaluate <model> <ontonotes_or_silver.spacy>
"""

import spacy

MODEL_PATH = "./output_stage2/model-best"

ONTO_TYPES = ["CARDINAL","DATE","EVENT","FAC","GPE","LANGUAGE","LAW","LOC","MONEY",
              "NORP","ORDINAL","ORG","PERCENT","PERSON","PRODUCT","QUANTITY","TIME","WORK_OF_ART"]
CAN_TYPES = ["CANADIAN_SOCIAL_INSURANCE_NUMBER","CANADIAN_INDIVIDUAL_TAX_NUMBER",
             "ALBERTA_PERSONAL_HEALTH_NUMBER","CANADIAN_BANK_ACCOUNT_NUMBER",
             "ALBERTA_DRIVERS_LICENCE_NUMBER","CANADIAN_PASSPORT_NUMBER",
             "CANADIAN_PROVIDER_IDENTIFIER"]
CAN_SET = set(CAN_TYPES)

SIN_H, SIN_S, SIN_P = "130-692-544", "130 692 544", "130692544"
PHN_H, PHN_S, PHN_P = "257-574-243", "257 574 243", "257574243"
ITN_H, ITN_S, ITN_P = "459-311-601", "459 311 601", "459311601"
PROV_A, PROV_B = "748159263", "390215476"
BANK_A, BANK_B, BANK_C = "18016-003-86796296", "12345 006 7891234", "23456-001-98765432"
DL_AB, DL_BC, DL_ON, DL_SK = "428913576", "1234567", "A12345678901234", "12345678"
PASS_A, PASS_B, PASS_C, PASS_D = "AB123456", "CD234567", "A123456BC", "X987654YZ"

SIN="CANADIAN_SOCIAL_INSURANCE_NUMBER"; ITN="CANADIAN_INDIVIDUAL_TAX_NUMBER"
PHN="ALBERTA_PERSONAL_HEALTH_NUMBER";  BANK="CANADIAN_BANK_ACCOUNT_NUMBER"
DL="ALBERTA_DRIVERS_LICENCE_NUMBER";   PASS="CANADIAN_PASSPORT_NUMBER"
PROV="CANADIAN_PROVIDER_IDENTIFIER"

# (text, canadian_expected_set_or_None, onto_gold_list[(text,label)])
CASES = [
    (f"Please update SIN {SIN_S} in the payroll system.", {SIN}, []),
    (f"Employee SIN: {SIN_H} has been verified by HR.", {SIN}, []),
    (f"Her social insurance number is {SIN_P}.", {SIN}, []),
    (f"Add {SIN_S} to the T4 batch for next week.", {SIN}, [("next week","DATE")]),
    (f"Service Canada confirmed SIN {SIN_H} belongs to the applicant.", {SIN}, [("Service Canada","ORG")]),
    (f"Please update PHN {PHN_H} in the clinic system.", {PHN}, []),
    (f"Patient AHCIP number {PHN_S} is on file.", {PHN}, []),
    (f"Verify Alberta health number {PHN_P} for the new registration.", {PHN}, []),
    (f"PHN: {PHN_H} billing approved.", {PHN}, []),
    (f"The personal health number on the form reads {PHN_S}.", {PHN}, []),
    (f"The CRA ITN is {ITN_H} for the non-resident filer.", {ITN}, [("CRA","ORG")]),
    (f"Individual tax number {ITN_S} was registered last quarter.", {ITN}, [("last quarter","DATE")]),
    (f"Please enter ITN: {ITN_H} in the tax software.", {ITN}, []),
    (f"Non-resident ITN {ITN_P} needs annual review.", {ITN}, []),
    (f"Tax number {ITN_H} is linked to this account.", {ITN}, []),
    (f"Wire transfer to {BANK_A}.", {BANK}, []),
    (f"Bank details: {BANK_B}.", {BANK}, []),
    (f"Direct deposit to {BANK_A} has been set up.", {BANK}, []),
    (f"Account {BANK_C} is the new payroll destination.", {BANK}, []),
    (f"Please verify {BANK_B} for the refund.", {BANK}, []),
    (f"Alberta DL {DL_AB} is on file for the new employee.", {DL}, []),
    (f"Driver's licence number {DL_BC} has expired.", {DL}, []),
    (f"Please update DL# {DL_SK} in the registry.", {DL}, []),
    (f"Operator licence {DL_ON} was renewed last week.", {DL}, [("last week","DATE")]),
    (f"AB licence {DL_AB} verified at the dealership.", {DL}, []),
    (f"Canadian passport {PASS_A} expires next year.", {PASS}, [("next year","DATE")]),
    (f"Passport number {PASS_C} was confirmed by CBSA.", {PASS}, [("CBSA","ORG")]),
    (f"Travel document {PASS_D} needs verification before boarding.", {PASS}, []),
    (f"Please scan passport {PASS_B} at check-in.", {PASS}, []),
    (f"The passport on file is {PASS_A}.", {PASS}, []),
    (f"Billing provider {PROV_A} submitted the claim.", {PROV}, []),
    (f"Practitioner ID {PROV_B} needs renewal.", {PROV}, []),
    (f"AHS provider {PROV_A} was added to the roster.", {PROV}, []),
    (f"Please verify provider no. {PROV_B} for this referral.", {PROV}, []),
    (f"CPSA {PROV_A} is registered for the new clinic.", {PROV}, []),
    (f"Patient SIN {SIN_H} and PHN {PHN_H} both updated.", {SIN,PHN}, []),
    (f"Process refund to {BANK_C} for SIN {SIN_S}.", {BANK,SIN}, []),
    (f"ITN {ITN_H} and passport {PASS_A} are on file.", {ITN,PASS}, []),
    (f"Provider {PROV_A} billed PHN {PHN_H} last Tuesday.", {PROV,PHN}, [("last Tuesday","DATE")]),
    (f"DL {DL_AB} and passport {PASS_C} verified at the border.", {DL,PASS}, []),
    ("Sarah Patel from CRA called on March 15, 2026.", set(),
        [("Sarah Patel","PERSON"),("CRA","ORG"),("March 15, 2026","DATE")]),
    ("Dr. Jonathan Smith updated the records yesterday.", set(),
        [("Dr. Jonathan Smith","PERSON"),("yesterday","DATE")]),
    ("The meeting at Foothills Medical Centre starts at 9:00 a.m.", set(),
        [("Foothills Medical Centre","FAC"),("9:00 a.m.","TIME")]),
    ("Alberta Health Services processed 2,500 applications in Q1 2026.", set(),
        [("Alberta Health Services","ORG"),("2,500","CARDINAL"),("Q1 2026","DATE")]),
    ("Please contact the Canada Revenue Agency in Edmonton by Friday.", set(),
        [("Canada Revenue Agency","ORG"),("Edmonton","GPE"),("Friday","DATE")]),
    ("Please update the system and restart the application.", set(), []),
    ("The meeting has been postponed to next week.", set(), [("next week","DATE")]),
    ("Submit your feedback through the online form.", set(), []),
    ("All systems are operating normally today.", set(), [("today","DATE")]),
    ("Thank you for reaching out to our team.", set(), []),
    (f"Reference {ITN_H} appears in the meeting notes.", set(), []),
    (f"Invoice {PHN_H} was paid on Tuesday.", set(), [("Tuesday","DATE")]),
    ("Page 257 of 574 contains the executive summary.", set(), []),
    ("Phone extension 4591 reached voicemail.", set(), []),
    (f"Order number {SIN_H} has shipped from the warehouse.", set(), []),
    (f"Lot {BANK_A} shipped from the warehouse.", set(), []),
    (f"Asset tag {PROV_A} was scanned during inventory.", set(), []),
    (f"Conference room {PASS_C} is booked for the morning.", set(), []),
    (f"PHN {PHN_H} is different from ITN {ITN_H} on the form.", {PHN,ITN}, []),
    (f"Update SIN {SIN_S} but do not confuse it with PHN {PHN_S}.", {SIN,PHN}, []),
    (f"The CRA ITN {ITN_H} is for tax purposes only, not health billing.", {ITN}, [("CRA","ORG")]),
    (f"Note: {SIN_H} is the SIN; {PHN_H} is the PHN.", {SIN,PHN}, []),
    (f"Hey, can you double-check the SIN {SIN_S} for me?", {SIN}, []),
    (f"FYI the PHN on the patient's file is {PHN_H}.", {PHN}, []),
    (f"Quick note \u2014 Sarah's ITN is {ITN_H}, please update.", {ITN}, [("Sarah","PERSON")]),
    (f"Heads up: the bank account {BANK_A} was flagged for review.", {BANK}, []),
    (f"Need to verify Alberta DL {DL_AB} before the test drive.", {DL}, []),
    (SIN_H, set(), []),
    (f"{PHN_S}.", set(), []),
    (f"{ITN_H} \u2014 ITN on file.", {ITN}, []),
    (f"Passport: {PASS_A}", {PASS}, []),
    (f"SIN {SIN_H}", {SIN}, []),
    (f"Maya Singh's SIN {SIN_S} was verified by HR on March 12, 2026.", {SIN},
        [("Maya Singh","PERSON"),("March 12, 2026","DATE")]),
    (f"Send the refund of $1,500.00 to {BANK_A} by Friday.", {BANK},
        [("$1,500.00","MONEY"),("Friday","DATE")]),
    (f"Dr. André Lévesque updated PHN {PHN_H} at Misericordia Community Hospital.", {PHN},
        [("Dr. André Lévesque","PERSON"),("Misericordia Community Hospital","FAC")]),
    (f"The Canada Revenue Agency processed ITN {ITN_H} last quarter.", {ITN},
        [("Canada Revenue Agency","ORG"),("last quarter","DATE")]),
    ("Officer Yang at Calgary International Airport verified passport AB123456.", {PASS},
        [("Officer Yang","PERSON"),("Calgary International Airport","FAC")]),
]

nlp = spacy.load(MODEL_PATH)

onto = {t: {"tp":0,"fp":0,"fn":0} for t in ONTO_TYPES}
can  = {t: {"tp":0,"fp":0,"fn":0} for t in CAN_TYPES}
onto_failures = []

for idx, (text, can_exp, onto_gold) in enumerate(CASES, 1):
    doc = nlp(text)
    pred_onto = {(e.text, e.label_) for e in doc.ents if e.label_ in ONTO_TYPES}
    gold_onto = set(onto_gold)

    for span in gold_onto & pred_onto: onto[span[1]]["tp"] += 1
    missing  = gold_onto - pred_onto
    spurious = pred_onto - gold_onto
    for _, lab in missing:  onto[lab]["fn"] += 1
    for _, lab in spurious: onto[lab]["fp"] += 1
    if missing or spurious:
        onto_failures.append((idx, text, sorted(missing), sorted(spurious)))

    if can_exp is not None:
        pred_can = {e.label_ for e in doc.ents if e.label_ in CAN_SET}
        for lab in CAN_TYPES:
            g, p = (lab in can_exp), (lab in pred_can)
            can[lab]["tp"] += g and p
            can[lab]["fp"] += p and not g
            can[lab]["fn"] += g and not p

def prf(d):
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    P = tp/(tp+fp) if tp+fp else None
    R = tp/(tp+fn) if tp+fn else None
    F = 2*P*R/(P+R) if P and R else (0.0 if (tp+fp+fn) else None)
    return P, R, F

def cell(x): return " -  " if x is None else f"{x*100:4.0f}"

print("="*78)
print("ONTONOTES (18 base types) — span-level over the original smoke cases")
print("="*78)
print(f"{'type':12s} {'P':>5s} {'R':>5s} {'F':>5s}   {'TP':>3s} {'FP':>3s} {'FN':>3s}   support")
print("-"*78)
fails = []
for t in ONTO_TYPES:
    P,R,F = prf(onto[t])
    sup = onto[t]["tp"] + onto[t]["fn"]
    flag = ""
    if onto[t]["fp"] or onto[t]["fn"]:
        flag = "  <-- FAILS"; fails.append(t)
    print(f"{t:12s} {cell(P):>5s} {cell(R):>5s} {cell(F):>5s}   "
          f"{onto[t]['tp']:>3d} {onto[t]['fp']:>3d} {onto[t]['fn']:>3d}   "
          f"{sup if sup else '-':>4}{flag}")

print("\nBase types with ANY error (FP or FN):")
print("  " + (", ".join(fails) if fails else "none"))
print("Base types with NO gold instance here (precision-only):")
print("  " + ", ".join(t for t in ONTO_TYPES if onto[t]['tp']+onto[t]['fn'] == 0))

print("\n" + "="*78)
print("OntoNotes failures (per sentence)")
print("="*78)
for idx, text, missing, spurious in onto_failures:
    print(f"\n[{idx}] {text}")
    if missing:  print("   MISSED   :", ", ".join(f'"{t}"->{l}' for t,l in missing))
    if spurious: print("   SPURIOUS :", ", ".join(f'"{t}"->{l}' for t,l in spurious))

print("\n" + "="*78)
print("Canadian types (sentence-level, for context)")
print("="*78)
print(f"{'type':35s} {'P':>5s} {'R':>5s} {'F':>5s}")
for t in CAN_TYPES:
    P,R,F = prf(can[t])
    print(f"{t:35s} {cell(P):>5s} {cell(R):>5s} {cell(F):>5s}")

print("\nNote: 10/18 base types have no gold positives in these sentences, so their")
print("recall is unmeasured here. Run `spacy evaluate` on an OntoNotes/silver dev")
print("set for true per-type numbers on all 18.")
