#!/usr/bin/env python3
"""
Real-world smoke test — Alberta Wallet register.

TIER A  False positives on authentic domain text (trigger words, no numbers) -> silent.
TIER B  Detection inside real review/support phrasing with SYNTHETIC values.
TIER C  Domain numeric decoys (versions, error codes, durations) -> no PII.

Sentences are original paraphrases of common review themes; all values synthetic.
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

PHN_H, PHN_S, PHN_P = "257-574-243", "257 574 243", "257574243"
SIN_S = "130 692 544"
ITN_H = "459-311-601"
DL_AB = "428913576"
BANK_A = "18016-003-86796296"

smoke_tests = [
    # TIER A — authentic register, NO PII numbers -> expect nothing
    ("Why do I have to type my password every single time I open the wallet app?", set()),
    ("Wish I could add my driver's licence and vehicle registration, not just the health card.", set()),
    ("The Mobile Health Card works, but it signs me out every time I close the app.", set()),
    ("Please let us store the health card in Apple Wallet instead of a whole separate app.", set()),
    ("Nice graphics, but building an entire app for one card feels like overkill.", set()),
    ("Carrying the physical health card is honestly faster than unlocking this each time.", set()),
    ("Add fingerprint or Face ID sign-in please; the password step is tedious.", set()),
    ("Couldn't finish setup without a verified Alberta.ca account, which took a while.", set()),
    ("Good idea to go digital, but it logs me out constantly and I have to verify again.", set()),
    ("I just want my personal health number on my phone without jumping through hoops.", set()),
    ("When will the driver's licence finally be available in the Alberta Wallet?", set()),
    ("The QR code wouldn't scan at the pharmacy, so I had to use my plastic card.", set()),
    ("It asks me to refresh the health card far too often for something I rarely use.", set()),
    ("Setup was confusing — the verify screen never said what it actually wanted.", set()),
    ("My AHCIP coverage is fine but the app still won't show my health card.", set()),
    ("Tried to add my SIN and passport too, but apparently only the health card is supported.", set()),

    # TIER B — real register WITH a synthetic value -> expect type
    (f"Tried adding my health card but it rejected PHN {PHN_H} as invalid.",            {PHN}),
    (f"The app finally shows my personal health number {PHN_P} after the update.",       {PHN}),
    (f"Support asked me to confirm driver's licence {DL_AB} before unlocking the wallet.", {DL}),
    (f"I typed AHCIP number {PHN_S} and it still refused to verify.",                    {PHN}),
    (f"It wouldn't accept Alberta DL {DL_AB} when I tried to add it.",                   {DL}),
    (f"The renewal email quoted my ITN {ITN_H}, which felt odd for a health app.",       {ITN}),
    (f"Why does verification ask for SIN {SIN_S} just to add a health card?",            {SIN}),
    (f"Setup wanted a deposit account, so I gave {BANK_A} and it still failed.",          {BANK}),

    # TIER C — domain numeric decoys -> expect nothing
    ("After updating to version 2.4.1 the app finally stopped crashing on launch.", set()),
    ("It signs me out after about 30 days and makes me verify all over again.", set()),
    ("I've been waiting since March 2026 for the driver's licence option to show up.", set()),
    ("Got error code 0x80004 three times before the health card would load.", set()),
    ("The download is around 48 MB and still takes forever to open.", set()),
    ("Support ticket 88231 has been open for two weeks with no reply.", set()),
    ("The QR code refreshes every 60 seconds which is annoying at the counter.", set()),
    ("Build 1.0.7 fixed the login loop I had on Android 14.", set()),
]

print(f"\n{'='*80}")
print(f"REAL-WORLD smoke test — Alberta Wallet register   (model: {MODEL_PATH})")
print(f"Running {len(smoke_tests)} cases")
print('='*80)

scored = passed = 0
failures = []
fp_on_clean = 0

for i, (text, expect) in enumerate(smoke_tests, 1):
    doc = nlp(text)
    found_can = {e.label_ for e in doc.ents if e.label_ in CANADIAN_LABELS}
    scored += 1
    ok = (found_can == expect)
    passed += ok
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append((i, text, expect, found_can))
        if expect == set():
            fp_on_clean += 1

    print(f"\n[{status}] [{i}] '{text}'")
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
print(f"SCORED: {passed}/{scored} passed   ({100*passed/scored:.1f}%)")
print(f"False positives on clean domain text (Tier A/C): {fp_on_clean}")
print('='*80)
if failures:
    print("Failures:")
    for i, text, exp, got in failures:
        print(f"  [{i}] expected {sorted(exp) or '\u2205'} | got {sorted(got) or '\u2205'}  ::  {text[:62]}")
