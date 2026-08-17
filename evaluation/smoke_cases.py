#!/usr/bin/env python3
"""
Shared smoke-test cases. Imported by both the model smoke test and the
Presidio comparison so the two are scored on identical inputs.

Each case is (text, expect) where `expect` is the set of Canadian-PII labels
that should be found. `expect=None` => informational only (not scored).
All identifier values are synthetic and conform to the project's generators.
"""

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

CASES = [
    # SIN
    (f"Please update SIN {SIN_S} in the payroll system.",            {SIN}),
    (f"Employee SIN: {SIN_H} has been verified by HR.",              {SIN}),
    (f"Her social insurance number is {SIN_P}.",                     {SIN}),
    (f"Service Canada confirmed SIN {SIN_H} belongs to the applicant.", {SIN}),
    # PHN
    (f"Please update PHN {PHN_H} in the clinic system.",             {PHN}),
    (f"Patient AHCIP number {PHN_S} is on file.",                    {PHN}),
    (f"Verify Alberta health number {PHN_P} for the new registration.", {PHN}),
    (f"PHN: {PHN_H} billing approved.",                              {PHN}),
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
    # DL
    (f"Alberta DL {DL_AB} is on file for the new employee.",         {DL}),
    (f"Driver's licence number {DL_BC} has expired.",                {DL}),
    (f"Please update DL# {DL_SK} in the registry.",                  {DL}),
    (f"AB licence {DL_AB} verified at the dealership.",              {DL}),
    # Passport
    (f"Canadian passport {PASS_A} expires next year.",               {PASS}),
    (f"Passport number {PASS_C} was confirmed by CBSA.",             {PASS}),
    (f"Please scan passport {PASS_B} at check-in.",                  {PASS}),
    (f"The passport on file is {PASS_A}.",                           {PASS}),
    # Provider
    (f"Billing provider {PROV_A} submitted the claim.",              {PROV}),
    (f"Practitioner ID {PROV_B} needs renewal.",                     {PROV}),
    (f"AHS provider {PROV_A} was added to the roster.",              {PROV}),
    (f"Please verify provider no. {PROV_B} for this referral.",      {PROV}),
    # Multiple
    (f"Patient SIN {SIN_H} and PHN {PHN_H} both updated.",           {SIN, PHN}),
    (f"Process refund to {BANK_C} for SIN {SIN_S}.",                 {BANK, SIN}),
    (f"ITN {ITN_H} and passport {PASS_A} are on file.",              {ITN, PASS}),
    (f"DL {DL_AB} and passport {PASS_C} verified at the border.",    {DL, PASS}),
    # Negatives
    ("Please update the system and restart the application.",        set()),
    ("The meeting has been postponed to next week.",                 set()),
    ("All systems are operating normally today.",                    set()),
    # Hard negatives (look like IDs, no keyword)
    (f"Reference {ITN_H} appears in the meeting notes.",             set()),
    (f"Invoice {PHN_H} was paid on Tuesday.",                        set()),
    ("Page 257 of 574 contains the executive summary.",              set()),
    (f"Order number {SIN_H} has shipped from the warehouse.",        set()),
    (f"Lot {BANK_A} shipped from the warehouse.",                    set()),
    (f"Asset tag {PROV_A} was scanned during inventory.",            set()),
    (f"Conference room {PASS_C} is booked for the morning.",         set()),
    # Disambiguation
    (f"PHN {PHN_H} is different from ITN {ITN_H} on the form.",      {PHN, ITN}),
    (f"Update SIN {SIN_S} but do not confuse it with PHN {PHN_S}.",  {SIN, PHN}),
    # Informal
    (f"Hey, can you double-check the SIN {SIN_S} for me?",           {SIN}),
    (f"FYI the PHN on the patient's file is {PHN_H}.",               {PHN}),
    # Edge / bare (should be negative)
    (SIN_H,                                                          set()),
    (f"{PHN_S}.",                                                    set()),
    (f"Passport: {PASS_A}",                                          {PASS}),
    (f"SIN {SIN_H}",                                                 {SIN}),
    # Domain numeric decoys (version/error/duration) -> negative
    ("After updating to version 2.4.1 the app finally stopped crashing.", set()),
    ("Got error code 0x80004 three times before the health card loaded.", set()),
    ("Support ticket 88231 has been open for two weeks with no reply.", set()),
]
