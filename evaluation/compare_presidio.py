#!/usr/bin/env python3
"""
Baseline comparison: rule-based Presidio vs the fine-tuned spaCy model, scored
on the SAME cases (smoke_cases.CASES). Columns: MODEL, PRESIDIO-stock,
PRESIDIO-custom (Presidio + hand-written Canadian recognizers).

Install:  pip install presidio-analyzer
"""

import os
from smoke_cases import (CASES, CANADIAN_LABELS, SIN, ITN, PHN, BANK, DL, PASS, PROV)

MODEL_PATH = "./output_stage2/model-best"

try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
except ImportError:
    raise SystemExit("Install Presidio first:  pip install presidio-analyzer")

PRESIDIO_TO_CANADIAN = {
    "CA_SIN": SIN, "CA_ITN": ITN, "CA_PHN": PHN, "CA_BANK": BANK,
    "CA_DL": DL, "CA_PASSPORT": PASS, "CA_PROVIDER": PROV,
    "US_SSN": SIN, "US_ITIN": ITN, "UK_NHS": PHN,
    "US_BANK_NUMBER": BANK, "IBAN_CODE": BANK,
    "US_DRIVER_LICENSE": DL, "US_PASSPORT": PASS, "MEDICAL_LICENSE": PROV,
}

NINE = r"(?:\b\d{3}[- ]\d{3}[- ]\d{3}\b|\b\d{9}\b)"
BANK_RE = r"(?:\b\d{5}[- ]\d{3}[- ]\d{7,12}\b|\b(?:account|acct\.?)\s+\d{7,12}\b)"
PASS_RE = r"(?:\b[A-Z]\d{6}[A-Z]{2}\b|\b[A-Z]{2}\d{6}\b)"
DL_RE   = r"(?:\b[A-Z]?\d{6,9}\b|\b[A-Z]\d{9,14}\b|\b[A-Z0-9]{12,14}\b)"

def rec(entity, regex, context, score=0.4):
    return PatternRecognizer(
        supported_entity=entity,
        patterns=[Pattern(name=entity.lower(), regex=regex, score=score)],
        context=context,
    )

CUSTOM_RECOGNIZERS = [
    rec("CA_SIN",      NINE,    ["sin", "social", "insurance"]),
    rec("CA_ITN",      NINE,    ["itn", "individual", "tax"]),
    rec("CA_PHN",      NINE,    ["phn", "ahcip", "health", "personal"]),
    rec("CA_PROVIDER", NINE,    ["provider", "practitioner", "cpsa", "ahs", "billing"]),
    rec("CA_BANK",     BANK_RE, ["account", "transit", "deposit", "bank", "wire"]),
    rec("CA_DL",       DL_RE,   ["licence", "license", "dl", "driver", "operator"]),
    rec("CA_PASSPORT", PASS_RE, ["passport", "travel", "document"]),
]
SCORE_THRESHOLD = 0.5

def canadian_labels_from(results):
    out = set()
    for r in results:
        lab = PRESIDIO_TO_CANADIAN.get(r.entity_type)
        if lab:
            out.add(lab)
    return out

print("Loading Presidio (spaCy en_core_web_lg under the hood)...")
analyzer = AnalyzerEngine()

def presidio_pass(texts):
    return [canadian_labels_from(
        analyzer.analyze(text=t, language="en", score_threshold=SCORE_THRESHOLD)
    ) for t, _ in texts]

stock_preds = presidio_pass(CASES)
for r in CUSTOM_RECOGNIZERS:
    analyzer.registry.add_recognizer(r)
custom_preds = presidio_pass(CASES)

model_preds = None
if os.path.isdir(MODEL_PATH):
    import spacy
    nlp = spacy.load(MODEL_PATH)
    model_preds = []
    for t, _ in CASES:
        doc = nlp(t)
        model_preds.append({e.label_ for e in doc.ents if e.label_ in CANADIAN_LABELS})
else:
    print(f"(model not found at {MODEL_PATH}; skipping MODEL column)")

def overall_pass(preds):
    scored = [(p, exp) for p, (_, exp) in zip(preds, CASES) if exp is not None]
    return sum(p == exp for p, exp in scored), len(scored)

def per_type(preds):
    rows = {}
    for lab in sorted(CANADIAN_LABELS):
        tp = fp = fn = 0
        for p, (_, exp) in zip(preds, CASES):
            if exp is None:
                continue
            g, pr = (lab in exp), (lab in p)
            tp += g and pr; fp += pr and not g; fn += g and not pr
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        rows[lab] = (P, R, F, tp, fp, fn)
    return rows

def fp_on_clean(preds):
    return sum(1 for p, (_, exp) in zip(preds, CASES) if exp == set() and p)

def short(lab):
    return {SIN:"SIN", ITN:"ITN", PHN:"PHN", BANK:"BANK", DL:"DL", PASS:"PASS", PROV:"PROV"}[lab]
def fmt(s):
    return "{" + ",".join(short(x) for x in sorted(s)) + "}" if s else "\u2205"

cols = [("PRESIDIO-stock", stock_preds), ("PRESIDIO-custom", custom_preds)]
if model_preds is not None:
    cols = [("MODEL", model_preds)] + cols

print("\n" + "=" * 100)
print("PER-CASE  (check/x vs expected; informational rows marked .)")
print("=" * 100)
for i, (text, exp) in enumerate(CASES):
    print(f"\n[{i+1}] {text}")
    print(f"      expected: {fmt(exp) if exp is not None else '(info)'}")
    for name, preds in cols:
        p = preds[i]
        mark = "." if exp is None else ("OK" if p == exp else "X ")
        print(f"      {mark} {name:16s} {fmt(p)}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
hdr = f"{'engine':18s} {'exact-match':>12s} {'clean-text FPs':>15s}"
print(hdr); print("-" * len(hdr))
for name, preds in cols:
    ok, n = overall_pass(preds)
    print(f"{name:18s} {ok:>4d}/{n:<4d} {100*ok/n:>5.1f}% {fp_on_clean(preds):>10d}")

print("\nPer-type F1 (sentence-level)")
header = f"{'type':6s}" + "".join(f"{name:>17s}" for name, _ in cols)
print(header); print("-" * len(header))
pertype = {name: per_type(preds) for name, preds in cols}
for lab in sorted(CANADIAN_LABELS):
    line = f"{short(lab):6s}"
    for name, _ in cols:
        P, R, F, tp, fp, fn = pertype[name][lab]
        line += f"{F*100:>10.1f} (R{R*100:3.0f})"
    print(line)
