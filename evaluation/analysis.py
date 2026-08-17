#!/usr/bin/env python3
"""
Statistical analysis of the Canadian PII NER model on a labelled .spacy set.

Produces:
  1. Per-type Precision/Recall/F1 (strict span match) with 95% bootstrap CIs.
  2. Document-level exact-match accuracy with a Wilson score interval.
  3. McNemar's test: model vs a rule (regex+context) baseline, on doc-level
     correctness — tells you if the difference is statistically real.
  4. Entity confusion matrix (gold label x predicted label, incl. MISS/SPURIOUS),
     saved as confusion_matrix.png.

Usage:
  python analysis.py --model ./output_stage2/model-best --gold corpus/test.spacy
"""
import argparse, re, json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import DocBin
from scipy import stats

CANADIAN = {
    "CANADIAN_SOCIAL_INSURANCE_NUMBER","CANADIAN_INDIVIDUAL_TAX_NUMBER",
    "ALBERTA_PERSONAL_HEALTH_NUMBER","CANADIAN_BANK_ACCOUNT_NUMBER",
    "ALBERTA_DRIVERS_LICENCE_NUMBER","CANADIAN_PASSPORT_NUMBER",
    "CANADIAN_PROVIDER_IDENTIFIER",
}
SHORT = {"CANADIAN_SOCIAL_INSURANCE_NUMBER":"SIN","CANADIAN_INDIVIDUAL_TAX_NUMBER":"ITN",
         "ALBERTA_PERSONAL_HEALTH_NUMBER":"PHN","CANADIAN_BANK_ACCOUNT_NUMBER":"BANK",
         "ALBERTA_DRIVERS_LICENCE_NUMBER":"DL","CANADIAN_PASSPORT_NUMBER":"PASS",
         "CANADIAN_PROVIDER_IDENTIFIER":"PROV"}

# ---- lightweight rule baseline (value-only, context-gated) for McNemar ----
NINE = re.compile(r"\b(?:\d{3}[- ]\d{3}[- ]\d{3}|\d{9})\b")
BANKRE = re.compile(r"\b\d{5}[- ]\d{3}[- ]\d{7,12}\b")
PASSRE = re.compile(r"\b(?:[A-Z]\d{6}[A-Z]{2}|[A-Z]{2}\d{6})\b")
DLRE = re.compile(r"\b(?:[A-Z]?\d{6,9}|[A-Z]\d{9,14}|[A-Z0-9]{12,14})\b")
CTX = {
    "CANADIAN_SOCIAL_INSURANCE_NUMBER": ["sin","social insurance"],
    "CANADIAN_INDIVIDUAL_TAX_NUMBER": ["itn","individual tax","tax number"],
    "ALBERTA_PERSONAL_HEALTH_NUMBER": ["phn","ahcip","health number","health card"],
    "CANADIAN_PROVIDER_IDENTIFIER": ["provider","practitioner","cpsa","ahs"],
    "CANADIAN_BANK_ACCOUNT_NUMBER": ["account","transit","deposit","bank"],
    "ALBERTA_DRIVERS_LICENCE_NUMBER": ["licence","license","dl","driver","operator"],
    "CANADIAN_PASSPORT_NUMBER": ["passport","travel document"],
}
def near(text, s, e, words, win=25):
    ctx = text[max(0, s-win):s].lower()
    return any(w in ctx for w in words)

def rule_spans(text):
    """Return set of (start,end,label) from the regex baseline (context-gated)."""
    out = set()
    for m in BANKRE.finditer(text):
        if near(text, m.start(), m.end(), CTX["CANADIAN_BANK_ACCOUNT_NUMBER"]):
            out.add((m.start(), m.end(), "CANADIAN_BANK_ACCOUNT_NUMBER"))
    for m in PASSRE.finditer(text):
        if near(text, m.start(), m.end(), CTX["CANADIAN_PASSPORT_NUMBER"]):
            out.add((m.start(), m.end(), "CANADIAN_PASSPORT_NUMBER"))
    for m in NINE.finditer(text):
        for lab in ("CANADIAN_SOCIAL_INSURANCE_NUMBER","CANADIAN_INDIVIDUAL_TAX_NUMBER",
                    "ALBERTA_PERSONAL_HEALTH_NUMBER","CANADIAN_PROVIDER_IDENTIFIER"):
            if near(text, m.start(), m.end(), CTX[lab]):
                out.add((m.start(), m.end(), lab)); break
    for m in DLRE.finditer(text):
        if near(text, m.start(), m.end(), CTX["ALBERTA_DRIVERS_LICENCE_NUMBER"]):
            out.add((m.start(), m.end(), "ALBERTA_DRIVERS_LICENCE_NUMBER"))
    return out

# --------------------------------------------------------------------------
def prf(tp, fp, fn):
    P = tp/(tp+fp) if tp+fp else 0.0
    R = tp/(tp+fn) if tp+fn else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return P, R, F

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k/n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return p, max(0, centre-half), min(1, centre+half)

def counts_for(gold_list, pred_list, labels):
    """tp/fp/fn per label across all docs (strict span match)."""
    c = {l: [0,0,0] for l in labels}   # tp,fp,fn
    for g, p in zip(gold_list, pred_list):
        gs, ps = set(g), set(p)
        for span in gs & ps: c[span[2]][0] += 1
        for span in ps - gs: c[span[2]][1] += 1
        for span in gs - ps: c[span[2]][2] += 1
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./output_stage2/model-best")
    ap.add_argument("--gold",  default="corpus/test.spacy")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--canadian-only", action="store_true", default=True)
    args = ap.parse_args()

    nlp = spacy.load(args.model)
    gold_docs = list(DocBin().from_disk(args.gold).get_docs(nlp.vocab))
    print(f"Loaded {len(gold_docs)} gold docs from {args.gold}")

    labels = sorted(CANADIAN) if args.canadian_only else \
             sorted({e.label_ for d in gold_docs for e in d.ents})

    gold, pred_model, pred_rule = [], [], []
    for d in gold_docs:
        text = d.text
        g = {(e.start_char, e.end_char, e.label_) for e in d.ents
             if (e.label_ in CANADIAN or not args.canadian_only)}
        pm = {(e.start_char, e.end_char, e.label_) for e in nlp(text).ents
              if (e.label_ in CANADIAN or not args.canadian_only)}
        gold.append(g); pred_model.append(pm); pred_rule.append(rule_spans(text))

    # ---- 1. per-type PRF + bootstrap CI ----
    def perF(idx):
        c = counts_for([gold[i] for i in idx], [pred_model[i] for i in idx], labels)
        return {l: prf(*c[l]) for l in labels}

    base = perF(range(len(gold)))
    n = len(gold)
    boot = {l: [] for l in labels}; boot_micro = []
    rng = np.random.default_rng(0)
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        c = counts_for([gold[i] for i in idx], [pred_model[i] for i in idx], labels)
        tp = sum(c[l][0] for l in labels); fp = sum(c[l][1] for l in labels); fn = sum(c[l][2] for l in labels)
        boot_micro.append(prf(tp, fp, fn)[2])
        for l in labels:
            boot[l].append(prf(*c[l])[2])

    print("\n" + "="*72)
    print("PER-TYPE  (strict span match)   F1 [95% bootstrap CI]")
    print("="*72)
    print(f"{'type':8s} {'P':>6s} {'R':>6s} {'F1':>6s}   {'95% CI (F1)':>16s}")
    for l in labels:
        P, R, F = base[l]
        lo, hi = np.percentile(boot[l], [2.5, 97.5])
        print(f"{SHORT.get(l,l[:8]):8s} {P*100:6.1f} {R*100:6.1f} {F*100:6.1f}   "
              f"[{lo*100:5.1f}, {hi*100:5.1f}]")
    lo, hi = np.percentile(boot_micro, [2.5, 97.5])
    micro = counts_for(gold, pred_model, labels)
    mtp = sum(micro[l][0] for l in labels); mfp = sum(micro[l][1] for l in labels); mfn = sum(micro[l][2] for l in labels)
    print(f"{'MICRO':8s} {'':6s} {'':6s} {prf(mtp,mfp,mfn)[2]*100:6.1f}   [{lo*100:5.1f}, {hi*100:5.1f}]")

    # ---- 2. document-level exact match + Wilson ----
    exact = sum(gold[i] == pred_model[i] for i in range(n))
    p, wlo, whi = wilson(exact, n)
    print("\n" + "="*72)
    print("DOCUMENT-LEVEL EXACT MATCH")
    print("="*72)
    print(f"{exact}/{n} = {p*100:.1f}%   Wilson 95% CI [{wlo*100:.1f}, {whi*100:.1f}]")

    # ---- 3. McNemar: model vs rule baseline (doc-level correctness) ----
    m_ok = [gold[i] == pred_model[i] for i in range(n)]
    r_ok = [gold[i] == pred_rule[i]  for i in range(n)]
    b = sum(mo and not ro for mo, ro in zip(m_ok, r_ok))  # model right, rule wrong
    c = sum(ro and not mo for mo, ro in zip(m_ok, r_ok))  # rule right, model wrong
    pval = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    print("\n" + "="*72)
    print("McNEMAR — model vs rule (regex+context) baseline")
    print("="*72)
    print(f"model correct: {sum(m_ok)}/{n}   rule correct: {sum(r_ok)}/{n}")
    print(f"discordant: model-only={b}  rule-only={c}   exact binomial p = {pval:.4g}")
    print("  (p<0.05 => the model/rule difference is unlikely to be chance)")

    # ---- 4. confusion matrix ----
    axis = [SHORT.get(l, l) for l in labels] + ["MISS"]
    idxmap = {l: i for i, l in enumerate(labels)}
    K = len(labels)
    M = np.zeros((K, K + 1), dtype=int)   # rows=gold, cols=pred (+MISS)
    spur = np.zeros(K, dtype=int)         # predicted with no overlapping gold
    for g, p in zip(gold, pred_model):
        gused = set()
        for (ps, pe, pl) in p:
            hit = None
            for (gs, ge, gl) in g:
                if not (pe <= gs or ps >= ge):   # overlap
                    hit = gl; gused.add((gs, ge, gl)); break
            if hit is None:
                spur[idxmap[pl]] += 1
            else:
                M[idxmap[hit], idxmap[pl]] += 1
        for (gs, ge, gl) in g - gused:
            M[idxmap[gl], K] += 1  # gold missed

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(K + 1)); ax.set_xticklabels(axis, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels([SHORT.get(l, l) for l in labels], fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title("Entity confusion (rows=gold, cols=predicted)")
    for i in range(K):
        for j in range(K + 1):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center",
                        color="white" if M[i, j] > M.max()/2 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=140)
    print("\nSaved confusion_matrix.png")
    print("Spurious (predicted, no gold overlap):",
          {SHORT.get(labels[i], labels[i]): int(spur[i]) for i in range(K) if spur[i]})

    # dump machine-readable summary
    out = {"per_type": {SHORT.get(l,l): base[l] for l in labels},
           "exact_match": {"k": exact, "n": n, "wilson95": [wlo, whi]},
           "mcnemar": {"model_only": b, "rule_only": c, "p": pval}}
    json.dump(out, open("analysis_summary.json","w"), indent=2, default=float)
    print("Saved analysis_summary.json")

if __name__ == "__main__":
    main()
