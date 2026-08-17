# Canadian PII Recognition & Redaction

A fine-tuned spaCy NER system that detects and redacts seven Canadian personally
identifiable identifiers in unstructured text, trained on synthetic data and
built on `en_core_web_lg`.

Detected types:

| Label | Meaning |
|-------|---------|
| `CANADIAN_SOCIAL_INSURANCE_NUMBER` | SIN (9 digits, Luhn-valid) |
| `CANADIAN_INDIVIDUAL_TAX_NUMBER` | ITN (9 digits) |
| `ALBERTA_PERSONAL_HEALTH_NUMBER` | Alberta PHN / AHCIP (9 digits) |
| `CANADIAN_BANK_ACCOUNT_NUMBER` | transit + institution + account |
| `ALBERTA_DRIVERS_LICENCE_NUMBER` | provincial driver's licence |
| `CANADIAN_PASSPORT_NUMBER` | Canadian passport |
| `CANADIAN_PROVIDER_IDENTIFIER` | Alberta practitioner / provider ID |

Design in one line: **value-only labelling** — the identifier value is the
entity, the trigger word ("SIN", "PHN", ...) is left as context, and the model
learns to gate on that context so bare look-alike numbers stay untagged.

---

## Repository structure

```
canadian-pii-ner/
├── README.md
├── .gitignore
├── requirements.txt
│
├── data_generation/
│   ├── synthetic_ner_generator.py   # your generator (produces JSONL + label.json)
│   └── convert_to_spacy.py          # JSONL (tokens+BIO) -> spaCy .spacy
│
├── configs/
│   ├── base_config.cfg              # stage 1 (frozen tok2vec, LR 1e-3)
│   └── base_config_stage2.cfg       # stage 2 (unfrozen, LR 1e-4, sources frozen model)
│
├── notebooks/
│   ├── canadian_pii_end2end.ipynb   # generate -> convert -> train -> eval (from scratch)
│   └── canadian_pii_finetune.ipynb  # upload .spacy -> train -> eval (data already built)
│
├── evaluation/
│   ├── smoke_cases.py               # shared test cases (imported below)
│   ├── smoke_test.py                # synthetic out-of-distribution smoke test (scored)
│   ├── smoke_test_realworld.py      # Alberta Wallet register (false-positive robustness)
│   ├── eval_ontonotes.py            # diagnostic for the 18 inherited base types
│   ├── compare_presidio.py          # model vs rule baseline (stock + custom Presidio)
│   └── analysis.py                  # bootstrap CIs, Wilson, McNemar, confusion matrix
│
├── api/
│   ├── app.py                       # FastAPI: /health /detect /redact
│   ├── static/index.html            # web UI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── corpus/            # generated .spacy    (gitignored)
├── synth/             # generated JSONL     (gitignored)
├── output_frozen/     # stage-1 model       (gitignored)
└── output_stage2/     # stage-2 model — model-best is the deliverable (gitignored)
```

Paths in the configs and eval scripts are **relative to the repo root**
(`corpus/…`, `./output_stage2/model-best`). Run all commands from the repo root
so they resolve, e.g. `python evaluation/smoke_test.py`, not from inside
`evaluation/`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

`requirements.txt` (top level):

```
spacy==3.7.5
click<8.2
typer<0.10
numpy<2
scipy
matplotlib
```

(The API has its own `api/requirements.txt`; Presidio comparison additionally
needs `pip install presidio-analyzer`.)

---

## The fast path: Google Colab (recommended)

Two notebooks in `notebooks/`. Both mount Drive first so a runtime disconnect
doesn't wipe your work.

**`canadian_pii_end2end.ipynb`** — from scratch. Upload your generator, it
generates JSONL, converts to `.spacy`, runs both training stages, evaluates,
smoke tests, and zips the model for download. Use this to reproduce everything.

**`canadian_pii_finetune.ipynb`** — training only. Upload pre-built
`train.spacy` / `valid.spacy` / `test.spacy` / `label.json` and it trains +
evaluates. Use this when the data is already generated.

Colab gives you a free GPU; open the notebook, Runtime → Run all, and follow the
upload prompts.

---

## The full pipeline locally

### 1. Generate synthetic data

```bash
python data_generation/synthetic_ner_generator.py \
    --output-dir synth --seed 42 \
    --train-size 12000 --validation-size 1600 --test-size 1600
```

Produces `synth/train.jsonl`, `synth/validation.jsonl`, `synth/test.jsonl`,
`synth/label.json` (each JSONL line is `{"tokens": [...], "tags": [int, ...]}`).

### 2. Convert JSONL → `.spacy`

```bash
python data_generation/convert_to_spacy.py --in synth --out corpus
# -> corpus/train.spacy, corpus/valid.spacy, corpus/test.spacy
```

### 3. Train — two stages

Stage 1 warms up the NER head with the encoder frozen (protects the pretrained
weights). Stage 2 unfreezes and fine-tunes at a low LR, sourcing both components
from the stage-1 model.

```bash
# stage 1 (frozen)
python -m spacy init fill-config configs/base_config.cfg configs/_stage1.cfg
python -m spacy debug config configs/_stage1.cfg        # confirm learn_rate = 0.001
python -m spacy train configs/_stage1.cfg --output ./output_frozen

# stage 2 (unfrozen, continues from output_frozen/model-best)
python -m spacy init fill-config configs/base_config_stage2.cfg configs/_stage2.cfg
python -m spacy debug config configs/_stage2.cfg        # confirm learn_rate = 0.0001
python -m spacy train configs/_stage2.cfg --output ./output_stage2
```

`patience = 1600` stops each stage when the dev score plateaus; `max_steps` is a
safety ceiling, not the intended endpoint. Your deliverable is
`output_stage2/model-best`.

### 4. Evaluate (per-type P/R/F on the held-out set)

```bash
python -m spacy evaluate ./output_stage2/model-best corpus/test.spacy --output metrics.json
```

Note: if `test.spacy` came from the same generator it is *in-distribution* and
will read high. Treat it as a convergence check, not a defensible accuracy
figure — see the smoke tests and analysis below for the honest numbers.

---

## Evaluation & analysis

All scripts default to `./output_stage2/model-best`; run from the repo root.

### Smoke tests (out-of-distribution, scored)

```bash
python evaluation/smoke_test.py            # hand-written synthetic sentences
python evaluation/smoke_test_realworld.py  # Alberta Wallet review register
```

`smoke_test_realworld.py` is the false-positive test: authentic app-review
phrasing full of trigger words ("health card", "driver's licence") but no
numbers — the model must stay silent. It reports a clean-text false-positive
count separately.

### Base-type (OntoNotes) diagnostic

```bash
python evaluation/eval_ontonotes.py
```

Scores the 18 inherited types (PERSON/ORG/DATE/...) on the original cases and
lists which fail. Note only ~8 of 18 are exercised here; use `spacy evaluate` on
an OntoNotes/silver set for real per-type numbers on all 18.

### Rule-baseline comparison (Presidio)

```bash
pip install presidio-analyzer
python evaluation/compare_presidio.py
```

Scores your model vs stock Presidio vs Presidio + custom Canadian recognizers on
identical cases (`smoke_cases.py`). Supports the "compare against rule-based
recognizers" objective.

### Statistical analysis

```bash
python evaluation/analysis.py --model ./output_stage2/model-best --gold corpus/test.spacy
```

Produces per-type F1 with **95% bootstrap confidence intervals**, document-level
exact-match with a **Wilson interval**, a **McNemar test** vs a regex+context
rule baseline (with a p-value), and a **confusion-matrix heatmap**
(`confusion_matrix.png`). Point `--gold` at an independent hand-labelled set for
a defensible number; against `test.spacy` the CIs measure generator-consistency,
not generalization.

---

## Deploy the API

See `api/README.md` for detail. Short version:

```bash
cp -r output_stage2/model-best api/model     # place the trained model
cd api
docker build -t pii-api .
docker run -p 8000:8000 pii-api
# open http://localhost:8000
```

Endpoints: `GET /health`, `POST /detect`, `POST /redact` (values replaced with
`[SIN]`, `[PHN]`, ...). Request bodies are not logged. Testing/synthetic data
only — for real use add auth, TLS, rate limiting, and a no-retention policy.

```bash
curl -s localhost:8000/redact -H 'Content-Type: application/json' \
  -d '{"text":"SIN 130 692 544 on file."}'
# -> {"redacted":"[SIN] on file.", ...}
```

---

## Notes & honest limitations

- **In-distribution vs generalization.** A high `spacy evaluate` / dev F on
  generator-derived data mainly measures memorization of your templates. The
  smoke tests and an independent hand-labelled held-out set are the numbers that
  reflect real performance.
- **Base-type drift.** Fine-tuning erodes some inherited OntoNotes types
  (notably relative DATE and government ORG, since org names sit in-context as
  `O`). If those types matter for your deliverable, add silver-label rehearsal
  (stock `en_core_web_lg` predictions over generic English mixed into training).
  If only the seven Canadian types matter, this is an accepted trade-off.
- **Privacy.** Training data is entirely synthetic. The API receives real PII at
  inference, so keep the no-logging default and don't deploy the demo as-is for
  production without hardening.
