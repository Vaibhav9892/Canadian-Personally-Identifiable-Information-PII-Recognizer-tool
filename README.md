# Canadian PII Detector — REST API + Web UI

FastAPI service wrapping the fine-tuned spaCy model, with a small web front-end.

## Layout
```
pii_api/
  app.py              FastAPI app (/health, /detect, /redact)
  static/index.html   web UI
  Dockerfile
  requirements.txt
  model/              <- put your trained model here (copy of model-best)
```

## 1. Add your model
```bash
cp -r /path/to/output_stage2/model-best ./model
```

## 2a. Run locally (no Docker)
```bash
pip install -r requirements.txt
MODEL_PATH=./model uvicorn app:app --reload --port 8000
# open http://localhost:8000
```

## 2b. Run with Docker
```bash
docker build -t pii-api .
docker run -p 8000:8000 pii-api
# open http://localhost:8000
```

## API
```bash
curl -s localhost:8000/health

curl -s localhost:8000/detect -H 'Content-Type: application/json' \
  -d '{"text":"Please update SIN 130 692 544 and PHN 257-574-243."}'

curl -s localhost:8000/redact -H 'Content-Type: application/json' \
  -d '{"text":"SIN 130 692 544 on file."}'
# -> {"redacted":"[SIN] on file.", ...}
```
`canadian_only` (default true) restricts output to the 7 target PII types.

## Notes
- Request bodies are **not logged** (uvicorn access log disabled).
- Testing/synthetic data only; this is not a hardened production service.
  For real use add auth, TLS, rate limiting, and a no-retention data policy.
