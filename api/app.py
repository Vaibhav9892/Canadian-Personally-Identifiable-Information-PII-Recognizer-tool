#!/usr/bin/env python3
"""
Canadian PII detection/redaction REST API.

Endpoints
---------
GET  /health           -> {"status": "ok", "labels": [...]}
POST /detect  {text}   -> {"entities": [{text,label,start,end,tag}], "count"}
POST /redact  {text}   -> {"redacted", "entities"}

Privacy: request bodies are NEVER logged. This service is intended for
synthetic / testing data. Do not send real personal information to a
non-hardened deployment.
"""

import os
import logging
from typing import List

import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- do not log request/response bodies ----
logging.getLogger("uvicorn.access").disabled = True

MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# short tag shown in redacted output, per label
TAGS = {
    "CANADIAN_SOCIAL_INSURANCE_NUMBER": "SIN",
    "CANADIAN_INDIVIDUAL_TAX_NUMBER": "ITN",
    "ALBERTA_PERSONAL_HEALTH_NUMBER": "PHN",
    "CANADIAN_BANK_ACCOUNT_NUMBER": "BANK_ACCT",
    "ALBERTA_DRIVERS_LICENCE_NUMBER": "DL",
    "CANADIAN_PASSPORT_NUMBER": "PASSPORT",
    "CANADIAN_PROVIDER_IDENTIFIER": "PROVIDER_ID",
}
CANADIAN_LABELS = set(TAGS)

print(f"Loading model from {MODEL_PATH} ...")
nlp = spacy.load(MODEL_PATH)

app = FastAPI(title="Canadian PII Detector", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str
    canadian_only: bool = True   # only return the 7 target PII types


def _entities(text: str, canadian_only: bool):
    doc = nlp(text)
    out = []
    for e in doc.ents:
        if canadian_only and e.label_ not in CANADIAN_LABELS:
            continue
        out.append({
            "text": e.text,
            "label": e.label_,
            "tag": TAGS.get(e.label_, e.label_),
            "start": e.start_char,
            "end": e.end_char,
        })
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "labels": sorted(CANADIAN_LABELS)}


@app.post("/detect")
def detect(inp: TextIn):
    ents = _entities(inp.text, inp.canadian_only)
    return {"count": len(ents), "entities": ents}


@app.post("/redact")
def redact(inp: TextIn):
    ents = _entities(inp.text, inp.canadian_only)
    # replace right-to-left so offsets stay valid
    red = inp.text
    for e in sorted(ents, key=lambda x: x["start"], reverse=True):
        red = red[:e["start"]] + f"[{e['tag']}]" + red[e["end"]:]
    return {"redacted": red, "count": len(ents), "entities": ents}


# ---- static front-end ----
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
