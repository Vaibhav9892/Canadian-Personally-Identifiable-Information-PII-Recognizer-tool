#!/usr/bin/env python
"""
Convert token+BIO-tag JSONL datasets to spaCy v3 binary (.spacy) format.

Usage:
    python convert_to_spacy.py                        # uses defaults
    python convert_to_spacy.py \
        --input-dir synthetic_ontonotes5_canadian_sensitive_identifiers \
        --output-dir spacy_output
"""

import argparse
import json
from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin


# ── helpers ──────────────────────────────────────────────────────────────────

def load_label_map(path: Path) -> dict[int, str]:
    """Return {int_id: BIO_label_string} from label.json."""
    with open(path, encoding="utf-8") as f:
        str_to_id = json.load(f)
    return {v: k for k, v in str_to_id.items()}


def bio_tags_to_token_spans(bio_labels: list[str]):
    """
    Convert BIO label list into token-index entity spans.

    Yields (start_tok, end_tok, entity_label) where end_tok is exclusive,
    compatible with doc[start:end].
    """
    i = 0
    while i < len(bio_labels):
        label = bio_labels[i]
        if label.startswith("B-"):
            ent_label = label[2:]
            start = i
            i += 1
            while i < len(bio_labels) and bio_labels[i] == f"I-{ent_label}":
                i += 1
            yield (start, i, ent_label)
        else:
            i += 1


def convert_split(
    jsonl_path: Path,
    id_to_label: dict[int, str],
    nlp: spacy.Language,
) -> DocBin:
    """Read one JSONL split and return a populated DocBin."""
    doc_bin = DocBin()

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            tokens: list[str] = record["tokens"]
            tag_ids: list[int] = record["tags"]

            # map integer tag IDs → BIO strings
            bio_labels = [id_to_label[t] for t in tag_ids]

            # build Doc from pre-tokenized words (bypass spaCy tokenizer)
            spaces = [True] * len(tokens)
            if spaces:
                spaces[-1] = False
            doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

            # convert BIO → token-index spans → spaCy Span objects
            ents = []
            for start, end, ent_label in bio_tags_to_token_spans(bio_labels):
                span = doc[start:end]
                span = spacy.tokens.Span(doc, start, end, label=ent_label)
                ents.append(span)

            try:
                doc.ents = ents
            except ValueError:
                filtered = spacy.util.filter_spans(ents)
                doc.ents = filtered

            doc_bin.add(doc)

    return doc_bin


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert token+tag JSONL to spaCy binary format."
    )
    parser.add_argument(
        "--input-dir",
        default="synthetic_ontonotes5_canadian_sensitive_identifiers",
        type=Path,
        help=(
            "Directory with train.jsonl, validation.jsonl, test.jsonl, label.json "
            "(default: synthetic_ontonotes5_canadian_sensitive_identifiers)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="spacy_output",
        type=Path,
        help="Directory to write .spacy files into (default: spacy_output)",
    )
    parser.add_argument(
        "--lang", default="en", help="spaCy language code (default: en)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nlp = spacy.blank(args.lang)
    id_to_label = load_label_map(args.input_dir / "label.json")

    splits = ["train", "validation", "test"]

    for split in splits:
        jsonl_path = args.input_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"⏭  {jsonl_path} not found – skipping")
            continue

        print(f"Converting {split} …")
        doc_bin = convert_split(jsonl_path, id_to_label, nlp)

        out_path = args.output_dir / f"{split}.spacy"
        doc_bin.to_disk(out_path)
        print(f"  ✓ {len(doc_bin)} docs → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()