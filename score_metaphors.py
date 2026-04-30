"""MLM metaphor scoring for immigration speeches.

For each segment in a topic CSV: split into sentences, find immigrant
mentions, mask each mention, run BERT MLM, and compute per-category
scores by summing P(mask = w) over each category's word list.

Method follows Card et al. 2022 ("Computational analysis of 140 years of
US political speeches…", PNAS) — reimplemented from scratch.

Usage:
    # Pilot run on 500 segments from topic 8
    python score_metaphors.py --topic 8 --pilot 500 --out scores_topic8_pilot.csv

    # Full run on a topic (may take hours on CPU)
    python score_metaphors.py --topic 8 --out scores_topic8.csv

    # Run on multiple topics
    python score_metaphors.py --topic 5 6 8 9 --out-dir scores/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from categories import METAPHOR_CATEGORIES, filter_categories_to_single_wordpiece
from mentions import build_mention_regex, find_masked_sentences


DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_TOPIC_DIR = Path(__file__).parent / "topic_speeches"
MAX_SEQ_LEN = 256  # truncate long sentences; mask token must be within this
TEXT_COLUMN = "text"


def load_model(model_name: str, device: torch.device):
    print(f"Loading {model_name} on {device}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def collect_mentions_for_topic(
    topic_csv: Path,
    pattern,
    mask_token: str,
    pilot: int | None = None,
) -> list[dict]:
    """Stream the topic CSV and return one record per mention found."""
    df = pd.read_csv(topic_csv, usecols=["segment_id", "speech_id", "date", "party", TEXT_COLUMN])
    if pilot is not None:
        df = df.head(pilot)
    print(f"  Read {len(df)} segments from {topic_csv.name}", file=sys.stderr)

    records: list[dict] = []
    for row in df.itertuples(index=False):
        text = getattr(row, TEXT_COLUMN)
        if not isinstance(text, str) or not text.strip():
            continue
        mentions = find_masked_sentences(text, pattern, mask_token=mask_token)
        for sent_idx, m in enumerate(mentions):
            records.append({
                "segment_id": row.segment_id,
                "speech_id": row.speech_id,
                "date": row.date,
                "party": row.party,
                "sentence_idx": sent_idx,
                "mention_text": m.mention_text,
                "mention_type": m.mention_type,
                "masked_sentence": m.masked_sentence,
            })
    print(f"  Found {len(records)} mentions", file=sys.stderr)
    return records


@torch.no_grad()
def score_batch(
    masked_sentences: list[str],
    tokenizer,
    model,
    device,
    category_token_ids: dict[str, list[int]],
) -> list[dict[str, float]]:
    """Run BERT MLM on a batch of [MASK]-containing sentences and return
    a list of {category: score} dicts. Sentences with no [MASK] surviving
    truncation get NaN scores for every category."""
    enc = tokenizer(
        masked_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)
    logits = model(**enc).logits  # (B, T, V)
    probs = torch.softmax(logits, dim=-1)

    mask_id = tokenizer.mask_token_id
    out: list[dict[str, float]] = []
    for i in range(enc["input_ids"].shape[0]):
        mask_positions = (enc["input_ids"][i] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            out.append({cat: float("nan") for cat in category_token_ids})
            continue
        # If multiple [MASK]s slipped in (shouldn't happen), use the first.
        pos = mask_positions[0].item()
        p = probs[i, pos]  # (V,)
        scores = {}
        for cat, ids in category_token_ids.items():
            if not ids:
                scores[cat] = 0.0
            else:
                scores[cat] = float(p[ids].sum().item())
        out.append(scores)
    return out


def score_records(
    records: list[dict],
    tokenizer,
    model,
    device,
    category_token_ids: dict[str, list[int]],
    batch_size: int,
) -> list[dict]:
    if not records:
        return []
    enriched = []
    for start in tqdm(range(0, len(records), batch_size), desc="  scoring"):
        batch = records[start:start + batch_size]
        sentences = [r["masked_sentence"] for r in batch]
        scores = score_batch(sentences, tokenizer, model, device, category_token_ids)
        for r, s in zip(batch, scores):
            enriched.append({**r, **s})
    return enriched


def write_csv(records: list[dict], out_path: Path, categories: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "segment_id", "speech_id", "date", "party",
        "sentence_idx", "mention_text", "mention_type", "masked_sentence",
    ]
    cols = base_cols + categories
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in cols})
    print(f"  Wrote {len(records)} rows to {out_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", type=int, nargs="+", required=True,
                   help="Topic id(s) to score (e.g. --topic 8 or --topic 5 6 8 9)")
    p.add_argument("--topic-dir", type=Path, default=DEFAULT_TOPIC_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path (only valid when --topic has one id)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir; one CSV per topic written here")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--pilot", type=int, default=None,
                   help="If set, only score the first N segments per topic")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps. Default: cuda if available else cpu.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out is not None and len(args.topic) > 1:
        sys.exit("--out only works with a single --topic; use --out-dir for multiple.")
    if args.out is None and args.out_dir is None:
        sys.exit("Provide --out (single topic) or --out-dir (one or more topics).")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer, model = load_model(args.model, device)
    pattern = build_mention_regex()
    filtered_words, category_token_ids = filter_categories_to_single_wordpiece(
        METAPHOR_CATEGORIES, tokenizer
    )
    categories = list(METAPHOR_CATEGORIES.keys())

    print("Category words kept after BERT-vocab filter:", file=sys.stderr)
    for cat in categories:
        print(f"  {cat:12s} ({len(filtered_words[cat])}): {filtered_words[cat]}", file=sys.stderr)

    for topic_id in args.topic:
        topic_csv = args.topic_dir / f"topic_{topic_id}.csv"
        if not topic_csv.exists():
            sys.exit(f"Missing input file: {topic_csv}")
        print(f"\nTopic {topic_id}: {topic_csv}", file=sys.stderr)

        records = collect_mentions_for_topic(
            topic_csv, pattern, tokenizer.mask_token, pilot=args.pilot
        )
        scored = score_records(records, tokenizer, model, device, category_token_ids, args.batch_size)

        out_path = args.out if args.out is not None else args.out_dir / f"scores_topic_{topic_id}.csv"
        write_csv(scored, out_path, categories)


if __name__ == "__main__":
    main()
