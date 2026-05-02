# Metaphors across topics in US Congressional Speech

Code accompanying our NLP for CSS final project. Per-subtopic analysis of
metaphorical framing in U.S. congressional immigration speeches, building on
the [Card et al. 2022](https://www.pnas.org/doi/10.1073/pnas.2120510119)
dataset and extending their masked-mention scoring approach to a per-topic
axis of comparison.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.7 (CUDA), and `transformers` 4.51.

## Data

Input data is not included due to size. To reproduce:

1. Obtain the Card et al. 2022 immigration-speeches dataset from
   [github.com/dallascard/us-immigration-speeches](https://github.com/dallascard/us-immigration-speeches).
2. Run `topic_model.py` to fit the CTM and produce per-topic CSVs in
   `topic_speeches/`.

## Pipeline

```bash
# 1. Score every immigrant-mention with bert-base-uncased
python score_metaphors.py --topic 0 1 2 3 4 5 6 7 8 9 --out-dir scores/

# 2. Aggregate to per-topic statistics + heatmap
python aggregate_scores.py --scores-dir scores/
```

Step 1 auto-detects CUDA. On an RTX 4070 Ti the full ten-topic run
(~270K segments, ~350K mentions) takes roughly 20 minutes.

## Files

| File | Purpose |
|---|---|
| `topic_model.py` | CTM training and per-topic data partitioning |
| `mentions.py` | Mention identification (direct / nationality / nationality+role) with head-noun-only masking |
| `categories.py` | Metaphor category word lists + WordPiece-vocab filter |
| `score_metaphors.py` | Per-mention BERT MLM scoring |
| `aggregate_scores.py` | Per-topic aggregation, log-ratios, permutation tests, heatmap |
| `ctm_topics.csv` / `.json` | Top-20 words per CTM topic |
| `requirements.txt` | Python dependencies |

## Output

`aggregate_scores.py` writes to `scores/`:

- `topic_summary.csv` — long-format means, log-ratios, BH-adjusted p-values
- `topic_means.csv` — wide-format mean matrix
- `topic_log_ratios.csv` — wide-format log-ratio matrix
- `heatmap.png` — heatmap visualization (Figure 1 of the report)

## Authors

Lily Wheeler, Gabriel Pernell — NLP for CSS, Spring 2026.
