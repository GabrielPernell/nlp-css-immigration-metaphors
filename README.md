# Immigration Metaphor Analysis (NLP for CSS Final Project)

Per-subtopic analysis of metaphorical framing in U.S. congressional speeches on
immigration. Builds on the dataset and dehumanization-detection method of
[Card et al. 2022 (PNAS)](https://www.pnas.org/doi/10.1073/pnas.2120510119).

## Project framing

Card et al. measure dehumanizing metaphor usage across **party** (Democrat vs.
Republican) and **nationality** (European vs. non-European) over the full
corpus of U.S. immigration speeches. We ask a different question: holding party
and nationality constant, **does metaphor usage vary across substantive
subtopics of immigration discourse** (labor, refugees, welfare, border
enforcement, etc.)?

The pipeline:

1. **Topic modeling** — a Contextualized Topic Model (CTM) is run over the
   immigration-speech corpus to recover ~10 latent subtopics. Each segment is
   labeled with its dominant topic.
2. **Mention identification** — for each segment, find references to immigrants
   (direct mentions, nationalities, nationality+role phrases).
3. **MLM metaphor scoring** — replace the mention with `[MASK]`, run BERT,
   score each metaphor category (animals, cargo, disease, flood/tide, machines,
   vermin, invasion, threat) by summing P(mask = w) over a curated word list.
4. **Per-topic comparison** — aggregate scores per topic, compute log-ratios
   vs. the corpus mean, test significance with permutation tests.

## Repo contents

| File | Purpose |
|---|---|
| [mentions.py](mentions.py) | Regex-based mention identification (direct / nationality / nationality+role) with single-token head-noun masking |
| [categories.py](categories.py) | Metaphor categories + word lists + WordPiece-vocab filter |
| [score_metaphors.py](score_metaphors.py) | Main MLM scoring script (BERT + GPU batching) |
| [requirements.txt](requirements.txt) | Python dependencies |
| [ctm_topics.csv](ctm_topics.csv) | Top-20 words per CTM topic |
| [ctm_topics.json](ctm_topics.json) | CTM topic metadata (JSON form) |
| [scores_topic8_pilot.csv](scores_topic8_pilot.csv) | Sample MLM output (Topic 8 pilot, 500 segments / 1097 mentions) |

## Setup

```bash
pip install -r requirements.txt
```

## Data

The data is **not committed** to the repo (~500MB+). To reproduce:

1. Download the Card et al. 2022 dataset from
   [github.com/dallascard/us-immigration-speeches](https://github.com/dallascard/us-immigration-speeches).
2. Run our CTM topic-modeling pipeline (see project notes — code lives in a
   separate notebook) to produce `all_speeches_with_topic_probs.csv` and the
   per-topic split files in `topic_speeches/topic_{0..9}.csv`.
3. Place those alongside the scripts.

## Running the MLM scorer

```bash
# Pilot on a small sample (recommended first run)
python score_metaphors.py --topic 8 --pilot 500 --out scores_topic8_pilot.csv

# Full run on a single topic
python score_metaphors.py --topic 8 --out scores/scores_topic_8.csv

# Multiple topics, one CSV per topic
python score_metaphors.py --topic 5 6 8 9 --out-dir scores/ --batch-size 64
```

The script auto-detects CUDA. On an RTX 4070 Ti, the 500-segment pilot takes
~3 seconds; full topic runs (50K-60K segments each) finish in a few minutes.

## Output schema

One row per (segment, sentence, mention). Columns:

- `segment_id`, `speech_id`, `date`, `party` — provenance from the source corpus
- `sentence_idx` — index of the sentence within the segment
- `mention_text`, `mention_type` (`direct` / `nationality` / `nat_role`)
- `masked_sentence` — the input fed to BERT
- `animal`, `cargo`, `disease`, `flood_tide`, `machine`, `vermin`,
  `invasion`, `threat` — per-category probability mass at the `[MASK]` slot

## Method notes

- **Single-token masking only.** For multi-token phrases like "Mexican
  laborers" we mask only the head noun (`Mexican [MASK]`), keeping the
  nationality as context. Replacing both tokens with one `[MASK]` would
  mismatch BERT's single-token mask slot.
- **WordPiece filtering.** Each category's word list is filtered at runtime
  to keep only terms that BERT tokenizes to a single piece — multi-piece
  words can't be cleanly scored at one mask position. (Card et al. do the
  same.)
- **Per-sentence scores are noisy.** The method is meaningful in aggregate,
  not at the individual-sentence level — Card et al. report a 0.73
  correlation between BERT scores and human annotation. Topic-level means
  with permutation tests are the right unit of analysis.

## Relation to Card et al. 2022

We **reimplement** their masked-mention scoring approach from scratch (no code
reuse from their repo). Differences:

- **New axis of comparison** — per-CTM-topic, rather than party or nationality.
- **Two new categories** — `invasion` (military metaphor: invader, army,
  attacker, troop) and `threat` (general danger: threat, menace, hazard,
  burden), in addition to their original six.
- **Topic coherence reporting** — NPMI / C_v scores for the CTM topics, which
  Card et al. don't compute (they don't run topic modeling).

## Authors

Gabe Pernell, Lily Wheeler — NLP for CSS, Spring 2026.
