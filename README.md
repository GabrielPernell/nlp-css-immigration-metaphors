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
| [scores/](scores/) | MLM output: one CSV per topic + run logs |
| [scores_topic8_pilot.csv](scores_topic8_pilot.csv) | Original 500-segment pilot output (kept as a smaller reference) |

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

## Run results

All ten CTM topics scored end-to-end on a single RTX 4070 Ti
(`bert-base-uncased`, batch size 64). Total: **273,475 segments processed,
354,600 mentions scored** in ~19 minutes of GPU time (across two runs).

| Topic | Theme | Segments | Mentions | Mentions / segment |
|---|---|---:|---:|---:|
| 0 | Chinese-era legal / citizenship | 30,974 | 44,168 | 1.43 |
| 1 | (procedural junk) | 24,663 | 21,074 | 0.85 |
| 2 | border / enforcement | 30,845 | 12,332 | **0.40** |
| 3 | refugees / asylum | 29,683 | 72,831 | **2.45** |
| 4 | heritage / contributions | 31,362 | 55,348 | 1.77 |
| 5 | welfare / services | 25,876 | 40,980 | 1.58 |
| 6 | DACA / Dreamers | 30,665 | 26,003 | 0.85 |
| 7 | (incoherent junk) | 34,647 | 14,522 | 0.42 |
| 8 | labor / agriculture | 26,951 | 25,652 | 0.95 |
| 9 | quotas / national origins | 24,844 | 41,690 | 1.68 |

### Early observations (pre-aggregation)

- **Mention density itself is a finding.** Topic 2 (border / enforcement) has
  the lowest mention density of any *substantive* topic — even lower than the
  procedural-junk topics. Border discourse evidently uses more abstract framing
  ("the border", "illegal immigration", "the law") rather than directly naming
  people. Topic 3 (refugees) is the opposite extreme at 2.45 mentions/segment
  — refugee discourse is heavily group-naming.
- **Junk topics (1, 7) produce mentions but at low density.** Useful as a
  baseline: any per-category metaphor scores from these topics should look
  near-uniform / low-signal. If a substantive topic's per-category profile
  doesn't look distinguishably different, that's a methodological warning.
- **Pilot validation** (Topic 8, 500 segments, 1,097 mentions): score
  distributions track intuition. Labor speeches show high signal on
  `machine` / `cargo` / `animal` / `invasion` (max scores 0.13–0.34) and
  near-zero signal on `flood_tide` / `vermin` / `disease` / `threat` (max
  ≤0.007). Top-scoring sentences contain a mixture of genuine metaphor
  ("the costs of *importing* Mexican [MASK]" → cargo) and BERT artifacts
  ("Mexican [MASK] must be *recruited*" → invasion via military
  vocabulary). Per-sentence scores are noisy by design; aggregate
  topic-level means are the right unit of analysis. See
  [scores_topic8_pilot.csv](scores_topic8_pilot.csv) for the full pilot
  output.
- **`invasion` and `threat` should stay separate categories.** On the
  Topic 8 pilot, the two are nearly uncorrelated (Pearson r = 0.07) and
  threat scores are ~100× smaller in magnitude than invasion scores.
  Combining the two would effectively reduce to just `invasion` and
  destroy any threat-specific signal in topics where it might fire (e.g.
  Topic 5 / welfare's "burden" language).

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
