"""Aggregate per-mention BERT MLM metaphor scores to per-topic statistics.

Pipeline:
  1. Load all scores/scores_topic_*.csv into one DataFrame.
  2. Dedupe by masked_sentence (the pilot showed exact-duplicate sentences
     in the source corpus — same speech entered into the Record twice).
  3. Per-topic per-category mean score.
  4. Log-ratio of each topic's mean vs. the corpus mean (Card et al.'s
     reporting style).
  5. Permutation test: for each (topic, category) cell, test whether the
     topic's mean differs from the rest of the corpus. BH-corrected.
  6. Write topic_summary.csv (means + log-ratios), topic_pvalues.csv
     (raw + BH-adjusted p-values), and a heatmap PNG.

Usage:
    python aggregate_scores.py --scores-dir scores/ --n-perm 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


CATEGORIES = [
    "animal", "cargo", "disease", "flood_tide",
    "machine", "vermin", "invasion", "threat",
]

TOPIC_THEMES = {
    0: "Chinese-era legal",
    1: "procedural (junk)",
    2: "border / enforcement",
    3: "refugees / asylum",
    4: "heritage / contributions",
    5: "welfare / services",
    6: "DACA / Dreamers",
    7: "incoherent (junk)",
    8: "labor / agriculture",
    9: "quotas / national origins",
}


def load_all_scores(scores_dir: Path) -> pd.DataFrame:
    dfs = []
    for csv_path in sorted(scores_dir.glob("scores_topic_*.csv")):
        topic_id = int(csv_path.stem.split("_")[-1])
        df = pd.read_csv(csv_path)
        df["topic_id"] = topic_id
        dfs.append(df)
        print(f"  loaded topic {topic_id}: {len(df):,} rows from {csv_path.name}", file=sys.stderr)
    if not dfs:
        sys.exit(f"No scores_topic_*.csv found under {scores_dir}")
    out = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(out):,} mention rows across {len(dfs)} topics", file=sys.stderr)
    return out


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=["topic_id", "masked_sentence"]).reset_index(drop=True)
    print(f"Deduped {before:,} → {len(out):,} (dropped {before - len(out):,} exact-duplicate (topic, masked_sentence) pairs)", file=sys.stderr)
    return out


def compute_topic_means(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    topic_means = df.groupby("topic_id")[CATEGORIES].mean()
    corpus_mean = df[CATEGORIES].mean()
    return topic_means, corpus_mean


def log_ratio(topic_means: pd.DataFrame, corpus_mean: pd.Series) -> pd.DataFrame:
    """log(topic_mean / corpus_mean). Positive → topic uses this metaphor more
    than corpus average; negative → less."""
    eps = 1e-12  # avoid log(0)
    return np.log((topic_means + eps).divide(corpus_mean + eps, axis=1))


def permutation_test(df: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    """For each (topic, category) cell, compute a two-sided permutation p-value
    on the statistic mean(in_topic) - mean(out_of_topic).

    Returns DataFrame indexed by topic_id, columns = CATEGORIES, values = p-value.
    """
    rng = np.random.default_rng(seed)
    topic_ids = df["topic_id"].values
    scores = df[CATEGORIES].values  # (N, K)
    unique_topics = np.array(sorted(df["topic_id"].unique()))
    K = len(CATEGORIES)
    T = len(unique_topics)
    N = len(df)

    # Observed stat: mean_in - mean_out per (topic, category)
    observed = np.zeros((T, K))
    overall_sum = scores.sum(axis=0)
    for i, t in enumerate(unique_topics):
        in_mask = topic_ids == t
        n_in = in_mask.sum()
        n_out = N - n_in
        sum_in = scores[in_mask].sum(axis=0)
        sum_out = overall_sum - sum_in
        observed[i] = sum_in / n_in - sum_out / n_out

    # Permutation null
    extreme_count = np.zeros((T, K), dtype=np.int64)
    abs_observed = np.abs(observed)
    for _ in tqdm(range(n_perm), desc="permutations"):
        shuffled = rng.permutation(topic_ids)
        for i, t in enumerate(unique_topics):
            in_mask = shuffled == t
            n_in = in_mask.sum()
            n_out = N - n_in
            sum_in = scores[in_mask].sum(axis=0)
            sum_out = overall_sum - sum_in
            stat = sum_in / n_in - sum_out / n_out
            extreme_count[i] += np.abs(stat) >= abs_observed[i]

    # Add-one smoothing so we never report p=0
    p_values = (extreme_count + 1) / (n_perm + 1)
    return pd.DataFrame(p_values, index=unique_topics, columns=CATEGORIES)


def benjamini_hochberg(p_flat: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Standard Benjamini-Hochberg adjusted p-values (q-values).
    Input/output 1-D arrays of equal length."""
    n = len(p_flat)
    order = np.argsort(p_flat)
    ranks = np.argsort(order) + 1  # rank of each original p-value (1-indexed)
    adjusted = p_flat * n / ranks
    # Enforce monotonicity: walk down sorted p-values, taking running min
    sorted_p = p_flat[order]
    sorted_adj = sorted_p * n / np.arange(1, n + 1)
    sorted_adj = np.minimum.accumulate(sorted_adj[::-1])[::-1]
    sorted_adj = np.clip(sorted_adj, 0, 1)
    out = np.empty_like(sorted_adj)
    out[order] = sorted_adj
    return out


def adjust_pvalues(p_df: pd.DataFrame) -> pd.DataFrame:
    flat = p_df.values.flatten()
    adjusted_flat = benjamini_hochberg(flat)
    return pd.DataFrame(adjusted_flat.reshape(p_df.shape), index=p_df.index, columns=p_df.columns)


def heatmap(log_ratios: pd.DataFrame, p_adj: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of log-ratios with significance markers from BH-adjusted p-values.
    * = q<0.05, ** = q<0.01, *** = q<0.001."""
    annot = log_ratios.copy().round(2).astype(str)
    for t in log_ratios.index:
        for c in log_ratios.columns:
            q = p_adj.loc[t, c]
            star = ""
            if q < 0.001:
                star = "***"
            elif q < 0.01:
                star = "**"
            elif q < 0.05:
                star = "*"
            annot.loc[t, c] = f"{log_ratios.loc[t, c]:.2f}{star}"

    row_labels = [f"{t}: {TOPIC_THEMES.get(t, '?')}" for t in log_ratios.index]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.heatmap(
        log_ratios.values,
        annot=annot.values,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-2, vmax=2,
        xticklabels=log_ratios.columns,
        yticklabels=row_labels,
        cbar_kws={"label": "log(topic mean / corpus mean)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(
        "Per-topic metaphor usage vs. corpus mean\n"
        "(BH-adjusted permutation test: * q<0.05, ** q<0.01, *** q<0.001)",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("metaphor category")
    ax.set_ylabel("CTM topic")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", type=Path, default=Path("scores"))
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_all_scores(args.scores_dir)
    df = dedupe(df)

    topic_means, corpus_mean = compute_topic_means(df)
    log_ratios = log_ratio(topic_means, corpus_mean)
    n_per_topic = df.groupby("topic_id").size().rename("n_mentions")

    print(f"Running permutation test ({args.n_perm:,} perms × {len(topic_means)} topics × {len(CATEGORIES)} categories)...", file=sys.stderr)
    p_raw = permutation_test(df, n_perm=args.n_perm, seed=args.seed)
    p_adj = adjust_pvalues(p_raw)

    # Build a long-format summary
    summary_long = []
    for t in topic_means.index:
        for c in CATEGORIES:
            summary_long.append({
                "topic_id": t,
                "theme": TOPIC_THEMES.get(t, ""),
                "category": c,
                "n_mentions": int(n_per_topic[t]),
                "topic_mean": topic_means.loc[t, c],
                "corpus_mean": corpus_mean[c],
                "log_ratio": log_ratios.loc[t, c],
                "p_raw": p_raw.loc[t, c],
                "p_bh": p_adj.loc[t, c],
            })
    summary_df = pd.DataFrame(summary_long)

    # Write outputs
    out_dir = args.scores_dir
    summary_path = out_dir / "topic_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}", file=sys.stderr)

    pivot_means = topic_means.copy()
    pivot_means.index = [f"{t}: {TOPIC_THEMES.get(t, '?')}" for t in pivot_means.index]
    pivot_means.to_csv(out_dir / "topic_means.csv")

    pivot_logr = log_ratios.copy()
    pivot_logr.index = pivot_means.index
    pivot_logr.to_csv(out_dir / "topic_log_ratios.csv")
    print(f"  wrote {out_dir / 'topic_means.csv'}", file=sys.stderr)
    print(f"  wrote {out_dir / 'topic_log_ratios.csv'}", file=sys.stderr)

    heatmap(log_ratios, p_adj, out_dir / "heatmap.png")

    # Print a console summary
    print("\n=== Per-topic mean scores ===", file=sys.stderr)
    print(topic_means.round(5).to_string(), file=sys.stderr)
    print("\n=== Log-ratios vs. corpus mean (significance markers from BH-adjusted permutation test) ===", file=sys.stderr)
    annotated = log_ratios.copy().round(2).astype(str)
    for t in log_ratios.index:
        for c in log_ratios.columns:
            q = p_adj.loc[t, c]
            star = ""
            if q < 0.001:
                star = "***"
            elif q < 0.01:
                star = "**"
            elif q < 0.05:
                star = "*"
            annotated.loc[t, c] = f"{log_ratios.loc[t, c]:+.2f}{star}"
    annotated.index = [f"{t} ({TOPIC_THEMES.get(t, '?')[:18]:<18s})" for t in annotated.index]
    print(annotated.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
