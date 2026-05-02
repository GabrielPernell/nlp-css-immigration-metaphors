from __future__ import annotations

import json
from pathlib import Path

import nltk
import pandas as pd

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from nltk.corpus import stopwords as stop_words


def load_records(text_file: Path, text_col: str = "text") -> list[dict]:
    records = []
    with text_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    text = row.get(text_col, "").strip()
                    if text:
                        row["raw_text"] = text
                        records.append(row)
                except json.JSONDecodeError:
                    print(f"Skipping line: {line[:100]}")
    return records


def get_stopwords() -> list[str]:
    nltk.download("stopwords")
    stopwords = list(stop_words.words("english"))
    custom_stopwords = ["mr", "madam", "speaker", "senate", "senator", "house", "committee", "subcommittee", "chairman", "chair", "record", "yield", "bill", "act", "section", "states", "united", "government", "resolution", "unanimous", "consent", "printed", "report", "page", "shall", "may", "upon"]
    return stopwords + custom_stopwords


def save_topics(topics: list[list[str]], output_dir: Path) -> None:
    rows = []
    for topic_id, words in enumerate(topics):
        rows.append({"topic_id": topic_id, "top_words": ", ".join(words)})
    pd.DataFrame(rows).to_csv(output_dir / "ctm_topics.csv", index=False)
    with (output_dir / "ctm_topics.json").open("w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)


def save_topic_speeches(
    ctm: CombinedTM,
    training_dataset,
    retained_records: list[dict],
    topics: list[list[str]],
    output_dir: Path,
    top_docs_per_topic: int,
) -> None:
    topic_predictions = ctm.get_thetas(training_dataset, n_samples=5)
    metadata_df = pd.DataFrame(retained_records)
    topic_cols = [f"topic_{i}_prob" for i in range(topic_predictions.shape[1])]
    topic_prob_df = pd.DataFrame(topic_predictions, columns=topic_cols)
    doc_topics_df = pd.concat([metadata_df.reset_index(drop=True), topic_prob_df], axis=1)
    doc_topics_df["dominant_topic"] = (
        topic_prob_df.idxmax(axis=1)
        .str.replace("topic_", "", regex=False)
        .str.replace("_prob", "", regex=False)
        .astype(int)
    )
    doc_topics_df["dominant_topic_prob"] = topic_prob_df.max(axis=1)
    doc_topics_df.to_csv(output_dir / "all_speeches_with_topic_probs.csv", index=False)
    topic_dir = output_dir / "topic_speeches"
    topic_dir.mkdir(parents=True, exist_ok=True)
    for topic_id, topic_words in enumerate(topics):
        topic_df = doc_topics_df[doc_topics_df["dominant_topic"] == topic_id].copy()
        topic_df["topic_id"] = topic_id
        topic_df["topic_words"] = ", ".join(topic_words)
        topic_df["topic_probability"] = topic_df["dominant_topic_prob"]
        topic_df = topic_df.sort_values("topic_probability", ascending=False)

        preferred_cols = [
            "topic_id",
            "topic_words",
            "topic_probability",
            "dominant_topic",
            "dominant_topic_prob",
            "speech_id",
            "segment_id",
            "date",
            "chamber",
            "party",
            "state",
            "speaker",
            "speaker_id",
            "congress",
            "anti",
            "neutral",
            "pro",
            "raw_text",
            "text",
        ]

        cols_to_save = [col for col in preferred_cols if col in topic_df.columns]
        remaining_cols = [
            col for col in topic_df.columns
            if col not in cols_to_save and not col.endswith("_prob")
        ]
        cols_to_save = cols_to_save + remaining_cols
        topic_df[cols_to_save].to_csv(topic_dir / f"topic_{topic_id}.csv", index=False)



def main() -> None:
    TEXT_FILE = Path("./data/speeches/Congress/imm_segments_with_tone_and_metadata.jsonlist")
    TEXT_COL = "text"
    OUTPUT_DIR = Path("test_split_output")

    N_TOPICS = 10
    TOP_N_WORDS = 20
    EPOCHS = 10
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    CONTEXTUAL_SIZE = 768
    TOP_DOCS_PER_TOPIC = 100

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records(TEXT_FILE, text_col=TEXT_COL)
    documents = [record["raw_text"] for record in records]
    print(f"Loaded {len(documents):,} documents.")

    stopwords = get_stopwords()

    sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
    preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()

    retained_records = [records[i] for i in retained_indices]

    print(f"Documents retained after preprocessing: {len(preprocessed_documents):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    print("\nFirst two preprocessed documents:")
    print(preprocessed_documents[:2])

    tp = TopicModelDataPreparation(EMBEDDING_MODEL)
    training_dataset = tp.fit(
        text_for_contextual=unpreprocessed_corpus,
        text_for_bow=preprocessed_documents,
    )

    print("\nFirst ten vocabulary words:")
    print(tp.vocab[:10])

    ctm = CombinedTM(
        bow_size=len(tp.vocab),
        contextual_size=CONTEXTUAL_SIZE,
        n_components=N_TOPICS,
        num_epochs=EPOCHS,
        num_data_loader_workers=0,
    )
    ctm.fit(training_dataset)

    print(f"Number of documents in training_dataset: {len(training_dataset)}")

    if len(training_dataset) > 0:
        sample = training_dataset[0]
        print(f"Sample keys: {sample.keys()}")

    topics = ctm.get_topic_lists(TOP_N_WORDS)

    print("\nRESULTING TOPICS!")
    for topic_id, topic_words in enumerate(topics):
        print(f"Topic {topic_id}: {', '.join(topic_words)}")

    save_topics(topics, OUTPUT_DIR)
    save_topic_speeches(
        ctm=ctm,
        training_dataset=training_dataset,
        retained_records=retained_records,
        topics=topics,
        output_dir=OUTPUT_DIR,
        top_docs_per_topic=TOP_DOCS_PER_TOPIC,
    )



if __name__ == "__main__":
    main()
