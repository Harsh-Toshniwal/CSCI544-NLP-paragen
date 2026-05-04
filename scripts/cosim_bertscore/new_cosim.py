"""
Compute Sentence-BERT cosine similarity for train/dev/test splits
from the balanced_label_no_lowercase dataset folder.

Folder layout expected:
.
â”œâ”€â”€ cosine_similarity_report.py
â””â”€â”€ balanced_label_no_lowercase/
    â”œâ”€â”€ train.tsv
    â”œâ”€â”€ dev.tsv
    â””â”€â”€ test.tsv

Outputs inside balanced_label_no_lowercase/:
    train_with_similarity.tsv
    dev_with_similarity.tsv
    test_with_similarity.tsv
    cosine_similarity_summary_by_label.tsv
    cosine_similarity_summary_overall.tsv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# â”€â”€ 1. Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_DIR = Path("balanced_label_no_lowercase")
SPLITS = ["train", "dev", "test"]

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
ROUND_DIGITS = 4
SAVE_EMBEDDINGS = False  # set to True only if you want .npy files too

# â”€â”€ 2. Load model once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

summary_by_label_rows = []
summary_overall_rows = []

# â”€â”€ 3. Quality flag helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def quality_flag(score: float) -> str:
    if score >= 0.85:
        return "strong"
    elif score >= 0.75:
        return "moderate"
    return "drift"

# â”€â”€ 4. Process each split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for split in SPLITS:
    input_file = DATA_DIR / f"{split}.tsv"
    output_file = DATA_DIR / f"{split}_with_similarity.tsv"

    print(f"\nProcessing: {input_file}")
    df = pd.read_csv(input_file, sep="\t")

    required_cols = {"sentence1", "sentence2", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{input_file} is missing required columns: {missing}")

    # Clean text columns
    df["sentence1"] = df["sentence1"].fillna("").astype(str).str.strip()
    df["sentence2"] = df["sentence2"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Drop invalid labels if any
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    after = len(df)

    if before != after:
        print(f"Dropped {before - after} rows with invalid labels.")

    print(f"Rows: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    # Encode both columns
    embeddings1 = model.encode(
        df["sentence1"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings2 = model.encode(
        df["sentence2"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Row-wise cosine similarity
    # Since embeddings are normalized, cosine similarity = dot product
    similarities = np.sum(embeddings1 * embeddings2, axis=1)

    df["semantic_similarity"] = np.round(similarities, ROUND_DIGITS)
    df["quality_flag"] = df["semantic_similarity"].apply(quality_flag)

    # Save enriched TSV
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved: {output_file}")

    # Optional embedding save
    if SAVE_EMBEDDINGS:
        np.save(DATA_DIR / f"{split}_embeddings1.npy", embeddings1)
        np.save(DATA_DIR / f"{split}_embeddings2.npy", embeddings2)

    # Overall split summary
    sim = df["semantic_similarity"]
    summary_overall_rows.append({
        "split": split,
        "count": int(len(df)),
        "mean_cossim": round(sim.mean(), ROUND_DIGITS),
        "std_cossim": round(sim.std(ddof=1), ROUND_DIGITS),
        "min_cossim": round(sim.min(), ROUND_DIGITS),
        "max_cossim": round(sim.max(), ROUND_DIGITS),
    })

    # Label-wise summary
    grouped = df.groupby("label")["semantic_similarity"]
    for label, group in grouped:
        summary_by_label_rows.append({
            "split": split,
            "label": int(label),
            "count": int(group.count()),
            "mean_cossim": round(group.mean(), ROUND_DIGITS),
            "std_cossim": round(group.std(ddof=1), ROUND_DIGITS),
            "min_cossim": round(group.min(), ROUND_DIGITS),
            "max_cossim": round(group.max(), ROUND_DIGITS),
        })

    print("\nOverall similarity stats:")
    print(sim.describe().round(4).to_string())

    print("\nMean similarity by label:")
    print(df.groupby("label")["semantic_similarity"].mean().round(4).to_string())

    print("\nQuality flag distribution:")
    print(df["quality_flag"].value_counts().to_string())

# â”€â”€ 5. Save summary tables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
summary_by_label_df = pd.DataFrame(summary_by_label_rows)
summary_overall_df = pd.DataFrame(summary_overall_rows)

summary_by_label_file = DATA_DIR / "cosine_similarity_summary_by_label.tsv"
summary_overall_file = DATA_DIR / "cosine_similarity_summary_overall.tsv"

summary_by_label_df.to_csv(summary_by_label_file, sep="\t", index=False)
summary_overall_df.to_csv(summary_overall_file, sep="\t", index=False)

print("\nSaved summary tables:")
print(f"  {summary_by_label_file}")
print(f"  {summary_overall_file}")

print("\nLabel-wise summary table:")
print(summary_by_label_df.to_string(index=False))

print("\nOverall split summary table:")
print(summary_overall_df.to_string(index=False))
