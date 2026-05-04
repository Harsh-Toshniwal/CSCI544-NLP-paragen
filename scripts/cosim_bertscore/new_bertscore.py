"""
Compute BERTScore F1 for train/dev/test splits
from the balanced_label_no_lowercase dataset folder.

Folder layout expected:
.
â”œâ”€â”€ bertscore_report.py
â””â”€â”€ balanced_label_no_lowercase/
    â”œâ”€â”€ train.tsv
    â”œâ”€â”€ dev.tsv
    â””â”€â”€ test.tsv

Outputs inside balanced_label_no_lowercase/:
    train_with_bertscore.tsv
    dev_with_bertscore.tsv
    test_with_bertscore.tsv
    bertscore_summary_by_label.tsv
    bertscore_summary_overall.tsv
"""

from pathlib import Path
import pandas as pd
from bert_score import score

# â”€â”€ 1. Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_DIR = Path("balanced_label_no_lowercase")
SPLITS = ["train", "dev", "test"]

MODEL_TYPE = "distilbert-base-uncased"
LANG = "en"
BATCH_SIZE = 64
ROUND_DIGITS = 4
USE_IDF = False
RESCALE_WITH_BASELINE = True

summary_by_label_rows = []
summary_overall_rows = []

# â”€â”€ 2. Quality flag helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def quality_flag(score_value: float) -> str:
    if score_value >= 0.90:
        return "strong"
    elif score_value >= 0.80:
        return "moderate"
    return "drift"

# â”€â”€ 3. Process each split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for split in SPLITS:
    input_file = DATA_DIR / f"{split}.tsv"
    output_file = DATA_DIR / f"{split}_with_bertscore.tsv"

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

    candidates = df["sentence2"].tolist()
    references = df["sentence1"].tolist()

    # Compute BERTScore
    print(f"Computing BERTScore F1 using model: {MODEL_TYPE}")
    P, R, F1 = score(
        cands=candidates,
        refs=references,
        model_type=MODEL_TYPE,
        lang=LANG,
        batch_size=BATCH_SIZE,
        idf=USE_IDF,
        rescale_with_baseline=RESCALE_WITH_BASELINE,
        verbose=True
    )

    df["bertscore_precision"] = P.cpu().numpy().round(ROUND_DIGITS)
    df["bertscore_recall"] = R.cpu().numpy().round(ROUND_DIGITS)
    df["bertscore_f1"] = F1.cpu().numpy().round(ROUND_DIGITS)
    df["quality_flag"] = df["bertscore_f1"].apply(quality_flag)

    # Save enriched TSV
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved: {output_file}")

    # Overall split summary
    f1_scores = df["bertscore_f1"]
    summary_overall_rows.append({
        "split": split,
        "count": int(len(df)),
        "mean_bertscore_f1": round(f1_scores.mean(), ROUND_DIGITS),
        "standard_deviation": round(f1_scores.std(ddof=1), ROUND_DIGITS),
        "min_bertscore_f1": round(f1_scores.min(), ROUND_DIGITS),
        "max_bertscore_f1": round(f1_scores.max(), ROUND_DIGITS),
    })

    # Label-wise summary
    grouped = df.groupby("label")["bertscore_f1"]
    for label, group in grouped:
        summary_by_label_rows.append({
            "split": split,
            "label": int(label),
            "count": int(group.count()),
            "mean_bertscore_f1": round(group.mean(), ROUND_DIGITS),
            "standard_deviation": round(group.std(ddof=1), ROUND_DIGITS),
            "min_bertscore_f1": round(group.min(), ROUND_DIGITS),
            "max_bertscore_f1": round(group.max(), ROUND_DIGITS),
        })

    print("\nOverall BERTScore F1 stats:")
    print(f1_scores.describe().round(4).to_string())

    print("\nMean BERTScore F1 by label:")
    print(df.groupby("label")["bertscore_f1"].mean().round(4).to_string())

    print("\nQuality flag distribution:")
    print(df["quality_flag"].value_counts().to_string())

# â”€â”€ 4. Save summary tables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
summary_by_label_df = pd.DataFrame(summary_by_label_rows)
summary_overall_df = pd.DataFrame(summary_overall_rows)

summary_by_label_file = DATA_DIR / "bertscore_summary_by_label.tsv"
summary_overall_file = DATA_DIR / "bertscore_summary_overall.tsv"

summary_by_label_df.to_csv(summary_by_label_file, sep="\t", index=False)
summary_overall_df.to_csv(summary_overall_file, sep="\t", index=False)

print("\nSaved summary tables:")
print(f"  {summary_by_label_file}")
print(f"  {summary_overall_file}")

print("\nLabel-wise summary table:")
print(summary_by_label_df.to_string(index=False))

print("\nOverall split summary table:")
print(summary_overall_df.to_string(index=False))
