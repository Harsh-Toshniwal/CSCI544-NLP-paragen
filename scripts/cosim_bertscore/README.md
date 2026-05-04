# Semantic Similarity Evaluation (ParaGen Project)

This repository contains two scripts used to compute semantic similarity metrics for paraphrase evaluation:

- `new_cosim.py` → computes Sentence-BERT cosine similarity  
- `new_bertscore.py` → computes BERTScore (Precision, Recall, F1)

Both scripts run on the same dataset and generate row-level outputs and summary tables.

---

## Folder Structure

Make sure your directory looks like this:

project_cs544/
├── new_cosim.py
├── new_bertscore.py
└── balanced_label_no_lowercase/
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv

Each `.tsv` file must contain:

- sentence1
- sentence2
- label

---

### Create Environment

#### Using Conda
conda create -n paragen_env python=3.10

#### Using venv (Python built-in)
python -m venv venv

## Install Dependencies

pip install pandas numpy sentence-transformers scikit-learn bert-score torch

---

## Running Cosine Similarity

python new_cosim.py

### What it does:
- Computes Sentence-BERT embeddings (all-MiniLM-L6-v2)
- Computes cosine similarity for each sentence pair
- Generates summary statistics

### Outputs (inside balanced_label_no_lowercase/):
- train_with_similarity.tsv
- dev_with_similarity.tsv
- test_with_similarity.tsv
- cosine_similarity_summary_by_label.tsv
- cosine_similarity_summary_overall.tsv

Main file used for report:
cosine_similarity_summary_by_label.tsv

---

## Running BERTScore

python new_bertscore.py

### What it does:
- Computes BERTScore Precision, Recall, and F1
- Evaluates semantic similarity at token level
- Generates summary statistics

### Outputs (inside balanced_label_no_lowercase/):
- train_with_bertscore.tsv
- dev_with_bertscore.tsv
- test_with_bertscore.tsv
- bertscore_summary_by_label.tsv
- bertscore_summary_overall.tsv

Main file used for analysis:
bertscore_summary_by_label.tsv

---

## Notes

### First Run
- Models will download automatically from Hugging Face
- This may take some time (~90MB for SBERT)

## Dataset Used

balanced_label_no_lowercase

---

## Interpretation

### Cosine Similarity
- Higher for paraphrases (label 1)
- Still high for non-paraphrases due to lexical overlap

### BERTScore F1
- Captures finer semantic differences
- Complements cosine similarity

---