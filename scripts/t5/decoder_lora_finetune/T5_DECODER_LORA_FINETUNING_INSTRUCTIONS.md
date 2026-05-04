# T5 Decoder LoRA Fine-tuning: Quick Guide

## Overview

Fine-tune the decoder side of FLAN-T5 with LoRA for paraphrase generation using STYLE and LENGTH control attributes.

This guide is written to reproduce the **best T5 decoder fine-tuning result** used in our experiments.

**Best configuration:**
- Model: `flan-t5-large`
- Learning rate: `2e-5`
- Epochs: `8`
- Batch size: `2`
- Result summary: `Inverse BLEU = 0.3764`, `Semantic Similarity = 0.9896`

---

## Prerequisites

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## Data Format

CSV files must contain:
```
source, target, style_label, length_label, label
```

Default files:
- `data/classification_splits/train.csv`
- `data/classification_splits/val.csv`
- `data/classification_splits/test.csv`

By default, the decoder fine-tuning script uses only positive paraphrase pairs (`label = 1`).

---

## Step 1: Fine-tune T5 Decoder with LoRA

Run the best-performing T5 decoder LoRA configuration:

```bash
python scripts/t5_decoder/lora_finetune/train_decoder_lora.py \
  --model checkpoints/flan-t5-large \
  --train-file data/classification_splits/train.csv \
  --val-file data/classification_splits/val.csv \
  --batch-size 2 \
  --learning-rate 2e-5 \
  --epochs 8 \
  --checkpoint-dir results/t5_decoder_ft/best/checkpoints
```

**Key Parameters:**
```
--model NAME              Base model (flan-t5-small/base/large)
--train-file PATH         Training CSV file
--val-file PATH           Validation CSV file
--batch-size INT          Batch size
--learning-rate FLOAT     Learning rate
--epochs INT              Training epochs
--checkpoint-dir PATH     Output directory for LoRA checkpoint
--max-length INT          Maximum sequence length
--all-labels              Use all rows instead of only label=1 rows
```

**Output:**
```
results/t5_decoder_ft/best/checkpoints/
├── best_decoder_lora/
└── ...
```

---

## Step 2: Run Inference

Generate predictions with the fine-tuned T5 decoder:

```bash
python scripts/t5_decoder/lora_finetune/inference_decoder_lora.py \
  --lora-dir results/t5_decoder_ft/best/checkpoints/best_decoder_lora \
  --source-file data/classification_splits/test.csv \
  --output-file results/t5_decoder_lora_best/predictions.csv \
  --base-model checkpoints/flan-t5-large \
  --max-length 96
```

**Output:**
```
results/t5_decoder_lora_best/
└── predictions.csv
```

---

## Step 3: Evaluate Predictions

Evaluate the generated outputs:

```bash
python scripts/t5_decoder/lora_finetune/evaluate_decoder_predictions.py \
  --prediction-file results/t5_decoder_lora_best/predictions.csv \
  --output-file results/t5_decoder_lora_best/evaluation.json
```

**Output:**
```
results/t5_decoder_lora_best/
├── predictions.csv
└── evaluation.json
```

---

## Recommended Workflow

```bash
# 1. Train
python scripts/t5_decoder/lora_finetune/train_decoder_lora.py \
  --model checkpoints/flan-t5-large \
  --train-file data/classification_splits/train.csv \
  --val-file data/classification_splits/val.csv \
  --batch-size 2 \
  --learning-rate 2e-5 \
  --epochs 8 \
  --checkpoint-dir results/t5_decoder_ft/best/checkpoints

# 2. Generate predictions
python scripts/t5_decoder/lora_finetune/inference_decoder_lora.py \
  --lora-dir results/t5_decoder_ft/best/checkpoints/best_decoder_lora \
  --source-file data/classification_splits/test.csv \
  --output-file results/t5_decoder_lora_best/predictions.csv \
  --base-model checkpoints/flan-t5-large \
  --max-length 96

# 3. Evaluate
python scripts/t5_decoder/lora_finetune/evaluate_decoder_predictions.py \
  --prediction-file results/t5_decoder_lora_best/predictions.csv \
  --output-file results/t5_decoder_lora_best/evaluation.json
```

---

## Notes

- This README covers only the **decoder-only LoRA fine-tuning** line.
- The T5 baseline is already documented separately in the encoder-side workflow and is not repeated here.
- Relative paths assume execution from the project root.
