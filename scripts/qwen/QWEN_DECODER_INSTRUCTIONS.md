# Qwen Decoder Experiments: Quick Guide

## Overview

This guide covers both:
- Qwen paraphrase baseline
- Qwen decoder-only LoRA fine-tuning

The fine-tuning commands below are written to reproduce the **best Qwen decoder fine-tuning result** used in our experiments.

**Best fine-tuned configuration:**
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Learning rate: `1e-4`
- Epochs: `4`
- Batch size: `1`
- Gradient accumulation steps: `8`
- LoRA: `r = 8`, `alpha = 16`, `dropout = 0.05`

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

CSV files should contain:
```
source, target, style_label, length_label, label
```

Default files:
- `data/classification_splits/train.csv`
- `data/classification_splits/val.csv`
- `data/classification_splits/test.csv`

---

## Part A: Qwen Baseline

### Run Baseline

```bash
python scripts/qwen/baseline/qwen_paraphrase_baseline.py \
  --test-file data/classification_splits/test.csv \
  --few-shot-file data/classification_splits/train.csv \
  --output-dir results/qwen_paraphrase_baseline \
  --modes zero_shot few_shot
```

### Few-shot Only

```bash
python scripts/qwen/baseline/qwen_paraphrase_baseline.py \
  --test-file data/classification_splits/test.csv \
  --few-shot-file data/classification_splits/train.csv \
  --output-dir results/qwen_paraphrase_baseline_fewshot \
  --modes few_shot \
  --examples-per-combo 3 \
  --max-few-shot-token-overlap 0.65
```

**Key Parameters:**
```
--test-file PATH                     Evaluation CSV
--few-shot-file PATH                 Source of few-shot examples
--output-dir PATH                    Output directory
--modes zero_shot few_shot           Baseline modes to run
--examples-per-combo INT             Few-shot examples per attribute combo
--max-few-shot-token-overlap FLOAT   Filter high-overlap examples
--limit INT                          Run a small subset for testing
```

---

## Part B: Qwen Decoder LoRA Fine-tuning

### Step 1: Fine-tune Qwen Decoder

Run the best-performing Qwen LoRA fine-tuning configuration:

```bash
python scripts/qwen/lora_finetune/train_qwen_decoder_lora.py \
  --train-file data/classification_splits/train.csv \
  --val-file data/classification_splits/val.csv \
  --model checkpoints/qwen2.5-1.5b-instruct \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --epochs 4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --checkpoint-dir results/qwen_decoder_lora/best/checkpoints
```

**Key Parameters:**
```
--train-file PATH                    Training CSV
--val-file PATH                      Validation CSV
--model NAME                         Qwen model name
--batch-size INT                     Per-step batch size
--gradient-accumulation-steps INT    Gradient accumulation
--learning-rate FLOAT                Learning rate
--epochs INT                         Training epochs
--lora-r INT                         LoRA rank
--lora-alpha INT                     LoRA alpha
--lora-dropout FLOAT                 LoRA dropout
--checkpoint-dir PATH                Output checkpoint directory
--all-labels                         Use all rows instead of only label=1 rows
```

**Output:**
```
results/qwen_decoder_lora/best/checkpoints/
|-- best_qwen_decoder_lora/
`-- training_history.csv
```

---

## Step 2: Run Inference

Generate predictions with the fine-tuned Qwen decoder:

```bash
python scripts/qwen/lora_finetune/inference_qwen_decoder_lora.py \
  --adapter-dir results/qwen_decoder_lora/best/checkpoints/best_qwen_decoder_lora \
  --source-file data/classification_splits/test.csv \
  --output-file results/qwen_decoder_lora_best/predictions.csv \
  --base-model checkpoints/qwen2.5-1.5b-instruct \
  --max-length 96
```

**Output:**
```
results/qwen_decoder_lora_best/
`-- predictions.csv
```

---

## Recommended Workflow

### Baseline
```bash
python scripts/qwen/baseline/qwen_paraphrase_baseline.py \
  --test-file data/classification_splits/test.csv \
  --few-shot-file data/classification_splits/train.csv \
  --output-dir results/qwen_paraphrase_baseline \
  --modes zero_shot few_shot
```

### Fine-tuning
```bash
# 1. Train
python scripts/qwen/lora_finetune/train_qwen_decoder_lora.py \
  --train-file data/classification_splits/train.csv \
  --val-file data/classification_splits/val.csv \
  --model checkpoints/qwen2.5-1.5b-instruct \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --epochs 4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --checkpoint-dir results/qwen_decoder_lora/best/checkpoints

# 2. Generate
python scripts/qwen/lora_finetune/inference_qwen_decoder_lora.py \
  --adapter-dir results/qwen_decoder_lora/best/checkpoints/best_qwen_decoder_lora \
  --source-file data/classification_splits/test.csv \
  --output-file results/qwen_decoder_lora_best/predictions.csv \
  --base-model checkpoints/qwen2.5-1.5b-instruct \
  --max-length 96
```

---

## Notes

- Qwen baseline and Qwen fine-tuning are separate experimental settings.
- Few-shot examples are drawn from `train.csv`.
- Relative paths assume execution from the project root.
