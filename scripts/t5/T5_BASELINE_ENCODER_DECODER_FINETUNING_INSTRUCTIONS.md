# T5 Encoder LoRA Fine-tuning: Quick Guide

## Overview

Fine-tune T5 encoder with LoRA adapters for duplicate detection using combined sentences + categorical features (style_label, length_label).

**Key Features:**
- Parameter-efficient LoRA fine-tuning
- Multi-modal input (text + categorical)
- Early stopping and checkpoint management
- Supports FLAN-T5-Small/Base/Large

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

CSV files must have these columns:
```
sentence_a, sentence_b, style_label, length_label, label
```

Default locations:
- `data/classification_splits/train.csv`
- `data/classification_splits/val.csv`
- `data/classification_splits/test.csv`

---

## Step 1: Download Models

Download T5 models from HuggingFace to local checkpoints:

```bash
python -c "
from transformers import T5EncoderModel, AutoTokenizer

models = ['google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large']

for model_name in models:
    print(f'Downloading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    
    save_name = model_name.split('/')[-1]
    model.save_pretrained(f'./checkpoints/{save_name}')
    tokenizer.save_pretrained(f'./checkpoints/{save_name}')
    print(f'✓ Saved {save_name}')
"
```

Or use the notebook:
```bash
jupyter notebook download_model.ipynb
```

---

## Step 2: Run Vanilla T5 Baseline

Get baseline results using pretrained T5 without fine-tuning:

```bash
python script_old/baseline/generate_evaluate_vanilla_t5.py \
  --csv data/classification_splits/test.csv \
  --model flan-t5-base \
  --output-dir results/vanilla_t5_baseline
```

**Options:**
```bash
--model flan-t5-small          # Fast, low memory
--model flan-t5-base           # Balanced (default)
--model flan-t5-large          # Best quality

--local-model-path CHECKPOINT  # Use local checkpoint
--num-beams 5                  # Beam search width (default: 5)
--device cuda                  # Force device (cuda or cpu)
```

**Output:**
```
results/vanilla_t5_baseline/
├── predictions.json           # Generated paraphrases
└── evaluation.json            # BLEU, ROUGE-L, metrics
```

---

## Step 3: Fine-tune Encoder with LoRA

Fine-tune T5 encoder on classification task:

```bash
python scripts/encoder/train_encoder_lora_multimodal_combined.py \
  --model flan-t5-base \
  --train-file data/classification_splits/train.csv \
  --val-file data/classification_splits/val.csv \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --epochs 10
```

**Common Configurations:**

Fast Prototyping:
```bash
python scripts/encoder/train_encoder_lora_multimodal_combined.py \
  --model flan-t5-small \
  --batch-size 32 \
  --learning-rate 5e-4 \
  --epochs 5
```

High Quality:
```bash
python scripts/encoder/train_encoder_lora_multimodal_combined.py \
  --model flan-t5-large \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --epochs 20
```

**Key Parameters:**
```
--model NAME                    Base model (flan-t5-small/base/large)
--train-file PATH              Training CSV file
--val-file PATH                Validation CSV file
--batch-size INT               Batch size (default: 16)
--learning-rate FLOAT          Learning rate (default: 1e-4)
--epochs INT                   Training epochs (default: 10)
--lora-r INT                   LoRA rank (default: 8)
--lora-alpha INT               LoRA alpha (default: 16)
--checkpoint-dir PATH          Output directory for checkpoints
--early-stopping-patience INT  Stop if no improvement for N epochs (default: 3)
```

**Output:**
```
checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5/
├── best_model.pt
├── best_encoder_lora/          # LoRA weights
├── model_config.json           # Vocabulary and config
└── training_history.json       # Loss and accuracy curves
```

---

## Training Monitoring

The script outputs real-time progress:
```
Epoch 1/10 [Train]: 100%|████| 500/500 [loss: 0.652, acc: 0.721]
Epoch 1/10 [Val]: 100%|████| 100/100
Epoch 1 - Train Loss: 0.6523 | Train Acc: 0.7210 | Val Loss: 0.6148 | Val Acc: 0.7456
```

Analyze training history:
```python
import json

with open('checkpoints/.../training_history.json') as f:
    history = json.load(f)
    
print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
print(f"Best validation loss: {min(history['val_loss']):.4f}")
```

---

## Step 4: Generate Text with Merged Encoder + Decoder

After fine-tuning, merge the fine-tuned encoder with the base T5 decoder to generate paraphrases.

**Important**: Only the encoder is fine-tuned with LoRA. The decoder remains the default (unchanged) from the base model.

### Generate from Single Text

```bash
python scripts/merge_encoder_lora_decoder.py \
  --checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
  --input-text "The quick brown fox jumps over the lazy dog" \
  --output-file results/generated_paraphrase.txt
```

### Generate from CSV File

```bash
python scripts/merge_encoder_lora_decoder.py \
  --checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
  --input-csv data/classification_splits/test.csv \
  --output-file results/test_generated.csv \
  --max-length 150 \
  --num-beams 5
```

**Key Parameters:**
```
--checkpoint PATH           Path to fine-tuned encoder checkpoint
--input-text TEXT          Single text to generate from
--input-csv PATH           CSV file with sentences to generate
--output-file PATH         Output file path (txt or csv)
--max-length INT           Max length of generated text (default: 128)
--num-beams INT            Beam search width (default: 4)
--temperature FLOAT        Sampling temperature (default: 1.0)
--task TEXT                Generation task prefix (default: "paraphrase")
--device DEVICE            Device to use (cuda or cpu)
```

**Architecture:**
- **Encoder**: Fine-tuned with LoRA weights ✓
- **Decoder**: Default unchanged from base T5 model

---

## Step 5: Fine-tune Decoder with Frozen Encoder

After generating with the base decoder, optionally fine-tune the decoder for better paraphrase generation while keeping the encoder frozen.

### Basic Training

```bash
python scripts/decoder/train_T5_decoder_with_frozen_encoder.py \
  --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_AdamW_1e-04_ep10 \
  --data-path sample_data/train.tsv \
  --num-epochs 5 \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --output-dir checkpoints/decoder_lora_with_frozen_encoder
```

### Custom Configuration

```bash
python scripts/decoder/train_T5_decoder_with_frozen_encoder.py \
  --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
  --data-path sample_data/train.tsv \
  --output-dir checkpoints/decoder_lora_frozen_encoder_v2 \
  --num-epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --max-source-length 256 \
  --max-target-length 256
```

**Key Parameters:**
```
--encoder-checkpoint PATH       Path to fine-tuned encoder checkpoint (required)
--data-path PATH               Training data TSV/CSV file (required)
--output-dir PATH              Output directory for checkpoints
--num-epochs INT               Number of training epochs (default: 5)
--batch-size INT               Batch size (default: 16)
--learning-rate FLOAT          Learning rate (default: 5e-4)
--lora-r INT                   LoRA rank for decoder (default: 8)
--lora-alpha INT               LoRA alpha (default: 16)
--lora-dropout FLOAT           LoRA dropout (default: 0.05)
--max-source-length INT        Max source length (default: 128)
--max-target-length INT        Max target length (default: 128)
--warmup-steps INT             Warmup steps (default: 500)
--weight-decay FLOAT           Weight decay (default: 0.01)
--train-split FLOAT            Train/val/test split (default: 0.8)
--val-split FLOAT              Validation split (default: 0.1)
```

**Output:**
```
checkpoints/decoder_lora_with_frozen_encoder/
├── best_model/                 # Best decoder LoRA weights
│   ├── adapter_config.json
│   └── adapter_model.bin
├── epoch_1/, epoch_2/, ...     # Checkpoint per epoch
├── config.json                 # Training config + encoder path
├── training_history.json       # Loss and metrics per epoch
└── test_results.json           # Final test set metrics
```

**What's Frozen:**
- Encoder remains completely frozen (no updates)
- Only decoder receives LoRA fine-tuning
- Encoder's LoRA weights stay unchanged

---

## Step 6: Inference with Frozen Encoder + Decoder LoRA

After fine-tuning the decoder, use both components together for paraphrase generation.

### Generate from Text

```bash
python scripts/decoder/infer_T5_decoder_frozen_encoder.py \
  --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_AdamW_1e-04_ep10 \
  --decoder-checkpoint checkpoints/decoder_lora_with_frozen_encoder/best_model \
  --input-texts "The quick brown fox jumps over the lazy dog"
```

### Compare Source vs Reference

```bash
python scripts/decoder/infer_T5_decoder_frozen_encoder.py \
  --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
  --decoder-checkpoint checkpoints/decoder_lora_with_frozen_encoder/best_model \
  --input-texts "The quick brown fox" "A fast brown fox" \
  --num-paraphrases 3
```

### Multiple Texts

```bash
python scripts/decoder/infer_T5_decoder_frozen_encoder.py \
  --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
  --decoder-checkpoint checkpoints/decoder_lora_with_frozen_encoder/best_model \
  --input-texts "Text 1" "Text 2" "Text 3" \
  --max-length 150 \
  --num-beams 5 \
  --num-paraphrases 2
```

**Key Parameters:**
```
--encoder-checkpoint PATH       Path to encoder checkpoint (required)
--decoder-checkpoint PATH       Path to decoder LoRA checkpoint (required)
--input-texts TEXTS            Input texts to generate paraphrases for (required)
--num-paraphrases INT          Paraphrases per input (default: 3)
--max-length INT               Max generated length (default: 128)
--num-beams INT                Beam search width (default: 5)
--device DEVICE                Device to use (cuda or cpu)
```

**Output:**
When 2 texts provided (source + reference):
```
Source: The quick brown fox
Reference: A fast brown fox

Generated 3 paraphrase(s):
  [1] A speedy brown fox leaps over the sleeping canine
      Similarity: 0.8234 | Diversity: 0.6512
  [2] The swift brown fox jumps across a lazy dog
      Similarity: 0.8512 | Diversity: 0.5234
  [3] A quick tan fox hops over the lazy hound
      Similarity: 0.8123 | Diversity: 0.7123

Average Similarity: 0.8290
Average Diversity: 0.6290
```

**Architecture:**
- **Encoder**: Fine-tuned with LoRA + frozen (no updates)
- **Decoder**: Fine-tuned with LoRA (trained in Step 5)

---

## Step 7: Evaluate Generation Results

After generating paraphrases, evaluate them using the `evaluate_paraphrase_outputs.py` script.

### Evaluate Generation Results

```bash
python scripts/evaluate_paraphrase_outputs.py \
  --predictions results/t5_large_2k_records/predictions.csv \
  --output-dir results/t5_large_2k_records/evaluation \
  --output-name "encoder_decoder_lora_eval" \
  --model-name "T5 Encoder+Decoder LoRA"
```

### Auto-detect Columns

The script auto-detects column names. These are recognized:
- **Source**: `source`, `sentence_a`, `input_text`, `original`
- **Generated**: `generated_output`, `generated`, `prediction`
- **Reference**: `target`, `sentence_b`, `reference`, `paraphrase`

```bash
# Simple auto-detection
python scripts/evaluate_paraphrase_outputs.py \
  --predictions results/generated_outputs.csv \
  --output-dir results/evaluation_outputs
```

### Custom Column Names

```bash
python scripts/evaluate_paraphrase_outputs.py \
  --predictions results/generated_outputs.csv \
  --output-dir results/evaluation_outputs \
  --output-name "my_evaluation" \
  --source-col sentence_a \
  --generated-col generated_output \
  --reference-col sentence_b \
  --model-name "Custom Model Name"
```

### Evaluate Without Reference (Diversity Only)

If you only have source and generated (no reference), the script evaluates diversity:

```bash
python scripts/evaluate_paraphrase_outputs.py \
  --predictions results/generated_no_reference.csv \
  --output-dir results/evaluation_outputs \
  --output-name "diversity_eval"
```

**Key Parameters:**
```
--predictions PATH             Input CSV or JSON file (required)
--output-dir PATH              Output directory for results
--output-name TEXT             Base name for output files
--model-name TEXT              Model name for reporting
--source-col TEXT              Source column name (auto-detected if omitted)
--generated-col TEXT           Generated column name (auto-detected if omitted)
--reference-col TEXT           Reference column name (optional)
--no-report                    Skip generating text report
```

### Output Files

The script generates 3 files:

1. **`evaluation_*.json`** - Full evaluation results with per-sample scores
2. **`predictions_*.json`** - Predictions with all metrics for each sample
3. **`report_*.txt`** - Human-readable report with summary and samples

**Example Report Content:**
```
PARAPHRASE EVALUATION REPORT
Model: T5 Encoder+Decoder LoRA
Number of samples: 500
Evaluation mode: Against reference paraphrases

SUMMARY METRICS
semantic_similarity.................................... 0.8234
inverse_bleu........................................... 0.5612
lexical_diversity...................................... 0.7234
fluency_score.......................................... 0.8912

DETAILED METRIC STATISTICS

semantic_similarity:
  Mean:   0.8234
  Std:    0.0856
  Min:    0.5234
  Max:    0.9876
  Median: 0.8456
```

### Example Workflow

```bash
# 1. Generate paraphrases (Step 6)
python scripts/decoder/infer_T5_decoder_frozen_encoder.py \
  --encoder-checkpoint checkpoints/encoder_lora_multimodal_combined_16_epc_50_lr_5e_5 \
  --decoder-checkpoint checkpoints/decoder_lora_with_frozen_encoder/best_model \
  --input-csv data/classification_splits/test.csv \
  --output-file results/generated_test.csv

# 2. Evaluate results (Step 7)
python scripts/evaluate_paraphrase_outputs.py \
  --predictions results/generated_test.csv \
  --output-dir results/generated_test_evaluation \
  --output-name "test_set_evaluation" \
  --model-name "T5 Encoder+Decoder LoRA (Best Checkpoint)"

# 3. Review results
cat results/generated_test_evaluation/report_test_set_evaluation.txt
```

### Evaluation Metrics

The script computes:
- **Semantic Similarity**: Sentence-BERT similarity to reference (0-1, higher is better)
- **Inverse BLEU**: Lexical diversity measure (0-1, higher = more diverse)
- **Lexical Diversity**: Vocabulary diversity ratio
- **METEOR**: Meteor score for paraphrase quality
- **Fluency Score**: Language model based fluency

---

## Workflow Summary

```
Step 1: Download Models
    ↓
Step 2: Run Vanilla T5 Baseline (compare reference)
    ↓
Step 3: Fine-tune Encoder with LoRA (understand sentence pairs)
    ↓
Step 4: Generate with Encoder + Default Decoder (initial generation)
    ↓
Step 5: Fine-tune Decoder with Frozen Encoder (improve generation)
    ↓
Step 6: Inference with Both Components (final results)
    ↓
Step 7: Evaluate Generation Results (compute metrics & report)
```

---

## Quick Troubleshooting

**Out of Memory:**
```bash
--model flan-t5-small           # Use smaller model
--batch-size 8                  # Reduce batch size
--lora-r 4                      # Reduce LoRA rank
```

**Model Download Fails:**
```bash
# Pre-download using notebook first, then use local path
--model checkpoints/flan-t5-base
```

**Training Not Converging:**
```bash
--learning-rate 5e-5            # Lower learning rate
--epochs 20                     # More epochs
--batch-size 32                 # Larger batch size
```

**GPU Not Detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Script automatically falls back to CPU if GPU unavailable
```

---

## File Locations

```
scripts/encoder/train_encoder_lora_multimodal_combined.py    # Main script
data/classification_splits/                                   # Data directory
checkpoints/flan-t5-base/                                    # Model checkpoints
results/vanilla_t5_baseline/                                  # Baseline results
```

---

**Last Updated**: May 2026
