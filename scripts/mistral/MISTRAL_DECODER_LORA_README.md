# Mistral Decoder LoRA Fine-tuning Script

This script fine-tunes the Mistral decoder using LoRA for paraphrase generation. It uses a pre-trained encoder checkpoint (from the encoder fine-tuning), merges it with the base model, and freezes the encoder while training only the decoder with LoRA.

## Features

- **Encoder Reuse**: Loads fine-tuned encoder weights from checkpoint
- **Encoder Freezing**: First half of layers (encoder) are frozen, only second half (decoder) is trainable
- **LoRA Efficiency**: Only ~1% of decoder parameters trainable
- **Paraphrase Generation**: Generates paraphrases conditioned on source text
- **PyTorch Lightning**: Clean training loop with callbacks and logging
- **Loss Tracking**: CSV and TensorBoard logging for analysis

## Architecture

```
Input: "Paraphrase: [source text]"
  ↓
[Frozen Encoder Layers (L1-L16)]  ← From encoder checkpoint
  ↓
[Trainable Decoder Layers (L17-L32)] + LoRA  ← Trained
  ↓
Output: [paraphrase text]
```

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0.0
- transformers>=4.35.0
- pytorch-lightning>=2.0.0
- peft>=0.4.0
- pandas>=2.0.3

## Usage

### Basic Usage (with fine-tuned encoder)

First, make sure you have a fine-tuned encoder checkpoint from running `train_mistral_encoder_lora.py`:

```bash
python scripts/mistral/train_mistral_decoder_lora.py \
  --local_model_path checkpoints/mistral7binstruct \
  --encoder_checkpoint checkpoints/mistral_lora_encoder_custom/final_model \
  --train_path data/processed/train.csv \
  --val_path data/processed/val.csv \
  --output_dir checkpoints/mistral_decoder_lora_custom
```

### Without Encoder (Base Model Only)

If you don't have an encoder checkpoint, just omit the `--encoder_checkpoint` argument:

```bash
python scripts/mistral/train_mistral_decoder_lora.py \
  --local_model_path checkpoints/mistral7binstruct \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --max_epochs 15
```

### Custom Configuration

```bash
python scripts/mistral/train_mistral_decoder_lora.py \
  --local_model_path checkpoints/mistral7binstruct \
  --encoder_checkpoint checkpoints/mistral_lora_encoder_custom/final_model \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --max_epochs 20 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_source_length 512 \
  --max_target_length 512
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `mistralai/Mistral-7B-Instruct-v0.1` | Base model identifier (HuggingFace) |
| `--local_model_path` | `None` | Local path to base model checkpoint (overrides --model_name) |
| `--encoder_checkpoint` | `None` | Path to fine-tuned encoder LoRA checkpoint |
| `--train_path` | `data/processed/train.csv` | Path to training CSV |
| `--val_path` | `data/processed/val.csv` | Path to validation CSV |
| `--test_path` | `data/processed/test.csv` | Path to test CSV |
| `--output_dir` | `checkpoints/mistral_decoder_lora` | Output directory for checkpoints |
| `--batch_size` | `8` | Batch size for training |
| `--learning_rate` | `1e-4` | Learning rate |
| `--max_epochs` | `10` | Maximum number of epochs |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha scaling factor |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--max_source_length` | `256` | Maximum source sequence length |
| `--max_target_length` | `256` | Maximum target sequence length |
| `--seed` | `42` | Random seed for reproducibility |
| `--num_workers` | `4` | Number of workers for DataLoader |

## Data Format

CSV files must have the following columns:
- `source`: Source sentence to paraphrase
- `target`: Target paraphrase

Example:
```csv
source,target,...
"The cat sat on the mat","The feline rested on the carpet",...
"I like apples","Apples are what I enjoy",...
```

## Training Pipeline

### 1. **Encoder Loading**
- Loads base Mistral model
- If encoder checkpoint provided, merges encoder LoRA weights
- Freezes encoder parameters (first 16 layers)

### 2. **Decoder LoRA Configuration**
- Applies LoRA to q_proj and v_proj in decoder attention layers
- Only decoder is trained (last 16 layers)
- Trainable parameters: ~1-2% of decoder

### 3. **Generation Setup**
- Input: Prompt + source text ("Paraphrase: [source]")
- Target: Expected paraphrase text
- Training uses causal language modeling loss

### 4. **Callbacks**
- **ModelCheckpoint**: Saves best 3 models by validation loss
- **EarlyStopping**: Stops if validation loss doesn't improve for 3 epochs
- **LossTrackingCallback**: Saves metrics to CSV

## Output Structure

```
checkpoints/mistral_decoder_lora_custom/
├── checkpoints/
│   ├── best-epoch_01-val_loss_0.95.ckpt
│   ├── best-epoch_02-val_loss_0.87.ckpt
│   └── best-epoch_03-val_loss_0.84.ckpt
├── logs/
│   └── events files for TensorBoard
├── losses.csv
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── hparams.yaml
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir checkpoints/mistral_decoder_lora_custom/logs
```

View at: http://localhost:6006

### Plot Losses

```bash
python scripts/plot_losses.py \
  --losses_path checkpoints/mistral_decoder_lora_custom/losses.csv
```

## Performance Tuning

### For Better Quality
- Increase `--lora_r` to 16 or 32
- Decrease `--learning_rate` to 5e-5
- Increase `--max_epochs` to 20-30
- Use `--batch_size` 16 for more stable gradients

### For Memory Constraints
- Decrease `--batch_size` to 4
- Decrease `--max_source_length` and `--max_target_length` to 128
- Use `--lora_r` 4 or 8

### For Faster Training
- Increase `--learning_rate` to 2e-4
- Decrease `--max_epochs` to 5-10
- Use `--batch_size` 4 for speed

## Inference with Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1"
)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1"
)

# Load fine-tuned decoder adapters
model = PeftModel.from_pretrained(
    base_model,
    "checkpoints/mistral_decoder_lora_custom/final_model"
)

# Generate paraphrases
prompt = "Paraphrase: The quick brown fox jumps over the lazy dog"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(paraphrase)
```

## Technical Notes

### Encoder Freezing Strategy
- Splits 32-layer Mistral into two halves
- First 16 layers (encoder): Keep fine-tuned encoder weights, freeze parameters
- Last 16 layers (decoder): Add LoRA, train with task-specific loss

### Why Freeze Encoder?
- Preserves learned semantic representations from classification task
- Reduces training time and memory
- Prevents catastrophic forgetting of encoder knowledge
- Focuses adaptation on generation capability

### Loss Function
- Uses standard causal language modeling loss
- Targets: Expected paraphrase tokens
- Predictions: Model's next-token predictions during training

## Troubleshooting

### CUDA Out of Memory
```bash
python scripts/mistral/train_mistral_decoder_lora.py \
  --batch_size 4 \
  --max_source_length 128 \
  --max_target_length 128 \
  --lora_r 4
```

### Model Not Found
Ensure checkpoint paths are correct and readable:
```bash
ls checkpoints/mistral7binstruct
ls checkpoints/mistral_lora_encoder_custom/final_model
```

### Poor Convergence
Try different learning rates:
```bash
# More conservative
python scripts/mistral/train_mistral_decoder_lora.py --learning_rate 5e-5

# More aggressive
python scripts/mistral/train_mistral_decoder_lora.py --learning_rate 2e-4
```

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09714)
- [PEFT Library](https://github.com/huggingface/peft)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
