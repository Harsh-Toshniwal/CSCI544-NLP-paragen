# Mistral Encoder LoRA Fine-tuning Script

This script fine-tunes the Mistral encoder using LoRA (Low-Rank Adaptation) for paraphrase classification on the data in `data/classification_splits/`.

## Features

- **LoRA Efficiency**: Uses PEFT library for parameter-efficient fine-tuning (~1% of model parameters)
- **PyTorch Lightning**: Clean training loop with automatic mixed precision, distributed training support, and callbacks
- **Classification Head**: Adds a learnable classification head on top of the frozen encoder
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Loss Tracking**: Logs losses and metrics to CSV and TensorBoard for visualization
- **Flexible Input**: Supports using source text alone or source + target concatenation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - torch>=2.0.0
# - transformers>=4.35.0
# - pytorch-lightning>=2.0.0
# - peft>=0.4.0
# - pandas>=2.0.3
# - tqdm>=4.66.1
```

## Usage

### Basic Usage (with defaults)

```bash
python scripts/mistral/train_mistral_encoder_lora.py
```

This will:
- Load Mistral-7B-Instruct-v0.1 from HuggingFace
- Train on `data/classification_splits/train.csv`
- Validate on `data/classification_splits/val.csv`
- Test on `data/classification_splits/test.csv`
- Save checkpoints to `checkpoints/mistral_encoder_lora/`

### Using Local Model Checkpoint

To use a local model checkpoint instead of downloading from HuggingFace:

```bash
python scripts/mistral/train_mistral_encoder_lora.py \
  --local_model_path checkpoints/mistral7binstruct
```

### Custom Configuration

```bash
python scripts/mistral/train_mistral_encoder_lora.py \
  --local_model_path checkpoints/mistral7binstruct \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --max_epochs 15 \
  --lora_r 16 \
  --lora_alpha 32 \
  --output_dir checkpoints/mistral_lora_custom
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `mistral` | Model name or HuggingFace ID |
| `--local_model_path` | `None` | Local path to model checkpoint (overrides --model_name) |
| `--train_path` | `data/classification_splits/train.csv` | Path to training CSV |
| `--val_path` | `data/classification_splits/val.csv` | Path to validation CSV |
| `--test_path` | `data/classification_splits/test.csv` | Path to test CSV |
| `--output_dir` | `checkpoints/mistral_encoder_lora` | Output directory for checkpoints |
| `--batch_size` | `16` | Batch size for training |
| `--learning_rate` | `1e-4` | Learning rate |
| `--max_epochs` | `10` | Maximum number of epochs |
| `--warmup_steps` | `500` | Number of warmup steps |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha scaling factor |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--max_length` | `512` | Maximum sequence length |
| `--seed` | `42` | Random seed for reproducibility |
| `--num_workers` | `4` | Number of workers for DataLoader |
| `--use_target` | `True` | Whether to include target text in input |

## Data Format

The script expects CSV files with the following columns:
- `source`: Source text
- `target`: Target text (paraphrase)
- `label`: Binary label (0 = not_paraphrase, 1 = paraphrase)

Example:
```
source,target,label,...
"These canons were later approved","These canons were rejected later",0,...
"Later that night Teddy met Phil","Later that night Teddy joined Phil",1,...
```

## Architecture

### Model Components

1. **Mistral Base Model**: Causal language model from HuggingFace
2. **LoRA Layers**: Applied to attention projections (`q_proj`, `v_proj`)
3. **Classification Head**: Single linear layer on top of encoder embeddings
   - Input: Hidden state of first token (similar to [CLS])
   - Output: Linear projection to 2 classes (binary classification)

### Training Details

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW with cosine annealing scheduler
- **Device**: Automatically uses GPU if available
- **Precision**: float16 on GPU, float32 on CPU
- **Callbacks**: 
  - ModelCheckpoint: Saves best 3 models based on validation loss
  - EarlyStopping: Stops if validation loss doesn't improve for 3 epochs

## Output

After training, checkpoints are saved to:
```
checkpoints/mistral_encoder_lora/
├── checkpoints/
│   ├── best-epoch_01-val_loss_1.23.ckpt
│   ├── best-epoch_02-val_loss_1.15.ckpt
│   └── best-epoch_03-val_loss_1.12.ckpt
└── final_model/
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── config.json
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Monitoring Training

### TensorBoard Logs

View real-time training metrics with TensorBoard:

```bash
# Start TensorBoard (run from project root)
tensorboard --logdir checkpoints/mistral_encoder_lora/logs

# Open browser to http://localhost:6006
```

TensorBoard logs include:
- Training loss and accuracy (updated every 10 steps)
- Validation loss and accuracy (updated every epoch)
- Learning rate schedule
- Gradients and weights (if enabled)

### Loss CSV File

Losses are automatically saved to `checkpoints/mistral_encoder_lora/losses.csv` with columns:
- `epoch`: Epoch number
- `stage`: "train" or "val"
- `loss`: Loss value
- `acc`: Accuracy (if available)

### Plotting Losses

Generate loss plots from CSV file:

```bash
# Plot losses with defaults
python scripts/plot_losses.py

# Specify custom paths
python scripts/plot_losses.py \
  --losses_path checkpoints/mistral_encoder_lora/losses.csv \
  --output_dir checkpoints/mistral_encoder_lora

# Customize figure size
python scripts/plot_losses.py --figsize 16 6
```

This generates two plots:
- `losses_plot.png`: 2-panel plot showing loss and accuracy
- `losses_detailed.png`: Detailed loss plot with best epoch marker

Example output:
```
Loaded losses from checkpoints/mistral_encoder_lora/losses.csv
Shape: (20, 4)

============================================================
TRAINING STATISTICS
============================================================

Train Loss - Min: 0.4231, Max: 1.2341, Mean: 0.7821
Train Acc - Min: 0.6234, Max: 0.9102, Mean: 0.8456

Val Loss - Min: 0.3987, Max: 1.1234, Mean: 0.7234
Val Acc - Min: 0.6512, Max: 0.9234, Mean: 0.8678

============================================================
```

## Performance Notes

- **GPU Memory**: Requires ~20-24GB GPU memory (e.g., A100, A40, RTX A6000)
- **Training Time**: ~2-4 hours per epoch on single GPU
- **LoRA Benefits**: 
  - Trainable parameters: ~1.3M (vs 7B full model)
  - Memory savings: ~60% less than full fine-tuning
  - Can merge LoRA adapters with base model for inference

## Loading and Using Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "checkpoints/mistral_encoder_lora/final_model")

# Or merge adapters with base model for inference
merged_model = model.merge_and_unload()
merged_model.save_pretrained("checkpoints/mistral_encoder_lora/merged_model")
```

## Hyperparameter Tuning Tips

### For Better Accuracy
- Increase `--lora_r` (rank) to 16 or 32
- Decrease `--learning_rate` to 5e-5
- Increase `--max_epochs` to 15-20
- Use `--batch_size` 32 for more stable gradients

### For Faster Training
- Decrease `--lora_r` to 4 or 8
- Increase `--learning_rate` to 2e-4
- Decrease `--max_epochs` to 5-8
- Use `--batch_size` 8 (faster but noisier)

### For Memory Constraints
- Decrease `--batch_size` to 8
- Decrease `--max_length` to 256
- Use `--lora_dropout` 0.1 (more regularization, less memory)

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python scripts/train_mistral_encoder_lora.py --batch_size 8

# Or reduce sequence length
python scripts/train_mistral_encoder_lora.py --max_length 256

# Or use smaller LoRA rank
python scripts/train_mistral_encoder_lora.py --lora_r 4
```

### Slow Training on CPU
```bash
# Ensure GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If no GPU, model will still work but be slow (use for debugging only)
```

### Model Not Found
```bash
# Update transformers cache location
export HF_HOME=/path/to/cache
python scripts/train_mistral_encoder_lora.py
```

## References

- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- [LoRA Paper](https://arxiv.org/abs/2106.09714)
