"""Baseline training script - vanilla T5 without controllable generation"""

import argparse
import logging
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

from paragen.config import Config, get_config
from paragen.data_loader import load_qqp, load_paws, ParaphraseDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig

# Auto-detect CUDA
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_local_pairs(sources_file: str, targets_file: str) -> list:
    """Load paraphrase pairs from local text files"""
    with open(sources_file, 'r', encoding='utf-8') as f:
        sources = [line.strip() for line in f if line.strip()]
    
    with open(targets_file, 'r', encoding='utf-8') as f:
        targets = [line.strip() for line in f if line.strip()]
    
    assert len(sources) == len(targets), f"Mismatch: {len(sources)} sources vs {len(targets)} targets"
    
    pairs = list(zip(sources, targets))
    logger.info(f"Loaded {len(pairs)} pairs from local files")
    return pairs


class VanillaT5Model:
    """Vanilla T5 without attribute tokens"""

    def __init__(self, model_name: str = "google/flan-t5-base", device: str = None, use_lora: bool = False):
        if device is None:
            device = DEFAULT_DEVICE
        self.device = device
        self.use_lora = use_lora
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        
        # Apply LoRA if requested
        if use_lora:
            self._apply_lora()
            logger.info(f"Loaded {model_name} with LoRA on {device}")
        else:
            logger.info(f"Loaded {model_name} on {device}")

    def _apply_lora(self):
        """Apply LoRA configuration to the decoder only"""
        lora_config = LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=16,
            target_modules=["q", "v"],  # Apply LoRA to query and value projections
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            modules_to_save=["lm_head"],  # Also train output layer
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Additionally freeze encoder to only train decoder
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        
        self.model.print_trainable_parameters()
        logger.info("LoRA applied to decoder + encoder frozen")

    def generate(self, source_text: str, num_beams: int = 5, max_length: int = 128):
        """Generate paraphrase without attribute tokens"""
        input_text = f"paraphrase: {source_text}"
        
        encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0]

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved model to {save_path}")

    def load(self, load_path: str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        logger.info(f"Loaded model from {load_path}")


class BaselineDataset(ParaphraseDataset):
    """Dataset without attribute tokens"""

    def __getitem__(self, idx: int):
        source, target = self.pairs[idx]

        # Build input WITHOUT attributes
        input_text = f"paraphrase: {source}"

        # Tokenize
        source_encoding = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "source": source,
            "target": target,
        }


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        optimizer.step()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, val_loader, device):
    """Validation"""
    model.model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def train_baseline(config: Config = None, use_lora: bool = True, use_full_data: bool = False):
    """Train vanilla T5 baseline (optionally with LoRA)"""
    if config is None:
        config = get_config()

    device = config.training.device
    
    # Create unique checkpoint directory based on hyperparameters
    lr = config.training.learning_rate
    optimizer_name = "AdamW"
    epochs = config.training.num_epochs
    lora_str = "lora" if use_lora else "full"
    data_str = "full" if use_full_data else "20pct"
    
    # Format learning rate for directory name
    lr_str = f"{lr:.0e}".replace("+", "").lower()  # e.g., 1e-04
    
    checkpoint_name = f"baseline_{optimizer_name}_{lr_str}_ep{epochs}_{lora_str}_{data_str}"
    baseline_dir = os.path.join(config.training.checkpoint_dir, checkpoint_name)
    os.makedirs(baseline_dir, exist_ok=True)
    
    logger.info(f"Checkpoint directory: {baseline_dir}")

    # Load data from local files
    logger.info("Loading local paraphrase data...")
    data_dir = config.data.data_dir
    
    # Use 20% subset by default for faster training
    suffix = "" if use_full_data else "_20pct"
    
    # Load training data from splits_no_test subdirectory
    train_pairs = load_local_pairs(
        os.path.join(data_dir, "splits_no_test", f"train_sources{suffix}.txt"),
        os.path.join(data_dir, "splits_no_test", f"train_targets{suffix}.txt")
    )
    
    # Load validation data
    val_pairs = load_local_pairs(
        os.path.join(data_dir, "splits_no_test", f"val_sources{suffix}.txt"),
        os.path.join(data_dir, "splits_no_test", f"val_targets{suffix}.txt")
    )
    
    data_type = "full data" if use_full_data else "20% subset"
    logger.info(f"Total train: {len(train_pairs)}, val: {len(val_pairs)} ({data_type})")

    # Create model (with LoRA if requested)
    logger.info(f"Loading model: {config.model.model_name}")
    
    # Try local checkpoint first, fallback to model name
    model_path = config.model.model_name
    if not model_path.startswith("./checkpoints") and not model_path.startswith("/"):
        # If not a local path, check if local checkpoint exists
        local_checkpoint = f"./checkpoints/{config.model.model_name}"
        if os.path.exists(local_checkpoint):
            model_path = local_checkpoint
            logger.info(f"Using local checkpoint: {model_path}")
    
    model = VanillaT5Model(
        model_name=model_path,
        device=device,
        use_lora=use_lora,
    )

    train_dataset = BaselineDataset(
        train_pairs,
        model.tokenizer,
        max_source_length=config.data.max_source_length,
        max_target_length=config.data.max_target_length,
        add_attributes=False,
    )

    val_dataset = BaselineDataset(
        val_pairs,
        model.tokenizer,
        max_source_length=config.data.max_source_length,
        max_target_length=config.data.max_target_length,
        add_attributes=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Setup AdamW optimizer (adaptive learning rate per parameter)
    optimizer = AdamW(model.model.parameters(), lr=config.training.learning_rate)
    
    # Use ReduceLROnPlateau for adaptive learning rate scheduling
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Reduce LR by 50% when no improvement
        patience=5,  # Wait 2 epochs without improvement before reducing
        eps=1e-6
    )
    logger.info(f"Optimizer: AdamW with ReduceLROnPlateau scheduler. LoRA enabled: {use_lora}")

    # Training loop
    best_val_loss = float("inf")
    train_history = {"loss": [], "val_loss": []}

    for epoch in range(config.training.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)

        train_history["loss"].append(train_loss)
        train_history["val_loss"].append(val_loss)
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(baseline_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)

            model.save(checkpoint_path)

            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "history": train_history,
                },
                os.path.join(checkpoint_path, "training_state.pt"),
            )

            logger.info(f"Saved baseline checkpoint to {checkpoint_path}")

    # Save final history
    with open(os.path.join(baseline_dir, "training_history.json"), "w") as f:
        json.dump(train_history, f, indent=2)
    
    # Save hyperparameters used for this run
    hyperparams = {
        "optimizer": "AdamW",
        "learning_rate": config.training.learning_rate,
        "momentum": None,  # Not used with AdamW
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        "num_epochs": config.training.num_epochs,
        "batch_size": config.data.batch_size,
        "use_lora": use_lora,
        "lora_rank": 8,
        "lora_alpha": 16,
        "use_full_data": use_full_data,
        "model_name": config.model.model_name,
        "warmup_steps": config.training.warmup_steps,
    }
    
    with open(os.path.join(baseline_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    logger.info("Baseline training completed!")
    logger.info(f"Best model saved to {os.path.join(baseline_dir, f'checkpoint_epoch_{epoch}')}")

    return baseline_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vanilla T5 baseline")
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to train on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model checkpoint path or name (default: uses config model_name or checkpoints/flan-t5-base)",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        default=False,
        help="Use full dataset instead of 20% subset (default: False)",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning (default: True)",
    )
    parser.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA and train full model",
    )

    args = parser.parse_args()

    config = get_config()
    config.training.device = args.device
    config.training.num_epochs = args.num_epochs
    
    # Override model if provided
    if args.model:
        config.model.model_name = args.model
    elif not os.path.exists(config.model.model_name) and os.path.exists("./checkpoints/flan-t5-base"):
        # Default to local flan-t5-base if available
        config.model.model_name = "./checkpoints/flan-t5-base"

    train_baseline(config=config, use_lora=args.use_lora, use_full_data=args.full_data)
