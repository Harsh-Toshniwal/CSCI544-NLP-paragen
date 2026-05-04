"""
Fine-tune Mistral encoder with LoRA for paraphrase classification using PyTorch Lightning.
"""

import argparse
import csv
import logging
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to HuggingFace identifier."""
    aliases = {
        "mistral": DEFAULT_MODEL,
        "mistral-7b": DEFAULT_MODEL,
        "mistral-7b-instruct": DEFAULT_MODEL,
    }
    return aliases.get(model_name.lower(), model_name)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text):
    """Clean text by removing special characters and extra whitespace."""
    return str(text).replace("\x01", " ").strip()


class LossTrackingCallback(pl.Callback):
    """Callback to track losses to a CSV file."""

    def __init__(self, output_path: Path):
        """
        Args:
            output_path: Path to save losses CSV
        """
        super().__init__()
        self.output_path = output_path
        self.losses_data = []
        self.current_epoch = 0

    def on_epoch_start(self, trainer, pl_module):
        """Called at epoch start."""
        self.current_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at end of training epoch."""
        if trainer.callback_metrics:
            metrics = {
                "epoch": trainer.current_epoch,
                "stage": "train",
                "loss": trainer.callback_metrics.get("train_loss", None),
                "acc": trainer.callback_metrics.get("train_acc", None),
            }
            self.losses_data.append(metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at end of validation epoch."""
        if trainer.callback_metrics:
            metrics = {
                "epoch": trainer.current_epoch,
                "stage": "val",
                "loss": trainer.callback_metrics.get("val_loss", None),
                "acc": trainer.callback_metrics.get("val_acc", None),
            }
            self.losses_data.append(metrics)

    def on_train_end(self, trainer, pl_module):
        """Called at end of training."""
        if self.losses_data:
            df = pd.DataFrame(self.losses_data)
            df.to_csv(self.output_path, index=False)
            logger.info(f"Loss tracking saved to {self.output_path}")


class ClassificationDataset(Dataset):
    """Dataset for paraphrase classification task with multimodal features."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        use_target: bool = True,
        style_to_idx: Optional[dict] = None,
        length_stats: Optional[dict] = None,
    ):
        """
        Args:
            df: DataFrame with 'source', 'target', 'label', 'length', and 'style' columns
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            use_target: Whether to include target text in input
            style_to_idx: Mapping of style values to indices (created if None)
            length_stats: Statistics for length normalization (mean, std)
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_target = use_target
        
        # Build style-to-index mapping if not provided
        if style_to_idx is None:
            unique_styles = df["style"].unique() if "style" in df.columns else ["unknown"]
            self.style_to_idx = {style: idx for idx, style in enumerate(unique_styles)}
        else:
            self.style_to_idx = style_to_idx
        
        # Compute length statistics if not provided
        if length_stats is None:
            if "length" in df.columns:
                lengths = df["length"].astype(float).values
                self.length_stats = {"mean": float(lengths.mean()), "std": float(lengths.std())}
            else:
                self.length_stats = {"mean": 0.0, "std": 1.0}
        else:
            self.length_stats = length_stats

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        source = clean_text(row["source"])
        label = int(row["label"])

        if self.use_target:
            target = clean_text(row["target"])
            text = f"Source: {source}\nTarget: {target}"
        else:
            text = source

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract length modality (normalize)
        length = float(row.get("length", 0))
        normalized_length = (length - self.length_stats["mean"]) / (self.length_stats["std"] + 1e-8)
        
        # Extract style modality (convert to index)
        style = str(row.get("style", "unknown"))
        style_idx = self.style_to_idx.get(style, 0)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(normalized_length, dtype=torch.float32),
            "style": torch.tensor(style_idx, dtype=torch.long),
        }


class MistralEncoderClassifier(pl.LightningModule):
    """PyTorch Lightning module for Mistral encoder classification with LoRA (multimodal)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        num_labels: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        warmup_steps: int = 500,
        max_epochs: int = 10,
        num_styles: int = 1,
        style_embedding_dim: int = 16,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of classification labels
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_epochs: Maximum training epochs
            num_styles: Number of unique style values
            style_embedding_dim: Dimension of style embeddings
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.num_styles = num_styles
        self.style_embedding_dim = style_embedding_dim

        # Load base model
        logger.info(f"Loading {model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Mistral attention layers
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()

        # Hidden size
        hidden_size = self.base_model.config.hidden_size
        
        # ===== Multimodal embeddings =====
        # Style embedding (categorical feature)
        self.style_embedding = torch.nn.Embedding(num_styles, style_embedding_dim)
        
        # Length projection (numeric feature -> 16 dims)
        self.length_projection = torch.nn.Linear(1, style_embedding_dim)
        
        # Combine modalities: text (hidden_size) + style (style_embedding_dim) + length (style_embedding_dim)
        combined_dim = hidden_size + style_embedding_dim + style_embedding_dim
        
        # Classification head
        self.classifier = torch.nn.Linear(combined_dim, num_labels)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        logger.info(f"Initialized multimodal classifier with combined_dim={combined_dim} (text={hidden_size} + style={style_embedding_dim} + length={style_embedding_dim})")

    def forward(self, input_ids, attention_mask, length=None, style=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (multimodal).

        Args:
            input_ids: Text token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            length: Normalized length values (batch_size,)
            style: Style indices (batch_size,)

        Returns:
            logits: Classification logits (batch_size, num_labels)
            last_hidden_state: Last layer hidden states (batch_size, seq_len, hidden_size)
        """
        # Text modality
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state of the first token (similar to [CLS])
        last_hidden_state = outputs.hidden_states[-1]
        cls_output = last_hidden_state[:, 0, :]  # Take first token (batch_size, hidden_size)
        
        # Style modality (embedding)
        if style is not None:
            style_emb = self.style_embedding(style)  # (batch_size, style_embedding_dim)
        else:
            style_emb = torch.zeros(cls_output.size(0), self.style_embedding_dim, device=cls_output.device)
        
        # Length modality (projection)
        if length is not None:
            length_emb = self.length_projection(length.unsqueeze(-1))  # (batch_size, 1) -> (batch_size, style_embedding_dim)
        else:
            length_emb = torch.zeros(cls_output.size(0), self.style_embedding_dim, device=cls_output.device)
        
        # Concatenate all modalities
        combined = torch.cat([cls_output, style_emb, length_emb], dim=1)  # (batch_size, combined_dim)
        
        logits = self.classifier(combined)
        return logits, last_hidden_state

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        length = batch.get("length")
        style = batch.get("style")

        logits, _ = self(input_ids, attention_mask, length=length, style=style)
        loss = self.loss_fn(logits, labels)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        """Validation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        length = batch.get("length")
        style = batch.get("style")

        logits, _ = self(input_ids, attention_mask, length=length, style=style)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return {"loss": loss, "acc": acc, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx) -> dict:
        """Test step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        length = batch.get("length")
        style = batch.get("style")

        logits, _ = self(input_ids, attention_mask, length=length, style=style)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {"loss": loss, "acc": acc, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class ClassificationDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for classification."""

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        tokenizer=None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 512,
        val_split: float = 0.1,
        use_target: bool = True,
    ):
        """
        Args:
            train_path: Path to training CSV
            val_path: Path to validation CSV (optional)
            test_path: Path to test CSV (optional)
            tokenizer: HuggingFace tokenizer
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            max_length: Maximum sequence length
            val_split: Validation split ratio if val_path is None
            use_target: Whether to include target text
        """
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.val_split = val_split
        self.use_target = use_target
        
        # Initialize multimodal feature tracking
        self.style_to_idx = None
        self.length_stats = None
        self.num_styles = 1

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.train_path)
            logger.info(f"Loaded {len(train_df)} training samples")
            
            # Build style mapping and length stats from training data
            if "style" in train_df.columns:
                unique_styles = train_df["style"].unique()
                self.style_to_idx = {style: idx for idx, style in enumerate(unique_styles)}
                self.num_styles = len(unique_styles)
                logger.info(f"Found {self.num_styles} unique styles: {unique_styles}")
            
            if "length" in train_df.columns:
                lengths = train_df["length"].astype(float).values
                self.length_stats = {"mean": float(lengths.mean()), "std": float(lengths.std())}
                logger.info(f"Length stats - mean: {self.length_stats['mean']:.2f}, std: {self.length_stats['std']:.2f}")

            if self.val_path and os.path.exists(self.val_path):
                val_df = pd.read_csv(self.val_path)
                logger.info(f"Loaded {len(val_df)} validation samples")
            else:
                # Split training data
                train_size = int(len(train_df) * (1 - self.val_split))
                train_df, val_df = random_split(
                    train_df,
                    [train_size, len(train_df) - train_size],
                    generator=torch.Generator().manual_seed(42),
                )
                train_df = pd.DataFrame(train_df)
                val_df = pd.DataFrame(val_df)
                logger.info(
                    f"Split into {len(train_df)} train and {len(val_df)} validation samples"
                )

            self.train_dataset = ClassificationDataset(
                train_df,
                self.tokenizer,
                max_length=self.max_length,
                use_target=self.use_target,
                style_to_idx=self.style_to_idx,
                length_stats=self.length_stats,
            )
            self.val_dataset = ClassificationDataset(
                val_df,
                self.tokenizer,
                max_length=self.max_length,
                use_target=self.use_target,
                style_to_idx=self.style_to_idx,
                length_stats=self.length_stats,
            )

        if stage == "test" or stage is None:
            if self.test_path and os.path.exists(self.test_path):
                test_df = pd.read_csv(self.test_path)
                logger.info(f"Loaded {len(test_df)} test samples")
                self.test_dataset = ClassificationDataset(
                    test_df,
                    self.tokenizer,
                    max_length=self.max_length,
                    use_target=self.use_target,
                    style_to_idx=self.style_to_idx,
                    length_stats=self.length_stats,
                )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral encoder with LoRA for classification"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral",
        help="Model name or HuggingFace ID",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to model checkpoint (overrides --model_name)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/classification_splits/train.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/classification_splits/val.csv",
        help="Path to validation CSV",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/classification_splits/test.csv",
        help="Path to test CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/mistral_encoder_lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--use_target",
        action="store_true",
        default=True,
        help="Whether to include target text",
    )
    parser.add_argument(
        "--no_use_target",
        action="store_false",
        dest="use_target",
        help="Do not include target text",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Optimize GPU matmul precision for Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        logger.info("Set float32 matmul precision to 'high' for GPU optimization")

    # Determine model path
    if args.local_model_path:
        model_name = args.local_model_path
        logger.info(f"Using local model path: {model_name}")
    else:
        model_name = normalize_model_name(args.model_name)
        logger.info(f"Using model: {model_name}")

    # Verify local model exists if specified
    if args.local_model_path and not os.path.exists(args.local_model_path):
        logger.error(f"Local model path does not exist: {args.local_model_path}")
        raise FileNotFoundError(f"Model not found at: {args.local_model_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data module
    data_module = ClassificationDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        use_target=args.use_target,
    )
    
    # Setup data module to compute style mappings and length stats
    data_module.setup(stage="fit")
    num_styles = data_module.num_styles
    logger.info(f"Number of unique styles: {num_styles}")

    # Create model (with num_styles from data module)
    model = MistralEncoderClassifier(
        model_name=model_name,
        num_labels=2,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        num_styles=num_styles,
        style_embedding_dim=16,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min",
    )

    loss_tracking_callback = LossTrackingCallback(output_dir / "losses.csv")

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="logs",
        version=None,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, loss_tracking_callback],
        logger=tb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Test
    logger.info("Starting testing...")
    trainer.test(model, data_module)

    # Save final model
    final_model_path = output_dir / "final_model"
    model.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
