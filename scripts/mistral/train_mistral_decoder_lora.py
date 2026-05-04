"""
Fine-tune Mistral decoder with LoRA for paraphrase generation using PyTorch Lightning.
Uses a fine-tuned encoder checkpoint and freezes it while training the decoder.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add scripts directory to path to import paraphrase_loss
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paraphrase_loss import HybridParaphraseLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


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

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at end of training epoch."""
        if trainer.callback_metrics:
            metrics = {
                "epoch": trainer.current_epoch,
                "stage": "train",
                "loss": trainer.callback_metrics.get("train_loss", None),
            }
            self.losses_data.append(metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at end of validation epoch."""
        if trainer.callback_metrics:
            metrics = {
                "epoch": trainer.current_epoch,
                "stage": "val",
                "loss": trainer.callback_metrics.get("val_loss", None),
            }
            self.losses_data.append(metrics)

    def on_train_end(self, trainer, pl_module):
        """Called at end of training."""
        if self.losses_data:
            df = pd.DataFrame(self.losses_data)
            df.to_csv(self.output_path, index=False)
            logger.info(f"Loss tracking saved to {self.output_path}")


class ParaphraseDataset(Dataset):
    """Dataset for paraphrase generation task."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256,
    ):
        """
        Args:
            df: DataFrame with 'source' and 'target' columns
            tokenizer: HuggingFace tokenizer
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        source = clean_text(row["source"])
        target = clean_text(row["target"])

        # Create prompt for paraphrase generation
        prompt = f"Paraphrase the following sentence:\n{source}\nParaphrase:"
        
        # Tokenize prompt (input)
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize full text with target (for generation)
        full_text = f"{prompt} {target}"
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_source_length + self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Calculate where the target starts (for loss computation)
        target_encoding = self.tokenizer(
            target,
            add_special_tokens=False,
            return_tensors="pt",
        )
        target_ids = target_encoding["input_ids"].squeeze(0)

        return {
            "input_ids": full_encoding["input_ids"].squeeze(0),
            "attention_mask": full_encoding["attention_mask"].squeeze(0),
            "prompt_length": prompt_encoding["input_ids"].shape[1],
            "target_ids": target_ids,
            "source_text": source,
        }


class MistralDecoderWithEncoderLora(pl.LightningModule):
    """PyTorch Lightning module for Mistral decoder with frozen encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        encoder_checkpoint: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 1e-4,
        max_epochs: int = 10,
    ):
        """
        Args:
            model_name: Base model identifier
            encoder_checkpoint: Path to fine-tuned encoder checkpoint (will be merged with base model)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout
            learning_rate: Learning rate
            max_epochs: Maximum training epochs
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.encoder_checkpoint = encoder_checkpoint

        # Load base model
        logger.info(f"Loading base model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # If encoder checkpoint is provided, merge the encoder weights
        if encoder_checkpoint:
            self._merge_encoder_weights(encoder_checkpoint)
            # Freeze encoder parameters
            self._freeze_encoder()

        # Configure LoRA - only on decoder attention layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Attention layers
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Initialize hybrid loss function
        self.loss_fn = HybridParaphraseLoss(
            use_semantic_loss=True,
            use_overlap_penalty=True,
            semantic_weight=0.2,
            overlap_weight=0.2,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info("Initialized HybridParaphraseLoss with semantic similarity and overlap penalty")

    def _merge_encoder_weights(self, encoder_checkpoint: str):
        """
        Merge encoder weights from checkpoint into the base model.
        Assumes the encoder uses LoRA adapters that can be loaded.
        """
        try:
            from peft import PeftModel
            
            logger.info(f"Loading encoder checkpoint: {encoder_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, encoder_checkpoint)
            
            # Merge encoder adapters into base model
            logger.info("Merging encoder adapters with base model...")
            self.model = self.model.merge_and_unload()
            
            logger.info("Encoder weights successfully merged")
        except Exception as e:
            logger.warning(f"Could not merge encoder weights: {e}")
            logger.info("Proceeding with base model only")

    def _freeze_encoder(self):
        """Freeze encoder parameters to only train decoder."""
        # For Mistral, encoder and decoder share the same transformer layers
        # This is a causal model, so we freeze the lower layers (encoder part)
        # and train only higher layers (decoder part)
        
        # Get model layers
        num_layers = self.model.config.num_hidden_layers
        freeze_until = num_layers // 2  # Freeze first half, train second half
        
        logger.info(f"Freezing first {freeze_until} layers (encoder), training last {num_layers - freeze_until} layers (decoder)")
        
        for i, layer in enumerate(self.model.model.layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """Forward pass - returns logits and outputs."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        
        # Compute hybrid loss
        losses = self.loss_fn(
            logits=outputs.logits,
            labels=batch["input_ids"],
            input_ids=batch["input_ids"],
            source_texts=batch.get("source_text"),
            tokenizer=self.tokenizer if hasattr(self, "tokenizer") else None,
        )
        
        # Log all loss components
        self.log("train_loss", losses["total_loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_clm_loss", losses["clm_loss"], on_step=False, on_epoch=True)
        self.log("train_overlap_penalty", losses["overlap_penalty"], on_step=False, on_epoch=True)
        if "semantic_loss" in losses:
            self.log("train_semantic_loss", losses["semantic_loss"], on_step=False, on_epoch=True)

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx) -> dict:
        """Validation step."""
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        
        # Compute hybrid loss
        losses = self.loss_fn(
            logits=outputs.logits,
            labels=batch["input_ids"],
            input_ids=batch["input_ids"],
            source_texts=batch.get("source_text"),
            tokenizer=self.tokenizer if hasattr(self, "tokenizer") else None,
        )
        
        # Log all loss components
        self.log("val_loss", losses["total_loss"], prog_bar=True)
        self.log("val_clm_loss", losses["clm_loss"])
        self.log("val_overlap_penalty", losses["overlap_penalty"])
        if "semantic_loss" in losses:
            self.log("val_semantic_loss", losses["semantic_loss"])

        return {"loss": losses["total_loss"]}

    def test_step(self, batch, batch_idx) -> dict:
        """Test step."""
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        
        # Compute hybrid loss
        losses = self.loss_fn(
            logits=outputs.logits,
            labels=batch["input_ids"],
            input_ids=batch["input_ids"],
            source_texts=batch.get("source_text"),
            tokenizer=self.tokenizer if hasattr(self, "tokenizer") else None,
        )
        
        # Log all loss components
        self.log("test_loss", losses["total_loss"])
        self.log("test_clm_loss", losses["clm_loss"])
        self.log("test_overlap_penalty", losses["overlap_penalty"])
        if "semantic_loss" in losses:
            self.log("test_semantic_loss", losses["semantic_loss"])

        return {"loss": losses["total_loss"]}

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


class ParaphraseDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for paraphrase generation."""

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        tokenizer=None,
        batch_size: int = 16,
        num_workers: int = 4,
        max_source_length: int = 256,
        max_target_length: int = 256,
        val_split: float = 0.1,
    ):
        """
        Args:
            train_path: Path to training CSV
            val_path: Path to validation CSV (optional)
            test_path: Path to test CSV (optional)
            tokenizer: HuggingFace tokenizer
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            val_split: Validation split ratio if val_path is None
        """
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            train_df = pd.read_csv(self.train_path)
            logger.info(f"Loaded {len(train_df)} training samples")

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

            self.train_dataset = ParaphraseDataset(
                train_df,
                self.tokenizer,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
            )
            self.val_dataset = ParaphraseDataset(
                val_df,
                self.tokenizer,
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
            )

        if stage == "test" or stage is None:
            if self.test_path and os.path.exists(self.test_path):
                test_df = pd.read_csv(self.test_path)
                logger.info(f"Loaded {len(test_df)} test samples")
                self.test_dataset = ParaphraseDataset(
                    test_df,
                    self.tokenizer,
                    max_source_length=self.max_source_length,
                    max_target_length=self.max_target_length,
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
        description="Fine-tune Mistral decoder with LoRA for paraphrase generation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Base model identifier",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to base model checkpoint",
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned encoder checkpoint (LoRA adapters)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/train.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/processed/val.csv",
        help="Path to validation CSV",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/processed/test.csv",
        help="Path to test CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/mistral_decoder_lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
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
        "--max_source_length",
        type=int,
        default=256,
        help="Maximum source sequence length",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="Maximum target sequence length",
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
        model_name = args.model_name
        logger.info(f"Using model: {model_name}")

    # Verify local model exists if specified
    if args.local_model_path and not os.path.exists(args.local_model_path):
        logger.error(f"Local model path does not exist: {args.local_model_path}")
        raise FileNotFoundError(f"Model not found at: {args.local_model_path}")

    # Verify encoder checkpoint exists if specified
    if args.encoder_checkpoint and not os.path.exists(args.encoder_checkpoint):
        logger.error(f"Encoder checkpoint path does not exist: {args.encoder_checkpoint}")
        raise FileNotFoundError(f"Encoder checkpoint not found at: {args.encoder_checkpoint}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Store tokenizer in module for loss computation
    # (Will be set on model after creation)

    # Create data module
    data_module = ParaphraseDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    # Create model
    model = MistralDecoderWithEncoderLora(
        model_name=model_name,
        encoder_checkpoint=args.encoder_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
    )
    
    # Attach tokenizer to model for loss computation
    model.tokenizer = tokenizer

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
