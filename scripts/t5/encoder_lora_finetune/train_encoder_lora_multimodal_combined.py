"""
Fine-tune T5 Encoder with LoRA using Multi-Modal Input (Text + Categorical Features).
COMBINED VERSION: Concatenates sentence_a and sentence_b with </s> separator

This script fine-tunes a T5 encoder with LoRA for duplicate detection using:
- Combined sentences (sentence_a </s> sentence_b) passed to single encoder
- Categorical modalities: style_label and length_label
- Concatenation-based fusion of sequence embedding + categorical modalities
- Binary classification output (is_duplicate)

The encoder is fine-tuned with LoRA (not end-to-end), preserving the ability to
later combine it with a custom-trained decoder.

Example Usage:

# Train with default settings
python scripts/encoder/train_encoder_lora_multimodal_combined.py

# Train with custom hyperparameters
python scripts/encoder/train_encoder_lora_multimodal_combined.py \
    --batch-size 16 --learning-rate 5e-4 --epochs 20 --save-every 2

# Use custom data files
python scripts/encoder/train_encoder_lora_multimodal_combined.py \
    --train-file data/my_train.csv --val-file data/my_val.csv

# Use a larger model
python scripts/encoder/train_encoder_lora_multimodal_combined.py \
    --model flan-t5-large
"""

import os
import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from collections import defaultdict

from transformers import T5EncoderModel, AutoTokenizer
from peft import get_peft_model, LoraConfig

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalSentencePairDataset(Dataset):
    """
    Dataset for dual-sentence pairs with categorical modalities.
    
    Expects CSV with columns:
    - sentence_a: First sentence
    - sentence_b: Second sentence
    - style_label: Categorical feature (e.g., CONSERVATIVE, CREATIVE)
    - length_label: Categorical feature (e.g., SAME, LONG, SHORT)
    - label: Binary label (0 or 1)
    """
    def __init__(self, sentence_a_list, sentence_b_list, style_labels, length_labels, 
                 binary_labels, tokenizer, style_label_vocab=None, length_label_vocab=None, 
                 max_length=256):
        self.sentence_a_list = sentence_a_list
        self.sentence_b_list = sentence_b_list
        self.style_labels = style_labels
        self.length_labels = length_labels
        self.binary_labels = binary_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create vocabulary mappings for categorical features
        self.style_label_vocab = style_label_vocab or self._create_vocab(style_labels)
        self.length_label_vocab = length_label_vocab or self._create_vocab(length_labels)

    def _create_vocab(self, labels):
        """Create a mapping from label to index"""
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.sentence_a_list)

    def __getitem__(self, idx):
        sentence_a = self.sentence_a_list[idx]
        sentence_b = self.sentence_b_list[idx]
        style_label = self.style_labels[idx]
        length_label = self.length_labels[idx]
        binary_label = self.binary_labels[idx]

        # Combine sentences with explicit task instruction and markers
        combined_text = f"paraphrase detection: sentence1: {sentence_a} </s> sentence2: {sentence_b}"
        
        # Encode combined text
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True  # Explicitly add special tokens (EOS, PAD, etc.)
        )

        # Encode categorical labels as indices
        style_idx = self.style_label_vocab[style_label]
        length_idx = self.length_label_vocab[length_label]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "style_label_idx": torch.tensor(style_idx, dtype=torch.long),
            "length_label_idx": torch.tensor(length_idx, dtype=torch.long),
            "label": torch.tensor(binary_label, dtype=torch.long)
        }


class LoRAMultiModalEncoder(nn.Module):
    """
    T5 Encoder with LoRA and multi-modal categorical features.
    
    Architecture:
    - T5 Encoder with LoRA (combined sequence: sentence_a </s> sentence_b)
    - Mean pooling over entire sequence
    - Categorical embeddings for style_label and length_label
    - Concatenation of sequence embedding + categorical embeddings
    - Classification head for duplicate detection
    """
    def __init__(self, model_path="checkpoints/flan-t5-small", num_classes=2, 
                 num_style_labels=2, num_length_labels=3, categorical_embed_dim=32,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, lora_target_modules=None):
        super().__init__()
        
        # Load base encoder
        self.encoder = T5EncoderModel.from_pretrained(model_path)
        
        # Apply LoRA configuration
        if lora_target_modules is None:
            lora_target_modules = ["q", "v"]
        logger.info("Applying LoRA configuration to T5 Encoder...")
        logger.info(f"LoRA Config - rank={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, target_modules={lora_target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / Total params: {total_params:,}")
        
        hidden_size = self.encoder.config.hidden_size
        
        # Categorical embeddings
        self.style_embedding = nn.Embedding(num_style_labels, categorical_embed_dim)
        self.length_embedding = nn.Embedding(num_length_labels, categorical_embed_dim)
        
        # Calculate classifier input size
        # = combined_emb (768) + style_emb (32) + length_emb (32)
        classifier_input_size = hidden_size + categorical_embed_dim * 2
        
        # Classification head (1 layer)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(classifier_input_size, num_classes)
        )
        
        logger.info(f"Classifier input size: {classifier_input_size}")

    def mean_pool(self, hidden_states, attention_mask):
        """Apply mean pooling with attention mask"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, style_label_idx, length_label_idx):
        """
        Forward pass combining text and categorical modalities.
        
        Args:
            input_ids: Input IDs for combined text (sentence_a </s> sentence_b)
            attention_mask: Attention mask for combined text
            style_label_idx: Style label indices
            length_label_idx: Length label indices
        
        Returns:
            logits: Classification logits [batch_size, 2]
        """
        # Encode combined text through the LoRA-enhanced encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling over sequence dimension
        embedding = self.mean_pool(outputs.last_hidden_state, attention_mask)
        
        # Categorical embeddings
        style_embedding = self.style_embedding(style_label_idx)
        length_embedding = self.length_embedding(length_label_idx)
        
        # Concatenate all modalities (text + categorical features)
        combined = torch.cat([
            embedding,             # (batch_size, 768)
            style_embedding,       # (batch_size, 32)
            length_embedding       # (batch_size, 32)
        ], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits


def load_multimodal_data(file_path):
    """
    Load data expecting columns: (sentence_a/source), (sentence_b/target), style_label, length_label, label
    Supports both naming conventions: sentence_a/sentence_b or source/target
    """
    df = pd.read_csv(file_path).dropna()
    
    # Check for sentence_a/sentence_b or source/target columns
    if 'sentence_a' in df.columns and 'sentence_b' in df.columns:
        sentence_a = df['sentence_a'].tolist()
        sentence_b = df['sentence_b'].tolist()
    elif 'source' in df.columns and 'target' in df.columns:
        sentence_a = df['source'].tolist()
        sentence_b = df['target'].tolist()
    else:
        available = list(df.columns)
        logger.warning(f"Available columns: {available}")
        raise ValueError("CSV must have either [sentence_a, sentence_b] or [source, target] columns")
    
    # Check for style_label and length_label
    required_cols = ['style_label', 'length_label', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        logger.warning(f"Available columns: {list(df.columns)}")
        raise ValueError(f"CSV must have columns: {required_cols}")
    
    style_labels = df['style_label'].tolist()
    length_labels = df['length_label'].tolist()
    labels = df['label'].astype(int).tolist()
    
    return sentence_a, sentence_b, style_labels, length_labels, labels


def train_encoder_lora(batch_size=16, epochs=10, learning_rate=1e-4, model_path="checkpoints/flan-t5-small",
                       train_file="data/classification_splits/train.csv", 
                       val_file="data/classification_splits/val.csv",
                       save_every=3, checkpoint_dir=None,
                       lora_r=8, lora_alpha=16, lora_dropout=0.1, lora_target_modules=None,
                       early_stopping_patience=5):
    """
    Fine-tune T5 encoder with LoRA using multi-modal input.
    
    Args:
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        model_path: Path to pretrained T5 model
        train_file: Path to training CSV file
        val_file: Path to validation CSV file
        save_every: Save checkpoint every N epochs
        checkpoint_dir: Output directory for checkpoints (auto-generated if None)
    """
    device = DEFAULT_DEVICE
    
    # Load data
    logger.info(f"Loading training data from {train_file}...")
    train_sent_a, train_sent_b, train_style, train_length, train_labels = load_multimodal_data(train_file)
    
    logger.info(f"Loading validation data from {val_file}...")
    val_sent_a, val_sent_b, val_style, val_length, val_labels = load_multimodal_data(val_file)
    
    logger.info(f"Train size: {len(train_sent_a)}, Val size: {len(val_sent_a)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure special tokens are configured
    # T5 uses </s> for EOS (end-of-sequence) by default
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer special tokens: EOS='{tokenizer.eos_token}', PAD='{tokenizer.pad_token}', UNK='{tokenizer.unk_token}'")
    
    # Create datasets (share vocabulary mappings)
    combined_style = train_style + val_style
    combined_length = train_length + val_length
    style_vocab = {label: idx for idx, label in enumerate(sorted(set(combined_style)))}
    length_vocab = {label: idx for idx, label in enumerate(sorted(set(combined_length)))}
    
    logger.info(f"Style labels: {style_vocab}")
    logger.info(f"Length labels: {length_vocab}")
    
    train_dataset = MultiModalSentencePairDataset(
        train_sent_a, train_sent_b, train_style, train_length, train_labels,
        tokenizer, style_vocab, length_vocab
    )
    val_dataset = MultiModalSentencePairDataset(
        val_sent_a, val_sent_b, val_style, val_length, val_labels,
        tokenizer, style_vocab, length_vocab
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LoRAMultiModalEncoder(
        model_path=model_path,
        num_style_labels=len(style_vocab),
        num_length_labels=len(length_vocab),
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, eps=1e-6)
    
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/encoder_lora_multimodal_combined_AdamW_{learning_rate:.0e}_ep{epochs}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info(f"Starting LoRA fine-tuning. Saving to {checkpoint_dir}")
    logger.info(f"Device: {device}")
    
    best_val_loss = float("inf")
    patience_counter = 0
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            style_idx = batch["style_label_idx"].to(device)
            length_idx = batch["length_label_idx"].to(device)
            batch_labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, style_idx, length_idx)
            
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
            
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            progress_bar.set_postfix({"loss": loss.item(), "acc": correct/total})
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                style_idx = batch["style_label_idx"].to(device)
                length_idx = batch["length_label_idx"].to(device)
                batch_labels = batch["label"].to(device)
                
                logits = model(input_ids, attention_mask, style_idx, length_idx)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Record history
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
            logger.info("Saving best model...")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            
            # Save LoRA weights separately for later inference
            model.encoder.save_pretrained(os.path.join(checkpoint_dir, "best_encoder_lora"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "best_encoder_lora"))
            
            # Save model config
            model_config = {
                'model_path': model_path,
                'num_style_labels': len(style_vocab),
                'num_length_labels': len(length_vocab),
                'style_vocab': style_vocab,
                'length_vocab': length_vocab,
                'categorical_embed_dim': 32
            }
            with open(os.path.join(checkpoint_dir, "model_config.json"), 'w') as f:
                json.dump(model_config, f, indent=2)
        else:
            # Validation loss did not improve
            patience_counter += 1
            logger.info(f"No improvement in val_loss. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered! Val loss did not improve for {early_stopping_patience} epochs.")
                break
        
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            epoch_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}")
            os.makedirs(epoch_checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(epoch_checkpoint_dir, "model.pt"))
            model.encoder.save_pretrained(os.path.join(epoch_checkpoint_dir, "encoder_lora"))
            logger.info(f"Saved epoch checkpoint: {epoch_checkpoint_dir}")
    
    # Save final training history
    with open(os.path.join(checkpoint_dir, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    logger.info(f"All checkpoints saved to: {os.path.abspath(checkpoint_dir)}")
    
    return checkpoint_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine-tune T5 Encoder with LoRA using Multi-Modal Input (Combined Version)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="flan-t5-small")
    parser.add_argument("--train-file", type=str, default="data/classification_splits/train.csv")
    parser.add_argument("--val-file", type=str, default="data/classification_splits/val.csv")
    parser.add_argument("--save-every", type=int, default=3, help="Save checkpoint every N epochs")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Output checkpoint directory")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q,v", help="Comma-separated target modules (e.g., 'q,v' or 'q,v,k')")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Stop training if val loss doesn't improve for N epochs")
    
    args = parser.parse_args()
    
    # Parse target modules from string
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")] if args.lora_target_modules else None
    
    train_encoder_lora(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_path=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        early_stopping_patience=args.early_stopping_patience
    )
