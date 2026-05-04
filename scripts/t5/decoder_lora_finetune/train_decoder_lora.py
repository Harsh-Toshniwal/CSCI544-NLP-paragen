"""
LoRA fine-tuning entry point for FLAN-T5 paraphrase generation.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL_ALIASES = {
    "google/flan-t5-large": PROJECT_ROOT / "checkpoints" / "flan-t5-large",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_model_name(model_name):
    aliases = {
        "flan-t5-small": "google/flan-t5-small",
        "flan-t5-base": "google/flan-t5-base",
        "flan-t5-large": "google/flan-t5-large",
    }
    return aliases.get(model_name, model_name)


def resolve_model_name(model_name):
    model_name = normalize_model_name(model_name)
    local_model = LOCAL_MODEL_ALIASES.get(model_name)
    if local_model and local_model.exists():
        logger.info("Using local T5 base model: %s", local_model)
        return str(local_model)
    return model_name


class ParaphraseSeq2SeqDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_length=256):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = f"paraphrase: {str(self.sources[idx])}"
        target = str(self.targets[idx])

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


def train_decoder_lora(
    batch_size=16,
    epochs=10,
    learning_rate=1e-4,
    model_path="flan-t5-small",
    train_file="data/classification_splits/train.csv",
    val_file="data/classification_splits/val.csv",
    checkpoint_dir=None,
    random_seed=42,
    max_length=256,
    positive_only=True,
):
    device = DEFAULT_DEVICE
    model_path = resolve_model_name(model_path)

    logger.info("Loading train data from %s", train_file)
    train_df = pd.read_csv(train_file).dropna()
    logger.info("Loading val data from %s", val_file)
    val_df = pd.read_csv(val_file).dropna()

    required_cols = {"source", "target"}
    if not required_cols.issubset(train_df.columns):
        raise ValueError("Decoder train CSV must contain 'source' and 'target' columns")
    if not required_cols.issubset(val_df.columns):
        raise ValueError("Decoder val CSV must contain 'source' and 'target' columns")

    if positive_only and "label" in train_df.columns:
        train_df = train_df[train_df["label"] == 1]
    if positive_only and "label" in val_df.columns:
        val_df = val_df[val_df["label"] == 1]

    train_src = train_df["source"].astype(str).tolist()
    train_tgt = train_df["target"].astype(str).tolist()
    val_src = val_df["source"].astype(str).tolist()
    val_tgt = val_df["target"].astype(str).tolist()
    logger.info("Train size: %s, Val size: %s", len(train_src), len(val_src))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    logger.info("Applying LoRA to decoder attention layers")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    train_dataset = ParaphraseSeq2SeqDataset(train_src, train_tgt, tokenizer, max_length=max_length)
    val_dataset = ParaphraseSeq2SeqDataset(val_src, val_tgt, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, eps=1e-6)

    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/decoder_lora_AdamW_{learning_rate:.0e}_ep{epochs}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    overlap_alpha = 0.5
    logger.info("Starting decoder LoRA fine-tuning -> %s", checkpoint_dir)

    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = outputs.loss

            vocab_size = outputs.logits.size(-1)
            src_multi_hot = torch.zeros(input_ids.size(0), vocab_size, device=device)
            src_multi_hot.scatter_(1, input_ids, 1.0)
            src_multi_hot[:, :3] = 0.0

            probs = torch.softmax(outputs.logits, dim=-1)
            overlap_prob = (probs * src_multi_hot.unsqueeze(1)).sum(dim=-1)
            valid_tgt_mask = (labels != -100).float()
            overlap_penalty = (overlap_prob * valid_tgt_mask).sum() / (valid_tgt_mask.sum() + 1e-8)

            loss = ce_loss + overlap_alpha * overlap_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info("Epoch %s - Val Loss: %.4f", epoch + 1, avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info("Saving best decoder adapter")
            model.save_pretrained(os.path.join(checkpoint_dir, "best_decoder_lora"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "best_decoder_lora"))

    return checkpoint_dir


def build_parser():
    parser = argparse.ArgumentParser("Train decoder-only LoRA for paraphrase generation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="flan-t5-small")
    parser.add_argument("--train-file", type=str, default="data/classification_splits/train.csv")
    parser.add_argument("--val-file", type=str, default="data/classification_splits/val.csv")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--all-labels", action="store_true", help="Use all rows instead of only label=1 rows")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    train_decoder_lora(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_path=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        checkpoint_dir=args.checkpoint_dir,
        random_seed=args.random_seed,
        max_length=args.max_length,
        positive_only=not args.all_labels,
    )


if __name__ == "__main__":
    main()
