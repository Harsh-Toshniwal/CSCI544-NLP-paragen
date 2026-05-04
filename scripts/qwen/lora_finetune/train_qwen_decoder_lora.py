"""LoRA fine-tuning for Qwen decoder-only paraphrase generation."""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "checkpoints" / "qwen2.5-1.5b-instruct"


def normalize_model_name(model_name):
    aliases = {
        "qwen": DEFAULT_MODEL,
        "qwen2.5-1.5b": DEFAULT_MODEL,
        "qwen2.5-1.5b-instruct": DEFAULT_MODEL,
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    }
    return aliases.get(model_name.lower(), model_name)


def resolve_model_name(model_name):
    model_name = normalize_model_name(model_name)
    if model_name == DEFAULT_MODEL and DEFAULT_LOCAL_MODEL.exists():
        logger.info("Using local Qwen base model: %s", DEFAULT_LOCAL_MODEL)
        return str(DEFAULT_LOCAL_MODEL)
    return model_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text):
    return str(text).replace("\x01", " ").strip()


def build_instruction(style_label, length_label):
    return (
        "Rewrite the sentence as a paraphrase. Preserve the meaning, avoid copying the wording too closely, "
        "and return only one paraphrased sentence.\n"
        f"Style label: {style_label}\n"
        f"Target length label: {length_label}"
    )


def build_messages(source, target=None, style_label="unspecified", length_label="unspecified"):
    user_content = (
        f"{build_instruction(style_label, length_label)}\n\n"
        f"Sentence: {source}\n"
        "Paraphrase:"
    )
    messages = [
        {
            "role": "system",
            "content": "You are a careful paraphrase generator.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


class CausalParaphraseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source = clean_text(row["source"])
        target = clean_text(row["target"])
        style_label = clean_text(row["style_label"]) if "style_label" in row else "unspecified"
        length_label = clean_text(row["length_label"]) if "length_label" in row else "unspecified"

        prompt_messages = build_messages(source, None, style_label, length_label)
        full_messages = build_messages(source, target, style_label, length_label)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full_encoding["input_ids"].squeeze(0)
        attention_mask = full_encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = min(prompt_encoding["input_ids"].shape[-1], labels.shape[-1])
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_split(path, positive_only=True, limit=None):
    df = pd.read_csv(path).dropna(subset=["source", "target"])
    if positive_only and "label" in df.columns:
        df = df[df["label"] == 1]
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def train_qwen_decoder_lora(
    train_file="data/classification_splits/train.csv",
    val_file="data/classification_splits/val.csv",
    model_name=DEFAULT_MODEL,
    checkpoint_dir=None,
    batch_size=1,
    gradient_accumulation_steps=8,
    epochs=2,
    learning_rate=0.0001,
    max_length=512,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    positive_only=True,
    train_limit=None,
    val_limit=None,
    seed=42,
):
    set_seed(seed)
    model_name = resolve_model_name(model_name)
    device = torch.device(DEFAULT_DEVICE)

    logger.info("Loading train data from %s", train_file)
    train_df = load_split(train_file, positive_only=positive_only, limit=train_limit)
    logger.info("Loading val data from %s", val_file)
    val_df = load_split(val_file, positive_only=positive_only, limit=val_limit)
    logger.info("Train size: %s, Val size: %s", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    train_dataset = CausalParaphraseDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = CausalParaphraseDataset(val_df, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, eps=1e-6)

    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/qwen_decoder_lora_{learning_rate:.0e}_ep{epochs}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    global_step = 0

    logger.info("Starting Qwen decoder LoRA fine-tuning -> %s", checkpoint_dir)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(train_bar, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * gradient_accumulation_steps

        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "global_step": global_step,
            }
        )
        logger.info("Epoch %s - Train Loss: %.4f - Val Loss: %.4f", epoch + 1, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info("Saving best Qwen decoder adapter")
            best_dir = os.path.join(checkpoint_dir, "best_qwen_decoder_lora")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, "training_history.csv"), index=False)
    return checkpoint_dir


def build_parser():
    parser = argparse.ArgumentParser("Train Qwen decoder-only LoRA for paraphrase generation")
    parser.add_argument("--train-file", default="data/classification_splits/train.csv")
    parser.add_argument("--val-file", default="data/classification_splits/val.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--all-labels", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_qwen_decoder_lora(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        positive_only=not args.all_labels,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        seed=args.seed,
    )
