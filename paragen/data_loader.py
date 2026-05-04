"""Data loading and preprocessing utilities"""

import os
from typing import Dict, Tuple, Optional
import pandas as pd
from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class ParaphraseDataset(Dataset):
    """Custom dataset for paraphrase pairs"""

    def __init__(
        self,
        pairs: list,
        tokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
        add_attributes: bool = True,
        attribute_cache: Optional[Dict] = None,
    ):
        """
        Args:
            pairs: List of (source, target) tuples
            tokenizer: Tokenizer to use
            max_source_length: Max source sequence length
            max_target_length: Max target sequence length
            add_attributes: Whether to add length/diversity attributes
            attribute_cache: Dict mapping pair indices to their attributes
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.add_attributes = add_attributes
        self.attribute_cache = attribute_cache or {}

    def __len__(self):
        return len(self.pairs)

    def _infer_attributes(self, source: str, target: str) -> Tuple[str, str]:
        """Infer length and diversity attributes from source-target pair"""
        source_len = len(source.split())
        target_len = len(target.split())

        # Determine length attribute
        if target_len < source_len * 0.8:
            length_attr = "[SHORT]"
        elif target_len > source_len * 1.2:
            length_attr = "[LONG]"
        else:
            length_attr = "[SAME]"

        # Determine diversity attribute based on token overlap
        source_tokens = set(source.lower().split())
        target_tokens = set(target.lower().split())
        overlap = len(source_tokens & target_tokens) / len(source_tokens | target_tokens)

        if overlap > 0.7:  # High overlap = conservative
            diversity_attr = "[CONSERVATIVE]"
        else:
            diversity_attr = "[CREATIVE]"

        return length_attr, diversity_attr

    def __getitem__(self, idx: int) -> Dict:
        source, target = self.pairs[idx]

        # Get or infer attributes
        if idx in self.attribute_cache:
            length_attr, diversity_attr = self.attribute_cache[idx]
        else:
            length_attr, diversity_attr = self._infer_attributes(source, target)

        # Build input with attributes
        if self.add_attributes:
            input_text = f"paraphrase: {length_attr} {diversity_attr} {source}"
        else:
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

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "source": source,
            "target": target,
            "length_attr": length_attr,
            "diversity_attr": diversity_attr,
        }


def load_qqp(data_dir: str = "./data") -> Tuple[list, list]:
    """Load Quora Question Pairs dataset"""
    logger.info("Loading QQP dataset...")
    dataset = load_dataset("qqp", cache_dir=data_dir)

    # Extract positive pairs (is_duplicate == 1)
    pairs = []

    for split in ["train", "validation"]:
        if split in dataset:
            data = dataset[split]
            for example in data:
                if example["is_duplicate"] == 1:  # Only semantic duplicates
                    if example["question1"] and example["question2"]:
                        pairs.append((example["question1"], example["question2"]))

    logger.info(f"Loaded {len(pairs)} QQP pairs")
    return pairs


def load_paws(data_dir: str = "./data") -> Tuple[list, list]:
    """Load PAWS dataset"""
    logger.info("Loading PAWS dataset...")
    dataset = load_dataset("paws", "labeled_final", cache_dir=data_dir)

    pairs = []

    for split in ["train", "validation"]:
        if split in dataset:
            data = dataset[split]
            for example in data:
                if example["label"] == 1:  # Only positive pairs
                    if example["sentence1"] and example["sentence2"]:
                        pairs.append((example["sentence1"], example["sentence2"]))

    logger.info(f"Loaded {len(pairs)} PAWS pairs")
    return pairs


def load_mrpc(data_dir: str = "./data") -> Tuple[list, list]:
    """Load MRPC evaluation dataset"""
    logger.info("Loading MRPC dataset...")
    dataset = load_dataset("glue", "mrpc", cache_dir=data_dir)

    pairs = []
    for example in dataset["test"]:
        if example["label"] == 1:  # Only paraphrase pairs
            pairs.append((example["sentence1"], example["sentence2"]))

    logger.info(f"Loaded {len(pairs)} MRPC pairs")
    return pairs


def create_data_loaders(
    pairs: list,
    tokenizer,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_source_length: int = 128,
    max_target_length: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders from a list of pairs
    """
    import random

    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    train_dataset = ParaphraseDataset(
        train_pairs,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_attributes=True,
    )

    val_dataset = ParaphraseDataset(
        val_pairs,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_attributes=True,
    )

    test_dataset = ParaphraseDataset(
        test_pairs,
        tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        add_attributes=False,  # Test without attributes for evaluation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
