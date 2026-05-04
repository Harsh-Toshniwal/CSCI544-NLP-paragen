"""Utility functions for ParaGen"""

import logging
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Setup logging configuration"""
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(level)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(level)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_file:
        root_logger.addHandler(file_handler)


def read_sentences(file_path: str) -> List[str]:
    """Read sentences from file (one per line)"""
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def write_sentences(sentences: List[str], file_path: str):
    """Write sentences to file (one per line)"""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent + "\n")


def read_pairs(file_path: str, delimiter: str = "\t") -> List[Tuple[str, str]]:
    """Read sentence pairs from file"""
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(delimiter)
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    return pairs
