"""Model wrapper for T5-based paraphrase generation"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ParaphraseModel:
    """T5-based paraphrase generation model with controllable attributes"""

    def __init__(
        self,
        model_name: str = "t5-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        length_tokens: List[str] = None,
        diversity_tokens: List[str] = None,
    ):
        """
        Args:
            model_name: Hugging Face model ID
            device: Device to load model on
            length_tokens: Length control tokens
            diversity_tokens: Diversity control tokens
        """
        self.device = device
        self.model_name = model_name

        self.length_tokens = length_tokens or ["[SHORT]", "[SAME]", "[LONG]"]
        self.diversity_tokens = diversity_tokens or ["[CONSERVATIVE]", "[CREATIVE]"]

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Add new tokens for attributes
        new_tokens = self.length_tokens + self.diversity_tokens
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(device)
        logger.info(f"Loaded {model_name} on {device}")

    def generate(
        self,
        source_text: str,
        length_control: Optional[str] = None,
        diversity_control: Optional[str] = None,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        max_length: int = 128,
        min_length: int = 5,
        diversity_penalty: float = 0.5,
        **kwargs,
    ) -> List[str]:
        """
        Generate paraphrases for source text

        Args:
            source_text: Source sentence to paraphrase
            length_control: One of ["[SHORT]", "[SAME]", "[LONG]"] or None
            diversity_control: One of ["[CONSERVATIVE]", "[CREATIVE]"] or None
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return
            max_length: Maximum generation length
            min_length: Minimum generation length
            diversity_penalty: Diversity penalty for diverse beam search

        Returns:
            List of generated paraphrases
        """
        # Build input with control tokens
        input_text = "paraphrase:"

        if length_control and length_control in self.length_tokens:
            input_text += f" {length_control}"

        if diversity_control and diversity_control in self.diversity_tokens:
            input_text += f" {diversity_control}"

        input_text += f" {source_text}"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_length=max_length,
                min_length=min_length,
                diversity_penalty=diversity_penalty,
                early_stopping=True,
                **kwargs,
            )

        # Decode
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def batch_generate(
        self,
        source_texts: List[str],
        length_control: Optional[str] = None,
        diversity_control: Optional[str] = None,
        **kwargs,
    ) -> List[List[str]]:
        """
        Generate paraphrases for multiple source texts

        Args:
            source_texts: List of source sentences
            length_control: Length control token
            diversity_control: Diversity control token
            **kwargs: Additional arguments for generate()

        Returns:
            List of lists of generated paraphrases
        """
        results = []
        for text in source_texts:
            paraphrases = self.generate(
                text,
                length_control=length_control,
                diversity_control=diversity_control,
                **kwargs,
            )
            results.append(paraphrases)
        return results

    def save(self, save_path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved model to {save_path}")

    def load(self, load_path: str):
        """Load model and tokenizer from checkpoint"""
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        logger.info(f"Loaded model from {load_path}")
