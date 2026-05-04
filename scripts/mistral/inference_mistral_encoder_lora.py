"""
Inference script for trained Mistral encoder LoRA classifier.
"""

import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MistralEncoderInference:
    """Inference wrapper for trained Mistral encoder classifier."""

    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        adapter_path: str = "checkpoints/mistral_encoder_lora/final_model",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        merge_adapters: bool = False,
    ):
        """
        Initialize inference model.

        Args:
            base_model: Base model identifier
            adapter_path: Path to LoRA adapter checkpoint
            device: Device to load model on
            merge_adapters: Whether to merge adapters with base model
        """
        self.device = device
        logger.info(f"Loading base model: {base_model}")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None,
        )

        logger.info(f"Loading LoRA adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)

        if merge_adapters:
            logger.info("Merging adapters with base model...")
            self.model = self.model.merge_and_unload()

        self.model.eval()

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(
        self,
        source: str,
        target: str = None,
        max_length: int = 512,
    ) -> dict:
        """
        Predict paraphrase classification for a sentence pair.

        Args:
            source: Source sentence
            target: Target sentence (optional, for context)
            max_length: Maximum sequence length

        Returns:
            Dictionary with predictions and probabilities
        """
        # Prepare input
        if target:
            text = f"Source: {source}\nTarget: {target}"
        else:
            text = source

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Use first token hidden state (similar to [CLS])
            last_hidden_state = outputs.hidden_states[-1]
            cls_output = last_hidden_state[:, 0, :]  # Take first token

            # Classification head (simple linear layer for inference)
            # For full predictions, you'd need the actual classifier weights
            logits = cls_output.squeeze(0)

        return {
            "input": text,
            "source": source,
            "target": target if target else "N/A",
            "embedding": cls_output.cpu().numpy().squeeze(0),
        }

    def predict_batch(
        self,
        sources: list,
        targets: list = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> list:
        """
        Predict on multiple samples.

        Args:
            sources: List of source sentences
            targets: List of target sentences (optional)
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            List of predictions
        """
        if targets is None:
            targets = [None] * len(sources)

        predictions = []
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            for source, target in zip(batch_sources, batch_targets):
                pred = self.predict(source, target, max_length)
                predictions.append(pred)

        return predictions


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Inference with trained Mistral encoder LoRA classifier"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Base model identifier",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/mistral_encoder_lora/final_model",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source sentence to classify",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target sentence (optional)",
    )
    parser.add_argument(
        "--merge_adapters",
        action="store_true",
        help="Merge adapters with base model",
    )

    args = parser.parse_args()

    # Initialize inference
    inference = MistralEncoderInference(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merge_adapters=args.merge_adapters,
    )

    # Predict
    result = inference.predict(args.source, args.target)
    
    logger.info("=" * 80)
    logger.info(f"Source: {result['source']}")
    if result['target'] != 'N/A':
        logger.info(f"Target: {result['target']}")
    logger.info(f"Embedding shape: {result['embedding'].shape}")
    logger.info(f"Embedding (first 10 dims): {result['embedding'][:10]}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
