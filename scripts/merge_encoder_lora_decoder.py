"""
Generation script using fine-tuned T5 Encoder with default T5 Decoder.

This script:
1. Loads the fine-tuned T5 encoder with LoRA weights from checkpoint
2. Uses the default decoder from flan-T5-small
3. Combines them into a full T5 model for text generation
4. Generates output sequences (e.g., paraphrase/rewritten text)

The encoder has been fine-tuned for understanding sentence pairs with categorical features,
and the decoder generates output based on the encoded representations.

Example Usage:

# Basic generation
python scripts/encoder/generation_encoder_lora_decoder.py \
    --checkpoint checkpoints/encoder_lora_multimodal_combined_AdamW_1e-04_ep30 \
    --input-text "The quick brown fox jumps over the lazy dog" \
    --output-file generated_paraphrases.txt

# Generation with custom parameters
python scripts/encoder/generation_encoder_lora_decoder.py \
    --checkpoint checkpoints/encoder_lora_multimodal_combined_AdamW_1e-04_ep30 \
    --input-text "The quick brown fox jumps over the lazy dog" \
    --output-file generated_paraphrases.txt \
    --max-length 128 \
    --num-beams 4 \
    --temperature 0.8 \
    --top-k 50

# Generation from CSV file
python scripts/encoder/generation_encoder_lora_decoder.py \
    --checkpoint checkpoints/encoder_lora_multimodal_combined_AdamW_1e-04_ep30 \
    --input-csv data/classification_splits/test.csv \
    --output-file generated_outputs.csv \
    --max-length 128 \
    --task "paraphrase"
"""

import os
import argparse
import logging
import json
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

from transformers import T5ForConditionalGeneration, T5EncoderModel, AutoTokenizer
from peft import PeftModel

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncoderLoRAWithDecoder(torch.nn.Module):
    """
    Combined model: Fine-tuned encoder with LoRA + Default T5 decoder.
    
    Uses the fine-tuned encoder and replaces the decoder with the default one.
    """
    def __init__(self, checkpoint_dir, base_model_name="flan-t5-small"):
        super().__init__()
        
        # Load model config
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        base_model_path = model_config['model_path']
        
        logger.info(f"Loading full T5 model from {base_model_path}...")
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_path)
        
        # Load fine-tuned encoder with LoRA
        lora_path = os.path.join(checkpoint_dir, "best_encoder_lora")
        logger.info(f"Loading fine-tuned encoder with LoRA from {lora_path}...")
        
        # Create a temporary encoder to load LoRA weights
        temp_encoder = T5EncoderModel.from_pretrained(base_model_path)
        temp_encoder = PeftModel.from_pretrained(temp_encoder, lora_path)
        
        # Replace the encoder in the full T5 model with the fine-tuned one
        self.model.encoder = temp_encoder
        
        logger.info("Successfully loaded fine-tuned encoder with default decoder")
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None, **kwargs):
        """Forward pass for training or inference"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=128, num_beams=4,
                 early_stopping=True, temperature=1.0, top_k=None, top_p=None,
                 do_sample=False, repetition_penalty=1.0, **kwargs):
        """Generate sequences"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            **kwargs
        )


def generate_paraphrase(encoder_decoder_model, tokenizer, text, task="paraphrase",
                       max_length=128, num_beams=4, temperature=1.0, top_k=None,
                       device="cuda"):
    """
    Generate paraphrase/output for input text.
    
    Args:
        encoder_decoder_model: Loaded EncoderLoRAWithDecoder model
        tokenizer: Tokenizer
        text: Input text to paraphrase
        task: Task prefix (e.g., "paraphrase", "summarize", "translate")
        max_length: Maximum length of generated text
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to use
    
    Returns:
        str: Generated paraphrase
    """
    # Prepare input with task prefix (same format as training)
    input_text = f"Generate {task} for this text, don't repeate the sentence as it is, try and add some lexical diversity but maitain the semantic similarity. Text to paraphrase: {text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length"
    ).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = encoder_decoder_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            temperature=temperature,
            top_k=top_k,
            do_sample=(temperature > 0 and top_k is not None),
            repetition_penalty=1.0
        )
    
    # Decode
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    return generated_text


def generate_from_text(checkpoint_dir, input_text, output_file, task="paraphrase",
                      max_length=128, num_beams=4, temperature=1.0, top_k=None,
                      num_return_sequences=1, device=DEFAULT_DEVICE):
    """
    Generate paraphrases from input text.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        input_text: Input text string
        output_file: Path to save outputs
        task: Generation task prefix
        max_length: Maximum length of generated text
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        num_return_sequences: Number of outputs per input
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    base_model_path = model_config['model_path']
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "best_encoder_lora")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load model
    logger.info("Loading encoder-decoder model...")
    model = EncoderLoRAWithDecoder(checkpoint_dir).to(device)
    model.eval()
    
    # Generate
    logger.info(f"Generating paraphrase for: '{input_text}'")
    generated = generate_paraphrase(
        model, tokenizer, input_text, task=task,
        max_length=max_length, num_beams=num_beams,
        temperature=temperature, top_k=top_k, device=str(device)
    )
    
    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"Input: {input_text}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Parameters: max_length={max_length}, num_beams={num_beams}, temp={temperature}\n")
        f.write(f"\nGenerated Output:\n{generated}\n")
    
    logger.info(f"Output saved to {output_file}")
    logger.info(f"Generated: {generated}")


def generate_from_csv(checkpoint_dir, input_csv, output_csv, task="paraphrase",
                     max_length=128, num_beams=4, temperature=1.0, top_k=None,
                     device=DEFAULT_DEVICE):
    """
    Generate paraphrases from CSV file.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        input_csv: Path to input CSV file
        output_csv: Path to save outputs
        task: Generation task prefix
        max_length: Maximum length of generated text
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "model_config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    base_model_path = model_config['model_path']
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "best_encoder_lora")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load model
    logger.info("Loading encoder-decoder model...")
    model = EncoderLoRAWithDecoder(checkpoint_dir).to(device)
    model.eval()
    
    # Load input data
    logger.info(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Identify text column
    if 'source' in df.columns:
        text_col = 'source'
    elif 'sentence_a' in df.columns:
        text_col = 'sentence_a'
    else:
        raise ValueError("CSV must have 'source' or 'sentence_a' column")
    
    # Generate outputs
    logger.info(f"Generating {len(df)} outputs...")
    generated_outputs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        input_text = row[text_col]
        
        try:
            output = generate_paraphrase(
                model, tokenizer, input_text, task=task,
                max_length=max_length, num_beams=num_beams,
                temperature=temperature, top_k=top_k, device=str(device)
            )
            generated_outputs.append(output)
        except Exception as e:
            logger.warning(f"Error generating for row {idx}: {str(e)}")
            generated_outputs.append("")
    
    # Create output dataframe
    output_df = df.copy()
    output_df['generated_output'] = generated_outputs
    
    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    
    logger.info(f"Generated outputs saved to {output_csv}")
    logger.info(f"Sample outputs:\n{output_df[['source', 'generated_output']].head(5).to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate text using fine-tuned encoder with default decoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--input-text", type=str, default=None, help="Input text for generation")
    parser.add_argument("--input-csv", type=str, default=None, help="Input CSV file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    parser.add_argument("--task", type=str, default="paraphrase", help="Task prefix (e.g., paraphrase, summarize)")
    parser.add_argument("--max-length", type=int, default=128, help="Max length of generated text")
    parser.add_argument("--num-beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.input_text is None and args.input_csv is None:
        raise ValueError("Either --input-text or --input-csv must be provided")
    
    if args.input_text is not None and args.input_csv is not None:
        raise ValueError("Only one of --input-text or --input-csv can be provided")
    
    # Run generation
    if args.input_text is not None:
        generate_from_text(
            checkpoint_dir=args.checkpoint,
            input_text=args.input_text,
            output_file=args.output_file,
            task=args.task,
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
    else:
        generate_from_csv(
            checkpoint_dir=args.checkpoint,
            input_csv=args.input_csv,
            output_csv=args.output_file,
            task=args.task,
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
