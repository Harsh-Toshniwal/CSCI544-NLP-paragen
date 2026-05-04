"""Run vanilla T5-base on validation set and evaluate"""

import argparse
import logging
import json
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path so we can import baseline and paragen modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.train_baseline import VanillaT5Model
from paragen.evaluation import ParaphraseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-detect CUDA
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sentences(file_path):
    """Load sentences from text file (one per line)"""
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def load_from_csv(csv_path):
    """Load source and target sentences from CSV file with 'source' and 'target' columns"""
    df = pd.read_csv(csv_path)
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError(f"CSV must contain 'source' and 'target' columns. Found: {df.columns.tolist()}")
    
    sources = df["source"].astype(str).tolist()
    targets = df["target"].astype(str).tolist()
    logger.info(f"Loaded {len(sources)} samples from {csv_path}")
    return sources, targets


def run_vanilla_t5_evaluation(
    val_sources_file="./data/splits_no_test/val_sources.txt",
    val_targets_file=None,
    csv_file=None,
    output_dir="./results/vanilla_t5_baseline",
    model_name="google/flan-t5-base",
    local_model_path=None,
    num_beams=5,
    device=None,
):
    if device is None:
        device = DEFAULT_DEVICE
    """
    Run vanilla T5-base on validation set and evaluate
    
    Args:
        val_sources_file: Path to validation source sentences (text file, one per line)
        val_targets_file: Path to validation target (reference) sentences (text file, one per line)
        csv_file: Path to CSV file with 'source' and 'target' columns (alternative to text files)
        output_dir: Directory to save results
        model_name: T5 model to use (default: google/flan-t5-base instruction-tuned)
        local_model_path: Path to local model checkpoint (overrides model_name)
        num_beams: Number of beams for generation
        device: Device to use (cuda or cpu)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine model path
    if local_model_path:
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model path does not exist: {local_model_path}")
        model_name = local_model_path
        logger.info(f"Using local model checkpoint: {local_model_path}")
    else:
        logger.info(f"Using model from HuggingFace: {model_name}")
    
    # Load validation data from either CSV or text files
    if csv_file:
        logger.info(f"Loading data from CSV: {csv_file}")
        val_sources, val_targets = load_from_csv(csv_file)
    else:
        logger.info("Loading validation data from text files...")
        val_sources = load_sentences(val_sources_file)
        if val_targets_file is None:
            val_targets_file = val_sources_file.replace("_sources", "_targets")
        val_targets = load_sentences(val_targets_file)
    
    assert len(val_sources) == len(val_targets), "Mismatch in source and target counts"
    logger.info(f"Loaded {len(val_sources)} validation pairs")
    
    # Initialize vanilla T5 model
    logger.info(f"Loading {model_name} (no fine-tuning on your data)...")
    model = VanillaT5Model(model_name=model_name, device=device)
    
    # Generate paraphrases
    logger.info("Generating paraphrases...")
    predictions = []
    
    # Structured prompt injection with explicit source context and anti-answering instructions
    if "flan" in model_name.lower():
        instruction_prefix = (
            "Generate a paraphrase for the following sentence. "
            "IMPORTANT: Do NOT answer the question or provide information. "
            "Do NOT solve the problem. Just rephrase and rewrite the sentence using different words and structure. "
            "PRESERVE the exact semantic meaning - do not change what the sentence says. "
            "However, MAXIMIZE lexical diversity - use as many different words and synonyms as possible while keeping the same meaning. "
            "Use varied sentence structures and vocabulary to increase diversity.\n"
        )
    else:
        instruction_prefix = ""
    
    for i, source in enumerate(tqdm(val_sources)):
        if (i + 1) % 500 == 0:
            logger.info(f"Generated {i + 1}/{len(val_sources)} paraphrases")
        
        # Create structured prompt with source as main input
        if "flan" in model_name.lower():
            # Prompt injection format: instruction + explicit source labeling + anti-answering rules + semantic/lexical constraints
            input_text = f"{instruction_prefix}Text: {source}\nParaphrase:"
        else:
            input_text = source
            
        paraphrase = model.generate(input_text, num_beams=num_beams, max_length=128)
        predictions.append({
            "source": source,
            "reference": val_targets[i],
            "generated": paraphrase,
            "input_prompt": input_text,  # Save the actual prompt used
        })
    
    # Save predictions
    pred_file = os.path.join(output_dir, "predictions.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions to {pred_file}")
    
    # Evaluate
    logger.info("Evaluating predictions")
    evaluator = ParaphraseEvaluator()
    
    sources = [p["source"] for p in predictions]
    generated = [p["generated"] for p in predictions]
    
    all_scores, summary = evaluator.evaluate_batch(
        sources, generated, compute_all=True
    )
    
    # Save evaluation results
    eval_file = os.path.join(output_dir, "evaluation.json")
    results = {
        "model": model_name,
        "num_samples": len(predictions),
        "summary": summary,
        "detailed_scores": {metric: scores for metric, scores in all_scores.items()},
    }
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation results to {eval_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"VANILLA T5 EVALUATION SUMMARY ({model_name})")
    print("=" * 80)
    print(f"Validation samples: {len(predictions)}")
    print("\nMetrics:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run T5-base on validation set and evaluate"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with 'source' and 'target' columns (recommended)",
    )
    parser.add_argument(
        "--val-sources",
        type=str,
        default="./data/splits_no_test/val_sources.txt",
        help="Path to validation source sentences (text file, one per line)",
    )
    parser.add_argument(
        "--val-targets",
        type=str,
        default=None,
        help="Path to validation target sentences (text file, one per line). If not provided, uses val_sources with '_targets' suffix",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/vanilla_t5_baseline",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-base",
        help="T5 model name from HuggingFace (default: google/flan-t5-base instruction-tuned)",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Path to local model checkpoint (overrides --model)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to use (default: cuda if available, else cpu)",
    )
    
    args = parser.parse_args()
    
    run_vanilla_t5_evaluation(
        val_sources_file=args.val_sources,
        val_targets_file=args.val_targets,
        csv_file=args.csv,
        output_dir=args.output_dir,
        model_name=args.model,
        local_model_path=args.local_model,
        num_beams=args.num_beams,
        device=args.device,
    )
