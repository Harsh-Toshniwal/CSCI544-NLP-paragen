"""
Mistral baseline paraphrase generator and evaluator
Generates zero-shot paraphrases for sentences in a CSV file and evaluates them
"""

import argparse
import json
import logging
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paragen.evaluation import ParaphraseEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for noisy_logger in ["httpx", "httpcore", "huggingface_hub", "sentence_transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def load_model(model_path=None):
    """Load Mistral model from local checkpoint"""
    if model_path is None:
        # Default: look in project root checkpoints directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # Go up to project root
        model_path = project_root / "checkpoints" / "mistral7binstruct"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def generate_paraphrase(tokenizer, model, source_text, max_new_tokens=96):
    """Generate a zero-shot paraphrase for the source text"""
    system = (
        "You are a careful paraphrase generator. Return only one paraphrased sentence. "
        "Preserve the meaning, avoid copying the wording too closely, and do not explain."
    )
    user = f"Rewrite the sentence as a paraphrase.\nSentence: {source_text}\nParaphrase:"
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    paraphrase = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return paraphrase


def evaluate_paraphrase(source, reference, paraphrase):
    """Evaluate the generated paraphrase with all metrics vs source and vs reference"""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1
    
    # Compute semantic similarity for both pairs
    evaluator = ParaphraseEvaluator()
    semantic_similarity_vs_source = evaluator.compute_bert_score_similarity(source, paraphrase)
    semantic_similarity_vs_reference = evaluator.compute_bert_score_similarity(reference, paraphrase)
    
    # Tokenize all texts
    ref_tokens = reference.lower().split()
    source_tokens_list = source.lower().split()
    pred_tokens = paraphrase.lower().split()
    
    # Metrics vs reference
    bleu_vs_reference = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother) if pred_tokens else 0.0
    rouge_l_vs_reference = scorer.score(reference, paraphrase)["rougeL"].fmeasure
    length_ratio_vs_reference = len(pred_tokens) / max(1, len(ref_tokens))
    
    source_token_set = set(source_tokens_list)
    pred_token_set = set(pred_tokens)
    ref_token_set = set(ref_tokens)
    
    # Lexical diversity metrics
    union_source = source_token_set | pred_token_set
    lexical_diversity_vs_source = 1.0 - (len(source_token_set & pred_token_set) / len(union_source)) if union_source else 0.0
    
    union_reference = ref_token_set | pred_token_set
    lexical_diversity_vs_reference = 1.0 - (len(ref_token_set & pred_token_set) / len(union_reference)) if union_reference else 0.0
    
    # Inverse BLEU metrics
    inverse_bleu_vs_source = (
        1.0
        - sentence_bleu(
            [source_tokens_list],
            pred_tokens,
            smoothing_function=smoother,
        )
        if pred_tokens
        else 1.0
    )
    
    inverse_bleu_vs_reference = (
        1.0
        - sentence_bleu(
            [ref_tokens],
            pred_tokens,
            smoothing_function=smoother,
        )
        if pred_tokens
        else 1.0
    )
    
    # ROUGE-L vs source
    rouge_l_vs_source = scorer.score(source, paraphrase)["rougeL"].fmeasure
    
    # Length ratio vs source
    length_ratio_vs_source = len(pred_tokens) / max(1, len(source_tokens_list))
    
    # BLEU vs source
    bleu_vs_source = sentence_bleu([source_tokens_list], pred_tokens, smoothing_function=smoother) if pred_tokens else 0.0
    
    return {
        # Metrics vs reference
        "bleu_vs_reference": float(bleu_vs_reference),
        "rougeL_f1_vs_reference": float(rouge_l_vs_reference),
        "lexical_diversity_vs_reference": float(lexical_diversity_vs_reference),
        "inverse_bleu_vs_reference": float(inverse_bleu_vs_reference),
        "length_ratio_vs_reference": float(length_ratio_vs_reference),
        "semantic_similarity_vs_reference": float(semantic_similarity_vs_reference),
        # Metrics vs source
        "bleu_vs_source": float(bleu_vs_source),
        "rougeL_f1_vs_source": float(rouge_l_vs_source),
        "lexical_diversity_vs_source": float(lexical_diversity_vs_source),
        "inverse_bleu_vs_source": float(inverse_bleu_vs_source),
        "length_ratio_vs_source": float(length_ratio_vs_source),
        "semantic_similarity_vs_source": float(semantic_similarity_vs_source),
    }


def load_csv(csv_path, limit=None):
    """Load CSV file with source sentences"""
    df = pd.read_csv(csv_path).dropna(subset=["source"])
    if limit is not None:
        df = df.head(limit)
    return df


def process_csv(csv_path, output_dir="results/mistral_paraphrase_baseline", limit=None, model_path=None):
    """Process CSV file and generate paraphrases with evaluations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n" + "=" * 80)
    print("Loading Mistral model...")
    print("=" * 80)
    tokenizer, model = load_model(model_path=model_path)
    print("✓ Model loaded successfully\n")
    
    # Load CSV
    logger.info(f"Loading CSV from: {csv_path}")
    df = load_csv(csv_path, limit=limit)
    logger.info(f"Loaded {len(df)} sentences")
    
    # Process each sentence
    results = []
    all_scores = {}
    
    print("=" * 80)
    print(f"Processing {len(df)} sentences...")
    print("=" * 80)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        source = str(row["source"]).strip()
        
        # Get reference/target if available, otherwise use source as fallback
        if "target" in row:
            reference = str(row["target"]).strip()
        elif "reference" in row:
            reference = str(row["reference"]).strip()
        else:
            reference = source  # Fallback to source
        
        # Generate paraphrase
        paraphrase = generate_paraphrase(tokenizer, model, source)
        
        # Evaluate with reference
        scores = evaluate_paraphrase(source, reference, paraphrase)
        
        # Store results
        result = {
            "source": source,
            "reference": reference,
            "prediction": paraphrase,
        }
        
        # Add scores to result
        for metric, score in scores.items():
            result[metric] = score
            if metric not in all_scores:
                all_scores[metric] = []
            all_scores[metric].append(score)
        
        # Add any additional columns from CSV
        for col in df.columns:
            if col not in ["source", "target", "reference"] and col in row:
                result[col] = row[col]
        
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = output_dir / "mistral_paraphrase_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved results to {results_file}")
    
    # Save detailed metrics as JSON (list of objects)
    detailed_metrics_json = output_dir / "mistral_paraphrase_detailed_metrics.json"
    with open(detailed_metrics_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed metrics JSON to {detailed_metrics_json}")
    
    # Compute and save summary statistics with detailed scores
    summary = {}
    for metric, scores in all_scores.items():
        valid_scores = [s for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]
        if valid_scores:
            summary[f"{metric}_mean"] = float(np.mean(valid_scores))
            summary[f"{metric}_std"] = float(np.std(valid_scores))
            summary[f"{metric}_median"] = float(np.median(valid_scores))
    
    # Convert detailed scores to JSON-serializable format
    detailed_scores = {}
    for metric, scores in all_scores.items():
        detailed_scores[metric] = [float(s) if isinstance(s, (int, float)) else s for s in scores]
    
    # Create summary results object with both summary and detailed scores
    summary_results = {
        "num_samples": len(results),
        "summary": summary,
        "detailed_scores": detailed_scores,
    }
    
    summary_file = output_dir / "mistral_paraphrase_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    logger.info(f"Saved summary with detailed scores to {summary_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser("Mistral paraphrase baseline generator and evaluator")
    parser.add_argument(
        "--csv-path",
        default="data/classification_splits/test.csv",
        help="Path to CSV file with sentences to paraphrase",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mistral_paraphrase_baseline",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to local Mistral model checkpoint (default: checkpoints/mistral7binstruct)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sentences to process (for testing)",
    )
    
    args = parser.parse_args()
    
    process_csv(args.csv_path, args.output_dir, args.limit, args.model_path)


if __name__ == "__main__":
    main()
