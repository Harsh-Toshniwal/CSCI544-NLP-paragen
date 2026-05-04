"""
Evaluate decoder predictions directly from the CSV produced by
scripts/decoder/inference_decoder_lora.py, while preserving the same
core evaluation logic used by script_old/evaluate.py.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paragen.evaluation import ParaphraseEvaluator
from transformers.utils import logging as transformers_logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for noisy_logger in [
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "sentence_transformers.base.model",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
transformers_logging.set_verbosity_error()


def evaluate_decoder_predictions(prediction_file, output_file="decoder_evaluation.json"):
    logger.info("Loading predictions from %s", prediction_file)
    df = pd.read_csv(prediction_file).dropna(subset=["source", "prediction"])

    if "reference" not in df.columns:
        raise ValueError("Prediction CSV must contain a 'reference' column for evaluation")

    sources = df["source"].astype(str).tolist()
    predictions = df["prediction"].astype(str).tolist()
    references = df["reference"].astype(str).tolist()

    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Mismatch between source, prediction, and reference lengths")

    logger.info("Evaluating %s decoder predictions", len(df))
    evaluator = ParaphraseEvaluator()
    all_scores, summary = evaluator.evaluate_batch(sources, predictions, compute_all=True)

    results = {
        "num_samples": len(df),
        "summary": summary,
        "detailed_scores": {metric: scores for metric, scores in all_scores.items()},
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation results saved to %s", output_file)

    print("\n" + "=" * 80)
    print("DECODER EVALUATION SUMMARY")
    print("=" * 80)
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate decoder predictions from CSV")
    parser.add_argument("--prediction-file", required=True, help="CSV file produced by decoder inference")
    parser.add_argument("--output-file", default="decoder_evaluation.json", help="Output JSON file")
    args = parser.parse_args()

    evaluate_decoder_predictions(
        prediction_file=args.prediction_file,
        output_file=args.output_file,
    )
