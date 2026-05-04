"""Inference script for vanilla T5 baseline"""

import argparse
import logging
import torch
from paragen.config import get_config
from scripts.baseline.train_baseline import VanillaT5Model
from paragen.data_loader import load_mrpc
from paragen.evaluation import ParaphraseEvaluator
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-detect CUDA
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_baseline_predictions(
    model_checkpoint: str,
    input_sentences: list = None,
    output_file: str = "baseline_predictions.json",
    device: str = None,
):
    if device is None:
        device = DEFAULT_DEVICE
    """Generate predictions using baseline model"""
    logger.info("Loading baseline model...")
    model = VanillaT5Model(model_name=model_checkpoint, device=device)

    if input_sentences is None:
        logger.info("Loading MRPC test set")
        mrpc_pairs = load_mrpc()
        sources = [pair[0] for pair in mrpc_pairs]
    else:
        sources = input_sentences

    logger.info(f"Generating predictions for {len(sources)} sentences...")

    predictions = []
    for i, source in enumerate(sources):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(sources)}")

        paraphrase = model.generate(source, num_beams=5)
        predictions.append(
            {
                "source": source,
                "paraphrase": paraphrase,
            }
        )

    # Save predictions
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Predictions saved to {output_file}")
    return predictions


def evaluate_baseline(
    predictions_file: str,
    output_file: str = "baseline_evaluation.json",
):
    """Evaluate baseline predictions"""
    logger.info("Loading predictions...")
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    sources = [p["source"] for p in predictions]
    paraphrases = [p["paraphrase"] for p in predictions]

    logger.info(f"Evaluating {len(predictions)} predictions...")

    evaluator = ParaphraseEvaluator()
    all_scores, summary = evaluator.evaluate_batch(sources, paraphrases, compute_all=True)

    # Compile results
    results = {
        "num_samples": len(predictions),
        "summary": summary,
        "detailed_scores": {metric: scores for metric, scores in all_scores.items()},
    }

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("VANILLA T5 BASELINE - EVALUATION SUMMARY")
    print("=" * 80)
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 80 + "\n")

    return results


def interactive_baseline(model_checkpoint: str, device: str = None):
    if device is None:
        device = DEFAULT_DEVICE
    """Interactive baseline inference"""
    logger.info("Loading baseline model...")
    model = VanillaT5Model(model_name=model_checkpoint, device=device)

    print("\n" + "=" * 80)
    print("Vanilla T5 Baseline - Interactive Mode")
    print("=" * 80)
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("Enter source sentence: ").strip()

            if user_input.lower() == "quit":
                break

            if not user_input:
                continue

            paraphrase = model.generate(user_input, num_beams=5)

            print(f"\nSource:     {user_input}")
            print(f"Paraphrase: {paraphrase}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline inference and evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "generate", "evaluate"],
        default="interactive",
        help="Inference mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file for batch generation",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="baseline_predictions.json",
        help="Output file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for interactive mode")
        interactive_baseline(args.checkpoint, device=args.device)

    elif args.mode == "generate":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for generate mode")

        input_sentences = None
        if args.input_file:
            with open(args.input_file, "r") as f:
                input_sentences = [line.strip() for line in f if line.strip()]

        generate_baseline_predictions(
            model_checkpoint=args.checkpoint,
            input_sentences=input_sentences,
            output_file=args.output_file,
            device=args.device,
        )

    elif args.mode == "evaluate":
        if not args.output_file:
            raise ValueError("--output-file (predictions JSON) required")
        evaluate_baseline(
            predictions_file=args.output_file,
            output_file=args.output_file.replace(".json", "_eval.json"),
        )
