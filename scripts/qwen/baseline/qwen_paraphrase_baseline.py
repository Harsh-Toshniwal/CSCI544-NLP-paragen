"""Zero-shot and few-shot Qwen paraphrase baselines."""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paragen.evaluation import ParaphraseEvaluator

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

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "checkpoints" / "qwen2.5-1.5b-instruct"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_model_name(model_name):
    aliases = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-1.5b": DEFAULT_MODEL,
        "qwen2.5-1.5b-instruct": DEFAULT_MODEL,
        "qwen": DEFAULT_MODEL,
    }
    return aliases.get(model_name.lower(), model_name)


def clean_text(text):
    return str(text).replace("\x01", " ").strip()


def load_test_rows(test_file, positive_only=True, limit=None):
    df = pd.read_csv(test_file).dropna(subset=["source", "target"])
    if positive_only and "label" in df.columns:
        df = df[df["label"] == 1]
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def token_overlap_ratio(source, target):
    source_tokens = set(clean_text(source).lower().split())
    target_tokens = set(clean_text(target).lower().split())
    union = source_tokens | target_tokens
    return (len(source_tokens & target_tokens) / len(union)) if union else 0.0


def length_ratio(source, target):
    source_len = max(1, len(clean_text(source).split()))
    target_len = len(clean_text(target).split())
    return target_len / source_len


def build_few_shot_examples(
    df,
    examples_per_combo=3,
    max_token_overlap=0.65,
    min_length_ratio=0.65,
    max_length_ratio=1.5,
):
    examples = defaultdict(list)
    if not {"style_label", "length_label"}.issubset(df.columns):
        return examples

    pool = df
    if "label" in pool.columns:
        pool = pool[pool["label"] == 1]

    for _, row in pool.iterrows():
        overlap = token_overlap_ratio(row["source"], row["target"])
        ratio = length_ratio(row["source"], row["target"])
        if overlap > max_token_overlap or ratio < min_length_ratio or ratio > max_length_ratio:
            continue

        key = (str(row["style_label"]), str(row["length_label"]))
        if len(examples[key]) >= examples_per_combo:
            continue
        examples[key].append(
            {
                "source": clean_text(row["source"]),
                "target": clean_text(row["target"]),
                "style_label": str(row["style_label"]),
                "length_label": str(row["length_label"]),
                "token_overlap_ratio": overlap,
                "length_ratio": ratio,
            }
        )
    return examples


def row_attributes(row):
    style = str(row["style_label"]) if "style_label" in row and pd.notna(row["style_label"]) else "unspecified"
    length = str(row["length_label"]) if "length_label" in row and pd.notna(row["length_label"]) else "unspecified"
    return style, length


def make_zero_shot_messages(row):
    style, length = row_attributes(row)
    source = clean_text(row["source"])
    system = (
        "You are a careful paraphrase generator. Return only one paraphrased sentence. "
        "Preserve the meaning, avoid copying the wording too closely, and do not explain."
    )
    user = (
        "Rewrite the sentence as a paraphrase.\n"
        f"Style label: {style}\n"
        f"Target length label: {length}\n"
        f"Sentence: {source}\n"
        "Paraphrase:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_few_shot_messages(row, examples):
    style, length = row_attributes(row)
    key = (style, length)
    selected = examples.get(key) or examples.get(("CONSERVATIVE", "SAME")) or []
    source = clean_text(row["source"])

    system = (
        "You are a careful paraphrase generator. Return only one paraphrased sentence. "
        "Preserve the meaning, avoid copying the wording too closely, and do not explain."
    )
    messages = [{"role": "system", "content": system}]
    for example in selected:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Rewrite the sentence as a paraphrase.\n"
                    f"Style label: {example['style_label']}\n"
                    f"Target length label: {example['length_label']}\n"
                    f"Sentence: {example['source']}\n"
                    "Paraphrase:"
                ),
            }
        )
        messages.append({"role": "assistant", "content": example["target"]})

    messages.append(
        {
            "role": "user",
            "content": (
                "Rewrite the sentence as a paraphrase.\n"
                f"Style label: {style}\n"
                f"Target length label: {length}\n"
                f"Sentence: {source}\n"
                "Paraphrase:"
            ),
        }
    )
    return messages


def load_model_from_path(model_name, dtype="auto"):
    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def load_model(model_name, dtype="auto"):
    model_name = normalize_model_name(model_name)
    try:
        return load_model_from_path(model_name, dtype=dtype)
    except Exception as exc:
        if not DEFAULT_LOCAL_MODEL.exists():
            raise
        logger.warning(
            "Could not load %s from Hugging Face/cache (%s). Falling back to local checkpoint: %s",
            model_name,
            exc,
            DEFAULT_LOCAL_MODEL,
        )
        return load_model_from_path(str(DEFAULT_LOCAL_MODEL), dtype=dtype)


def generate_one(tokenizer, model, messages, max_new_tokens=96, temperature=0.2, top_p=0.9):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_similarity_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        logger.warning("sentence-transformers not installed; semantic similarity will be skipped")
        return None


def compute_semantic_similarities(similarity_model, sources, predictions, batch_size=32):
    if similarity_model is None:
        return [None] * len(sources)

    source_embeddings = similarity_model.encode(
        sources,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    prediction_embeddings = similarity_model.encode(
        predictions,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    similarities = torch.nn.functional.cosine_similarity(source_embeddings, prediction_embeddings)
    return [safe_float(value) for value in similarities.detach().cpu().tolist()]


def safe_float(value):
    if isinstance(value, (np.floating, float)) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def evaluate_generation(sources, predictions, references, similarity_model=None):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1
    semantic_similarities = compute_semantic_similarities(similarity_model, sources, predictions)

    rows = []
    for source, prediction, reference, semantic_similarity in zip(
        sources,
        predictions,
        references,
        semantic_similarities,
    ):
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother) if pred_tokens else 0.0
        rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
        source_tokens = set(source.lower().split())
        pred_token_set = set(prediction.lower().split())
        union = source_tokens | pred_token_set
        lexical_diversity = 1.0 - (len(source_tokens & pred_token_set) / len(union)) if union else 0.0
        inverse_bleu_vs_source = (
            1.0
            - sentence_bleu(
                [source.lower().split()],
                pred_tokens,
                smoothing_function=smoother,
            )
            if pred_tokens
            else 1.0
        )
        length_ratio = len(pred_tokens) / max(1, len(ref_tokens))

        rows.append(
            {
                "source": source,
                "reference": reference,
                "prediction": prediction,
                "bleu_vs_reference": bleu,
                "rougeL_f1_vs_reference": rouge_l,
                "lexical_diversity_vs_source": lexical_diversity,
                "inverse_bleu_vs_source": inverse_bleu_vs_source,
                "length_ratio_vs_reference": length_ratio,
                "semantic_similarity_vs_source": semantic_similarity,
            }
        )

    semantic_values = [
        row["semantic_similarity_vs_source"]
        for row in rows
        if row["semantic_similarity_vs_source"] is not None
    ]
    summary = {
        "num_samples": len(rows),
        "mean_bleu_vs_reference": safe_float(np.mean([row["bleu_vs_reference"] for row in rows])),
        "mean_rougeL_f1_vs_reference": safe_float(np.mean([row["rougeL_f1_vs_reference"] for row in rows])),
        "mean_lexical_diversity_vs_source": safe_float(np.mean([row["lexical_diversity_vs_source"] for row in rows])),
        "mean_inverse_bleu_vs_source": safe_float(np.mean([row["inverse_bleu_vs_source"] for row in rows])),
        "mean_length_ratio_vs_reference": safe_float(np.mean([row["length_ratio_vs_reference"] for row in rows])),
        "mean_semantic_similarity_vs_source": safe_float(np.mean(semantic_values)) if semantic_values else None,
    }
    return rows, summary


def evaluate_with_paragen(sources, predictions):
    evaluator = ParaphraseEvaluator()
    all_scores, summary = evaluator.evaluate_batch(sources, predictions, compute_all=True)
    return {
        "num_samples": len(sources),
        "summary": summary,
        "detailed_scores": {metric: scores for metric, scores in all_scores.items()},
    }


def run_baseline(
    test_file,
    output_dir,
    model_name=DEFAULT_MODEL,
    few_shot_file=None,
    modes=None,
    examples_per_combo=3,
    max_few_shot_token_overlap=0.65,
    min_few_shot_length_ratio=0.65,
    max_few_shot_length_ratio=1.5,
    positive_only=True,
    limit=None,
    max_new_tokens=96,
    temperature=0.2,
    top_p=0.9,
    dtype="auto",
):
    modes = modes or ["zero_shot", "few_shot"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_test_rows(test_file, positive_only=positive_only, limit=limit)
    few_shot_source = few_shot_file or test_file
    examples = build_few_shot_examples(
        pd.read_csv(few_shot_source).dropna(subset=["source", "target"]),
        examples_per_combo=examples_per_combo,
        max_token_overlap=max_few_shot_token_overlap,
        min_length_ratio=min_few_shot_length_ratio,
        max_length_ratio=max_few_shot_length_ratio,
    )
    tokenizer, model = load_model(model_name, dtype=dtype)
    similarity_model = load_similarity_model()

    with open(output_dir / "few_shot_examples.json", "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in examples.items()}, f, indent=2, ensure_ascii=False)

    for mode in modes:
        rows = []
        logger.info("Running %s on %s rows", mode, len(df))
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Rows"):
            messages = make_zero_shot_messages(row) if mode == "zero_shot" else make_few_shot_messages(row, examples)
            prediction = generate_one(
                tokenizer,
                model,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            rows.append(
                {
                    "source": clean_text(row["source"]),
                    "reference": clean_text(row["target"]),
                    "prediction": prediction,
                    "mode": mode,
                    "style_label": row.get("style_label", ""),
                    "length_label": row.get("length_label", ""),
                    "label": row.get("label", ""),
                    "combo_label": row.get("combo_label", ""),
                }
            )

        output_file = output_dir / f"qwen_{mode}_predictions.csv"
        pd.DataFrame(rows).to_csv(output_file, index=False)
        logger.info("Saved %s", output_file)

        metric_rows, metric_summary = evaluate_generation(
            sources=[row["source"] for row in rows],
            predictions=[row["prediction"] for row in rows],
            references=[row["reference"] for row in rows],
            similarity_model=similarity_model,
        )
        metrics_file = output_dir / f"qwen_{mode}_evaluation.json"
        detailed_metrics_file = output_dir / f"qwen_{mode}_detailed_metrics.csv"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metric_summary, f, indent=2, ensure_ascii=False)
        pd.DataFrame(metric_rows).to_csv(detailed_metrics_file, index=False)
        logger.info("Saved %s", metrics_file)
        logger.info("Saved %s", detailed_metrics_file)

        paragen_metrics = evaluate_with_paragen(
            sources=[row["source"] for row in rows],
            predictions=[row["prediction"] for row in rows],
        )
        paragen_metrics_file = output_dir / f"qwen_{mode}_paragen_evaluation.json"
        with open(paragen_metrics_file, "w", encoding="utf-8") as f:
            json.dump(paragen_metrics, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s", paragen_metrics_file)


def build_parser():
    parser = argparse.ArgumentParser("Run Qwen zero-shot/few-shot paraphrase baselines")
    parser.add_argument("--test-file", default="data/classification_splits/test.csv")
    parser.add_argument("--output-dir", default="results/qwen_paraphrase_baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--few-shot-file", default=None, help="CSV used to select few-shot examples; defaults to --test-file")
    parser.add_argument("--modes", nargs="+", default=["zero_shot", "few_shot"], choices=["zero_shot", "few_shot"])
    parser.add_argument("--examples-per-combo", type=int, default=3)
    parser.add_argument("--max-few-shot-token-overlap", type=float, default=0.65)
    parser.add_argument("--min-few-shot-length-ratio", type=float, default=0.65)
    parser.add_argument("--max-few-shot-length-ratio", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None, help="Optional quick-run row limit")
    parser.add_argument("--all-labels", action="store_true", help="Use all test rows instead of only label=1 rows")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_baseline(
        test_file=args.test_file,
        output_dir=args.output_dir,
        model_name=args.model,
        few_shot_file=args.few_shot_file,
        modes=args.modes,
        examples_per_combo=args.examples_per_combo,
        max_few_shot_token_overlap=args.max_few_shot_token_overlap,
        min_few_shot_length_ratio=args.min_few_shot_length_ratio,
        max_few_shot_length_ratio=args.max_few_shot_length_ratio,
        positive_only=not args.all_labels,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
    )
