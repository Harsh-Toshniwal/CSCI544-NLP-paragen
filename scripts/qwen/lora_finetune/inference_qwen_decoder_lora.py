"""Batch inference for Qwen decoder-only LoRA adapters."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "checkpoints" / "qwen2.5-1.5b-instruct"


def normalize_model_name(model_name):
    aliases = {
        "qwen": DEFAULT_MODEL,
        "qwen2.5-1.5b": DEFAULT_MODEL,
        "qwen2.5-1.5b-instruct": DEFAULT_MODEL,
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    }
    return aliases.get(model_name.lower(), model_name)


def resolve_model_name(model_name):
    model_name = normalize_model_name(model_name)
    if model_name == DEFAULT_MODEL and DEFAULT_LOCAL_MODEL.exists():
        return str(DEFAULT_LOCAL_MODEL)
    return model_name


def clean_text(text):
    return str(text).replace("\x01", " ").strip()


def build_instruction(style_label, length_label):
    return (
        "Rewrite the sentence as a paraphrase. Preserve the meaning, avoid copying the wording too closely, "
        "and return only one paraphrased sentence.\n"
        f"Style label: {style_label}\n"
        f"Target length label: {length_label}"
    )


def build_messages(source, style_label="unspecified", length_label="unspecified"):
    return [
        {
            "role": "system",
            "content": "You are a careful paraphrase generator.",
        },
        {
            "role": "user",
            "content": (
                f"{build_instruction(style_label, length_label)}\n\n"
                f"Sentence: {source}\n"
                "Paraphrase:"
            ),
        },
    ]


def generate_predictions(
    adapter_dir,
    source_file,
    output_file,
    base_model=DEFAULT_MODEL,
    positive_only=True,
    limit=None,
    max_new_tokens=96,
    temperature=0.2,
    top_p=0.9,
):
    adapter_dir = Path(adapter_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    base_model = resolve_model_name(base_model)

    df = pd.read_csv(source_file).dropna(subset=["source", "target"])
    if positive_only and "label" in df.columns:
        df = df[df["label"] == 1]
    if limit is not None:
        df = df.head(limit)
    df = df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(DEFAULT_DEVICE)
    model.eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rows"):
        source = clean_text(row["source"])
        style_label = clean_text(row["style_label"]) if "style_label" in row else "unspecified"
        length_label = clean_text(row["length_label"]) if "length_label" in row else "unspecified"
        messages = build_messages(source, style_label, length_label)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEFAULT_DEVICE)

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
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

        rows.append(
            {
                "source": source,
                "reference": clean_text(row["target"]),
                "prediction": prediction,
                "style_label": row.get("style_label", ""),
                "length_label": row.get("length_label", ""),
                "label": row.get("label", ""),
                "combo_label": row.get("combo_label", ""),
            }
        )

    if output_file.suffix.lower() == ".json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    else:
        pd.DataFrame(rows).to_csv(output_file, index=False)
    return rows


def build_parser():
    parser = argparse.ArgumentParser("Run Qwen decoder LoRA inference")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--source-file", default="data/classification_splits/test.csv")
    parser.add_argument("--output-file", default="results/qwen_decoder_lora_predictions.csv")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--all-labels", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate_predictions(
        adapter_dir=args.adapter_dir,
        source_file=args.source_file,
        output_file=args.output_file,
        base_model=args.base_model,
        positive_only=not args.all_labels,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
