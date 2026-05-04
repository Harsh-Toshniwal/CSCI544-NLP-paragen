"""
Batch inference for the new scripts/decoder layout.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL_ALIASES = {
    "google/flan-t5-large": PROJECT_ROOT / "checkpoints" / "flan-t5-large",
}


def normalize_model_name(model_name):
    aliases = {
        "flan-t5-small": "google/flan-t5-small",
        "flan-t5-base": "google/flan-t5-base",
        "flan-t5-large": "google/flan-t5-large",
    }
    return aliases.get(model_name, model_name)


def resolve_model_name(model_name):
    model_name = normalize_model_name(model_name)
    local_model = LOCAL_MODEL_ALIASES.get(model_name)
    if local_model and local_model.exists():
        return str(local_model)
    return model_name


def generate_predictions(lora_dir, source_file, output_file, base_model="flan-t5-small", batch_size=8, max_length=96):
    del batch_size  # kept for CLI compatibility if batching is added later
    base_model = resolve_model_name(base_model)

    lora_dir = Path(lora_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sources = []
    references = []

    if source_file.endswith(".txt"):
        with open(source_file, "r", encoding="utf-8") as f:
            sources = [line.strip() for line in f if line.strip()]
    else:
        df = pd.read_csv(source_file).dropna()
        if {"source", "target"}.issubset(df.columns):
            sources = df["source"].astype(str).tolist()
            references = df["target"].astype(str).tolist()
        elif {"question1", "question2"}.issubset(df.columns):
            sources = df["question1"].astype(str).tolist()
            references = df["question2"].astype(str).tolist()
        else:
            raise ValueError("Unsupported source file format for decoder inference")

    tokenizer = AutoTokenizer.from_pretrained(str(lora_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model.to(DEFAULT_DEVICE)
    model.eval()

    rows = []
    for idx, source in enumerate(sources):
        inputs = tokenizer(
            f"paraphrase: {source}",
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(DEFAULT_DEVICE) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        row = {"source": source, "prediction": prediction}
        if references:
            row["reference"] = references[idx]
        rows.append(row)

    if output_file.suffix.lower() == ".json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    else:
        pd.DataFrame(rows).to_csv(output_file, index=False)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Decoder LoRA inference")
    parser.add_argument("--lora-dir", required=True, help="Path to best_decoder_lora")
    parser.add_argument("--source-file", required=True, help="TXT or CSV file containing source text")
    parser.add_argument("--output-file", required=True, help="CSV or JSON output path")
    parser.add_argument("--base-model", default="flan-t5-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    generate_predictions(
        lora_dir=args.lora_dir,
        source_file=args.source_file,
        output_file=args.output_file,
        base_model=args.base_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
