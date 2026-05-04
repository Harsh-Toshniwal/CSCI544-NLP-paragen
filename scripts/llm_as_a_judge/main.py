import json
import time
import pandas as pd
from typing import Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()


def build_prompt(source: str, candidate: str) -> str:
    return f"""
Evaluate the following paraphrase with respect to the source sentence.

Source: {source}
Paraphrase: {candidate}

Rate each dimension on a scale of 1 to 5:

Semantic Equivalence
1 = Completely different meaning
2 = Major meaning loss or change
3 = Mostly similar meaning but noticeable differences
4 = Meaning preserved with minor nuance differences
5 = Perfect or near-perfect semantic equivalence

Fluency
1 = Incomprehensible or broken
2 = Very awkward and hard to read
3 = Understandable but somewhat unnatural
4 = Fluent with minor awkwardness
5 = Fully natural and grammatically correct

Diversity
1 = Almost identical to the source
2 = Very small wording changes
3 = Moderate rewording
4 = Meaningful rephrasing with clear variation
5 = Strong variation while preserving meaning

Return ONLY valid JSON in this format:
{{
  "semantic_equivalence": <1-5>,
  "fluency": <1-5>,
  "diversity": <1-5>,
  "justification": "<short explanation>"
}}
""".strip()


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise paraphrase evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=180
    )
    return response.choices[0].message.content.strip()


def parse_response(text: str) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    try:
        data = json.loads(text)
        semantic = int(data.get("semantic_equivalence"))
        fluency = int(data.get("fluency"))
        diversity = int(data.get("diversity"))
        justification = str(data.get("justification", "")).strip()
        return semantic, fluency, diversity, justification
    except Exception:
        return None, None, None, text.strip()


def test_single_example():
    source = "How can I learn Python fast?"
    candidate = "What is the quickest way to learn Python?"

    prompt = build_prompt(source, candidate)
    raw = call_llm(prompt)
    semantic, fluency, diversity, justification = parse_response(raw)

    print("\nRAW RESPONSE:")
    print(raw)
    print("\nPARSED OUTPUT:")
    print("Semantic Equivalence:", semantic)
    print("Fluency:", fluency)
    print("Diversity:", diversity)
    print("Justification:", justification)


def run_csv_pipeline(
    input_csv="paraphrase_pairs.csv",
    output_csv="llm_judge_results.csv",
    limit=None
):
    df = pd.read_csv(input_csv)

    if limit is not None:
        df = df.head(limit).copy()

    results = []

    for _, row in df.iterrows():
        source = str(row["sentence1"])
        candidate = str(row["sentence2"])

        prompt = build_prompt(source, candidate)

        try:
            raw = call_llm(prompt)
            semantic, fluency, diversity, justification = parse_response(raw)
        except Exception as e:
            semantic = None
            fluency = None
            diversity = None
            justification = f"ERROR: {str(e)}"

        result = row.to_dict()
        result["llm_semantic_equivalence"] = semantic
        result["llm_fluency"] = fluency
        result["llm_diversity"] = diversity
        result["llm_justification"] = justification

        results.append(result)

        print(f"Processed row {len(results)}")
        time.sleep(0.5)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    # test_single_example()
    run_csv_pipeline(limit=10)