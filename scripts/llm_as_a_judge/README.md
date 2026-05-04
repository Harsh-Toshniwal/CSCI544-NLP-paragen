# LLM-as-a-Judge for Paraphrase Evaluation

This project uses an LLM to evaluate how good a paraphrase is.

Given a sentence pair, the model rates:
- Semantic Equivalence (meaning preserved?)
- Fluency (natural and grammatical?)
- Diversity (how different is the wording?)

Each is scored from 1–5, along with a short explanation.

---

## Project Structure

project/
├── main.py  
├── paraphrase_pairs.csv  
├── llm_judge_results.csv  
├── .env  
└── requirements.txt  

---

## Input Format

paraphrase_pairs.csv

sentence1,sentence2,gold_label  
How can I learn Python fast?,What is the quickest way to learn Python?,1  

---

## Output Format

llm_judge_results.csv

sentence1,sentence2,llm_semantic_equivalence,llm_fluency,llm_diversity,llm_justification  

---

## How to Run

1. Open project in VS Code terminal (Ctrl + `)

2. Create environment:
python -m venv judge_env

3. Activate environment (Windows PowerShell):
.\judge_env\Scripts\Activate

4. Install dependencies:
pip install -r requirements.txt

5. Create `.env` file:
OPENAI_API_KEY=your_api_key_here

6. Run:
python main.py

---

## Config

In main.py:
run_csv_pipeline(limit=10)

- limit=10 → runs only 10 rows (safe)
- set to None → runs full dataset

---