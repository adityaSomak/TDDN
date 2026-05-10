"""
Generate LLM eval JSONL from nqueens_v2.csv + questions.yaml.

Each CSV row produces one eval record per question (7 questions × 100 puzzles = 700 records).

Usage:
    python generate_eval_jsonl.py

Output:
    nqueens_eval.jsonl
"""

import csv
import json
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "nqueens_v2.csv"
YAML_PATH = BASE_DIR / "questions.yaml"
OUTPUT_PATH = BASE_DIR / "nqueens_eval.jsonl"

# Answer type per question
ANSWER_TYPES = {
    "q1": "integer_list_rows",
    "q2": "integer_list_cols",
    "q3": "coordinate_list",
    "q4": "integer_list_rows",
    "q5": "integer_list_cols",
    "q6": "cell_dict_empty_first",
    "q7": "cell_dict_occupied_first",
}

# Q4 answers are identical to Q1, Q5 to Q2
REDUNDANT_OF = {
    "q4": "q1",
    "q5": "q2",
}


def main():
    with open(YAML_PATH) as f:
        questions = yaml.safe_load(f)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        image_path = row["image_path"]
        puzzle_id = image_path.split("/")[1]  # e.g. "1860" from "images/1860/nqueens.jpg"

        for qnum in range(1, 8):
            qid = f"q{qnum}"
            ans_col = f"{qid}-ans"
            answer = row[ans_col]

            record = {
                "puzzle_id": puzzle_id,
                "question_id": qid,
                "question": questions[qid]["question"],
                "options": None,
                "answer": answer,
                "answer_type": ANSWER_TYPES[qid],
                "is_redundant_of": REDUNDANT_OF.get(qid),
                "image_path": image_path,
            }
            records.append(record)

    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Generated {len(records)} eval records → {OUTPUT_PATH}")

    from collections import Counter
    type_counts = Counter(r["answer_type"] for r in records)
    redundant_count = sum(1 for r in records if r["is_redundant_of"])
    print(f"  Answer types: {dict(type_counts)}")
    print(f"  Redundant records (Q4≡Q1, Q5≡Q2): {redundant_count} / {len(records)}")


if __name__ == "__main__":
    main()
