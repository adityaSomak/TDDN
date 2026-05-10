"""
Generate LLM eval JSONL from maze_solve_v2.csv + questions.yaml.

Each CSV row produces one eval record per question (15 questions × 100 puzzles = 1500 records).

Usage:
    python generate_eval_jsonl.py

Output:
    maze_solve_eval.jsonl
"""

import csv
import json
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "maze_solve_v2.csv"
YAML_PATH = BASE_DIR / "questions.yaml"
OUTPUT_PATH = BASE_DIR / "maze_solve_eval.jsonl"

# Answer type per question
ANSWER_TYPES = {
    "q1": "coordinate",
    "q2": "coordinate",
    "q3": "dimensions",
    "q4": "number",
    "q5": "number",
    "q6": "number",
    "q7": "number",
    "q8": "number",
    "q9": "number",
    "q10": "coordinate_list",
    "q11": "coordinate_list",
    "q12": "coordinate_list",
    "q13": "coordinate_list",
    "q14": "coordinate_list_long",
    "q15": "coordinate_list_long",
}

# Questions that have the same answer across all puzzles in the dataset
CONSTANT_QUESTIONS = {"q1", "q6", "q7", "q8", "q9", "q10", "q11"}


def main():
    with open(YAML_PATH) as f:
        questions = yaml.safe_load(f)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        image_path = row["image_path"]
        puzzle_id = image_path.split("/")[1]  # e.g. "1860" from "images/1860/maze_1860.jpg"

        for qnum in range(1, 16):
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
                "is_constant": qid in CONSTANT_QUESTIONS,
                "image_path": image_path,
            }
            records.append(record)

    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Generated {len(records)} eval records → {OUTPUT_PATH}")

    # Summary
    from collections import Counter
    type_counts = Counter(r["answer_type"] for r in records)
    const_count = sum(1 for r in records if r["is_constant"])
    print(f"  Answer types: {dict(type_counts)}")
    print(f"  Constant-answer records: {const_count} / {len(records)}")


if __name__ == "__main__":
    main()
