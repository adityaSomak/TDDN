"""
Generate LLM eval JSONL from wood_slide_v2.csv + questions.yaml.

Column naming pattern:
    q1-ans_start-position           → Q1, start image
    q2-ans_1x1_start-position       → Q2 dim=1x1, start image
    q5-ans_2x2_end-position         → Q5 dim=2x2, end image

Usage:
    python generate_eval_jsonl.py

Output:
    wood_slide_eval.jsonl
"""

import csv
import json
import re
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "wood_slide_v2.csv"
YAML_PATH = BASE_DIR / "questions.yaml"
OUTPUT_PATH = BASE_DIR / "wood_slide_eval.jsonl"

# Answer type per base question
ANSWER_TYPES = {
    "q1": "coordinate_list",
    "q2": "number",
    "q3": "dimension_list",
    "q4": "number",
    "q5": "boolean",
}

# Q2 answers are constant across all puzzles
CONSTANT_QUESTIONS = {"q2"}


def parse_column(col):
    """Parse 'q2-ans_1x1_start-position' into (qid, dimension, position)."""
    m = re.match(r'^(q\d+)-ans(?:_(\d+x\d+))?_(start-position|end-position)$', col)
    if not m:
        return None
    return {
        "qid": m.group(1),
        "dimension": m.group(2),  # None for Q1/Q3/Q4
        "position": m.group(3),
    }


def main():
    with open(YAML_PATH) as f:
        questions = yaml.safe_load(f)

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    # Identify all answer columns
    ans_cols = [c for c in rows[0].keys() if re.match(r'^q\d+-ans', c)]

    records = []
    for row in rows:
        start_image = row["start_image_path"]
        end_image = row["end_image_path"]
        puzzle_id = start_image.split("/")[1]  # e.g. "0247"

        for col in ans_cols:
            parsed = parse_column(col)
            if not parsed:
                continue

            qid = parsed["qid"]
            dim = parsed["dimension"]
            pos = parsed["position"]

            # Build question text
            template = questions[qid]["question"]
            if dim:
                question_text = template.replace("{dimension}", dim)
            else:
                question_text = template

            # Build question_id like "q2_1x1_start" or "q1_start"
            parts = [qid]
            if dim:
                parts.append(dim)
            parts.append(pos.replace("-position", ""))
            question_id = "_".join(parts)

            # Image path
            image_path = start_image if pos == "start-position" else end_image

            record = {
                "puzzle_id": puzzle_id,
                "question_id": question_id,
                "question": question_text,
                "options": None,
                "answer": row[col],
                "answer_type": ANSWER_TYPES[qid],
                "is_constant": qid in CONSTANT_QUESTIONS,
                "image_position": pos.replace("-position", ""),
                "image_path": image_path,
            }
            records.append(record)

    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Generated {len(records)} eval records → {OUTPUT_PATH}")

    from collections import Counter
    type_counts = Counter(r["answer_type"] for r in records)
    const_count = sum(1 for r in records if r["is_constant"])
    pos_counts = Counter(r["image_position"] for r in records)
    print(f"  Answer types: {dict(type_counts)}")
    print(f"  Constant records (Q2): {const_count} / {len(records)}")
    print(f"  Positions: {dict(pos_counts)}")


if __name__ == "__main__":
    main()
