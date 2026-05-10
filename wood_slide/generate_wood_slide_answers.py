import csv
import json
import re
import yaml
from collections import defaultdict
from typing import Dict, List, Tuple, Set

Coord = Tuple[int, int]


def detect_position_columns(fieldnames: List[str]) -> Dict[str, str]:
    """Detect board columns for start-position and end-position."""
    if not fieldnames:
        raise ValueError("CSV has no header row")

    positions = {}
    for name in fieldnames:
        lowered = name.lower()
        if "text" in lowered and "start-position" in lowered:
            positions["start-position"] = name
        elif "text" in lowered and "end-position" in lowered:
            positions["end-position"] = name

    if "start-position" not in positions or "end-position" not in positions:
        raise ValueError(
            "Could not detect both start/end board columns from headers: "
            f"{fieldnames}"
        )
    return positions


def parse_board(text: str) -> List[List[str]]:
    if text is None:
        raise ValueError("Board text is None")

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        raise ValueError("Board text is empty")

    board: List[List[str]] = []
    for line in lines:
        if ";" in line:
            row = [cell.strip() for cell in line.split(";")]
        else:
            row = [cell.strip() for cell in line.split()]
        if row:
            board.append(row)

    if not board:
        raise ValueError("Board text did not produce any rows")

    width = len(board[0])
    if any(len(row) != width for row in board):
        raise ValueError(f"Non-rectangular board: {[len(row) for row in board]}")

    return board


def empty_cells(board: List[List[str]]) -> List[Coord]:
    return [
        (r, c)
        for r, row in enumerate(board)
        for c, value in enumerate(row)
        if value == "0"
    ]


def block_cells(board: List[List[str]]) -> Dict[str, List[Coord]]:
    cells: Dict[str, List[Coord]] = defaultdict(list)
    for r, row in enumerate(board):
        for c, value in enumerate(row):
            if value != "0":
                cells[value].append((r, c))
    return dict(cells)


def block_dimension(cells: List[Coord]) -> str:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1

    # Verify rectangular occupancy
    expected = {(r, c) for r in range(min(rows), max(rows) + 1) for c in range(min(cols), max(cols) + 1)}
    actual = set(cells)
    if expected != actual:
        raise ValueError(f"Block cells are not a solid rectangle: {sorted(actual)}")

    return f"{height}x{width}"


def board_block_dimensions(board: List[List[str]]) -> Dict[str, str]:
    return {block_id: block_dimension(cells) for block_id, cells in block_cells(board).items()}


def adjacent_block_ids(board: List[List[str]]) -> List[str]:
    m, n = len(board), len(board[0])
    adjacent: Set[str] = set()
    for r, c in empty_cells(board):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                val = board[nr][nc]
                if val != "0":
                    adjacent.add(val)
    return sorted(adjacent, key=lambda x: int(x) if x.isdigit() else x)


def dimensions_adjacent_to_empty(board: List[List[str]]) -> List[str]:
    dim_map = board_block_dimensions(board)
    block_ids = adjacent_block_ids(board)
    dims = [dim_map[block_id] for block_id in block_ids]
    return dims


def count_blocks_by_dimension(board: List[List[str]], dimension: str) -> int:
    return sum(1 for dim in board_block_dimensions(board).values() if dim == dimension)


def any_adjacent_block_with_dimension(board: List[List[str]], dimension: str) -> bool:
    dim_map = board_block_dimensions(board)
    return any(dim_map[block_id] == dimension for block_id in adjacent_block_ids(board))


def board_answers_for_position(board: List[List[str]], position: str) -> Dict[str, str]:
    answers: Dict[str, str] = {}

    # q1
    answers[f"q1-ans_{position}"] = str(empty_cells(board))

    # q2
    for dimension in ["1x1", "1x2", "2x1", "2x2"]:
        answers[f"q2-ans_{dimension}_{position}"] = str(count_blocks_by_dimension(board, dimension))

    # q3
    answers[f"q3-ans_{position}"] = str(dimensions_adjacent_to_empty(board))

    # q4
    answers[f"q4-ans_{position}"] = str(len(adjacent_block_ids(board)))

    # q5
    for dimension in ["1x1", "1x2", "2x1", "2x2"]:
        answers[f"q5-ans_{dimension}_{position}"] = str(any_adjacent_block_with_dimension(board, dimension))

    return answers


def load_questions(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("questions.yaml must parse to a mapping")
    return data


def expected_answer_fields() -> List[str]:
    fields: List[str] = []
    positions = ["start-position", "end-position"]
    dimensions = ["1x1", "1x2", "2x1", "2x2"]

    for pos in positions:
        fields.append(f"q1-ans_{pos}")
    for dim in dimensions:
        for pos in positions:
            fields.append(f"q2-ans_{dim}_{pos}")
    for pos in positions:
        fields.append(f"q3-ans_{pos}")
    for pos in positions:
        fields.append(f"q4-ans_{pos}")
    for dim in dimensions:
        for pos in positions:
            fields.append(f"q5-ans_{dim}_{pos}")

    return fields


def main() -> None:
    csv_path = "wood_slide.csv"
    yaml_path = "questions.yaml"
    out_csv = "wood_slide_v2.csv"
    # out_jsonl = "wood_slide_answers.jsonl"

    questions = load_questions(yaml_path)
    expected_question_keys = [f"q{i}" for i in range(1, 6)]
    missing = [k for k in expected_question_keys if k not in questions]
    if missing:
        raise ValueError(f"questions.yaml is missing keys: {missing}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        original_fields = list(reader.fieldnames)
        position_cols = detect_position_columns(original_fields)

        rows: List[Dict[str, str]] = []
        answer_fields = expected_answer_fields()

        for row_idx, row in enumerate(reader, start=1):
            out_row = dict(row)

            for position, col_name in position_cols.items():
                if col_name not in row:
                    raise ValueError(
                        f"Detected board column '{col_name}' missing in row {row_idx}. "
                        f"Available keys: {list(row.keys())}"
                    )

                board = parse_board(row[col_name])
                out_row.update(board_answers_for_position(board, position))

            rows.append(out_row)

    if not rows:
        raise ValueError("No rows found in CSV")

    # with open(out_jsonl, "w", encoding="utf-8") as f:
    #     for row in rows:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=original_fields + answer_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
