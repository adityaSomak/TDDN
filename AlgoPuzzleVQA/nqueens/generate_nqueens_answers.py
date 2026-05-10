import csv
import json
import re
import yaml
from typing import Dict, List, Tuple

Coord = Tuple[int, int]


def detect_board_column(fieldnames: List[str]) -> str:
    """Auto-detect the column containing the n-queens board text."""
    if not fieldnames:
        raise ValueError("CSV has no header row")

    preferred_patterns = [
        r"text.*representation",
        r"board",
        r"grid",
        r"state",
        r"puzzle",
    ]

    for pattern in preferred_patterns:
        for original in fieldnames:
            if re.search(pattern, original, flags=re.IGNORECASE):
                return original

    # Fallback: prefer the first non-image/path/id style column.
    skip_patterns = [r"^id$", r"image", r"path", r"file", r"question"]
    for original in fieldnames:
        if not any(re.search(p, original, flags=re.IGNORECASE) for p in skip_patterns):
            return original

    raise ValueError(f"Could not auto-detect board column from headers: {fieldnames}")


def parse_board(text: str) -> List[List[str]]:
    """Parse board text into a rectangular 2D list.

    Supports semicolon-separated rows like:
        0;Q;0
        0;0;0
        Q;0;0
    and whitespace-separated rows.
    """
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


def queen_cells(board: List[List[str]]) -> List[Coord]:
    return [
        (r, c)
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
        if str(cell).strip().upper() == "Q"
    ]


def empty_cells(board: List[List[str]]) -> List[Coord]:
    return [
        (r, c)
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
        if str(cell).strip().upper() != "Q"
    ]


def rows_without_queen(board: List[List[str]]) -> List[int]:
    return [
        r for r, row in enumerate(board)
        if all(str(cell).strip().upper() != "Q" for cell in row)
    ]


def cols_without_queen(board: List[List[str]]) -> List[int]:
    m, n = len(board), len(board[0])
    return [
        c for c in range(n)
        if all(str(board[r][c]).strip().upper() != "Q" for r in range(m))
    ]


def row_has_queen(board: List[List[str]], r: int) -> bool:
    return any(str(cell).strip().upper() == "Q" for cell in board[r])


def col_has_queen(board: List[List[str]], c: int) -> bool:
    return any(str(board[r][c]).strip().upper() == "Q" for r in range(len(board)))


def diagonal_has_queen(board: List[List[str]], r: int, c: int) -> bool:
    m, n = len(board), len(board[0])

    # Main diagonal and anti-diagonal checks through (r, c)
    rr, cc = r - 1, c - 1
    while rr >= 0 and cc >= 0:
        if str(board[rr][cc]).strip().upper() == "Q":
            return True
        rr -= 1
        cc -= 1

    rr, cc = r + 1, c + 1
    while rr < m and cc < n:
        if str(board[rr][cc]).strip().upper() == "Q":
            return True
        rr += 1
        cc += 1

    rr, cc = r - 1, c + 1
    while rr >= 0 and cc < n:
        if str(board[rr][cc]).strip().upper() == "Q":
            return True
        rr -= 1
        cc += 1

    rr, cc = r + 1, c - 1
    while rr < m and cc >= 0:
        if str(board[rr][cc]).strip().upper() == "Q":
            return True
        rr += 1
        cc -= 1

    return False


def safe_rows_for_insertion(board: List[List[str]]) -> List[int]:
    """Rows that contain at least one empty cell where a queen can be inserted safely."""
    m, n = len(board), len(board[0])
    safe_rows: List[int] = []

    for r in range(m):
        row_is_safe = False
        for c in range(n):
            if str(board[r][c]).strip().upper() == "Q":
                continue
            if not row_has_queen(board, r) and not col_has_queen(board, c) and not diagonal_has_queen(board, r, c):
                row_is_safe = True
                break
        if row_is_safe:
            safe_rows.append(r)

    return safe_rows


def safe_cols_for_insertion(board: List[List[str]]) -> List[int]:
    """Columns that contain at least one empty cell where a queen can be inserted safely."""
    m, n = len(board), len(board[0])
    safe_cols: List[int] = []

    for c in range(n):
        col_is_safe = False
        for r in range(m):
            if str(board[r][c]).strip().upper() == "Q":
                continue
            if not row_has_queen(board, r) and not col_has_queen(board, c) and not diagonal_has_queen(board, r, c):
                col_is_safe = True
                break
        if col_is_safe:
            safe_cols.append(c)

    return safe_cols


def board_answers(board: List[List[str]]) -> Dict[str, str]:
    queens = queen_cells(board)
    empties = empty_cells(board)

    answers: Dict[str, str] = {
        "q1-ans": str(rows_without_queen(board)),
        "q2-ans": str(cols_without_queen(board)),
        "q3-ans": str(queens),
        "q4-ans": str(safe_rows_for_insertion(board)),
        "q5-ans": str(safe_cols_for_insertion(board)),
        "q6-ans": str({"empty_cells": empties, "occupied_cells": queens}),
        "q7-ans": str({"occupied_cells": queens, "empty_cells": empties}),
    }
    return answers


def load_questions(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("questions.yaml must parse to a mapping")
    return data


def main() -> None:
    csv_path = "nqueens.csv"
    yaml_path = "questions.yaml"
    out_csv = "nqueens_v2.csv"
    # out_jsonl = "nqueens_answers_updated.jsonl"

    questions = load_questions(yaml_path)
    expected_keys = [f"q{i}" for i in range(1, 8)]
    missing = [k for k in expected_keys if k not in questions]
    if missing:
        raise ValueError(f"questions.yaml is missing keys: {missing}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        original_fields = list(reader.fieldnames)
        board_col = detect_board_column(original_fields)

        rows: List[Dict[str, str]] = []
        answer_fields = [f"q{i}-ans" for i in range(1, 8)]

        for row_idx, row in enumerate(reader, start=1):
            if board_col not in row:
                raise ValueError(
                    f"Detected board column '{board_col}' missing in row {row_idx}. "
                    f"Available keys: {list(row.keys())}"
                )

            board = parse_board(row[board_col])
            answers = board_answers(board)

            out_row = dict(row)
            out_row.update(answers)
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
