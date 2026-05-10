import csv
import json
import yaml
from typing import List, Tuple, Optional, Dict, Any

Coord = Tuple[int, int]


def parse_maze(text: str) -> List[List[str]]:
    """Parse a maze text representation into a rectangular grid.

    Supported row formats:
    - semicolon-delimited cells, e.g. '1;0;S;1'
    - compact character rows, e.g. '10S1'

    Coordinates are 0-based: (row, col), with (0, 0) at the top left.
    """
    if text is None:
        raise ValueError("Maze text is None")

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        raise ValueError("Maze text is empty")

    rows: List[List[str]] = []
    for line in lines:
        if ";" in line:
            cells = [cell.strip() for cell in line.split(";") if cell.strip() != ""]
        else:
            cells = [ch for ch in line if not ch.isspace()]
        if cells:
            rows.append(cells)

    if not rows:
        raise ValueError("Maze text did not contain any cells")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError(f"Non-rectangular maze: row lengths = {[len(r) for r in rows]}")

    allowed = {"0", "1", "S", "E"}
    invalid = sorted({cell for row in rows for cell in row if cell not in allowed})
    if invalid:
        raise ValueError(f"Unexpected maze tokens: {invalid}")

    return rows


def find_cell(grid: List[List[str]], target: str) -> Optional[Coord]:
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == target:
                return (r, c)
    return None


def get_neighbors(grid: List[List[str]], r: int, c: int) -> List[Coord]:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors: List[Coord] = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            neighbors.append((nr, nc))
    return neighbors


def classify_cells(grid: List[List[str]]) -> Tuple[List[Coord], List[Coord]]:
    white: List[Coord] = []
    black: List[Coord] = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val in {"0", "S", "E"}:
                white.append((r, c))
            elif val == "1":
                black.append((r, c))
    return white, black


def solve_maze(grid: List[List[str]]) -> Dict[str, Any]:
    m, n = len(grid), len(grid[0])
    entry = find_cell(grid, "S")
    exit_ = find_cell(grid, "E")

    if entry is None:
        raise ValueError("Entry cell 'S' not found")
    if exit_ is None:
        raise ValueError("Exit cell 'E' not found")

    white_cells, black_cells = classify_cells(grid)

    def analyze(pt: Coord) -> Tuple[List[Coord], List[Coord]]:
        r, c = pt
        white_adj: List[Coord] = []
        black_adj: List[Coord] = []
        for nr, nc in get_neighbors(grid, r, c):
            if grid[nr][nc] in {"0", "S", "E"}:
                white_adj.append((nr, nc))
            else:
                black_adj.append((nr, nc))
        return white_adj, black_adj

    entry_white, entry_black = analyze(entry)
    exit_white, exit_black = analyze(exit_)

    return {
        "q1-ans": entry,
        "q2-ans": exit_,
        "q3-ans": f"{m}x{n}",
        "q4-ans": len(white_cells),
        "q5-ans": len(black_cells),
        "q6-ans": len(entry_white),
        "q7-ans": len(entry_black),
        "q8-ans": len(exit_white),
        "q9-ans": len(exit_black),
        "q10-ans": entry_white,
        "q11-ans": entry_black,
        "q12-ans": exit_white,
        "q13-ans": exit_black,
        "q14-ans": white_cells,
        "q15-ans": black_cells,
    }


def main() -> None:
    csv_path = "maze_solve.csv"
    yaml_path = "questions.yaml"
    out_csv = "maze_solve_v2.csv"
    # out_jsonl = "maze_solve_v2.jsonl"
    maze_col = "text_representation_start-position"

    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml.safe_load(f)

    ans_keys = [f"q{i}-ans" for i in range(1, 16)]
    results: List[Dict[str, Any]] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        if maze_col not in reader.fieldnames:
            raise ValueError(f"Missing required column '{maze_col}'. Found: {reader.fieldnames}")

        original_fields = list(reader.fieldnames)

        for row_idx, row in enumerate(reader, start=1):
            grid = parse_maze(row.get(maze_col))
            answers = solve_maze(grid)
            out_row = dict(row)
            out_row.update({k: str(answers[k]) for k in ans_keys})
            results.append(out_row)

    if not results:
        raise ValueError("No rows found in CSV")

    # with open(out_jsonl, "w", encoding="utf-8") as f:
    #     for record in results:
    #         f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=original_fields + ans_keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. Wrote {len(results)} rows to {out_csv} and {out_jsonl}.")


if __name__ == "__main__":
    main()
