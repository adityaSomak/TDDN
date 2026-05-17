"""Maze-solve prompts: full-eval (Q1..Q15) and seg-eval (grid reconstruction).

SEG_EVAL has three modes: raw / oracle_mask / tddn_mask. Oracle masks are
ground-truth halo overlays and are accurate; TDDN masks are model-predicted
and may be noisy.
"""

EVAL_JSONL = 'maze_solve/maze_solve_eval.jsonl'

SYSTEM_PROMPT = """You are evaluating a maze puzzle. The image shows a grid-based maze. Black cells are walls. White cells are empty/passable. A green arrow marks the entry cell. A blue arrow marks the exit cell. Look at the image carefully and answer the question.
IMPORTANT: The entire grid includes outer border walls (black cells). The outermost rows and columns are typically all-black wall borders, but they ARE part of the grid — count them when determining dimensions, coordinates, and cell counts.
Indexing starts from 0 at the very first black border cell. The top-left black cell is (0, 0). The entry cell with the green arrow is typically at (1, 0) — row 1, column 0 — because row 0 is the top border wall and column 0 is where the entry sits (not column 1). Similarly, the exit cell with the blue arrow can be on the border edge.
Rows increase downward, columns increase rightward.
Adjacent cells means only the 4 direct neighbors (up, down, left, right) that exist within the grid. Cells on the grid edge have fewer than 4 neighbors."""

FORMAT_INSTRUCTIONS = {
        'coordinate': '\nRespond with only the (row, col) coordinate, e.g. (3, 5).',
        'dimensions': '\nRespond with only the dimensions as RxC, e.g. 7x13.',
        'coordinate_list': '\nRespond with only a Python-style list of (row, col) coordinates, e.g. [(1, 1), (2, 3)].',
        'coordinate_list_long': '\nRespond with only a complete Python-style list of all (row, col) coordinates. List every single coordinate. e.g. [(0, 0), (1, 2), (3, 4)].',
    }

_BASE = {
    'raw': (
        "You are given an image of a maze puzzle."
        " Note that the maze can be of any size — do not assume the dimensions."
    ),
    'tddn_mask': (
        "You are given an image of a maze puzzle with segmentation mask overlays"
        " that highlight different regions or objects as visual annotations."
        " Note that some of these annotations may be noisy, incomplete, or incorrect."
        " Also note that the maze can be of any size — do not assume the dimensions."
    ),
    'oracle_mask': (
        "You are given an image of a maze puzzle with ground-truth segmentation mask overlays"
        " that correctly highlight different regions or objects as visual annotations."
        " These annotations are accurate and trustworthy — use them as the authoritative source"
        " when identifying cell types."
        " Note that the maze can be of any size — do not assume the dimensions."
    ),
}

_STEPS = {
    'raw': (
        "1. Carefully analyze the image to understand the structure, layout, and relationships"
        " between different elements of the puzzle.\n"
        "2. Identify and describe the key components — wall cells, path cells, entry cell, and exit cell.\n"
        "3. Count the number of rows and columns carefully to determine the exact maze dimensions."
        " Do not assume a fixed size.\n"
        "4. Based on your understanding, construct a clear and consistent string representation of the puzzle.\n"
        "5. Use this representation to reason about the puzzle and answer the following question:"
    ),
    'tddn_mask': (
        "1. Carefully analyze the annotated image to understand the structure, layout, and relationships"
        " between different elements of the puzzle, while critically evaluating the reliability of the annotations.\n"
        "2. Identify and describe the key components or objects indicated by the segmentation masks,"
        " correcting any obvious annotation errors when necessary.\n"
        "3. Count the number of rows and columns carefully to determine the exact maze dimensions."
        " Do not assume a fixed size.\n"
        "4. Based on your understanding, construct a clear and consistent string representation of the puzzle.\n"
        "5. Use this representation to reason about the puzzle and answer the following question:"
    ),
    'oracle_mask': (
        "1. Carefully analyze the annotated image and use the segmentation masks as a reliable guide"
        " to the structure, layout, and relationships between different elements of the puzzle.\n"
        "2. Identify and describe the key components or objects indicated by the segmentation masks."
        " Trust the mask labels.\n"
        "3. Count the number of rows and columns carefully to determine the exact maze dimensions."
        " Do not assume a fixed size.\n"
        "4. Based on your understanding, construct a clear and consistent string representation of the puzzle.\n"
        "5. Use this representation to reason about the puzzle and answer the following question:"
    ),
}

_OUTPUT_FORMAT = """Output the full maze as a grid of semicolon-separated values — one row per line. Each cell is either:
- 1 = wall cell (black, impassable)
- 0 = path cell (white, passable)
- S = entry cell (green arrow)
- E = exit cell (blue arrow)

After your analysis, output the grid clearly labelled as:
GRID:
<your grid here>"""

SEG_EVAL = {
    mode: f"{_BASE[mode]}\n\n{_STEPS[mode]}\n\n{_OUTPUT_FORMAT}"
    for mode in ('raw', 'oracle_mask', 'tddn_mask')
}

TDDN_CLASS_PROMPTS = {
        'path': 'a plain empty white maze cell with no markings',
        'S': 'a small green right-pointing arrow inside a white maze cell, marking the entry',
        'E': 'a small blue right-pointing arrow inside a white maze cell, marking the exit',
    }
