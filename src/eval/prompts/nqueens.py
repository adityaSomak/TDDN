"""nqueens prompts."""

EVAL_JSONL = 'nqueens/nqueens_eval.jsonl'
ANSWER_TYPE_SOURCE = 'record_field'

SYSTEM_PROMPT = """You are evaluating an N-Queens puzzle. The image shows a chess-style board with two alternating cell colors. Queen pieces (chess crown icons) are placed on some cells. Cells without a queen are empty. Each queen has been highlighted with a colored circle and a unique numeric ID (1, 2, 3, ...) to help you identify its exact position on the board. Use these IDs to locate each queen precisely by counting its row and column. Look at the image carefully and answer the question.
IMPORTANT: The board size varies (8x8, 9x9, 10x10, or 11x11). There are no borders or margins outside the board — every cell you see is part of the grid. You MUST first count the number of rows and columns to determine the board size before answering any question.
Indexing is 0-based. The top-left cell is (0, 0). For example, on a 10x10 board: the top-left cell is (0, 0) and the bottom-right cell is (9, 9). Row 0 is the topmost row. Column 0 is the leftmost column. Rows increase downward, columns increase rightward.
A 'row without a queen' means no queen appears anywhere in that entire row. A 'column without a queen' means no queen appears anywhere in that entire column."""

FORMAT_INSTRUCTIONS = {
    'integer_list_rows': '\nRespond with only a Python-style list of 0-indexed row numbers (row 0 is the top row), e.g. [2, 6].',
    'integer_list_cols': '\nRespond with only a Python-style list of 0-indexed column numbers (column 0 is the leftmost column), e.g. [9, 10].',
    'coordinate_list': '\nRespond with only a Python-style list of (row, col) coordinates where row 0 is the top and column 0 is the left, e.g. [(0, 2), (1, 7)].',
    'cell_dict_empty_first': "\nRespond with only a Python dictionary with exactly two keys: 'empty_cells' listing all cells without a queen, then 'occupied_cells' listing all cells with a queen. Each value is a list of (row, col) tuples. e.g. {'empty_cells': [(0, 1), (0, 2)], 'occupied_cells': [(0, 0)]}.",
    'cell_dict_occupied_first': "\nRespond with only a Python dictionary with exactly two keys: 'occupied_cells' listing all cells with a queen, then 'empty_cells' listing all cells without a queen. Each value is a list of (row, col) tuples. e.g. {'occupied_cells': [(0, 0)], 'empty_cells': [(0, 1), (0, 2)]}.",
}

_BASE = {
    'raw': (
        "You are given an image of a N-Queens chess board puzzle."
        " Note that this is NOT necessarily a standard 8x8 board —"
        " it could be any size (8x8, 9x9, 10x10, 11x11, or larger)."
        " Do not assume the dimensions."
    ),
    'tddn_mask': (
        "You are given an image of a N-Queens chess board puzzle with segmentation mask overlays"
        " that highlight different regions or objects as visual annotations."
        " Note that some of these annotations may be noisy, incomplete, or incorrect."
        " Also note that this is NOT necessarily a standard 8x8 board — it could be any size."
        " Do not assume the dimensions."
    ),
    'oracle_mask': (
        "You are given an image of a N-Queens chess board puzzle with ground-truth segmentation"
        " mask overlays that correctly highlight different regions or objects as visual annotations."
        " These annotations are accurate and trustworthy — use them as the authoritative source"
        " when locating queens."
        " Also note that this is NOT necessarily a standard 8x8 board — it could be any size."
        " Do not assume the dimensions."
    ),
}

_STEPS = {
    'raw': (
        "1. Carefully analyze the image to understand the structure, layout, and relationships"
        " between different elements of the puzzle.\n"
        "2. Identify and describe the key components — the board squares and any queen pieces placed on them.\n"
        "3. Count the number of rows and columns carefully to determine the exact board dimensions."
        " Do not assume it is 8x8.\n"
        "4. Based on your understanding, construct a clear and consistent string representation of the puzzle.\n"
        "5. Use this representation to reason about the puzzle and answer the following question:"
    ),
    'tddn_mask': (
        "1. Carefully analyze the annotated image to understand the structure, layout, and relationships"
        " between different elements of the puzzle, while critically evaluating the reliability of the annotations.\n"
        "2. Identify and describe the key components or objects indicated by the segmentation masks,"
        " correcting any obvious annotation errors when necessary.\n"
        "3. Count the number of rows and columns carefully to determine the exact board dimensions."
        " Do not assume it is 8x8.\n"
        "4. Based on your understanding, construct a clear and consistent string representation of the puzzle.\n"
        "5. Use this representation to reason about the puzzle and answer the following question:"
    ),
    'oracle_mask': (
        "1. Carefully analyze the annotated image and use the segmentation masks as a reliable guide"
        " to the structure, layout, and relationships between different elements of the puzzle.\n"
        "2. Identify and describe the key components or objects indicated by the segmentation masks."
        " Trust the mask labels.\n"
        "3. Count the number of rows and columns carefully to determine the exact board dimensions."
        " Do not assume it is 8x8.\n"
        "4. Based on your understanding, construct a clear and consistent string representation of the puzzle.\n"
        "5. Use this representation to reason about the puzzle and answer the following question:"
    ),
}

_OUTPUT_FORMAT = """Output the full board as a grid of semicolon-separated values — one row per line. Each cell is either:
- Q = a queen piece is placed here
- 0 = the cell is empty

After your analysis, output the grid clearly labelled as:
GRID:
<your grid here>"""

SEG_EVAL = {
    mode: f"{_BASE[mode]}\n\n{_STEPS[mode]}\n\n{_OUTPUT_FORMAT}"
    for mode in ('raw', 'oracle_mask', 'tddn_mask')
}

TDDN_CLASS_PROMPTS = {
    'empty_square': 'an empty pink or blue square on a checkered chess board, with no piece on it',
    'queen': 'a chess queen — a tall white piece with a many-pointed crown on top',
}
