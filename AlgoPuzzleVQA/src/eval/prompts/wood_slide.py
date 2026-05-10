"""wood_slide prompts."""

EVAL_JSONL = 'wood_slide/wood_slide_eval.jsonl'
ANSWER_TYPE_SOURCE = 'record_field'

SYSTEM_PROMPT = """You are evaluating a sliding block puzzle (Klotski). The image shows a 5-row by 4-column grid containing colored wooden blocks of varying sizes and white empty cells. Look at the image carefully and answer the question.
IMPORTANT: The grid is always exactly 5 rows and 4 columns. There are no borders or margins — every colored rectangle and white space is part of the 5x4 grid. A single block is one contiguous rectangle of the same color.
Block dimensions are HEIGHT x WIDTH (rows x columns). Pay very careful attention to the difference between 1x2 and 2x1:
  - 1x1 = a small square (1 row tall, 1 column wide)
  - 1x2 = a HORIZONTAL rectangle, WIDER than tall (1 row tall, 2 columns wide). It looks like a flat lying brick.
  - 2x1 = a VERTICAL rectangle, TALLER than wide (2 rows tall, 1 column wide). It looks like an upright standing brick.
  - 2x2 = a large square (2 rows tall, 2 columns wide)
The puzzle always contains: one 2x2, four 1x2 (horizontal), two 2x1 (vertical), and two 1x1 blocks.
Coordinates are 0-indexed (row, col). The top-left cell is (0, 0). Row 0 is the topmost row. Column 0 is the leftmost column. Rows increase downward, columns increase rightward.
'Adjacent to empty cells' means the block has at least one cell that shares an edge (up, down, left, or right) with a white empty cell. Diagonal does not count. Count each block only once even if it touches multiple empty cells."""

FORMAT_INSTRUCTIONS = {
        'coordinate_list': '\nRespond with only a Python-style list of (row, col) coordinates, e.g. [(3, 2), (3, 3)].',
        'dimension_list': "\nRespond with only a Python-style list of block dimension strings, e.g. ['2x2', '1x1', '1x2'].",
    }
