"""Checker-move prompts: full-eval only (Q1..Q13)."""

EVAL_JSONL = 'checker_move/checker_move_eval.jsonl'

SYSTEM_PROMPT = 'You are evaluating a checkers puzzle. The image shows a horizontal checker board row. Cells contain red checkers, green checkers, or are empty (white). Look at the image carefully and answer the question.\nPositions are 0-indexed: the leftmost cell is position 0, the next is position 1, and so on.'

FORMAT_INSTRUCTIONS = {
    'mcq':           '\nRespond with only the letter (A, B, C, or D).',
    'color':         '\nRespond with only the color name (red or green).',
    'position':      '\nRespond with only the position number (0-indexed, where the leftmost cell is 0).',
    'position_list': '\nRespond with only a Python-style list of positions (0-indexed, where the leftmost cell is 0), e.g. [0, 3, 5].',
    'color_list':    "\nRespond with only a Python-style list of unique color names, e.g. ['red'] or ['red', 'green'].",
    'list':          '\nRespond with only a Python-style list.',
}
