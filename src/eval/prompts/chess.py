"""Chess prompts for seg-eval (pieces, empty, grid_black, grid_white).

Three modes per sub-task: raw / oracle_mask / tddn_mask.
Oracle masks are ground-truth outlines drawn from the board's known piece positions
and are accurate and trustworthy. TDDN masks are model-predicted and may be noisy.
"""

_BASE = {
    'raw': (
        "You are given an image of an 8x8 chess board."
    ),
    'oracle_mask': (
        "You are given an image of an 8x8 chess board with ground-truth segmentation mask overlays"
        " that outline the location and type of every chess piece."
        " These annotations are accurate and trustworthy — use them as the authoritative source"
        " for piece identification."
    ),
    'tddn_mask': (
        "You are given an image of an 8x8 chess board with segmentation mask overlays"
        " that highlight different regions or objects as visual annotations."
        " Note that some of these annotations may be noisy, incomplete, or incorrect."
    ),
}

_STEPS = {
    'raw': (
        "1. Carefully analyze the image and identify every chess piece on the board.\n"
        "2. A \"chess piece\" is any pawn, knight, bishop, rook, queen, or king of either colour."
    ),
    'oracle_mask': (
        "1. Use the outlines to identify every piece and its square.\n"
        "2. A \"chess piece\" is any pawn, knight, bishop, rook, queen, or king of either colour."
    ),
    'tddn_mask': (
        "1. Carefully analyze the annotated image, critically evaluating the reliability of the annotations.\n"
        "2. A \"chess piece\" is any pawn, knight, bishop, rook, queen, or king of either colour."
    ),
}

_TASKS = {
    'pieces': (
        "3. Count them exactly once each.\n\n"
        "After your analysis, output your final answer on a single line as:\n"
        "ANSWER: <integer>"
    ),
    'empty': (
        "3. Count the squares that are NOT occupied by any chess piece.\n"
        "   An empty square is one with no pawn, knight, bishop, rook, queen, or king —"
        " regardless of square colour.\n\n"
        "After your analysis, output your final answer on a single line as:\n"
        "ANSWER: <integer>"
    ),
    'grid_black': (
        "3. Produce a string representation of the BLACK pieces on the board using these rules:\n"
        "   - Encode each BLACK piece by its type:\n"
        "       pawn = 1\n"
        "       knight = 2\n"
        "       bishop = 3\n"
        "       rook = 4\n"
        "       queen = 5\n"
        "       king = 6\n"
        "   - A square that contains a WHITE piece is \"-\".\n"
        "   - An empty square is \"0\".\n\n"
        "Output exactly 8 rows. Each row contains 8 semicolon-separated values, one per column."
        " End each row with a newline character.\n\n"
        "After your analysis, output the grid clearly labelled as:\n"
        "GRID:\n"
        "<your grid here>"
    ),
    'grid_white': (
        "3. Produce a string representation of the WHITE pieces on the board using these rules:\n"
        "   - Encode each WHITE piece by its type:\n"
        "       pawn = 1\n"
        "       knight = 2\n"
        "       bishop = 3\n"
        "       rook = 4\n"
        "       queen = 5\n"
        "       king = 6\n"
        "   - A square that contains a BLACK piece is \"-\".\n"
        "   - An empty square is \"0\".\n\n"
        "Output exactly 8 rows. Each row contains 8 semicolon-separated values, one per column."
        " End each row with a newline character.\n\n"
        "After your analysis, output the grid clearly labelled as:\n"
        "GRID:\n"
        "<your grid here>"
    ),
}

SEG_EVAL = {
    task: {
        mode: f"{_BASE[mode]}\n\n{_STEPS[mode]}\n{_TASKS[task]}"
        for mode in ('raw', 'oracle_mask', 'tddn_mask')
    }
    for task in ('pieces', 'empty', 'grid_black', 'grid_white')
}


TDDN_CLASS_PROMPTS = {
    'background':   'the wooden frame around a chess board',
    'light_square': 'an empty light-coloured pink chess square with no piece on it',
    'dark_square':  'an empty dark-coloured blue chess square with no piece on it',
    'w_pawn':   'a white chess pawn, a short rounded piece with a small ball on top',
    'w_knight': 'a white chess knight, shaped like a horse head',
    'w_bishop': 'a white chess bishop, a tall slim piece with a pointed top and a slit',
    'w_rook':   'a white chess rook, a short cylindrical piece with battlements on top',
    'w_queen':  'a white chess queen, a tall piece with a many-pointed crown',
    'w_king':   'a white chess king, the tallest piece, topped with a small cross',
    'b_pawn':   'a black chess pawn, a short rounded dark piece with a small ball on top',
    'b_knight': 'a black chess knight, shaped like a dark horse head',
    'b_bishop': 'a black chess bishop, a tall slim dark piece with a pointed top and a slit',
    'b_rook':   'a black chess rook, a short cylindrical dark piece with battlements on top',
    'b_queen':  'a black chess queen, a tall dark piece with a many-pointed crown',
    'b_king':   'a black chess king, the tallest dark piece, topped with a small cross',
}
