"""
Generate a checker‐move puzzle image from a semicolon‑separated input string.

This script accepts an input string where each semicolon‑separated value
represents the content of a cell on a rectangular board.  The values

* ``0`` – an empty cell (no checker)
* ``1`` – a red checker
* ``2`` – a green checker

The script draws a grid of cells and places coloured checkers on top of
the cells according to the provided input.  The size of the board is
determined automatically: all factor pairs of the total number of cells
are calculated and the pair with the smallest difference is chosen.  For
example, an input string with 18 entries produces a 3×6 board.  Once
created, the resulting image is saved to the output file specified.

Requirements:
    * Pillow (the Python Imaging Library fork).  Pillow’s ``Image`` and
      ``ImageDraw`` modules are used to create and draw on the image.  The
      ``Image.new()`` method constructs a new blank image, while
      ``ImageDraw.Draw()`` returns a drawing context.  To draw the
      checkers we use ``ellipse()`` which, according to the Pillow
      documentation, draws an ellipse within a bounding box and can fill
      it with a colour and optionally draw an outline【196590540375716†L523-L590】.

Usage:

    python checker_puzzle.py "2;2;2;2;2;2;2;2;2;0;1;2;1;2;2;2;2;2" output.png

If no output filename is provided, the script will default to
``checker_puzzle.png``.
"""

import math
import sys
from typing import Tuple
import pandas as pd
from tqdm import tqdm
import glob

from PIL import Image, ImageDraw  # Pillow is used for image creation
import shutil
import random

random.seed(42)


def factor_pair_closest_to_square(n: int) -> Tuple[int, int]:
    """Return the factor pair of ``n`` with the smallest difference.

    The returned tuple is ordered such that the smaller factor comes first
    (rows, cols).  For example, for ``n = 18``, the function returns
    ``(3, 6)`` because ``3 × 6 = 18`` and the difference between 3 and 6
    is smaller than the difference between the other factor pairs
    (1×18 and 2×9).

    :param n: total number of items (cells)
    :returns: a tuple of integers (rows, cols)
    """
    # Start with a large difference so any factor pair improves upon it.
    best_pair = (1, n)
    best_diff = abs(n - 1)
    # Only need to check up to sqrt(n) for factorisation.
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            j = n // i
            diff = abs(i - j)
            if diff < best_diff:
                best_diff = diff
                best_pair = (i, j)
    return best_pair


def generate_checker_puzzle(
    board_string: str,
    cell_size: int = 500,
    margin: int = 10,
    output_path: str = "checker_puzzle.png",
    *,
    light_color: Tuple[int, int, int] = (238, 238, 210),
    dark_color: Tuple[int, int, int] = (118, 150, 86),
    red_colour: Tuple[int, int, int] = (220, 20, 60),
    red_outline: Tuple[int, int, int] = (139, 0, 0),
    green_colour: Tuple[int, int, int] = (34, 139, 34),
    green_outline: Tuple[int, int, int] = (0, 100, 0),
    horizontal: bool = False,
    cell_bg: Tuple[int, int, int] = (255, 255, 255),
    cell_border: Tuple[int, int, int] = (0, 0, 0),
    cell_border_width: int = 5
) -> None:
    """Create a checker puzzle image based on the supplied board string.

    By default, the board is arranged in the most square‑like factor pair of
    the number of items (for example, 18 items becomes a 3×6 grid).  Set
    ``horizontal=True`` to force the puzzle into a single row, which is
    useful for simple “checker move” puzzles that require a horizontal
    arrangement.  When ``horizontal`` is enabled, each cell is drawn with a
    white background and a black border as required for certain puzzle
    representations.  The coloured discs are drawn using ``ellipse()``,
    which fills an ellipse (circle) with a colour and allows an optional
    outline【196590540375716†L523-L590】.  Rectangles for cell backgrounds are drawn
    using ``rectangle()`` with both fill and outline parameters【973090375821017†L27-L36】.

    :param board_string: A semicolon‑separated string of integers
                         representing the board contents (0, 1, 2).
    :param cell_size: Side length of each square cell in pixels.
    :param margin: Padding between the checker and the edge of the square.
    :param output_path: File path where the image will be saved.
    :param light_color: RGB triple for the light squares (used in grid mode).
    :param dark_color: RGB triple for the dark squares (used in grid mode).
    :param red_colour: RGB triple for the fill colour of red checkers.
    :param red_outline: RGB triple for the outline of red checkers.
    :param green_colour: RGB triple for the fill colour of green checkers.
    :param green_outline: RGB triple for the outline of green checkers.
    :param horizontal: If ``True``, draw the puzzle as a single row with
                       white backgrounds and black borders.
    :param cell_bg: Background colour for each cell in horizontal mode.
    :param cell_border: Border colour for each cell in horizontal mode.
    :returns: None (writes the image to disk).
    """
    # Parse the input into a list of values, trimming any whitespace.
    cells = [c.strip() for c in board_string.split(';') if c.strip()]
    total_cells = len(cells)
    if total_cells == 0:
        raise ValueError("The input string must contain at least one cell value.")

    # Decide on board dimensions.  If horizontal, use a single row.
    if horizontal:
        rows, cols = 1, total_cells
    else:
        # Determine board dimensions: choose factor pair with smallest difference.
        rows, cols = factor_pair_closest_to_square(total_cells)
        # Validate that the number of values fits neatly into the grid.
        if rows * cols != total_cells:
            raise ValueError(
                f"Cannot determine a rectangular board for {total_cells} values; "
                f"tried {rows}×{cols}.")

    # Create the blank image.  In horizontal mode each cell background will be
    # drawn explicitly, so we initialise with a neutral colour (white).
    width, height = cols * cell_size, rows * cell_size
    base_color = cell_bg if horizontal else light_color
    image = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(image)

    # Iterate through each cell and draw the square and, if applicable, the checker.
    for idx, cell_value in enumerate(cells):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_size
        y0 = row * cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        if horizontal:
            # In horizontal mode each cell has a white background and a black border.
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=cell_bg,
                outline=cell_border,
                width=cell_border_width
            )
        else:
            # In grid mode, alternate colours for a checkerboard pattern.
            square_color = dark_color if (row + col) % 2 else light_color
            draw.rectangle((x0, y0, x1, y1), fill=square_color)

        # Draw the checker disc if the cell is not empty.
        if cell_value == '1':
            # Green checker
            draw.ellipse(
                (x0 + margin, y0 + margin, x1 - margin, y1 - margin),
                fill=green_colour,
                outline=green_outline if not horizontal else None,
            )
        elif cell_value == '2':
            # Red checker
            draw.ellipse(
                (x0 + margin, y0 + margin, x1 - margin, y1 - margin),
                fill=red_colour,
                outline=red_outline if not horizontal else None,
            )
        # else: 0 – no checker; leave cell blank

    # Save the image to the specified path.
    image.save(output_path)

def count_checkers_relative_to_empty_cell(s, color, direction):
    parts = s.split(";")
    
    if "0" not in parts:
        return 0  # or None if you prefer
    
    zero_index = parts.index("0")
    
    if direction == "left":
        return sum(1 for x in parts[:zero_index] if x == color)
    elif direction == "right":
        return sum(1 for x in parts[zero_index+1:] if x == color)
    else:
        raise ValueError("direction must be 'left' or 'right'")

def find_empty_cell_index(s):
    try:
        parts = [int(checker.strip()) for checker in str(s).split(";")]
    except Exception as e:
        print("Bad row:", s)
        raise e
    
    return parts.index(0) if 0 in parts else -1
        
def find_colors_adjacent_to_empty_cell(s):
    empty_cell_index = find_empty_cell_index(s)
    colors = []
    if empty_cell_index - 1 >= 0:
        colors.append("green" if s.split(";")[empty_cell_index - 1] == "1" else "red")
    if empty_cell_index + 1 <= len(s.split(";")) - 1:
        colors.append("green" if s.split(";")[empty_cell_index + 1] == "1" else "red")
    return colors

def count_color1_between_adjacent_color2s(s, color1, color2):
    parts = [x.strip() for x in str(s).split(";")]
    color1 = str(color1)
    color2 = str(color2)

    color2_positions = [i for i, x in enumerate(parts) if x == color2]

    if len(color2_positions) < 2:
        return 0

    count = 0
    for i in range(len(color2_positions) - 1):
        left = color2_positions[i]
        right = color2_positions[i + 1]
        count += sum(x == color1 for x in parts[left + 1:right])

    return count

def make_mcq_options_q1(x, min_num_checkers):
    choices = [i for i in range(x - min_num_checkers, x + min_num_checkers) if i != x]
    options = [int(x)] + random.sample(choices, 3)
    random.shuffle(options)
    return options

import random

import random
import pandas as pd

def make_mcq_options_q2(x, y, min_num_colored_checkers):
    if pd.isna(y) or pd.isna(min_num_colored_checkers):
        return None

    y = int(y)
    min_num_colored_checkers = max(0, int(min_num_colored_checkers))

    range_start = max(0, y - min_num_colored_checkers)
    range_end = y + min_num_colored_checkers + 1

    choices = [i for i in range(range_start, range_end) if i != y]

    while len(choices) < 2:
        range_start = max(0, range_start - 1)
        range_end += 1
        choices = [i for i in range(range_start, range_end) if i != y]

    options = [x-y, y] + random.sample(choices, 2)
    random.shuffle(options)
    return options

def main(argv: list[str]) -> None:
    
    horizontal = True
    filepath = "checker_move.csv"
    df_text = pd.read_csv(filepath)
    df_text["start_image_path_v2"] = df_text["start_image_path"].apply(lambda x: x.replace(".jpg", "_v2.jpg"))
    df_text["end_image_path_v2"] = df_text["end_image_path"].apply(lambda x: x.replace(".jpg", "_v2.jpg"))
    
    # all_start_images = glob.glob("images/*/*start.jpg")
    # all_end_images = glob.glob("images/*/*end.jpg")
    # print(f"Total number of start/end images: {len(all_start_images)}")
    # start_images_to_remove = set(all_start_images) - set(list(df_text["start_image_path"]))
    # end_images_to_remove = set(all_end_images) - set(list(df_text["end_image_path"]))
    # print(f"Total number of start/end images to remove: {len(start_images_to_remove)}")
    df_text["q1-ans"] = df_text["text-representation_start-position"].apply(lambda x: len([checker for checker in x.split(";") if checker in ["1","2"]]))
    min_num_checkers = df_text["q1-ans"].min()
    df_text["q1-mcq-options"] = df_text["q1-ans"].apply(make_mcq_options_q1, args=(min_num_checkers,))
    
    df_text["q2-ans_red"] = df_text["text-representation_start-position"].apply(lambda x: len([checker for checker in x.split(";") if checker in ["2"]]))
    min_num_red_checkers = df_text["q2-ans_red"].min()
    df_text["q2-mcq-options_red"] = df_text.apply(lambda row: make_mcq_options_q2(row["q1-ans"], row["q2-ans_red"], min_num_red_checkers), axis=1)
    df_text["q2-ans_green"] = df_text["text-representation_start-position"].apply(lambda x: len([checker for checker in x.split(";") if checker in ["1"]]))
    min_num_green_checkers = df_text["q2-ans_green"].min()
    df_text["q2-mcq-options_green"] = df_text.apply(lambda row: make_mcq_options_q2(row["q1-ans"], row["q2-ans_green"], min_num_green_checkers), axis=1)
    
    df_text["q3-ans_red-green"] = df_text["q2-ans_red"] > df_text["q2-ans_green"]
    df_text["q3-ans_green-red"] = df_text["q2-ans_green"] > df_text["q2-ans_red"]
    
    df_text["q4-ans"] = df_text["q3-ans_red-green"].apply(lambda x: "red" if x else "green")
    
    df_text["q5-ans_red-left_start-position"] = df_text["text-representation_start-position"].apply(count_checkers_relative_to_empty_cell, args=("2", "left"))
    df_text["q5-ans_red-right_start-position"] = df_text["text-representation_start-position"].apply(count_checkers_relative_to_empty_cell, args=("2", "right"))
    df_text["q5-ans_green-left_start-position"] = df_text["text-representation_start-position"].apply(count_checkers_relative_to_empty_cell, args=("1", "left"))
    df_text["q5-ans_green-right_start-position"] = df_text["text-representation_start-position"].apply(count_checkers_relative_to_empty_cell, args=("1", "right"))
    df_text["q5-ans_red-left_end-position"] = df_text["text-representation_end-position"].apply(count_checkers_relative_to_empty_cell, args=("2", "left"))
    df_text["q5-ans_red-right_end-position"] = df_text["text-representation_end-position"].apply(count_checkers_relative_to_empty_cell, args=("2", "right"))
    df_text["q5-ans_green-left_end-position"] = df_text["text-representation_end-position"].apply(count_checkers_relative_to_empty_cell, args=("1", "left"))
    df_text["q5-ans_green-right_end-position"] = df_text["text-representation_end-position"].apply(count_checkers_relative_to_empty_cell, args=("1", "right"))


    # print(df_text.loc[0, "text-representation_start-position"].split(";").index("0"))
    
    df_text["q6-ans_left-greater_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) > len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q6-ans_left-lower_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) < len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q6-ans_right-greater_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) < len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q6-ans_right-lower_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) > len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q6-ans_left-greater_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) > len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q6-ans_left-lower_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) < len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q6-ans_right-greater_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) < len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q6-ans_right-lower_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]) > len(x.split(";")[find_empty_cell_index(x)+1:]))
    
    df_text["q7-ans"] = df_text["text-representation_start-position"].apply(lambda x: x.count("1") == x.count("2"))
    
    df_text["q8-ans_left_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]))
    df_text["q8-ans_right_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")[find_empty_cell_index(x)+1:]))
    df_text["q8-ans_left_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")[:find_empty_cell_index(x)]))
    df_text["q8-ans_right_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")[find_empty_cell_index(x)+1:]))
    
    df_text["q9-ans_green_start-position"] = df_text["text-representation_start-position"].apply(count_color1_between_adjacent_color2s, args=("1", "2"))
    df_text["q9-ans_red_start-position"] = df_text["text-representation_start-position"].apply(count_color1_between_adjacent_color2s, args=("2", "1"))
    df_text["q9-ans_green_end-position"] = df_text["text-representation_end-position"].apply(count_color1_between_adjacent_color2s, args=("1", "2"))
    df_text["q9-ans_red_end-position"] = df_text["text-representation_end-position"].apply(count_color1_between_adjacent_color2s, args=("2", "1"))
    
    
    df_text["q10-ans_start-position"] = df_text["text-representation_start-position"].apply(find_colors_adjacent_to_empty_cell)
    df_text["q10-ans_end-position"] = df_text["text-representation_end-position"].apply(find_colors_adjacent_to_empty_cell)
    
    df_text["q11-ans_start-position"] = df_text["text-representation_start-position"].apply(find_empty_cell_index)
    df_text["q11-ans_end-position"] = df_text["text-representation_end-position"].apply(find_empty_cell_index)
    
    df_text["q12-ans_green-first_start-position"] = df_text["text-representation_start-position"].apply(lambda x: x.split(";").index("1"))
    df_text["q12-ans_red-first_start-position"] = df_text["text-representation_start-position"].apply(lambda x: x.split(";").index("2"))    
    df_text["q12-ans_green-last_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")) - 1 - x.split(";")[::-1].index("1"))
    df_text["q12-ans_red-last_start-position"] = df_text["text-representation_start-position"].apply(lambda x: len(x.split(";")) - 1 - x.split(";")[::-1].index("2"))    
    df_text["q12-ans_green-first_end-position"] = df_text["text-representation_end-position"].apply(lambda x: x.split(";").index("1"))
    df_text["q12-ans_red-first_end-position"] = df_text["text-representation_end-position"].apply(lambda x: x.split(";").index("2"))
    df_text["q12-ans_green-last_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")) - 1 - x.split(";")[::-1].index("1"))
    df_text["q12-ans_red-last_end-position"] = df_text["text-representation_end-position"].apply(lambda x: len(x.split(";")) - 1 - x.split(";")[::-1].index("2"))

    df_text["q13-ans_green_start-position"] = df_text["text-representation_start-position"].apply(lambda x: [i for i, checker in enumerate(x.split(";")) if checker=="1"])
    df_text["q13-ans_red_start-position"] = df_text["text-representation_start-position"].apply(lambda x: [i for i, checker in enumerate(x.split(";")) if checker=="2"])
    df_text["q13-ans_green_end-position"] = df_text["text-representation_end-position"].apply(lambda x: [i for i, checker in enumerate(x.split(";")) if checker=="1"])
    df_text["q13-ans_red_end-position"] = df_text["text-representation_end-position"].apply(lambda x: [i for i, checker in enumerate(x.split(";")) if checker=="2"])

    # for idx in tqdm(list(df_text.index), desc="Processing input text representations..."):
    #     text_representation_start_position = df_text.loc[idx, "text-representation_start-position"]
    #     text_representation_end_position = df_text.loc[idx, "text-representation_end-position"]
    #     output_start_image_path = df_text.loc[idx, "start_image_path_v2"]
    #     output_end_image_path = df_text.loc[idx, "end_image_path_v2"]
        
    #     # print(output_start_image_path)
    #     generate_checker_puzzle(text_representation_start_position, output_path=output_start_image_path, horizontal=horizontal)
    #     generate_checker_puzzle(text_representation_end_position, output_path=output_end_image_path, horizontal=horizontal)
    
    df_text.to_csv(filepath.replace(".csv", "_v2.csv"), index=False)

    


if __name__ == "__main__":
    main(sys.argv[1:])