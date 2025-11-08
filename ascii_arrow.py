import time
import math
from collections import defaultdict
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

console = Console()


def generate_base_dots(arrow_map):
    """Converts the character map into a canonical set of high-resolution dot coordinates."""
    dots = set()
    for y, row in enumerate(arrow_map):
        for x, char in enumerate(row):
            if char != " ":
                # A single character in the map corresponds to a 2x4 grid of dots
                for dy in range(4):
                    for dx in range(2):
                        dots.add((x * 2 + dx, y * 4 + dy))
    return dots


def rotate_dots(dots, direction):
    """Rotates the set of dot coordinates and re-centers them at (0,0)."""
    if not dots:
        return set(), (0, 0)

    if direction == 'up':
        # No rotation needed, just translate to origin
        min_x = min(d[0] for d in dots)
        min_y = min(d[1] for d in dots)
        final_dots = {(x - min_x, y - min_y) for x, y in dots}
        width = max(d[0] for d in final_dots) + 1
        height = max(d[1] for d in final_dots) + 1
        return final_dots, (width, height)

    # Find center for rotation
    min_x, max_x = min(d[0] for d in dots), max(d[0] for d in dots)
    min_y, max_y = min(d[1] for d in dots), max(d[1] for d in dots)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    angle = {
        'right': math.radians(90),
        'down': math.radians(180),
        'left': math.radians(-90)
    }[direction]

    rotated_dots = set()
    for x, y in dots:
        # Translate to origin, rotate, then translate back
        adj_x, adj_y = x - center_x, y - center_y
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        new_x = adj_x * cos_a - adj_y * sin_a
        new_y = adj_x * sin_a + adj_y * cos_a
        rotated_dots.add((round(new_x + center_x), round(new_y + center_y)))

    # Translate the entire rotated shape so its top-left corner is at (0,0)
    final_min_x = min(d[0] for d in rotated_dots)
    final_min_y = min(d[1] for d in rotated_dots)
    final_dots = {(x - final_min_x, y - final_min_y) for x, y in rotated_dots}

    width = max(d[0] for d in final_dots) + 1
    height = max(d[1] for d in final_dots) + 1

    return final_dots, (width, height)


def create_braille_arrow(arrow_dots, shape_dims, progress, fill_direction, color="white"):
    """Generates a braille arrow from a pre-calculated set of dots."""
    if not arrow_dots:
        return Text("")

    lines = defaultdict(list)
    grouping_axis = 1 if fill_direction in ['up', 'down'] else 0

    for dot in arrow_dots:
        lines[dot[grouping_axis]].append(dot)

    # --- THIS IS THE FIX ---
    # The condition for reversing the sort is now correct for all directions.
    # Left arrow fills right-to-left (descending/reverse sort on X-axis).
    # Right arrow fills left-to-right (ascending/normal sort on X-axis).
    reverse_sort = True if fill_direction in ['up', 'left'] else False
    sorted_line_coords = sorted(lines.keys(), reverse=reverse_sort)

    num_lit_lines = int(len(sorted_line_coords) * progress)

    lit_dots_set = set()
    for i in range(num_lit_lines):
        line_coord = sorted_line_coords[i]
        for dot in lines[line_coord]:
            lit_dots_set.add(dot)

    canvas_width, canvas_height = shape_dims
    braille_map = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

    final_text = Text(justify="center")
    for r in range(0, math.ceil(canvas_height / 4)):
        for c in range(0, math.ceil(canvas_width / 2)):
            char_code = 0x2800
            is_lit = False
            for i in range(4):
                for j in range(2):
                    dot_pos = (c * 2 + j, r * 4 + i)
                    if dot_pos in arrow_dots:
                        char_code |= braille_map[i][j]
                        if dot_pos in lit_dots_set:
                            is_lit = True

            style = f"bold {color}" if is_lit else "dim grey50"
            final_text.append(chr(char_code), style=style)
        final_text.append("\n")

    return final_text


def generate_layout() -> Layout:
    """Defines the 2x2 layout for the four arrows."""
    layout = Layout(name="root")
    layout.split_row(Layout(name="left_col"), Layout(name="right_col"))
    layout["left_col"].split_column(Layout(name="up"), Layout(name="down"))
    layout["right_col"].split_column(Layout(name="left"), Layout(name="right"))
    return layout


def main():
    """Main function to run the four-arrow animation."""
    base_up_arrow_map = [
        "       XX       ",
        "      XXXX      ",
        "     XXXXXX     ",
        "    XXXXXXXX    ",
        "   XXXXXXXXXX   ",
        "  XXXXXXXXXXXX  ",
        " XXXXXXXXXXXXXX ",
        "     XXXXXX     ",
        "     XXXXXX     ",
        "     XXXXXX     ",
    ]

    base_dots = generate_base_dots(base_up_arrow_map)
    up_dots, up_dims = rotate_dots(base_dots, 'up')
    down_dots, down_dims = rotate_dots(base_dots, 'down')
    left_dots, left_dims = rotate_dots(base_dots, 'left')
    right_dots, right_dims = rotate_dots(base_dots, 'right')

    layout = generate_layout()

    with Live(layout, auto_refresh=False, screen=True, vertical_overflow="visible") as live:
        try:
            animation_speed = 0.8
            animation_period = 2.0

            while True:
                cycle_time = (time.perf_counter() * animation_speed) % animation_period
                progress = 1.0 - abs(1.0 - 2 * (cycle_time / animation_period))

                up_text = create_braille_arrow(up_dots, up_dims, progress, 'up', "bright_green")
                down_text = create_braille_arrow(down_dots, down_dims, progress, 'down', "bright_red")
                left_text = create_braille_arrow(left_dots, left_dims, progress, 'left', "bright_blue")
                right_text = create_braille_arrow(right_dots, right_dims, progress, 'right', "bright_yellow")

                layout["up"].update(Panel(Align.center(up_text, vertical="middle"), title="Up", border_style="green"))
                layout["down"].update(
                    Panel(Align.center(down_text, vertical="middle"), title="Down", border_style="red"))
                layout["left"].update(
                    Panel(Align.center(left_text, vertical="middle"), title="Left", border_style="blue"))
                layout["right"].update(
                    Panel(Align.center(right_text, vertical="middle"), title="Right", border_style="yellow"))

                live.refresh()
                time.sleep(1 / 60)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()