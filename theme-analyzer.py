#!/usr/bin/env python3
"""
Terminal Theme Accessibility Analyzer

Analyzes terminal color themes for WCAG contrast compliance and generates
visual mockups to preview themes with 24-bit true colors.

Themes are loaded from ./themes/ directory in lexicographical order.

Usage:
    python theme-analyzer.py         # Interactive mode
    python theme-analyzer.py --help  # Show help
"""

import math
import re
import sys
import termios
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

import pyte
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.text import Text

# =============================================================================
# Constants & Configuration
# =============================================================================

RESET = "\033[0m"

COLOR_NAMES = {
    0: "black", 1: "red", 2: "green", 3: "yellow",
    4: "blue", 5: "magenta", 6: "cyan", 7: "white",
    8: "br.blk", 9: "br.red", 10: "br.grn", 11: "br.yel",
    12: "br.blu", 13: "br.mag", 14: "br.cyn", 15: "br.wht",
}

COLOR_NAMES_LONG = {
    0: "black", 1: "red", 2: "green", 3: "yellow",
    4: "blue", 5: "magenta", 6: "cyan", 7: "white",
    8: "br.black", 9: "br.red", 10: "br.green", 11: "br.yellow",
    12: "br.blue", 13: "br.magenta", 14: "br.cyan", 15: "br.white",
}

PYTE_COLOR_MAP = {
    'default': 7,
    'black': 0, 'red': 1, 'green': 2, 'brown': 3,
    'blue': 4, 'magenta': 5, 'cyan': 6, 'white': 7,
    'brightblack': 8, 'brightred': 9, 'brightgreen': 10, 'brightbrown': 11,
    'brightblue': 12, 'brightmagenta': 13, 'brightcyan': 14, 'brightwhite': 15,
}

# =============================================================================
# Color Utilities
# =============================================================================

class ColorUtils:
    """Static utilities for color conversion and analysis."""

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    @staticmethod
    def is_valid_hex_color(s: str) -> bool:
        """Check if string is a valid hex color (#RRGGBB)."""
        if not s.startswith("#") or len(s) != 7:
            return False
        try:
            int(s[1:], 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def rgb_to_oklab(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB to Oklab color space."""
        def linearize(c: int) -> float:
            c_srgb = c / 255.0
            if c_srgb <= 0.04045:
                return c_srgb / 12.92
            return ((c_srgb + 0.055) / 1.055) ** 2.4

        lr, lg, lb = linearize(r), linearize(g), linearize(b)

        l_ = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb
        m_ = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb
        s_ = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb

        l_ = l_ ** (1/3) if l_ > 0 else 0
        m_ = m_ ** (1/3) if m_ > 0 else 0
        s_ = s_ ** (1/3) if s_ > 0 else 0

        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        return L, a, ok_b

    @staticmethod
    def rgb_to_oklch(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB to OKLCH (Lightness, Chroma, Hue)."""
        L, a, ok_b = ColorUtils.rgb_to_oklab(r, g, b)
        C = math.sqrt(a * a + ok_b * ok_b)
        H = math.degrees(math.atan2(ok_b, a))
        if H < 0:
            H += 360
        return L, C, H

    @staticmethod
    def hue_distance(h1: float, h2: float) -> float:
        """Angular distance between two hues (handles wraparound)."""
        diff = abs(h1 - h2)
        if diff > 180:
            diff = 360 - diff
        return diff

    @staticmethod
    def relative_luminance(hex_color: str) -> float:
        """Calculate relative luminance per WCAG 2.1 specification."""
        r, g, b = ColorUtils.hex_to_rgb(hex_color)

        def linearize(c: int) -> float:
            c_srgb = c / 255
            if c_srgb <= 0.03928:
                return c_srgb / 12.92
            return ((c_srgb + 0.055) / 1.055) ** 2.4

        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

    @staticmethod
    def contrast_ratio(c1: str, c2: str) -> float:
        """Calculate WCAG contrast ratio between two colors."""
        l1 = ColorUtils.relative_luminance(c1)
        l2 = ColorUtils.relative_luminance(c2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)


# =============================================================================
# Models
# =============================================================================

@dataclass
class Theme:
    """Terminal color theme with 16 ANSI colors."""
    name: str
    colors: Dict[int, str]  # 0-15 -> hex color

    def fg(self, idx: int) -> str:
        """Get foreground escape sequence using 24-bit true color."""
        hex_color = self.colors.get(idx, "#ffffff")
        r, g, b = ColorUtils.hex_to_rgb(hex_color)
        return f"\033[38;2;{r};{g};{b}m"

    def bg(self, idx: int) -> str:
        """Get background escape sequence using 24-bit true color."""
        hex_color = self.colors.get(idx, "#000000")
        r, g, b = ColorUtils.hex_to_rgb(hex_color)
        return f"\033[48;2;{r};{g};{b}m"

    def to_ghostty(self) -> str:
        """Export theme in Ghostty format."""
        lines = [f"# {self.name}", "#"]
        for i in range(16):
            lines.append(f"palette = {i}={self.colors[i]}")

        # Add standard theme properties
        lines.append("")
        lines.append(f"background = {self.colors[0]}")
        lines.append(f"foreground = {self.colors[7]}")
        lines.append("")
        lines.append(f"cursor-color = {self.colors[11]}")  # bright yellow
        lines.append(f"cursor-text = {self.colors[0]}")
        lines.append("")
        lines.append(f"selection-background = {self.colors[4]}")
        lines.append("selection-foreground = #ffffff")

        return "\n".join(lines)


@dataclass
class EditState:
    """State for inline palette editing."""
    active: bool = False
    selected: int = 0
    editing: bool = False
    buffer: str = ""
    colors: Optional[Dict[int, str]] = None  # Working copy of colors


class KeyEvent(NamedTuple):
    type: str
    value: str


# =============================================================================
# Theme Management
# =============================================================================

class ThemeManager:
    """Handles loading and saving of themes."""

    def __init__(self, themes_dir: Path):
        self.themes_dir = themes_dir

    def load_themes(self) -> List[Theme]:
        """Load all themes from themes directory in lexicographical order."""
        if not self.themes_dir.exists():
            return []

        themes = []
        for filepath in sorted(self.themes_dir.iterdir()):
            if filepath.is_file() and not filepath.name.startswith("."):
                try:
                    themes.append(self._parse_ghostty_theme(filepath))
                except Exception as e:
                    # Silently skip malformed files or print error?
                    # Printing might disrupt TUI, better to skip or log elsewhere.
                    pass
        return themes

    def _parse_ghostty_theme(self, filepath: Path) -> Theme:
        """Parse a Ghostty theme file into a Theme object."""
        content = filepath.read_text()
        lines = content.split("\n")

        name = filepath.name
        colors: Dict[int, str] = {}

        for line in lines:
            line = line.strip()
            if line.startswith("palette = "):
                rest = line[10:]
                if "=" in rest:
                    idx_str, color = rest.split("=", 1)
                    idx = int(idx_str.strip())
                    colors[idx] = color.strip()

        if len(colors) != 16:
            # Fallback or strict? Strict for now as per original code
            raise ValueError(f"Theme {name} has {len(colors)} colors, expected 16")
        
        return Theme(name=name, colors=colors)

    def save_theme(self, theme: Theme) -> Path:
        """Save theme as Ghostty theme file."""
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        safe_name = theme.name.replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")
        filepath = self.themes_dir / safe_name
        filepath.write_text(theme.to_ghostty())
        return filepath


class ScreenshotManager:
    """Manages loading and caching of screenshot text files."""

    def __init__(self, screenshots_dir: Path):
        self.screenshots_dir = screenshots_dir
        self.cache: Dict[str, str] = {}

    def get_raw(self, name: str) -> Optional[str]:
        """Load a screenshot file from the screenshots directory."""
        if name in self.cache:
            return self.cache[name]

        filepath = self.screenshots_dir / f"{name}.txt"
        if filepath.exists():
            content = filepath.read_text()
            self.cache[name] = content
            return content
        return None


# =============================================================================
# Rendering & TUI
# =============================================================================

class ThemeRenderer:
    """Handles generation of Rich tables and mockup rendering."""

    def __init__(self, screenshot_manager: ScreenshotManager):
        self.screenshot_manager = screenshot_manager

    @staticmethod
    def get_contrast_color(cr: float) -> str:
        if cr >= 4.5: return "green"
        if cr >= 3.0: return "yellow"
        return "red"

    @staticmethod
    def get_status_icon(cr: float) -> str:
        if cr >= 4.5: return "[green]✓[/]"
        if cr >= 3.0: return "[yellow]~[/]"
        return "[red]✗[/]"

    def _create_contrast_cell(self, text: str, fg: str, bg: str) -> Tuple[Text, str, str]:
        """Create a styled text cell with contrast info."""
        cr = ColorUtils.contrast_ratio(fg, bg)
        fg_rgb = ColorUtils.hex_to_rgb(fg)
        bg_rgb = ColorUtils.hex_to_rgb(bg)
        
        styled = Text(text, style=Style(
            color=f"rgb({fg_rgb[0]},{fg_rgb[1]},{fg_rgb[2]})",
            bgcolor=f"rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})"
        ))
        
        return styled, f"[{self.get_contrast_color(cr)}]{cr:.1f}[/]", self.get_status_icon(cr)

    def create_palette_table(self, theme: Theme, edit_state: Optional[EditState] = None) -> Table:
        colors = edit_state.colors if edit_state and edit_state.colors else theme.colors
        bg_color = colors[0]

        title = "Palette [Edit Mode]" if edit_state and edit_state.active else "Palette"
        table = Table(title=title, box=None, show_header=True, padding=(0, 1))
        table.add_column("#", style="dim", width=2)
        table.add_column("Name", width=10)
        table.add_column("Hex", width=9)
        table.add_column("", width=6)  # Swatch
        table.add_column("CR", width=5)
        table.add_column("", width=2)  # Status

        for i in range(16):
            color = colors[i]
            is_selected = edit_state and edit_state.active and edit_state.selected == i

            if is_selected and edit_state.editing:
                display_hex = edit_state.buffer + "_"
                test_color = edit_state.buffer if edit_state.buffer.startswith("#") else "#" + edit_state.buffer
                if ColorUtils.is_valid_hex_color(test_color):
                    r, g, b = ColorUtils.hex_to_rgb(test_color)
                else:
                    r, g, b = 128, 128, 128
            else:
                display_hex = color
                r, g, b = ColorUtils.hex_to_rgb(color)

            cr = ColorUtils.contrast_ratio(color, bg_color)
            swatch = Text("      ", style=Style(bgcolor=f"rgb({r},{g},{b})"))
            
            cr_color = self.get_contrast_color(cr)
            status = self.get_status_icon(cr)

            if i == 0:
                cr_str = "---"
                status = ""
            else:
                cr_str = f"[{cr_color}]{cr:.1f}[/]"

            row_style = "reverse" if is_selected else None
            table.add_row(
                str(i), COLOR_NAMES_LONG[i], display_hex, swatch, cr_str, status, style=row_style
            )
        return table

    def create_constraint_table(self, theme: Theme) -> Table:
        table = Table(title="Constraints", box=None, padding=(0, 1))
        table.add_column("Color", width=7)
        table.add_column("CR", width=5)
        table.add_column("", width=4)

        bg = theme.colors[0]
        # Base colors (1-6)
        base_ratios = []
        for i in range(1, 7):
            cr = ColorUtils.contrast_ratio(theme.colors[i], bg)
            base_ratios.append(cr)
            color = self.get_contrast_color(cr)
            status = self.get_status_icon(cr)
            table.add_row(COLOR_NAMES[i], f"[{color}]{cr:.1f}[/]", status)

        min_base = min(base_ratios) if base_ratios else 0
        min_color = self.get_contrast_color(min_base)
        table.add_row("", "", "")
        table.add_row("[dim]min[/]", f"[{min_color}]{min_base:.1f}[/]", "")
        return table

    def create_bright_table(self, theme: Theme) -> Table:
        table = Table(title="Bright/Reg", box=None, padding=(0, 1))
        table.add_column("Pair", width=18)
        table.add_column("CR", width=5)
        table.add_column("", width=2)

        pairs = [(8, 0)] + [(i + 8, i) for i in range(1, 7)] + [(15, 7)]

        for bright_idx, base_idx in pairs:
            base = theme.colors[base_idx]
            bright = theme.colors[bright_idx]
            
            pair_name = f" {COLOR_NAMES[bright_idx]} on {COLOR_NAMES[base_idx]} "
            styled, cr_str, status = self._create_contrast_cell(pair_name, bright, base)
            table.add_row(styled, cr_str, status)

        return table

    def create_fm_pairs_table(self, theme: Theme) -> Table:
        table = Table(title="FM Pairs", box=None, padding=(0, 1))
        table.add_column("On Blue", width=18)
        table.add_column("CR", width=5)
        table.add_column("", width=2)
        table.add_column("On Green", width=18)
        table.add_column("CR", width=5)
        table.add_column("", width=2)

        on_blue_idxs = [7, 3, 5, 6, 2, 1]
        on_green_idxs = [7, 3, 5, 4, 1, 6]

        for i in range(len(on_blue_idxs)):
            row = []
            # Blue column
            fg = theme.colors[on_blue_idxs[i]]
            bg = theme.colors[4]
            name = f" {COLOR_NAMES[on_blue_idxs[i]]} on blue "
            styled, cr_str, status = self._create_contrast_cell(name, fg, bg)
            row.extend([styled, cr_str, status])

            # Green column
            fg = theme.colors[on_green_idxs[i]]
            bg = theme.colors[2]
            name = f" {COLOR_NAMES[on_green_idxs[i]]} on green "
            styled, cr_str, status = self._create_contrast_cell(name, fg, bg)
            row.extend([styled, cr_str, status])

            table.add_row(*row)
        return table

    def create_sample_matrix_lines(self, theme: Theme) -> List[str]:
        lines = ["Sample Matrix (FG on BG)"]
        header = "FG\\BG"
        for bg in range(16):
            header += f" {bg:02d} "
        lines.append(header)

        for fg in range(16):
            row = f"  {fg:02d} "
            fg_r, fg_g, fg_b = ColorUtils.hex_to_rgb(theme.colors[fg])
            for bg in range(16):
                bg_r, bg_g, bg_b = ColorUtils.hex_to_rgb(theme.colors[bg])
                row += f"\033[38;2;{fg_r};{fg_g};{fg_b};48;2;{bg_r};{bg_g};{bg_b}m {fg:02d} {RESET}"
            lines.append(row)
        return lines

    def create_swatches_lines(self, theme: Theme) -> List[str]:
        row1 = " 0-7: "
        for i in range(8):
            row1 += f"{theme.bg(i)}  {i}  {RESET}"
        row2 = "8-15: "
        for i in range(8, 16):
            row2 += f"{theme.bg(i)} {i:2d}  {RESET}"
        return [row1, row2]

    def create_hue_spacing_lines(self, theme: Theme) -> List[str]:
        base_hues = []
        for i in range(1, 7):
            r, g, b = ColorUtils.hex_to_rgb(theme.colors[i])
            _, _, H = ColorUtils.rgb_to_oklch(r, g, b)
            base_hues.append(H)

        min_dist = 360.0
        min_pair = ("", "")
        for i in range(6):
            for j in range(i + 1, 6):
                dist = ColorUtils.hue_distance(base_hues[i], base_hues[j])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (COLOR_NAMES[i + 1], COLOR_NAMES[j + 1])

        status = "\033[32m✓ Good\033[0m" if min_dist >= 50 else \
                 "\033[33m~ OK\033[0m" if min_dist >= 30 else \
                 "\033[31m✗ Close\033[0m"

        return [
            "Hue Spacing (ideal: 60°)",
            f"  Min: {min_dist:.0f}° ({min_pair[0]}/{min_pair[1]}) {status}"
        ]

    def render_screenshot(self, name: str, width: int, height: int, theme: Theme) -> List[str]:
        raw = self.screenshot_manager.get_raw(name)
        if not raw:
            return []

        screen = pyte.Screen(width, height)
        stream = pyte.Stream(screen)
        
        # Position cursor at start of lines to emulate file dump
        for i, line in enumerate(raw.split("\n")[:height]):
            stream.feed(f"\033[{i+1};1H{line}")
            # extend bg color
            if screen.cursor.x < width:
                stream.feed(" " * (width - screen.cursor.x))

        result = []
        for y in range(height):
            line_str = ""
            last_fg, last_bg = None, None
            
            for x in range(width):
                char = screen.buffer[y][x]
                fg_idx = PYTE_COLOR_MAP.get(char.fg, 7)
                bg_idx = PYTE_COLOR_MAP.get(char.bg, 0) if char.bg != 'default' else 0
                
                if (fg_idx, bg_idx) != (last_fg, last_bg):
                    line_str += theme.fg(fg_idx) + theme.bg(bg_idx)
                    last_fg, last_bg = fg_idx, bg_idx
                
                line_str += char.data if char.data else " "
            
            line_str += RESET
            result.append(line_str)
        
        return result


# =============================================================================
# Main Application
# =============================================================================

class App:
    def __init__(self):
        self.console = Console()
        self.capture_console = Console(force_terminal=True, width=55, no_color=False)
        
        base_dir = Path(__file__).parent
        self.theme_manager = ThemeManager(base_dir / "themes")
        self.screenshot_manager = ScreenshotManager(base_dir / "screenshots")
        self.renderer = ThemeRenderer(self.screenshot_manager)
        
        self.themes: List[Theme] = []
        self.current_index = 0
        self.modified_themes: Dict[int, Theme] = {}
        self.edit_state = EditState()
        self.running = True

    def run(self):
        self.themes = self.theme_manager.load_themes()
        if not self.themes:
            self.console.print("[red]No themes found in themes/ directory[/]")
            return

        # Start interactive loop
        self.current_index = len(self.themes) - 1
        
        while self.running:
            try:
                self.draw()
                self.handle_input()
            except KeyboardInterrupt:
                self.running = False
        
        self.clear_screen()

    def get_current_theme(self) -> Theme:
        if self.current_index in self.modified_themes:
            return self.modified_themes[self.current_index]
        return self.themes[self.current_index]

    def get_working_theme(self) -> Theme:
        base = self.get_current_theme()
        if self.edit_state.active and self.edit_state.colors:
            return Theme(name=base.name, colors=dict(self.edit_state.colors))
        return base

    def draw(self):
        self.clear_screen()
        theme = self.get_current_theme()
        working_theme = self.get_working_theme()
        
        # 1. Header
        name_display = f"{theme.name}-modified" if self.current_index in self.modified_themes else theme.name
        left = f"═══ THEME ANALYZER │ [{self.current_index + 1}/{len(self.themes)}] {name_display} ═══"
        
        if self.edit_state.active:
            if self.edit_state.editing:
                right = "Type hex │ Enter = confirm │ Esc = cancel edit"
            else:
                right = "↑↓ = select │ Type to edit │ Enter = apply │ Esc = cancel"
        else:
            right = "n/p/Space = next/prev │ e = edit │ s = save │ u = reload │ q = quit"
            
        gap = self.console.width - len(left) - len(right) - 1
        print(f"{left}{' ' * max(gap, 2)}{right}")
        print()

        # 2. Render Tables (captured to strings)
        palette_t = self.renderer.create_palette_table(working_theme, self.edit_state)
        constraint_t = self.renderer.create_constraint_table(working_theme)
        bright_t = self.renderer.create_bright_table(working_theme)
        fm_t = self.renderer.create_fm_pairs_table(working_theme)

        palette_lines = self._render_table_to_lines(palette_t)
        constraint_lines = self._render_table_to_lines(constraint_t)
        bright_lines = self._render_table_to_lines(bright_t)
        fm_lines = self._render_table_to_lines(fm_t)
        
        # 3. Text Components
        swatches = self.renderer.create_swatches_lines(working_theme)
        hue_lines = self.renderer.create_hue_spacing_lines(working_theme)
        matrix_lines = self.renderer.create_sample_matrix_lines(working_theme)

        # 4. Layout Assembly
        # Grid: Top Right (Constraints + Bright)
        grid_top = []
        max_top = max(len(constraint_lines), len(bright_lines))
        for i in range(max_top):
            c1 = constraint_lines[i] if i < len(constraint_lines) else ""
            c2 = bright_lines[i] if i < len(bright_lines) else ""
            grid_top.append(f"{self._pad_ansi(c1, 30)}{c2}")
        
        grid_lines = grid_top + [""] + fm_lines

        # Left Column: Row 1 (Swatches + Hue)
        left_row1 = []
        swatch_width = 60
        max_r1 = max(len(swatches), len(hue_lines))
        for i in range(max_r1):
            sw = swatches[i] if i < len(swatches) else ""
            hu = hue_lines[i] if i < len(hue_lines) else ""
            left_row1.append(f"{self._pad_ansi(sw, swatch_width)} {hu}")

        # Left Column: Row 2 (Palette + Matrix)
        left_row2 = []
        palette_width = 52
        matrix_width = 53
        max_r2 = max(len(palette_lines), len(matrix_lines))
        for i in range(max_r2):
            pal = palette_lines[i] if i < len(palette_lines) else ""
            mat = matrix_lines[i] if i < len(matrix_lines) else ""
            left_row2.append(f"{self._pad_ansi(pal, palette_width)} {self._pad_ansi(mat, matrix_width)}")

        left_lines = left_row1 + [""] + left_row2

        # Print Main Section
        left_width = 125
        max_lines = max(len(left_lines), len(grid_lines))
        for i in range(max_lines):
            left = left_lines[i] if i < len(left_lines) else ""
            grid = grid_lines[i] if i < len(grid_lines) else ""
            print(f"{self._pad_ansi(left, left_width)} {grid}")
        
        print()

        # 5. Render Screenshots (Row 3)
        mc_width, mcedit_width, htop_width = 80, 80, 50
        h = 20
        
        mc = self.renderer.render_screenshot("mc", mc_width, h, working_theme)
        mce = self.renderer.render_screenshot("mcedit", mcedit_width, h, working_theme)
        ht = self.renderer.render_screenshot("htop", htop_width, h, working_theme)
        
        div = f"{working_theme.fg(0)}{working_theme.bg(0)}│{RESET}"
        max_s = max(len(mc), len(mce), len(ht))
        
        for i in range(max_s):
            col1 = mc[i] if i < len(mc) else " " * mc_width
            col2 = mce[i] if i < len(mce) else " " * mcedit_width
            col3 = ht[i] if i < len(ht) else " " * htop_width
            print(f"{col1}{div}{col2}{div}{col3}")

        # Tig (Row 4)
        tig = self.renderer.render_screenshot("tig", mc_width, 10, working_theme)
        if tig:
            print()
            for line in tig:
                print(line)

        print()
        print("─" * self.console.width)

    def handle_input(self):
        key = self._read_key()
        
        if self.edit_state.active:
            self._handle_edit_input(key)
        else:
            self._handle_normal_input(key)

    def _handle_normal_input(self, key: KeyEvent):
        if key.value == 'q':
            self.running = False
        elif key.value in ('n', ' '):
            self.current_index = (self.current_index + 1) % len(self.themes)
        elif key.value == 'p':
            self.current_index = (self.current_index - 1) % len(self.themes)
        elif key.type == 'char' and key.value.isdigit() and 1 <= int(key.value) <= len(self.themes):
            self.current_index = int(key.value) - 1
        elif key.value == 'e':
            self.edit_state.active = True
            self.edit_state.selected = 0
            self.edit_state.editing = False
            self.edit_state.buffer = ""
            self.edit_state.colors = dict(self.get_current_theme().colors)
        elif key.value == 's':
            self._save_current_theme()
        elif key.value == 'u':
            self._reload_themes()

    def _handle_edit_input(self, key: KeyEvent):
        es = self.edit_state
        
        if es.editing:
            if key.type == 'escape':
                es.editing = False
                es.buffer = ""
            elif key.type == 'enter':
                if es.buffer:
                    val = es.buffer if es.buffer.startswith("#") else "#" + es.buffer
                    val = val.lower()
                    if ColorUtils.is_valid_hex_color(val):
                        es.colors[es.selected] = val
                es.editing = False
                es.buffer = ""
            elif key.type == 'backspace':
                es.buffer = es.buffer[:-1]
            elif key.type == 'char' and (key.value.isalnum() or key.value == '#'):
                if len(es.buffer) < 7:
                    es.buffer += key.value
        else:
            if key.type == 'escape':
                es.active = False
                es.colors = None
            elif key.type == 'enter':
                if es.colors != self.get_current_theme().colors:
                    self.modified_themes[self.current_index] = Theme(
                        name=self.get_current_theme().name,
                        colors=dict(es.colors)
                    )
                es.active = False
                es.colors = None
            elif key.type == 'arrow' and key.value == 'down':
                es.selected = (es.selected + 1) % 16
            elif key.type == 'arrow' and key.value == 'up':
                es.selected = (es.selected - 1) % 16
            elif key.type == 'tab':
                es.selected = (es.selected + 1) % 16
            elif key.type == 'shift-tab':
                es.selected = (es.selected - 1) % 16
            elif key.type == 'char':
                es.editing = True
                es.buffer = key.value if key.value != " " else ""

    def _save_current_theme(self):
        theme_to_save = self.get_current_theme()
        timestamp = datetime.now().strftime("%H%M%S")
        save_name = f"{theme_to_save.name.split('-')[0]}-{timestamp}"
        new_theme = Theme(name=save_name, colors=dict(theme_to_save.colors))
        
        filepath = self.theme_manager.save_theme(new_theme)
        
        # Reload and jump to new theme
        self.themes = self.theme_manager.load_themes()
        for i, t in enumerate(self.themes):
            if t.name == save_name:
                self.current_index = i
                break
        self.modified_themes.pop(self.current_index, None)
        
        print(f"\n  Saved: {filepath}")
        self._wait_key()

    def _reload_themes(self):
        self.themes = self.theme_manager.load_themes()
        if self.themes:
            self.current_index = len(self.themes) - 1
            self.modified_themes.clear()

    # --- Low Level Helpers ---

    def clear_screen(self):
        print("\033[2J\033[H", end="")

    def _wait_key(self):
        print("  Press any key...")
        self._getch()

    def _getch(self) -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def _read_key(self) -> KeyEvent:
        ch = self._getch()
        if ch == "\x1b":
            ch2 = self._getch()
            if ch2 == "[":
                ch3 = self._getch()
                if ch3 == "A": return KeyEvent("arrow", "up")
                if ch3 == "B": return KeyEvent("arrow", "down")
                if ch3 == "C": return KeyEvent("arrow", "right")
                if ch3 == "D": return KeyEvent("arrow", "left")
                if ch3 == "Z": return KeyEvent("shift-tab", "")
            return KeyEvent("escape", "")
        elif ch == "\t": return KeyEvent("tab", "")
        elif ch in ("\r", "\n"): return KeyEvent("enter", "")
        elif ch in ("\x7f", "\x08"): return KeyEvent("backspace", "")
        return KeyEvent("char", ch)

    def _render_table_to_lines(self, table: Table) -> List[str]:
        with self.capture_console.capture() as capture:
            self.capture_console.print(table)
        return capture.get().split("\n")

    def _pad_ansi(self, s: str, width: int) -> str:
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        visible = len(ansi_escape.sub('', s))
        if visible >= width:
            return s
        return s + " " * (width - visible)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
    else:
        App().run()
