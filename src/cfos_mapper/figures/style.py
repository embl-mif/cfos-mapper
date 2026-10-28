"""
One look for all figures (results and method notebooks): `from cfos_mapper.figures import style as fs`.

Colours come from a colour-blind-safe palette:
  - CELL / NOT_CELL      orange / blue  (on microscopy images)
  - OURS / PRETRAINED    blue / gray
  - BRAINS               fixed colour per brain (never reassigned)
  - BLUE_RAMP            one-hue colormap for amounts (e.g. cells per volume)
  - ORANGE_RAMP          the same in orange (atlas maps of cells per mm³)
  - BLUE_ORANGE          diverging colormap: more in one group (blue) or the other (orange)
Text stays in neutral inks; colour is only used for data.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---- palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # titles, values
INK_2 = "#52514e"        # labels
MUTED = "#898781"        # axis ticks, notes
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#a6a59e"

OURS, PRETRAINED = BLUE, GRAY
# on dark microscopy images use the brighter (dark-mode) steps
CELL, NOT_CELL = "#d95926", "#3987e5"
BRAIN_COLORS = [BLUE, ORANGE, AQUA, YELLOW]  # O15, O16, O17, O18 ... in this fixed order

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# the second amount next to BLUE_RAMP (e.g. cells per mm³ on atlas maps): one hue, light -> dark.
# Starts at a pale tint, not white: on atlas maps white is tissue without a value (fiber tracts, ventricles)
ORANGE_RAMP = LinearSegmentedColormap.from_list(
    "orange_ramp", ["#fdefe7", "#fbd3bf", "#f5a986", "#eb6834", "#c44a1b", "#8a3111", "#4a1a08"])
# more in one group (blue end) or the other (orange end), gray = the same
BLUE_ORANGE = LinearSegmentedColormap.from_list(
    "blue_orange", ["#184f95", "#3987e5", "#9ec5f4", "#e6e5df", "#f5b496", "#eb6834", "#a3390f"])


def brain_color(brain, all_brains):
    """Fixed colour per brain: by its position in the sorted list of all brains."""
    return BRAIN_COLORS[sorted(all_brains).index(brain) % len(BRAIN_COLORS)]


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        # PDF editable in Illustrator: embed fonts as TrueType (text stays text, not outlines)
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def title(fig, text):
    """Figure title on the left, just above the panels (matplotlib places both, no overlap/gap)."""
    fig.set_layout_engine("constrained")
    fig.suptitle(text, x=0.01, ha="left", fontweight="bold", color=INK, fontsize=14)


def save(fig, folder, name):
    """Save a figure as PDF (editable in Illustrator) and PNG (quick use in slides)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    background = fig.get_facecolor()  # keep the figure's own background (e.g. black for 3D views)
    fig.savefig(folder / f"{name}.pdf", facecolor=background)
    fig.savefig(folder / f"{name}.png", facecolor=background)
    print(f"saved {folder / name}.pdf / .png")


def image_axes(ax):
    """Axes for a microscopy image: no ticks, no grid, no frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def scale_bar(ax, um, pixel_um, label=None, color="white"):
    """Scale bar in the bottom right of an image (pixel_um = size of one pixel)."""
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    length = um / pixel_um
    x_end = max(x1, x2) - 0.05 * abs(x2 - x1)
    y = max(y1, y2) - 0.06 * abs(y2 - y1)
    ax.plot([x_end - length, x_end], [y, y], color=color, linewidth=3, solid_capstyle="butt")
    ax.text(x_end - length / 2, y - 0.025 * abs(y2 - y1), label or f"{um:g} µm",
            color=color, ha="center", va="bottom", fontsize=10)


def bar_labels(ax, bars, fmt="{:.0f}"):
    """Value labels on bars (neutral ink, not the bar colour)."""
    for bar in bars:
        value = bar.get_height()
        if np.isnan(value):
            continue
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK_2)
