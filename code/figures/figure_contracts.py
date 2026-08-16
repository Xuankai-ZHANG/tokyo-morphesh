"""Shared constants for manuscript Figures 2–4 (colors, labels, panel geometry).

The three plotters (``plot_figure{2,3,4}.py``) import the constants they need.
Per-figure staged-data and panel directories are namespaced (``FIG*_SOURCE_DATA``,
``FIG*_PANELS``) because they point to different locations; everything else that is
shared across figures is defined once.
"""

from pathlib import Path

from map_contract import MAP_CONTRACT

ROOT = Path(__file__).resolve().parents[2]
AUDITS = ROOT / "figures" / "audits"

# Per-figure staged-data + panel output directories
FIG2_SOURCE_DATA = ROOT / "data" / "figure_source" / "figure2"
FIG2_PANELS = ROOT / "figures" / "panels" / "figure2"
FIG3_SOURCE_DATA = ROOT / "data" / "figure_source" / "figure3"
FIG3_PANELS = ROOT / "figures" / "panels" / "figure3"
FIG4_SOURCE_DATA = ROOT / "data" / "figure_source" / "figure4"
FIG4_PANELS = ROOT / "figures" / "panels" / "figure4"

# Panel geometry — keys are disjoint across figures, so one shared dict per quantity.
PANEL_SIZES_MM = {
    "fig2a": (178.0, 36.0),
    "fig2b": (89.0, 72.0),
    "fig2c": (89.0, 95.0),
    "fig2d": (195.0, 54.0),
    "fig2e": (89.0, 72.0),
    "fig2f": (89.0, 95.0),
    "fig3a": (178.0, 28.0),
    "fig3c": (71.2, 84.0),
    "fig3b": (106.8, 84.0),
    "fig3d": (53.4, 84.0),
    "fig3e": (57.85, 96.6),
    "fig3f": (71.2, 90.3),
    "fig3g": (195.0, 40.0),
    "fig4a": (194.0, 65.0),
    "fig4b": (88.0, 56.0),
    "fig4c": (78.0, 56.0),
    "fig4d": (78.0, 48.0),
    "fig4e": (100.0, 48.0),
}

PANEL_DPI = {
    "fig2b": 600,
    "fig2e": 600,
    "fig3b": 600,
    "fig4a": 600,
}
for _panel in PANEL_SIZES_MM:
    PANEL_DPI.setdefault(_panel, 300)

# Map frame / ticks / north arrow — derived from the shared MapContract
MAP_LIMITS = {"x": MAP_CONTRACT.xlim, "y": MAP_CONTRACT.ylim}
MAP_TICKS = {"x": MAP_CONTRACT.xticks, "y": MAP_CONTRACT.yticks}
MAP_FRAME = {
    "left": MAP_CONTRACT.axes_left,
    "right": MAP_CONTRACT.axes_left + MAP_CONTRACT.axes_width,
    "bottom": MAP_CONTRACT.axes_bottom,
    "top": MAP_CONTRACT.axes_bottom + MAP_CONTRACT.axes_height,
    "tick_size": MAP_CONTRACT.tick_size,
    "tick_pad": MAP_CONTRACT.tick_pad,
    "aspect_latitude": MAP_CONTRACT.aspect_latitude,
}
NORTH_ARROW = {
    "x": MAP_CONTRACT.north_x,
    "y": MAP_CONTRACT.north_y,
    "half_width": MAP_CONTRACT.north_half_width,
    "height": MAP_CONTRACT.north_height,
    "label_gap": MAP_CONTRACT.north_label_gap,
    "linewidth": MAP_CONTRACT.north_linewidth,
    "font_size": MAP_CONTRACT.north_font_size,
}

# Category scheme (shared across figures)
CATEGORY_ORDER = ("metropolitan_core", "regional_city", "rural")
CATEGORY_LABELS = {
    "metropolitan_core": "Metro Core",
    "regional_city": "Regional City",
    "rural": "Rural",
}
CATEGORY_COLORS = {  # Figure 2 + Figure 4 palette
    "metropolitan_core": "#c47c76",
    "regional_city": "#D5A83F",
    "rural": "#7d9fc4",
}
FIG3_CATEGORY_COLORS = {  # Figure 3 lighter palette
    "metropolitan_core": "#e08886",
    "regional_city": "#fbcca1",
    "rural": "#aecbe1",
}

# Figure 2
CREAM_PANEL = "#F4EEE2"
INK = "#3A352E"
REFERENCE = "#6B6357"
FONT_SM = 6.5
FIG2_LISA_COLORS = {
    "HH": "#c47c76",
    "LL": "#7d9fc4",
    "HL": "#D5A83F",
    "LH": "#8588b9",
    "NS": "#F4EEE2",
}

# Figure 3
AUX_COLORS = {
    "green": "#5FABA0",
    "green_light": "#98D4C6",
    "green_pale": "#D5EFE9",
    "brown": "#6B6357",
    "brown_dark": "#3A352E",
    "yellow": "#E8B84B",
    "warm_grid": "#D8CFBF",
    "cream": "#F4EEE2",
}

# Figure 4
SCENARIO_LABELS = {
    "S1": "S1 Compact vertical development",
    "S2": "S2 Transit-oriented redistribution",
    "S3": "S3 Moderate deconcentration",
}
SCENARIO_COLORS = {
    "S1": "#c47c76",
    "S2": "#D5A83F",
    "S3": "#7d9fc4",
}


def ensure_directories() -> None:
    for directory in (
        FIG2_SOURCE_DATA,
        FIG2_PANELS,
        FIG3_SOURCE_DATA,
        FIG3_PANELS,
        FIG4_SOURCE_DATA,
        FIG4_PANELS,
        AUDITS,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
