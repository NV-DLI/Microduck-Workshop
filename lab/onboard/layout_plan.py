"""Pure layout arithmetic for the hero capture (no Kit, no GPU; importable anywhere).

All sizes are omni.ui points (pixels = points * Workspace.get_dpi_scale()). The Kit side
(display_layout.py) measures the real dock nodes and feeds them here; nothing here draws.
"""

# Fractions of the Kit main window (points). Bands are the acceptance ranges from the brief.
RIGHT_COLUMN = dict(target=0.12, band=(0.10, 0.14))   # Stage over Property/IsaacLab
BOTTOM_ROW = dict(target=0.07, band=(0.06, 0.08))     # Content/Console
PANEL_SHARE = dict(target=0.28, band=(0.25, 0.30))    # sensor panel / (viewport + panel)
TOLERANCE_PT = 3.0
SCENE_IMAGE = dict(band=(0.35, 0.45))  # root acceptance plan; reported, judged on the screenshot
TAB_PT = 20  # dock tab bar inside a dock node's height
VIEWPORT_ASPECT = 16 / 9  # realtime_view.py render texture 960x540, letterboxed in the Viewport

# Panel content (see sensor_panel.SensorPanel, which uses 15/13 pt fonts, 128 pt RGB, 28 pt cells).
BASE = dict(bold=15, small=13, cell_font=13, cell=28, rgb_h=128, legend_h=10)
PAD_PT = 16           # window frame padding + scrollbar gutter left free
MAX_CELL = 46
MIN_RGB_H = 150
RGB_ASPECT = 16 / 9   # delivered head camera frame 960x540


def targets(main_w, main_h):
    """Dock sizes to request for a main window of main_w x main_h points."""
    return dict(right_w=round(main_w * RIGHT_COLUMN["target"], 1),
                bottom_h=round(main_h * BOTTOM_ROW["target"], 1))


def panel_target(upper_w, upper_h=None):
    """Sensor panel dock width once viewport + panel together span upper_w x upper_h points.

    When the 16:9 viewport image would be width-limited at the target share (a tall window),
    the panel takes the low end of its band so the scene keeps the width."""
    share = PANEL_SHARE["target"]
    if upper_h is not None and upper_w * (1 - share) < (upper_h - TAB_PT) * VIEWPORT_ASPECT:
        share = PANEL_SHARE["band"][0] + 0.002
    # Never narrower than the course panel's 320 pt text lines (+ padding and 2 pt frame), never above the band.
    return round(min(max(upper_w * share, 320 + PAD_PT + 2), upper_w * PANEL_SHARE["band"][1]), 1)


def fractions(main_w, main_h, right_w, bottom_h, viewport_w, panel_w):
    upper = viewport_w + panel_w
    return dict(right=right_w / main_w, bottom=bottom_h / main_h,
                panel=panel_w / upper if upper else 0.0,
                viewport=viewport_w / upper if upper else 0.0,
                viewport_of_main=viewport_w / main_w, panel_of_main=panel_w / main_w)




def scene_image_fraction(viewport_w, viewport_h, main_w, main_h):
    """Share of the whole Kit window covered by the letterboxed 16:9 viewport image (estimate)."""
    h = max(viewport_h - TAB_PT, 0.0)
    w = min(viewport_w, h * VIEWPORT_ASPECT)
    return w * (w / VIEWPORT_ASPECT) / (main_w * main_h)


def within(f):
    """Which acceptance bands a measured layout satisfies."""
    def ok(v, band):
        return band[0] <= v <= band[1]
    return dict(right=ok(f["right"], RIGHT_COLUMN["band"]), bottom=ok(f["bottom"], BOTTOM_ROW["band"]),
                panel=ok(f["panel"], PANEL_SHARE["band"]))


def _text_block(k):
    """Height of every text row of the panel at font scale k (rows as in SensorPanel)."""
    bold, small = round(BASE["bold"] * k), round(BASE["small"] * k)
    bold_row, small_row = bold + 6, small + 6
    # status, rgb_title, rgb_role, tof_title, tof_zones, tof_role, min/max, 2 legend rows, orientation
    rows = 3 * bold_row + 3 * small_row + small_row + 2 * (small_row + 2) + small_row
    spacers = 2 * 4 + 10 * 1  # two Spacer(4) and VStack spacing=1
    return rows + spacers + round(BASE["legend_h"] * k)


def panel_metrics(panel_w, panel_h):
    """Widget sizes so the unchanged panel content fills a panel_w x panel_h dock without scrolling.

    Text scale k never exceeds avail/320 so every line still fits (SensorPanel sizes lines for
    320 pt at k = 1). Returns None-free dict; 'fits' is False if even the base sizes overflow."""
    avail = max(panel_w - PAD_PT, 1.0)
    cell = int(min(MAX_CELL, max(BASE["cell"], (avail - 7) // 8)))
    k = max(1.0, min(cell / BASE["cell"], avail / 320.0, 1.6))
    text = _text_block(k)
    rgb_h = min(avail / RGB_ASPECT, panel_h - text - (8 * cell + 7))
    while rgb_h < MIN_RGB_H and cell > BASE["cell"]:
        cell -= 2
        k = max(1.0, min(cell / BASE["cell"], avail / 320.0, 1.6))
        text = _text_block(k)
        rgb_h = min(avail / RGB_ASPECT, panel_h - text - (8 * cell + 7))
    rgb_h = int(max(rgb_h, BASE["rgb_h"]))
    total = text + rgb_h + 8 * cell + 7
    return dict(cell=cell, grid=8 * cell + 7, cell_font=max(BASE["cell_font"], round(cell * 0.42)),
                bold=round(BASE["bold"] * k), small=round(BASE["small"] * k), text_scale=round(k, 3),
                rgb_h=rgb_h, rgb_w=int(min(avail, rgb_h * RGB_ASPECT)), legend_h=round(BASE["legend_h"] * k),
                content_h=total, fits=total <= panel_h, panel_w=panel_w, panel_h=panel_h)


def pixel_check(main_pt, dpi, window_px, tol_px=4):
    """Kit's main window (points) and the X client window (pixels) describe the same surface."""
    w, h = main_pt[0] * dpi, main_pt[1] * dpi
    return abs(w - window_px[0]) <= tol_px and abs(h - window_px[1]) <= tol_px
