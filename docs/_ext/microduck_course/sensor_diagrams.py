"""Onboard sensor figure: an RGB frame with detector boxes beside an 8×8 ToF distance grid.

The two sensor panels draw recorded values supplied by the lesson YAML (one
synchronized capture of a completed simulation run): detector boxes in original
960 × 540 pixels and the 64 ToF zone ranges in meters, ``null`` for zones without
a valid return. The camera panel is a line drawing positioned by those boxes, not a
rendered image. Each sensor lane leads to the processing step that uses it, and all
lanes meet in one result card. ``tof.used: false`` greys the ToF lane and draws no
connection from it, for a learner that only observes it; ``rgb.detections: false``
drops the detector boxes for a learner that reads the raw frame. Both describe the
GR00T input in Part 5.
"""

from html import escape as e

from .widgets import balanced, inline_md

W, MARGIN, END = 760, 24, 4
PANEL_W = 288  # both sensor panels: 960 × 540 frame at 0.3, and 8 columns of 36
CARD_X, CARD_W = 352, 196
FUSE_X, FUSE_W = 588, 148
MUTED = "var(--nve-sys-text-muted-color)"
ACCENT = "var(--nve-sys-support-success-color)"
CARD = "var(--nve-sys-layer-container-accent-background)"


def _text(x, y, text, size=13, anchor="start", weight=None, opacity=None):
    extra = f' font-weight="{weight}"' if weight else ""
    extra += f' opacity="{opacity}"' if opacity is not None else ""
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" text-anchor="{anchor}"{extra}>{e(text)}</text>'


def _height(title, body, limit, minimum):
    """Card height that fits the wrapped title and body with 14 units of padding."""
    body_lines = [line for part in body.split("\n") for line in balanced(part, limit + 4)] if body else []
    block = len(balanced(title, limit)) * 18 + (6 if body_lines else 0) + len(body_lines) * 16
    return max(minimum, block + 28)


def _card(x, y, w, h, title, body, limit, dim=False):
    """Card with a centred title and wrapped body lines, vertically centred."""
    title_lines = balanced(title, limit)
    body_lines = [line for part in body.split("\n") for line in balanced(part, limit + 4)] if body else []
    block = len(title_lines) * 18 + (6 if body_lines else 0) + len(body_lines) * 16
    top = y + (h - block) / 2 + 13
    opacity = 0.45 if dim else None
    dash = ' stroke-dasharray="5 4"' if dim else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{CARD}" stroke="{MUTED}"{dash}/>']
    for i, line in enumerate(title_lines):
        out.append(_text(x + w / 2, top + i * 18, line, 14, "middle", "600", opacity))
    first = top + len(title_lines) * 18 + 6
    for i, line in enumerate(body_lines):
        out.append(_text(x + w / 2, first + i * 16, line, 12, "middle", opacity=opacity if dim else 0.85))
    return "".join(out)


def _arrow(x1, x2, y, marker):
    return f'<path d="M{x1 + END} {y}H{x2 - END}" stroke="{MUTED}" stroke-width="1.4" fill="none" marker-end="url(#{marker})"/>'


def _rgb_panel(rgb, x, y, uid):
    width, height = rgb.get("size", [960, 540])
    scale = PANEL_W / width
    h = height * scale
    out = [f'<rect x="{x}" y="{y}" width="{PANEL_W}" height="{h:g}" rx="3" fill="none" stroke="{MUTED}"/>']
    for name, confidence, (x1, y1, x2, y2) in rgb.get("boxes", []):
        bx, by, bw, bh = x + x1 * scale, y + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale
        if name == "ball":
            r = min(bw, bh) / 2 * 0.92
            out.append(f'<circle cx="{bx + bw / 2:.1f}" cy="{by + bh / 2:.1f}" r="{r:.1f}" fill="none" stroke="{MUTED}" stroke-width="1.6"/>')
        else:
            # Two posts and a crossbar inside the goal box, as seen from in front.
            inset = bw * 0.03
            out.append(f'<path d="M{bx + inset:.1f} {by + bh * 0.88:.1f}V{by + bh * 0.04:.1f}H{bx + bw - inset:.1f}V{by + bh * 0.88:.1f}" fill="none" stroke="{MUTED}" stroke-width="1.6"/>')
        if rgb.get("detections", True):
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="{ACCENT}" stroke-width="2"/>')
            label_y = by + bh + 13 if name == "ball" else by - 4
            out.append(_text(bx + 2, label_y, f"{name} {confidence:.2f}", 11))
    return "".join(out), h


def _tof_panel(tof, x, y, used):
    grid = tof["grid"]
    if len(grid) != 8 or any(len(row) != 8 for row in grid):
        raise ValueError("onboard-sensors: tof.grid must be 8 rows of 8 values")
    cw, ch = PANEL_W / 8, 20
    out = []
    near, far = tof.get("scale", [0.35, 1.7])
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            cx, cy = x + c * cw, y + r * ch
            if value is None:
                out.append(f'<rect x="{cx:.1f}" y="{cy}" width="{cw:.1f}" height="{ch}" fill="none" stroke="{MUTED}" stroke-width="0.6" stroke-dasharray="2 3"/>')
                out.append(_text(cx + cw / 2, cy + 14, "–", 11, "middle", opacity=0.6))
                continue
            # Nearer zones are drawn darker; the printed number is the recorded range.
            level = max(0.0, min(1.0, (far - float(value)) / (far - near)))
            fill = 0.04 + 0.66 * level ** 1.5 if used else 0.06
            out.append(f'<rect x="{cx:.1f}" y="{cy}" width="{cw:.1f}" height="{ch}" fill="{ACCENT}" fill-opacity="{fill:.2f}" stroke="{MUTED}" stroke-width="0.6"/>')
            out.append(_text(cx + cw / 2, cy + 14, f"{float(value):.2f}", 11, "middle", opacity=None if used else 0.45))
    return "".join(out), 8 * ch


def onboard_sensors(d, lang):
    uid = "sensors-" + ("groot" if d.get("tof", {}).get("used") is False else "onboard")
    marker = uid + "-arrow"
    rgb, tof, proprio = d["rgb"], d["tof"], d["proprio"]
    steps = d["steps"]
    if len(steps) != 3:
        raise ValueError("onboard-sensors: steps lists the camera, ToF and body processing cards")
    used = tof.get("used", True) is not False
    parts = []
    # Lane 1: camera.
    y_rgb = 40
    parts.append(_text(MARGIN, y_rgb - 12, rgb["title"], 13, weight="600"))
    panel, h_rgb = _rgb_panel(rgb, MARGIN, y_rgb, uid)
    parts.append(panel)
    mid_rgb = y_rgb + h_rgb / 2
    parts.append(_arrow(MARGIN + PANEL_W, CARD_X, mid_rgb, marker))
    h = _height(*steps[0], 22, 98)
    parts.append(_card(CARD_X, mid_rgb - h / 2, CARD_W, h, *steps[0], 22))
    # Lane 2: ToF distance grid.
    y_tof = y_rgb + h_rgb + 44
    parts.append(_text(MARGIN, y_tof - 12, tof["title"], 13, weight="600", opacity=None if used else 0.6))
    panel, h_tof = _tof_panel(tof, MARGIN, y_tof, used)
    parts.append(panel)
    mid_tof = y_tof + h_tof / 2
    if used:
        parts.append(_arrow(MARGIN + PANEL_W, CARD_X, mid_tof, marker))
    h = _height(*steps[1], 22, 98)
    parts.append(_card(CARD_X, mid_tof - h / 2, CARD_W, h, *steps[1], 22, dim=not used))
    # Lane 3: joint encoders and IMU.
    y_body = y_tof + h_tof + 28
    h_body = max(_height(*proprio, 36, 64), _height(*steps[2], 22, 64))
    parts.append(_card(MARGIN, y_body, PANEL_W, h_body, *proprio, 36))
    mid_body = y_body + h_body / 2
    parts.append(_arrow(MARGIN + PANEL_W, CARD_X, mid_body, marker))
    parts.append(_card(CARD_X, y_body, CARD_W, h_body, *steps[2], 22))
    # All used lanes meet in one result card.
    top, bottom = y_rgb - 28, y_body + h_body
    parts.append(_card(FUSE_X, top, FUSE_W, bottom - top, *d["result"], 16))
    # An observation-only lane has no connection to the model at all.
    for mid in (mid_rgb, mid_tof, mid_body) if used else (mid_rgb, mid_body):
        parts.append(_arrow(CARD_X + CARD_W, FUSE_X, mid, marker))
    height = bottom + MARGIN
    label = d["label"]
    svg = (f'<svg viewBox="0 0 {W} {height:g}" class="sensor-svg" style="min-width:560px" role="img" aria-labelledby="{uid}-title {uid}-desc">'
           f'<title id="{uid}-title">{e(label)}</title><desc id="{uid}-desc">{e(d.get("description", ""))}</desc>'
           f'<defs><marker id="{marker}" markerWidth="6" markerHeight="6" refX="5.25" refY="3" orient="auto">'
           f'<path d="M0 0L5.25 3L0 6" fill="none" stroke="{MUTED}" stroke-width="1.4"/></marker></defs>'
           + "".join(parts) + "</svg>")
    caption = inline_md(d.get("caption", ""))
    return (f'<figure class="concept-figure sensor-figure" data-diagram="onboard-sensors"><div class="diagram-scroll" tabindex="0" '
            f'role="region" aria-label="{e(label)}">{svg}</div><figcaption class="diagram-caption">{caption}</figcaption></figure>')
