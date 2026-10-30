"""HTML renderers for the course's interactive figures.

The markup, class names and data attributes match the earlier generator so the
existing browser scripts (``web/assets/*.js``) keep working unchanged. Every
learner-visible string comes from the Markdown page that uses the directive.
"""

from __future__ import annotations

import hashlib
import json
import re
from html import escape as e
from pathlib import Path

import yaml
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from .course_diagrams import lines, network_svg, platform_svg, vla_joint_svg
from .navigation_diagrams import navigation_diagram

_INLINE_TAGS = ("q", "sub", "sup", "br", "wbr")

# Set by the directive while it renders: turns \\sym marks in `$...$` into KaTeX htmlData.
MATH_RESOLVER = None


def inline_md(text: str) -> str:
    """Render the small inline Markdown subset used inside widget YAML.

    Supports `code`, **strong**, *emphasis*, [label](url), $inline TeX$ and the
    raw inline tags <q>, <sub>, <sup>, <br>. Everything else is HTML-escaped.
    """
    if text is None:
        return ""
    text = str(text)
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            out.append(e(text[i + 1]))
            i += 2
            continue
        if ch == "$":
            end = text.find("$", i + 1)
            if end != -1:
                tex = text[i + 1:end]
                if MATH_RESOLVER is not None:
                    tex = MATH_RESOLVER(tex)
                out.append("\\(" + e(tex) + "\\)")
                i = end + 1
                continue
        if ch == "`":
            run = len(text[i:]) - len(text[i:].lstrip("`"))
            fence = "`" * run
            end = text.find(fence, i + run)
            if end != -1:
                body = text[i + run:end]
                if body.startswith(" ") and body.endswith(" ") and body.strip():
                    body = body[1:-1]
                out.append("<code>" + e(body) + "</code>")
                i = end + run
                continue
        if text.startswith("**", i):
            end = text.find("**", i + 2)
            if end != -1:
                out.append("<strong>" + inline_md(text[i + 2:end]) + "</strong>")
                i = end + 2
                continue
        if ch == "[":
            m = re.match(r"\[((?:[^\[\]\\]|\\.|`[^`]*`)*)\]\((<[^>]*>|[^)\s]*)\)", text[i:])
            if m:
                url = m.group(2)
                if url.startswith("<"):
                    url = url[1:-1]
                external = "://" in url
                attrs = ' target="_blank" rel="noopener"' if external else ""
                out.append(f'<a href="{e(url)}"{attrs}>' + inline_md(m.group(1)) + "</a>")
                i += m.end()
                continue
        if ch == "<":
            m = re.match(r"</?(%s)\s*/?>" % "|".join(_INLINE_TAGS), text[i:])
            if m:
                out.append(m.group(0))
                i += m.end()
                continue
        if ch == "*" and i + 1 < n and text[i + 1] != " ":
            end = text.find("*", i + 1)
            if end != -1:
                out.append("<em>" + inline_md(text[i + 1:end]) + "</em>")
                i = end + 1
                continue
        out.append(e(ch))
        i += 1
    return "".join(out)


def pygmentize(code: str, language: str = "python") -> str:
    """Highlight with the theme's Pygments classes (light and dark styles)."""
    formatter = HtmlFormatter(nowrap=True)
    return highlight(code, get_lexer_by_name(language), formatter).rstrip("\n")


# ---------------------------------------------------------------- flows


def wrap_label(text, limit):
    """Greedy wrap by display units: CJK characters count 2, other characters 1.
    Latin words (with arrows, digits and sub/superscripts) never split."""
    import unicodedata
    tokens = re.findall(r"[\x21-\x7e\u00b0-\u00ff\u2070-\u209f\u2190-\u21ff\u2032\u00d7]+|\s+|.", text)
    units = lambda t: sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in t)
    out, line = [], ""
    for tok in tokens:
        if line.strip() and units(line + tok.rstrip()) > limit and not tok.isspace():
            out.append(line.strip())
            line = ""
        line += tok
    if line.strip():
        out.append(line.strip())
    return out


def balanced(text, limit):
    """Wrap to the fewest lines that fit `limit`, then even out line lengths
    so no line ends with a lone character or word."""
    first = wrap_label(text, limit)
    if len(first) < 2:
        return first
    for width in range(max(1, limit // 2), limit + 1):
        candidate = wrap_label(text, width)
        if len(candidate) == len(first):
            return candidate
    return first


def learning_flow(d, lang):
    """A complete flow diagram with one full-width explanation panel below it.

    Layout and interaction follow the SO-101 Newton workshop's interactive
    diagrams: select (or hover/focus to preview) a numbered module and the panel
    under the figure shows its explanation; Close or Escape clears it. Node
    cards keep the course's numbered c-fx style. Script: web/assets/teaching.js.
    """
    nodes = d["nodes"]
    details = d.get("details", [text for _, text in nodes])
    uid = "learn-" + hashlib.sha1((lang + d["label"]).encode()).hexdigest()[:10]
    n = len(nodes)
    architecture = d.get("layout") == "groot"
    # Four cards per row at every size, so all flows share one scale. Card
    # 180 wide, column gap 44, row gap 80, margin 24. Card height grows with the
    # longest wrapped label (no truncation). Connectors leave and enter cards
    # with straight segments of at least 36 units and never cross a card.
    cols, w, gap_x, gap_y, margin, end = 4, 180, 44, 80, 24, 4
    wrapped = [(balanced(title, 17), balanced(subtitle, 22)) for title, subtitle in nodes]
    title_rows = max(len(t) for t, _ in wrapped)
    body_rows = max(len(b) for _, b in wrapped)
    h = 34 + title_rows * 20 + 6 + body_rows * 17 + 12
    if architecture:
        # Images → System 2 → System 1 → execution on the first row; observe
        # again under images and robot state under System 1 on the second.
        cells = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (0, 1)]
        links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (3, 5), (5, 0)]
    else:
        # Snake order: left to right, then right to left on the next row.
        cells = [((i % cols) if (i // cols) % 2 == 0 else cols - 1 - i % cols, i // cols) for i in range(n)]
        links = [(i, i + 1) for i in range(n - 1)]
        if d.get("loop"):
            links.append((n - 1, d.get("loop_to", 0)))
    rows = max(r for _, r in cells) + 1
    used_cols = max(c for c, _ in cells) + 1
    if n == 3 and not architecture:
        # Three cards span the full row: first and last at the outer margins.
        shift, step = 0, (cols - 1) * (w + gap_x) / (n - 1)
    else:
        shift, step = (cols - used_cols) * (w + gap_x) / 2, w + gap_x  # center short flows on the same scale
    col = lambda c: margin + shift + c * step
    row = lambda r: margin + r * (h + gap_y)
    positions = [(col(c), row(r)) for c, r in cells]
    nodes_bottom = row(rows - 1) + h
    lane = nodes_bottom + 40  # routing lane below the cards
    uses_lane = architecture or (rows == 1 and d.get("loop"))
    height = (lane if uses_lane else nodes_bottom) + margin
    width = 2 * margin + cols * w + (cols - 1) * gap_x
    cx = lambda i: positions[i][0] + w / 2
    cy = lambda i: positions[i][1] + h / 2
    top = lambda i: positions[i][1]
    bottom = lambda i: positions[i][1] + h
    left = lambda i: positions[i][0]
    right = lambda i: positions[i][0] + w
    forward = [(i, i + 1) for i in range(n - 1)]
    paths = []
    for a, z in links:
        (ca, ra), (cz, rz) = cells[a], cells[z]
        if architecture and (a, z) == (3, 4):
            # Down from execution into the empty cell below, then left into robot state.
            path = f"M{cx(a)} {bottom(a)+end}V{cy(z)}H{right(z)+end}"
        elif architecture and (a, z) == (3, 5):
            # Beside the 3→4 link, down to the lane, back to the first column, up.
            path = f"M{cx(a)+40} {bottom(a)+end}V{lane}H{cx(z)}V{bottom(z)+end}"
        elif ra == rz and (a, z) in forward:
            path = f"M{right(a)+end} {cy(a)}H{left(z)-end}" if cz > ca else f"M{left(a)-end} {cy(a)}H{right(z)+end}"
        elif ca == cz and rz > ra:
            path = f"M{cx(a)} {bottom(a)+end}V{top(z)-end}"
        elif ca == cz and rz < ra:
            path = f"M{cx(a)} {top(a)-end}V{bottom(z)+end}"
        elif rows == 1:
            # Single row loop: down to the lane, back along it, up into the target.
            path = f"M{cx(a)} {bottom(a)+end}V{lane}H{cx(z)}V{bottom(z)+end}"
        else:
            # Lower-row loop: across empty cells to the target column, then up.
            start = left(a) - end if cz < ca else right(a) + end
            path = f"M{start} {cy(a)}H{cx(z)}V{bottom(z)+end}"
        paths.append(f'<path class="learning-edge" data-from="{a}" data-to="{z}" d="{path}" marker-end="url(#{uid}-arrow)"/>')
    cards = []
    for i, (((title, _), (title_lines, body_lines)), (x, y)) in enumerate(zip(zip(nodes, wrapped), positions)):
        mid = x + w / 2
        texts = f'<text x="{mid}" y="{y+24}" class="learning-node-index">{i+1:02d}</text>'
        texts += "".join(f'<text x="{mid}" y="{y+50+j*20}" class="learning-node-title">{e(t)}</text>' for j, t in enumerate(title_lines))
        first = y + 50 + len(title_lines) * 20 + 4
        texts += "".join(f'<text x="{mid}" y="{first+j*17}" class="learning-node-subtitle">{e(t)}</text>' for j, t in enumerate(body_lines))
        cards.append(f'<g class="learning-node" data-node="{i}" role="button" tabindex="0" aria-label="{i+1}. {e(title)}" aria-controls="{uid}-detail-{i}" aria-expanded="false"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>{texts}</g>')
    zh = lang == "zh"
    empty = "选择一个模块，说明会显示在这里。" if zh else "Select a module; its explanation appears here."
    scroll_hint = "左右滑动查看完整图示" if zh else "Swipe horizontally to see the full diagram"
    reset = "关闭说明" if zh else "Close explanation"
    articles = "".join(f'<article id="{uid}-detail-{i}" data-node-detail="{i}" class="learning-definition"><h3><span>{i+1}</span>{e(nodes[i][0])}</h3><p>{inline_md(detail)}</p></article>' for i, detail in enumerate(details))
    return (f'<figure class="learning-figure" data-learning-flow="{uid}"><p class="learning-help learning-scroll-hint">{scroll_hint}</p>'
            f'<div class="diagram-scroll" tabindex="0" role="region" aria-label="{e(d["label"])}"><svg class="learning-svg" viewBox="0 0 {width} {height}" role="group" aria-labelledby="{uid}-title"><title id="{uid}-title">{e(d["label"])}</title>'
            f'<defs><marker id="{uid}-arrow" markerWidth="6" markerHeight="6" refX="5.25" refY="3" orient="auto"><path d="M0 0L5.25 3L0 6" fill="none" stroke="currentColor" stroke-width="1.4"/></marker></defs>{"".join(paths)}{"".join(cards)}</svg></div>'
            f'<div class="learning-details" aria-label="{e(d["label"])}"><p class="learning-empty">{empty}</p><button type="button" class="learning-reset" aria-label="{reset}" title="{reset}" hidden>×</button>{articles}</div>'
            f'<span class="sr-only learning-status" role="status"></span>{f'<figcaption>{inline_md(d["caption"])}</figcaption>' if d.get("caption") else ""}</figure>')


# ---------------------------------------------------------------- concept diagrams


def concept_diagram(d, lang):
    name = d["kind"]
    zh = lang == "zh"
    caption = inline_md(d.get("caption", ""))
    if name in ("navigation-stance", "navigation-costmap"):
        return navigation_diagram(name, lang, caption)
    controls = ""
    start = (
        '<svg viewBox="0 0 640 320" role="img" aria-label="' + e(d["label"])
        + '" font-family="Open Sans, Noto Sans SC, sans-serif" font-size="15"><defs><marker id="arrow-' + name
        + '" markerWidth="6" markerHeight="6" refX="5.25" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="var(--nve-sys-text-muted-color)"/></marker></defs>'
    )
    if name == "ppo":
        L = (dict(title="单个样本的代理目标", ratio="新旧动作概率之比 ρ", adv="优势", good="A = +1（好于预期）",
                  bad="A = −1（差于预期）", rho="概率比 ρ", eps="裁剪范围 ε") if zh else
             dict(title="Surrogate objective of one sample", ratio="New / old action probability ratio ρ", adv="Advantage",
                  good="A = +1 (better than expected)", bad="A = −1 (worse than expected)", rho="Probability ratio ρ",
                  eps="Clip range ε"))
        svg = start + f"""<rect x="233" y="30" width="164" height="230" fill="var(--nve-sys-layer-container-accent-background)" class="clip-band"/><path d="M64 30V260H574" stroke="var(--nve-sys-text-muted-color)" fill="none"/><path class="ppo-original" d="" stroke="var(--nve-sys-text-muted-color)" stroke-width="2" stroke-dasharray="5 5" fill="none"/><path class="ppo-clipped" d="" stroke="var(--nve-sys-support-success-color)" stroke-width="3" fill="none"/><circle class="ppo-point" r="6" fill="var(--nve-sys-text-muted-color)" stroke="white" stroke-width="2"/><text x="65" y="18">{L["title"]}</text><text x="58" y="281">0.5</text><text x="312" y="281">1.0</text><text x="565" y="281">1.5</text><text x="319" y="308" text-anchor="middle">{L["ratio"]}</text><text class="ppo-low" x="205" y="50" fill="var(--nve-sys-text-muted-color)">0.8</text><text class="ppo-high" x="404" y="50" fill="var(--nve-sys-text-muted-color)">1.2</text><text class="ppo-ymax" x="54" y="38" text-anchor="end">1.6</text><text class="ppo-ymin" x="54" y="263" text-anchor="end">0.4</text></svg>"""
        controls = (f'<div class="concept-controls"><label>{L["adv"]} <select class="advantage"><option value="1">{L["good"]}</option>'
                    f'<option value="-1">{L["bad"]}</option></select></label><label>{L["rho"]} <input class="prob-ratio" type="range" min="0.5" max="1.5" step="0.01" value="1.3"></label>'
                    f'<label>{L["eps"]} <input class="clip-epsilon" type="range" min="0.05" max="0.4" step="0.05" value="0.2"></label></div><p class="concept-output" aria-live="polite"></p>')
    elif name == "timing":
        # One policy period over four physics steps; step labels sit centered in their boxes.
        L = (dict(policy="策略：每 20 ms 输出一次动作（50 Hz）", action="aₜ：关节目标", physics="物理：每 5 ms 推进一步（200 Hz）",
                  step="第 {n} 步", hold="动作保持不变；位置、速度与接触力仍在逐步变化") if zh else
             dict(policy="Policy: one action every 20 ms (50 Hz)", action="aₜ: joint targets", physics="Physics: one step every 5 ms (200 Hz)",
                  step="Step {n}", hold="The action stays fixed; positions, velocities and contact forces keep changing"))
        boxes = "".join(
            f'<rect x="{60 + i * 130}" y="182" width="124" height="48" rx="4" fill="var(--nve-sys-layer-container-background)" stroke="var(--nve-sys-text-muted-color)"/>'
            f'<text x="{122 + i * 130}" y="206" text-anchor="middle" dominant-baseline="central">{L["step"].format(n=i + 1)}</text>'
            f'<text x="{60 + i * 130}" y="255" text-anchor="middle" font-size="13" fill="var(--nve-sys-text-muted-color)">{i * 5} ms</text>'
            for i in range(4))
        svg = (start + f"""<text x="36" y="43">{L["policy"]}</text><rect x="60" y="70" width="514" height="52" rx="5" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-text-muted-color)"/><text x="317" y="96" text-anchor="middle" dominant-baseline="central">{L["action"]}</text><path d="M317 123V169" stroke="var(--nve-sys-text-muted-color)" marker-end="url(#arrow-timing)"/><text x="36" y="156">{L["physics"]}</text>"""
               + boxes + f"""<text x="574" y="255" text-anchor="middle" font-size="13" fill="var(--nve-sys-text-muted-color)">20 ms</text><text x="317" y="296" text-anchor="middle" font-size="14" fill="var(--nve-sys-text-muted-color)">{L["hold"]}</text></svg>""")
    elif name == "projection":
        svg = start + """<path d="M45 265H610" stroke="var(--nve-sys-text-muted-color)" stroke-width="2"/><text x="495" y="292">地面 z = 0</text><circle cx="102" cy="65" r="8" fill="var(--nve-sys-support-success-color)"/><text x="28" y="40">相机原点 o</text><path d="M102 65L475 265" stroke="var(--nve-sys-support-success-color)" stroke-width="2" marker-end="url(#arrow-projection)"/><path d="M205 65V210" stroke="var(--nve-sys-text-muted-color)" stroke-width="3"/><text x="180" y="237">图像平面</text><circle cx="205" cy="120" r="6" fill="var(--nve-sys-text-muted-color)"/><text x="230" y="105">检测像素 (u, v)</text><circle cx="475" cy="265" r="8" fill="var(--nve-sys-text-muted-color)"/><text x="410" y="235">地面交点 p</text><path d="M102 75V265H460" stroke="var(--nve-sys-support-muted-color)" stroke-dasharray="5 5" fill="none"/><text x="320" y="155">沿射线 d 延伸</text><text x="25" y="285">地面投影</text></svg>"""
    elif name == "stance":
        svg = start + """<path d="M60 270H570M60 270V35" stroke="var(--nve-sys-text-muted-color)" fill="none"/><text x="579" y="275">x</text><text x="53" y="25">y</text><path d="M234 196L545 72" stroke="var(--nve-sys-support-success-color)" stroke-width="2" marker-end="url(#arrow-stance)"/><rect x="526" y="37" width="22" height="76" rx="2" stroke="var(--nve-sys-text-muted-color)" fill="none" stroke-width="3"/><text x="550" y="65">球门 g</text><circle cx="353" cy="149" r="12" fill="var(--nve-sys-text-muted-color)"/><text x="363" y="182">足球 b</text><circle class="stance-robot" cx="234" cy="224" r="14" fill="var(--nve-sys-text-muted-color)"/><text class="stance-label" x="117" y="246">出脚位置 p*</text><path class="stance-offset" d="M234 196L234 224" stroke="var(--nve-sys-text-muted-color)" stroke-dasharray="4 4"/><text x="180" y="162">沿射门方向后退 d</text><text x="305" y="110">单位方向 u</text><text x="88" y="303">俯视示意：先留出出脚距离，再调整横向偏移</text></svg>"""
        controls = '<div class="concept-controls"><label>后退距离 d / m <input class="stance-distance" type="range" min="0.12" max="0.5" step="0.01" value="0.26"></label><label>横向偏移 ℓ / m <input class="stance-lateral" type="range" min="-0.14" max="0.14" step="0.01" value="0.04"></label></div><p class="concept-output" aria-live="polite"></p>'
    elif name == "terrain":
        # The height equation is shown, with explained symbols, in the formula below the figure.
        labels = (["平地 0.35 m", "坡道水平长度 1 m", "落脚平台", "坡角 / °", "侧视图：平地、坡道与落脚平台"] if zh
                  else ["Flat: 0.35 m", "Ramp run: 1 m", "Landing", "Slope angle / °", "Side view: flat ground, ramp and landing"])
        svg = start + f"""<path d="M60 165H585" stroke="var(--nve-sys-text-muted-color)" stroke-dasharray="5 5"/><path class="terrain-fill" fill="var(--nve-sys-layer-container-accent-background)" stroke="none"/><path class="terrain-surface" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="4"/><path d="M60 280H165M165 274V286M60 274V286M165 280H465M465 274V286" fill="none" stroke="var(--nve-sys-text-muted-color)"/><text x="110" y="305" text-anchor="middle" font-size="13">{labels[0]}</text><text x="315" y="305" text-anchor="middle">{labels[1]}</text><text class="terrain-landing" x="520" y="100" text-anchor="middle" font-size="13">{labels[2]}</text><circle cx="90" cy="148" r="10" fill="var(--nve-sys-text-muted-color)"/><path d="M85 157L82 165M95 157L98 165" stroke="var(--nve-sys-text-muted-color)" stroke-width="3"/><path class="terrain-height" stroke="var(--nve-sys-text-color)" stroke-width="2"/><text class="terrain-height-label" x="485" y="140" fill="var(--nve-sys-text-color)" font-size="13"/><text x="24" y="28">{labels[4]}</text></svg>"""
        controls = '<div class="concept-controls"><label>' + labels[3] + ' <input class="terrain-angle" type="range" min="-15" max="15" step="1" value="5"></label></div><p class="concept-output" aria-live="polite"></p>'
    else:
        raise ValueError("Unknown diagram " + name)
    return f'<figure class="concept-figure" data-diagram="{name}">{svg}{controls}<figcaption class="diagram-caption">{caption}</figcaption></figure>'


def _network_interface(lang):
    """Input and output tables shown when the input or output column is selected.

    Content lives in docs/_data/network-interface.yaml; this only builds markup.
    """
    data = yaml.safe_load((Path(__file__).resolve().parents[2] / "_data" / "network-interface.yaml").read_text(encoding="utf-8"))
    d, joints = data[lang], data["joints"]
    assert all(len(r) == len(d["input_columns"]) for r in d["input_rows"]), "quote YAML cells that contain commas"
    assert sum(int(r[1]) for r in d["input_rows"]) == 61 and len(joints) == 14
    head = lambda cols: "<thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr></thead>"
    rows = "".join("<tr>" + "".join(f"<td>{inline_md(str(c))}</td>" for c in r) + "</tr>" for r in d["input_rows"])
    inputs = (f'<div class="network-detail" data-for="layer-0"><p class="network-detail-title">{e(d["input_title"])}</p>'
              f'<p>{inline_md(d["input_intro"])}</p><table class="network-table">{head(d["input_columns"])}<tbody>{rows}</tbody></table>'
              f'<p class="network-note">{inline_md(d["input_note"])}</p></div>')
    groups = []
    for name, (a, b) in zip(d["output_groups"], data["output_groups"]):
        body = "".join(f'<tr><td>{i}</td><td><code>{e(joints[i - 1][0])}</code><span class="network-joint-name">{e(joints[i - 1][1 if lang == "zh" else 2])}</span></td></tr>'
                       for i in range(a, b + 1))
        groups.append(f'<table class="network-table"><caption>{e(name)} · {a}–{b}</caption>{head(d["output_columns"])}<tbody>{body}</tbody></table>')
    outputs = (f'<div class="network-detail" data-for="layer-4"><p class="network-detail-title">{e(d["output_title"])}</p>'
               f'<p>{inline_md(d["output_intro"])}</p><div class="network-output-groups">{"".join(groups)}</div></div>')
    return inputs + outputs


def network_figure_open(lang):
    default = ("观测 61 维，经过 512、256、128 个神经元的三个隐藏层，输出 14 维关节动作。指向或选中某一层、某一组连线，可查看它的维度；输入层和输出层还会列出每一维的含义。"
               if lang == "zh" else "61 observation values pass through three hidden layers of 512, 256 and 128 neurons to 14 joint actions. Point to or select a layer or a band of connections to see its size; the input and output layers also list what every value means.")
    return ('<figure class="concept-figure network-figure" data-network><div class="diagram-scroll">' + network_svg(lang)
            + f'</div><div class="network-panel"><p class="network-readout" aria-live="polite" data-default="{e(default)}">{e(default)}</p>'
            + _network_interface(lang) + '</div><figcaption>')


def vla_joint_figure_open(lang):
    """Direct-joint VLA figure (Section 4.4): node columns and bands like the walking-actor
    network; each output group lists its actual joints of the walking interface."""
    zh = lang == "zh"
    data = yaml.safe_load((Path(__file__).resolve().parents[2] / "_data" / "network-interface.yaml").read_text(encoding="utf-8"))
    d, joints = data[lang], data["joints"]
    head = "<thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in d["output_columns"]) + "</tr></thead>"
    groups = []
    for g, (name, (a, b)) in enumerate(zip(d["output_groups"], data["output_groups"])):
        body = "".join(f'<tr><td>{i}</td><td><code>{e(joints[i - 1][0])}</code><span class="network-joint-name">{e(joints[i - 1][1 if zh else 2])}</span></td></tr>'
                       for i in range(a, b + 1))
        groups.append(f'<div class="network-detail" data-for="out-{g}"><table class="network-table"><caption>{e(name)} · {a}–{b}</caption>{head}<tbody>{body}</tbody></table></div>')
    if zh:
        default = ("RGB 图像和语言指令进入视觉语言模型，机器人状态单独进入动作头，动作头每步输出 14 个舵机目标。"
                   "指向或选中一组节点或一束连线，可以查看它的作用；三组输出会列出对应的关节，顺序与 2.1 节行走策略的输出相同。")
        state_title = "机器人状态可以包含什么"
        state_rows = [("14 个关节相对默认姿态的角度", "关节编码器"), ("14 个关节的角速度", "关节编码器"),
                      ("机身角速度与重力方向", "身体 IMU"), ("上一步发出的 14 个舵机目标", "控制器自己的记录")]
        state_cols = ("内容", "来源")
        state_note = "这些都是机器人上真实可以测到或记录的量；球和球门的仿真坐标不在其中。"
    else:
        default = ("RGB images and the instruction enter the vision-language model, the robot state goes separately into the action head, "
                   "and the action head outputs 14 servo targets per step. Point to or select a group of nodes or a band of lines to see what it does; "
                   "the three output groups list their joints, in the same order as the walking-policy output in Section 2.1.")
        state_title = "What the robot state can contain"
        state_rows = [("Angles of the 14 joints relative to the default pose", "Joint encoders"), ("Velocities of the 14 joints", "Joint encoders"),
                      ("Body angular velocity and gravity direction", "Body IMU"), ("The 14 servo targets sent at the previous step", "The controller's own record")]
        state_cols = ("Content", "Source")
        state_note = "All of these can really be measured or recorded on the robot; simulated ball and goal coordinates are not among them."
    state = (f'<div class="network-detail" data-for="state"><p class="network-detail-title">{e(state_title)}</p><table class="network-table">'
             + "<thead><tr>" + "".join(f"<th>{e(c)}</th>" for c in state_cols) + "</tr></thead><tbody>"
             + "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td></tr>" for a, b in state_rows)
             + f'</tbody></table><p class="network-note">{e(state_note)}</p></div>')
    return ('<figure class="concept-figure network-figure vla-figure" data-network><div class="diagram-scroll">' + vla_joint_svg(lang)
            + f'</div><div class="network-panel"><p class="network-readout" aria-live="polite" data-default="{e(default)}">{e(default)}</p>'
            + state + "".join(groups) + '</div><figcaption>')


def platform_figure_open(lang):
    hint = "左右滑动查看完整图示" if lang == "zh" else "Scroll horizontally to view the full diagram"
    return f'<figure class="concept-figure"><p class="diagram-scroll-hint">{hint}</p><div class="diagram-scroll">' + platform_svg(lang) + '</div><figcaption class="diagram-caption">'


# ---------------------------------------------------------------- media


def video_open(src, poster, alt, stages=None, comparison=None):
    labels = ""
    if comparison:
        labels = '<div class="comparison-labels">' + "".join(f"<span>{e(x)}</span>" for x in comparison) + "</div>"
    stage = f' data-stages="{e(stages)}"' if stages else ""
    return f'<figure class="media">{labels}<video controls playsinline preload="metadata" poster="{e(poster)}" aria-label="{e(alt)}"{stage}><source src="{e(src)}" type="video/mp4"></video><figcaption>'


def image_open(src, alt, zoom_label):
    label = f' aria-label="{e(zoom_label)}"' if zoom_label else ""
    return f'<figure class="media"><a class="still-link" href="{e(src)}"{label}><img loading="lazy" src="{e(src)}" alt="{e(alt)}"></a><figcaption>'


def scene_cards_open(cards):
    items = "".join(f'<div class="scene-card"><h4>{e(label)}</h4><a href="{e(src)}"><img loading="lazy" src="{e(src)}" alt="{e(alt)}"></a></div>' for src, label, alt in cards)
    return f'<figure class="media"><div class="scene-card-grid">{items}</div><figcaption>'


def demonstration_open(prefix, lang, columns):
    zh = lang == "zh"
    return (
        f'<figure class="media demonstration" data-demonstration="{e(prefix)}.json"><video controls playsinline preload="metadata" poster="{e(prefix)}.jpg" aria-label="'
        + ("训练示范的相机画面" if zh else "Demonstration camera recording")
        + f'"><source src="{e(prefix)}.mp4" type="video/mp4"></video>'
        + '<p class="demonstration-time">' + ("加载样本…" if zh else "Loading samples…")
        + '</p><div class="table-scroll"><table><thead><tr>'
        + "".join("<th>" + e(c) + "</th>" for c in columns)
        + "</tr></thead><tbody><tr>" + '<td class="demonstration-action">-</td>' * len(columns)
        + "</tr></tbody></table></div><details><summary>"
        + ("查看当前样本的身体状态与动作" if zh else "Inspect this sample’s body state and action")
        + '</summary><pre class="demonstration-sample"></pre></details><figcaption>'
    )


CUBE_ICON = ('<svg class="load-robot-icon" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" '
             'stroke-width="1.6" stroke-linejoin="round" aria-hidden="true" focusable="false">'
             '<path d="M12 2.8 4.2 7.2v9.6L12 21.2l7.8-4.4V7.2L12 2.8Z"/><path d="M4.2 7.2 12 11.6l7.8-4.4M12 11.6v9.6"/></svg>')


def anatomy(lang, parts, megabytes, poster="robot/studio-hd.webp"):
    """Poster first; the learner opens the real CAD model on request (with progress and retry).

    After loading: an exploded overview with labels, and joint assembly demonstrations
    (neck, hip, ankle) whose steps and tracks come from robot/anatomy-layout.json.
    """
    zh = lang == "zh"
    t = lambda a, b: a if zh else b
    meta = t(f"{parts} 个可选部件 · 约 {megabytes:.0f} MB", f"{parts} selectable parts · about {megabytes:.0f} MB")
    return (
        '<div class="anatomy" data-robot-viewer><div class="anatomy-modes" role="group" aria-label="'
        + t("选择视图", "Choose a view") + '"></div><div class="viewer-stage">'
        f'<img class="robot-fallback" src="../assets/{poster}" alt="{t("Microduck 结构展开图", "Exploded view of Microduck")}">'
        '<div class="viewer-launch">'
        f'<button type="button" class="load-robot" data-label-retry="{t("重新加载", "Try again")}">{CUBE_ICON}<span class="load-robot-label">{t("打开 3D 模型", "Open 3D model")}</span></button>'
        f'<p class="load-robot-meta">{meta}</p></div>'
        f'<div class="viewer-progress" hidden><div class="viewer-progress-track"><span class="viewer-progress-bar"></span></div>'
        f'<p class="viewer-progress-text" data-label="{t("正在加载 3D 模型", "Loading 3D model")}">{t("正在加载 3D 模型", "Loading 3D model")}…</p></div>'
        '<div class="viewer-status" role="status" aria-live="polite"></div></div>'
        '<div class="viewer-toolbar"><label class="explode-control">' + t("装配", "Assembled")
        + ' <input type="range" min="0" max="100" value="100" class="explode" aria-label="' + t("结构展开程度", "Explosion amount") + '"> '
        + t("展开", "Exploded") + '</label>'
        '<div class="assembly-controls" hidden>'
        f'<button type="button" class="assembly-play" data-play="{t("播放", "Play")}" data-pause="{t("暂停", "Pause")}" aria-pressed="false">{t("播放", "Play")}</button>'
        f'<button type="button" class="assembly-replay">{t("重播", "Replay")}</button>'
        f'<input type="range" min="0" max="1000" value="0" class="assembly-scrub" aria-label="{t("动画进度", "Animation progress")}">'
        '<span class="assembly-time" aria-hidden="true"></span></div>'
        '<button type="button" class="view-reset">' + t("返回总览", "Reset view")
        + '</button></div><div class="component-info" aria-live="polite"><strong>'
        + t("点击部件或名称，即可放大查看", "Click a part or its label to inspect it") + "</strong><p>"
        + t("左键拖动旋转，按住滚轮拖动平移，滚轮缩放；滑块在装配与展开之间切换。", "Left-drag to orbit, middle-drag to pan, scroll to zoom; the slider moves between assembled and exploded.")
        + '</p></div><div class="assembly-panel" hidden><ol class="assembly-steps"></ol><p class="assembly-note">'
        + t("演示使用原始 CAD 零件，用来说明零件之间的连接和转动关系，不是官方装配说明。",
            "The demonstration uses the original CAD parts to show how the pieces connect and turn; it is not an official build guide.")
        + "</p></div></div>"
    )


# ---------------------------------------------------------------- charts and cards


def training_chart(d, lang, data_dir: Path):
    zh = lang == "zh"
    data = json.loads((data_dir / d["data"]).read_text())
    metrics = [(m["key"], m["label"]) for m in d["metrics"]]
    panels = []
    for key, label in metrics:
        rows = data[key]
        values = [r["value"] for r in rows]
        low, high = min(0, min(values)), max(values) * 1.1
        # Long runs show every logged point as a line; dots stay for short runs.
        dense = len(rows) > 60
        last = max(r["step"] for r in rows)
        x = lambda step: 44 + step / last * 300
        y = lambda value: 190 - (value - low) / (high - low) * 150
        grid = "".join(f'<path class="chart-grid" d="M44 {y(low+(high-low)*i/4):.2f}H344"/><text x="36" y="{y(low+(high-low)*i/4)+4:.2f}" text-anchor="end">{low+(high-low)*i/4:.1f}</text>' for i in range(5))
        axis = "".join(f'<text x="{x(i-1):.2f}" y="213" text-anchor="middle">{i}</text>' for i in d.get("ticks", (1, 5, 10, 15, 20)))
        line = " ".join(f'{x(r["step"]):.2f},{y(r["value"]):.2f}' for r in rows)
        points = "".join(f'<circle class="chart-dot" cx="{x(r["step"]):.2f}" cy="{y(r["value"]):.2f}" r="{1.6 if dense else 3.5}" tabindex="0" data-value="{e(label)} · {r["step"]+1}: {r["value"]:.4f}" aria-label="{e(label)}, {r["step"]+1}, {r["value"]:.4f}"><title>{r["step"]+1}: {r["value"]:.4f}</title></circle>' for r in rows)
        panels.append(f'<div class="chart-panel"><h4>{e(label)}</h4><svg viewBox="0 0 370 242" role="group" aria-label="{e(label)}">{grid}{axis}<polyline class="chart-line" points="{line}"/>{points}<text x="194" y="238" text-anchor="middle">{e(d["x_label"])}</text></svg></div>')
    count = len(data[metrics[0][0]])
    table ="<table><thead><tr><th>" + e(d["iteration_label"]) + "</th>" + "".join("<th>" + e(label) + "</th>" for _, label in metrics) + "</tr></thead><tbody>"
    for i in range(count):
        table += f'<tr><td>{data[metrics[0][0]][i]["step"]+1}</td>' + "".join(f'<td>{data[key][i]["value"]:.6f}</td>' for key, _ in metrics) + "</tr>"
    table += "</tbody></table>"
    readout = "悬停、点选或用 Tab 选择数据点，查看原始数值。" if zh else "Hover, tap or Tab to a point to inspect its logged value."
    all_values = "查看全部数据" if zh else "View all values"
    return f'<figure class="course-chart" data-chart><div class="chart-header"><h3>{e(d["title"])}</h3><span>{e(d["badge"])}</span></div><div class="chart-grid-pair">{"".join(panels)}</div><p class="chart-readout" aria-live="polite">{readout}</p><details class="chart-data"><summary>{all_values}</summary><div class="table-scroll">{table}</div></details><figcaption>{inline_md(d["caption"])}</figcaption></figure>'


def benchmark_chart(d, headers, rows):
    bars = []
    for row in rows:
        counts = [re.search(r"(\d+)\s*/\s*(\d+)", cell) for cell in row[1:3]]
        if len(counts) != 2 or not all(counts):
            continue
        tracks = []
        for i, m in enumerate(counts):
            a, n = map(int, m.groups())
            # Value sits in its own column beside the track, in the text color.
            tracks.append(f'<div class="benchmark-series"><div class="benchmark-track" role="img" aria-label="{e(headers[i+1])}: {a}/{n}"><div class="benchmark-bar {"secondary" if i==0 else ""}" style="width:{100*a/n:.2f}%"></div></div><span class="benchmark-value" aria-hidden="true">{a}/{n}</span></div>')
        bars.append(f'<div class="benchmark-row"><strong>{e(row[0])}</strong><div class="benchmark-bars">{"".join(tracks)}</div></div>')
    if not bars:
        raise ValueError("benchmark-chart found no 'passes / total' cells in the following table")
    return ('<figure class="course-chart"><div class="chart-header"><h3>' + e(d["title"]) + '</h3></div><div class="chart-legend"><span><i class="chart-swatch chart-swatch-bar secondary" aria-hidden="true"></i>'
            + e(headers[1]) + '</span><span><i class="chart-swatch chart-swatch-bar" aria-hidden="true"></i>' + e(headers[2]) + "</span></div>" + "".join(bars)
            + "<figcaption>" + inline_md(d["caption"]) + "</figcaption></figure>")


def model_card(d):
    facts = "".join(f"<div><dt>{e(k)}</dt><dd>{inline_md(v)}</dd></div>" for k, v in d["facts"])
    return (f'<article class="course-model-card"><div class="model-card-header"><div><p class="model-card-family">{e(d["family"])}</p><h3 class="model-card-title">{e(d["name"])}</h3></div>'
            f'<a class="model-card-hub" href="{e(d["url"])}" target="_blank" rel="noopener">{e(d.get("hub_label", "🤗 Hugging Face ↗"))}</a></div><dl class="model-card-facts">{facts}</dl>'
            f'<p class="model-card-note">{inline_md(d["note"])}</p></article>')


def control_timing(d, lang):
    """Recorded update order for GR00T and the onboard policy over rounds k−1, k, k+1.

    Every time comes from the page YAML, which copies a recorded simulation
    excerpt (see docs/_static/data/control-timing-evidence.json).

    GR00T: each round is a new inference (simulation paused while the server
    computes) and four 80 ms commands; before commands 2–4 the simulation also
    waits briefly while a cached command is fetched. Inference and cache-wait
    columns are a schematic of order; within a command, positions follow
    simulation time. Onboard: a true simulation-time axis; sensor capture,
    decision and command are linked events; the measured decision time is written
    under each event and takes no axis width, because simulation time does not
    advance during the synchronous decision.

    Each item carries progress positions: data-t0/t1 (active) and data-x (no
    longer in effect). course-widgets.js marks items future/active/live/expired
    and draws the dependency arrows (data-from/data-to rows).
    """
    zh = lang == "zh"
    t = lambda a, b: a if zh else b
    pct = lambda u: f"{100 * u:.3f}%"
    FL = 0.012  # how long an instant event stays highlighted, in progress units

    def item(cls, u0, u1=None, *, t0, t1, x, rnd=None, label="", say=None, status=None, title=None):
        width = f";width:{pct(u1 - u0)}" if u1 is not None else ""
        attrs = f' data-t0="{t0:.5f}" data-t1="{t1:.5f}" data-x="{x:.5f}"'
        if rnd is not None:
            attrs += f' data-round="{rnd}"'
        if say:
            attrs += f' data-say="{e(say)}"'
        if status:
            attrs += f' data-status="{status}"'
        if title:
            attrs += f' title="{e(title)}"'
        inner = f'<span class="seq-text">{label}</span>' if label else ""
        return f'<span class="seq-item {cls}" style="left:{pct(u0)}{width}"{attrs}>{inner}</span>'

    def link(src, dst, x1, x2, *, t0, x, rnd):
        return (f'<g class="seq-link" data-from="{src}" data-to="{dst}" data-x1="{x1:.5f}" data-x2="{x2:.5f}" '
                f'data-t0="{t0:.5f}" data-t1="{t0 + FL:.5f}" data-x="{x:.5f}" data-round="{rnd}">'
                '<path class="seq-link-line"/><path class="seq-link-head"/></g>')

    def row(key, lane, items, cls=""):
        name, rate = (lane.get("name", ""), lane.get("rate", "")) if lane else ("", "")
        rate_html = f"<small>{inline_md(rate)}</small>" if rate else ""
        return (f'<div class="seq-row {cls}" data-row="{key}"><div class="seq-label"><strong>{inline_md(name)}</strong>{rate_html}</div>'
                f'<div class="seq-track">{"".join(items)}<span class="seq-cursor" aria-hidden="true"></span></div></div>')

    def board(key, rows, links, note, legend, hidden):
        return (f'<div class="seq-board" data-board="{key}"{" hidden" if hidden else ""}>'
                f'<ul class="seq-legend">{"".join(f"<li>{inline_md(x)}</li>" for x in legend)}</ul>'
                f'<div class="seq-stage">{"".join(rows)}<svg class="seq-links" aria-hidden="true">{"".join(links)}</svg></div>'
                f'<p class="seq-note">{inline_md(note)}</p></div>')

    rounds = d["rounds"]
    modes = d["modes"]
    first = next(iter(modes))  # default mode: its board is visible without JavaScript
    fmt = lambda v: f"{v:.1f}"
    fill = lambda text, **kw: text.format(**kw)

    # ---- GR00T ---------------------------------------------------------------
    g = d["groot"]
    starts, infer_ms = g["round_start_ms"], g["inference_ms"]
    cmd_ms = g["command_ms"]
    period = starts[1] - starts[0]
    per = round(period / cmd_ms)
    assert len(rounds) == 3 and len(starts) == 4 and all(b - a == period for a, b in zip(starts, starts[1:]))
    assert len(infer_ms) == 3 and all(len(c) == per - 1 for c in g["cached_ms"])
    A, C = float(g["infer_share"]), float(g["cache_share"])
    W = (1 - A - (per - 1) * C) / per          # one command, as a share of a round
    U = 1 / 3
    seg_left = lambda r, j: r * U + (A + j * (W + C)) * U
    infer_end = lambda r: r * U + A * U

    def gx(ms, after=True):
        """Board position of simulation time ms; `after` places boundary times after the cache wait."""
        rel = ms - starts[0]
        s = int(rel // cmd_ms) if after else -int(-rel // cmd_ms) - 1
        s = max(0, min(3 * per - 1, s))
        return seg_left(s // per, s % per) + (rel - s * cmd_ms) / cmd_ms * W * U

    lanes = g["lanes"]
    head, phase, cam, cmd, act, links = [], [], [], [], [], []
    for r, rd in enumerate(rounds):
        r0, r1, ie = r * U, (r + 1) * U, infer_end(r)
        head.append(item("seq-round", r0, r1, t0=r0, t1=r1, x=r1, rnd=r, label=inline_md(rd["label"])))
        phase.append(item("seq-phase is-infer", r0, ie, t0=r0, t1=ie, x=ie, rnd=r, status="infer",
                          label=inline_md(fill(g["infer_label"], ms=fmt(infer_ms[r])))))
        phase.append(item("seq-phase", ie, r1, t0=ie, t1=r1, x=r1, rnd=r, label=inline_md(g["exec_label"])))
        cam.append(item("is-pause", r0, ie, t0=r0, t1=ie, x=ie, rnd=r, label=e(g["paused_label"])))
        act.append(item("is-pause", r0, ie, t0=r0, t1=ie, x=ie, rnd=r, label=e(g["actor_paused_label"])))
        hist = [starts[r] - h for h in g["history_frame_ms"][r]]
        say = fill(g["say_infer"], round=rd["name"], ms=fmt(infer_ms[r]), age=f"{hist[-1]:g}",
                   history=t("、", ", ").join(f"{v / 1000:.2f}" for v in sorted(hist[:-1])))
        cmd.append(item("is-predict", r0, ie, t0=r0, t1=ie, x=r1, rnd=r, status="infer", say=say,
                        label=inline_md(g["predict_label"])))
        for j in range(per):
            a0 = seg_left(r, j)
            if j:
                w0, wait = a0 - C * U, g["cached_ms"][r][j - 1]
                tip = fill(g["cache_title"], ms=fmt(wait))
                waiting = fill(g["say_wait"], round=rd["name"], n=j + 1, ms=fmt(wait))
                cmd.append(item("is-pause is-wait", w0, a0, t0=w0, t1=a0, x=a0, rnd=r,
                                title=tip, status="cache", say=waiting))
            say = fill(g["say_first"] if j == 0 else g["say_cached"], round=rd["name"], n=j + 1,
                       ms=fmt(g["cached_ms"][r][j - 1]) if j else "")
            cmd.append(item("is-command", a0, a0 + W * U, t0=a0, t1=a0 + W * U, x=a0 + W * U, rnd=r, say=say, label=str(j + 1)))
    # Camera: instant capture marks; only the frame read by a new inference keeps its round color.
    caps, latest = g["capture_ms"], g["latest_frame_ms"]
    for i, c in enumerate(caps):
        u = gx(c, after=False)
        nxt = gx(caps[i + 1], after=False) if i + 1 < len(caps) else 1.0
        if c in latest:
            q = latest.index(c)
            cam.append(item("is-capture is-used", u, t0=u, t1=u + FL, x=infer_end(q), rnd=q,
                            title=fill(g["used_title"], round=rounds[q]["name"])))
            links.append(link("cam", "cmd", u, q * U + A * U / 2, t0=u, x=infer_end(q), rnd=q))
        else:
            cam.append(item("is-capture", u, t0=u, t1=u + FL, x=nxt, title=g["capture_title"]))
    if latest[0] < starts[0]:
        cam.append(item("is-capture is-used is-lead", 0, t0=0, t1=FL, x=infer_end(0), rnd=0,
                        title=fill(g["lead_title"], age=f"{starts[0] - latest[0]:g}")))
        links.append(link("cam", "cmd", 0, A * U / 2, t0=0, x=infer_end(0), rnd=0))
    ticks = g["actor_ms"]
    assert len(ticks) == 3 * per * round(cmd_ms / g["actor_step_ms"])
    for a in ticks:
        s = int((a - starts[0]) // cmd_ms)
        u = gx(a)
        act.append(item("is-tick", u, t0=u, t1=u + FL, x=seg_left(s // per, s % per) + W * U, rnd=s // per))
    groot = board("groot", [
        row("head", None, head, "seq-head"), row("phase", {"name": g["phase_name"]}, phase, "seq-phases"),
        row("cam", lanes["camera"], cam), row("cmd", lanes["commands"], cmd), row("act", lanes["actor"], act),
    ], links, g["note"], g["legend"], hidden=first != "groot")

    # ---- Onboard: true simulation-time axis --------------------------------------
    o = d["onboard"]
    caps = [float(v) for v in o["capture_ms"]]
    assert len(caps) == 4
    span = caps[-1] - caps[0]
    ux = lambda ms: (ms - caps[0]) / span
    lanes = o["lanes"]
    head, sens, dec, cmd, act, links = [], [], [], [], [], []
    ticks = [float(v) for v in o["actor_ms"]]
    owner = o["actor_round"]  # recorded: the decision whose command each motor update used
    decide, detect, navigate = o["decision_ms"], o["detector_ms"], o["navigation_ms"]
    assert len(owner) == len(ticks) and len(decide) == len(detect) == len(navigate) == 3
    for r, rd in enumerate(rounds):
        u0, u1 = ux(caps[r]), ux(caps[r + 1])
        n = owner.count(r)
        times = dict(ms=fmt(decide[r]), det=fmt(detect[r]), nav=fmt(navigate[r]))
        head.append(item("seq-round", u0, u1, t0=u0, t1=u1, x=u1, rnd=r, label=inline_md(rd["label"])))
        sens.append(item("is-capture", u0, t0=u0, t1=u0 + FL, x=u1, rnd=r, title=o["capture_title"]))
        sens.append(item("is-mark-label", u0, u1, t0=u0, t1=u0 + FL, x=u1, rnd=r, label=inline_md(o["capture_label"])))
        dec.append(item("is-event", u0, t0=u0, t1=u0 + FL, x=u1, rnd=r, title=fill(o["decision_title"], **times)))
        dec.append(item("is-mark-label", u0, u1, t0=u0, t1=u0 + FL, x=u1, rnd=r,
                        label=f'{inline_md(o["decision_label"])}<small>{inline_md(fill(o["decision_note"], **times))}</small>'))
        say = fill(o["say"], round=rd["name"], k=rd["plain"], n=n, **times)
        cmd.append(item("is-command", u0, u1, t0=u0, t1=u1, x=u1, rnd=r, say=say, status="frame",
                        label=inline_md(rounds[r]["command"])))
        links.append(link("sens", "dec", u0, u0, t0=u0, x=u1, rnd=r))
        links.append(link("dec", "cmd", u0, u0, t0=u0, x=u1, rnd=r))
        for k, who in zip(ticks, owner):
            if who == r:
                act.append(item("is-tick", ux(k), t0=ux(k), t1=ux(k) + FL, x=u1, rnd=r))
    sens.append(item("is-capture is-next", 1.0, t0=1.0, t1=1.0, x=2.0, title=o["next_title"]))
    onboard = board("onboard", [
        row("head", None, head, "seq-head"), row("sens", lanes["sensors"], sens), row("dec", lanes["decision"], dec),
        row("cmd", lanes["commands"], cmd), row("act", lanes["actor"], act),
    ], links, o["note"], o["legend"], hidden=first != "onboard")

    buttons = "".join(
        f'<button type="button" role="radio" aria-checked="{"true" if key == first else "false"}" data-mode="{e(key)}">{e(label)}</button>'
        for key, label in modes.items())
    facts = "".join(f'<li data-modes="{e(" ".join(f.get("modes", list(modes))))}">' + inline_md(f["text"]) + "</li>" for f in d["facts"])
    config = {"replay": d["replay_ms"]}
    return (f'<figure class="course-chart seq-timing" data-seq-timing data-config=\'{e(json.dumps(config))}\'>'
            f'<div class="chart-header"><h3>{e(d["title"])}</h3><span>{e(d["subtitle"])}</span></div>'
            f'<div class="timing-modes" role="radiogroup" aria-label="{t("控制方式", "Control approach")}">{buttons}</div>'
            + "".join({"groot": groot, "onboard": onboard}[key] for key in modes)
            + '<p class="timing-status" aria-live="polite"></p>'
            f'<div class="timing-controls"><button type="button" class="course-button timing-replay" data-play="{e(d["play_label"])}" data-pause="{t("暂停", "Pause")}">{e(d["play_label"])}</button>'
            f'<label class="timing-scrub-label">{t("进度", "Progress")}<input class="timing-scrub" type="range" min="0" max="1000" value="0"></label></div>'
            f'<details class="timing-details"><summary>{t("详解", "Details")}</summary><ul class="timing-facts">{facts}</ul></details>'
            f'<figcaption>{inline_md(d["caption"])}</figcaption></figure>')


def code_story(d, lang):
    """Numbered explanations paired with the code chunks they explain.

    Each step and its chunk are adjacent cells of one grid row, so a pair always shares
    its top and bottom and the row takes the taller of the two. On desktop the steps form
    the left column and the chunks the right; a narrow viewport reads them in document
    order (step, then its code). Each chunk appears once.
    """
    zh = lang == "zh"
    code_lines = d["code"].rstrip("\n").splitlines()
    rows = []
    for i, step in enumerate(d["steps"]):
        a, z = step["lines"]
        snippet = pygmentize("\n".join(code_lines[a:z]), d.get("language", "python"))
        rows.append(f'<article class="story-step" data-story-step="{i}" tabindex="0"><div class="story-text">'
                    f'<h4><span>{i+1}</span>{e(step["title"])}</h4> <p>{inline_md(step["body"])}</p></div></article>'
                    f'<div class="story-code-chunk" data-story-target="{i}"><span class="story-code-number" aria-hidden="true">{i+1}</span>'
                    f'<div class="highlight"><pre><code>{snippet}</code></pre></div></div>')
    link = f'<a class="source-link" href="{e(d["source"])}" target="_blank" rel="noopener">{"完整代码 ↗" if zh else "Full source ↗"}</a>' if d.get("source") else ""
    return (f'<div class="code-story" data-story><div class="story-corner" aria-hidden="true"></div>'
            f'<div class="code-toolbar"><span>{e(d["label"])}</span>{link}</div>{"".join(rows)}</div>')


def scenario_matrix(d):
    factors = "".join(f'<label><input type="checkbox" data-factor="{int(f)}"> {e(label)}</label>' for f, label in d["factors"])
    return (f'<div class="scenario-matrix" data-scenario-matrix><p><strong>{e(d["title"])}</strong></p><div class="matrix-options">{factors}</div>'
            f'<label>{e(d["seeds_label"])} <input class="matrix-seeds" type="number" min="1" max="100" value="{int(d.get("seeds", 5))}"></label>'
            f'<p class="matrix-output" aria-live="polite"></p><p class="matrix-scope">{inline_md(d["scope"])}</p></div>')


def recorded_run(d, lang, image_url=None):
    """A command we ran, shown with its real output; learners read it instead of running it.

    No copy control (the copy button only targets terminal blocks); the output and
    interpretation start collapsed below the command.
    """
    zh = lang == "zh"
    badge = d.get("badge") or ("录制示例 · 无需运行" if zh else "Recorded run · no need to run")
    meta = f'<span class="recorded-meta">{inline_md(d["recorded"])}</span>' if d.get("recorded") else ""
    where = f'<span class="recorded-where">{e(d["where"])}</span>' if d.get("where") else ""
    command = pygmentize(d["command"].rstrip("\n"), d.get("language", "bash"))
    parts = []
    if image_url:
        parts.append(f'<img class="recorded-image" src="{e(image_url)}" alt="{e(d.get("image_alt", ""))}" loading="lazy">')
    if d.get("output"):
        label = d.get("output_label") or ("终端输出" if zh else "Terminal output")
        parts.append(f'<p class="recorded-output-label">{e(label)}</p><pre class="recorded-output"><code>{e(d["output"].rstrip())}</code></pre>')
    for paragraph in [p for p in str(d.get("result", "")).split("\n\n") if p.strip()]:
        parts.append(f'<p>{inline_md(paragraph.strip())}</p>')
    summary = d.get("result_label") or ("展开查看真实输出与结果解读" if zh else "Show the actual output and how to read it")
    caption = f'<figcaption>{inline_md(d["caption"])}</figcaption>' if d.get("caption") else ""
    return (f'<figure class="recorded-run"><div class="recorded-head"><span class="recorded-badge">{e(badge)}</span>{where}{meta}</div>'
            f'<div class="recorded-command highlight"><pre><code>{command}</code></pre></div>'
            f'<details class="recorded-result"><summary>{e(summary)}</summary><div class="recorded-body">{"".join(parts)}</div></details>'
            f'{caption}</figure>')
