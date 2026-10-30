"""Generate accessible SVG diagrams with explicit ports and orthogonal connectors."""

from html import escape as e
import hashlib
from pathlib import Path
import unicodedata

import yaml


def lines(text, limit=23):
    if text.isascii():
        import textwrap

        return textwrap.wrap(
            text, width=limit, break_long_words=False, break_on_hyphens=False
        )
    result = []
    line = ""
    width = 0
    for char in text:
        n = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if width + n > limit and line:
            result.append(line)
            line = ""
            width = 0
        line += char
        width += n
    if line:
        result.append(line)
    return result


def flow_svg(b):
    if b.get("layout") == "adaptive":
        return adaptive_flow_svg(b) + adaptive_flow_svg(b, compact=True)
    nodes = b["nodes"]
    n = len(nodes)
    uid = "flow-" + hashlib.sha1(b["label"].encode()).hexdigest()[:9]
    six = n == 6
    loop = b.get("loop", False)
    height = 436 if six else (264 if loop else 228)
    positions = [(24, 64), (280, 64), (536, 64), (536, 268), (280, 268), (24, 268)][:n]
    colors = [
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-muted-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-support-success-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-muted-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-muted-color)"),
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg" viewBox="0 0 760 {height}" role="img" aria-labelledby="{uid}-title {uid}-desc"><title id="{uid}-title">{e(b["label"])}</title><desc id="{uid}-desc">'
        + e("；".join(a + "：" + c for a, c in nodes))
        + "</desc>",
        f'<defs><marker id="{uid}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L6 3.5L0 7" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.3"/></marker></defs>',
    ]
    groups = b.get("groups", [])
    if six:
        for y, text in [
            (20, groups[0] if groups else "任务执行"),
            (224, groups[1] if len(groups) > 1 else "结果与反馈"),
        ]:
            out.append(
                f'<rect x="8" y="{y}" width="744" height="176" rx="10" fill="var(--nve-sys-layer-container-background)" stroke="var(--nve-sys-layer-container-accent-background)" stroke-dasharray="4 4"/><text x="24" y="{y+25}" fill="var(--nve-sys-text-muted-color)" font-size="13">{e(text)}</text>'
            )
    else:
        out.append(
            f'<text x="24" y="32" fill="var(--nve-sys-text-muted-color)" font-size="13">{e(b["label"])}</text>'
        )
    labels = b.get("edges", [])
    for i in range(n - 1):
        x, y = positions[i]
        nx, ny = positions[i + 1]
        if y == ny:
            right = nx > x
            start = x + 200 if right else x
            end = nx - 6 if right else nx + 206
            path = f"M{start} {y+56}H{end}"
            lx = (start + end) / 2
            ly = y + 40
        else:
            path = f"M{x+100} {y+112}V{ny-6}"
            lx = x + 112
            ly = (y + 112 + ny) / 2
        out.append(
            f'<path d="{path}" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.5" stroke-linejoin="round" marker-end="url(#{uid}-arrow)"/>'
        )
        if i < len(labels) and labels[i]:
            out.append(
                f'<text x="{lx}" y="{ly}" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="12">{e(labels[i])}</text>'
            )
    if loop and not six:
        out.append(
            f'<path d="M636 176V214Q636 222 628 222H132Q124 222 124 214V182" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.5" marker-end="url(#{uid}-arrow)"/><rect x="290" y="209" width="180" height="24" fill="var(--nve-sys-layer-container-background)"/><text x="380" y="226" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="13">{e(b.get("loop_label","下一次观测"))}</text>'
        )
    for i, ((title, body), (x, y)) in enumerate(zip(nodes, positions)):
        fill, stroke = colors[i]
        out.append(
            f'<rect x="{x}" y="{y}" width="200" height="112" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )
        for j, t in enumerate(lines(title, 23)):
            out.append(
                f'<text x="{x+100}" y="{y+29+j*21}" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="16" font-weight="600">{e(t)}</text>'
            )
        for j, t in enumerate(sum((lines(part, 26) for part in body.split("，")), [])):
            out.append(
                f'<text x="{x+100}" y="{y+66+j*20}" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="13">{e(t)}</text>'
            )
    return "".join(out) + "</svg>"


def platform_svg(lang="zh"):
    zh = lang == "zh"
    L = (dict(title="Isaac Lab、Newton 与 Isaac Sim 的职责",
              desc="Isaac Lab 定义机器人学习任务，Newton 在 GPU 上推进物理状态。需要图像或交互界面时，由 Isaac Sim 提供渲染和传感器能力。Newton 物理执行不必依赖 Kit。",
              group="任务接口、物理后端与渲染能力分工", left="动作 ⇄ 物理状态", right="按需接入图像与界面",
              lab="任务 · 观测 · 奖励 · 并行环境", newton="Warp / CUDA · 动力学与接触", newton_note="物理执行可不依赖 Kit",
              sim="OpenUSD · RTX · 相机与交互", sim_note="本课程可视化任务使用此路径") if zh else
         dict(title="What Isaac Lab, Newton and Isaac Sim each do",
              desc="Isaac Lab defines the robot learning task and Newton advances the physical state on the GPU. When images or an interactive window are needed, Isaac Sim provides rendering and sensors. Newton physics does not need Kit.",
              group="Task interface, physics backend and rendering", left="actions ⇄ physical state", right="images and window when needed",
              lab="Task · observations · rewards · parallel envs", newton="Warp / CUDA · dynamics and contact", newton_note="Physics can run without Kit",
              sim="OpenUSD · RTX · cameras and interaction", sim_note="The course's visual labs use this path"))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg" viewBox="0 0 760 384" role="img" aria-labelledby="platform-title platform-desc"><title id="platform-title">{L["title"]}</title><desc id="platform-desc">{L["desc"]}</desc><defs><marker id="platform-arrow" markerWidth="5.25" markerHeight="5.25" refX="4.5" refY="2.625" orient="auto-start-reverse"><path d="M0 0L4.5 2.625L0 5.25" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.3"/></marker></defs>
<rect x="16" y="16" width="728" height="352" rx="10" fill="var(--nve-sys-layer-container-background)" stroke="var(--nve-sys-layer-container-accent-background)" stroke-dasharray="4 4"/>
<text x="36" y="42" fill="var(--nve-sys-text-muted-color)" font-size="13">{L["group"]}</text>
<path marker-start="url(#platform-arrow)" d="M310 150V176Q310 184 302 184H180Q172 184 172 192V226" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<path d="M588 232V192Q588 184 580 184H458Q450 184 450 176V156" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<text x="176" y="174" fill="var(--nve-sys-text-muted-color)" font-size="13">{L["left"]}</text><text x="470" y="174" fill="var(--nve-sys-text-muted-color)" font-size="13">{L["right"]}</text>
<rect x="240" y="62" width="280" height="88" rx="7" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-text-color)"/><text x="380" y="95" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="20" font-weight="600">Isaac Lab</text><text x="380" y="124" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">{L["lab"]}</text>
<rect x="40" y="232" width="264" height="94" rx="7" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-support-success-color)"/><text x="172" y="264" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="19" font-weight="600">Newton</text><text x="172" y="292" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">{L["newton"]}</text><text x="172" y="348" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="13">{L["newton_note"]}</text>
<rect x="456" y="232" width="264" height="94" rx="7" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-text-muted-color)"/><text x="588" y="264" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="19" font-weight="600">Isaac Sim</text><text x="588" y="292" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">{L["sim"]}</text><text x="588" y="348" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="13">{L["sim_note"]}</text></svg>"""


def network_parts(lang):
    """Explanations for each layer column and each fully connected band."""
    zh = lang == "zh"
    widths = [61, 512, 256, 128, 14]
    layers = (["观测输入：61 维，进入网络前先做归一化；每一维的顺序与含义见下表。",
               "隐藏层 1：512 个神经元，激活函数 ELU。",
               "隐藏层 2：256 个神经元，激活函数 ELU。",
               "隐藏层 3：128 个神经元，激活函数 ELU。",
               "动作输出：14 维，每个关节一个目标角度偏移；对应的关节见下表。"] if zh else
              ["Observation input: 61 values, normalized before entering the network; the table below lists the order and meaning of each.",
               "Hidden layer 1: 512 neurons with ELU activation.",
               "Hidden layer 2: 256 neurons with ELU activation.",
               "Hidden layer 3: 128 neurons with ELU activation.",
               "Action output: 14 values, one target angle offset per joint; the table below lists the joints."])
    bands = []
    for a, b in zip(widths, widths[1:]):
        bands.append(f"全连接 {a} → {b}：每个输入都连到每个神经元，共 {a} × {b} = {a * b:,} 个连接权重。" if zh
                     else f"Fully connected {a} → {b}: every input connects to every neuron, {a} × {b} = {a * b:,} connection weights.")
    return layers, bands


def network_svg(lang):
    """Actual actor widths, square-root-compressed heights and sampled neurons.

    Each layer column and each band of connections between two layers is a
    focusable region; course-widgets.js highlights it and shows its text below.
    """
    zh = lang == "zh"
    names = (["观测输入", "隐藏层 1", "隐藏层 2", "隐藏层 3", "动作输出"]
             if zh else ["Observation", "Hidden 1", "Hidden 2", "Hidden 3", "Action"])
    widths = [61, 512, 256, 128, 14]
    xs = [70, 225, 380, 535, 690]
    counts = [5, 11, 9, 7, 3]
    heights = [280 * (width / 512) ** .5 for width in widths]
    center = 230
    layer_text, band_text = network_parts(lang)
    # The center slot is an ellipsis; all remaining slots are representative units.
    positions = [[center - height / 2 + j * height / (count - 1)
                  for j in range(count) if j != count // 2]
                 for height, count in zip(heights, counts)]
    title = "行走策略网络：61 → 512 → 256 → 128 → 14" if zh else "Walking actor: 61 → 512 → 256 → 128 → 14"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" class="network-svg" viewBox="0 0 760 428" role="group" aria-labelledby="actor-network-title" font-family="Open Sans, Noto Sans SC, sans-serif"><title id="actor-network-title">{e(title)}</title>']
    half = 30  # half-width of a layer column's hit region
    for i in range(4):
        edges = "".join(f'<path d="M{xs[i]+7} {y1:.2f}L{xs[i+1]-7} {y2:.2f}"/>' for y1 in positions[i] for y2 in positions[i + 1])
        out.append(f'<g class="network-part network-band" data-part="band-{i}" data-info="{e(band_text[i])}" role="button" tabindex="0" aria-label="{e(band_text[i])}">'
                   f'<rect class="network-hit" x="{xs[i]+half}" y="10" width="{xs[i+1]-xs[i]-2*half}" height="408"/>'
                   f'<g class="network-edges">{edges}</g></g>')
    for i, (x, width, name, height) in enumerate(zip(xs, widths, names, heights)):
        out.append(f'<g class="network-part network-layer" data-part="layer-{i}" data-dimension="{width}" data-span="{height:.3f}" data-info="{e(layer_text[i])}" role="button" tabindex="0" aria-label="{e(layer_text[i])}">'
                   f'<rect class="network-hit" x="{x-half}" y="10" width="{2*half}" height="408"/>'
                   f'<text class="network-name" x="{x}" y="26" text-anchor="middle" font-size="15">{name}</text>'
                   f'<text class="network-width" x="{x}" y="56" text-anchor="middle" font-size="23" font-weight="600">{width}</text>')
        for y in positions[i]:
            out.append(f'<circle class="network-neuron" cx="{x}" cy="{y:.2f}" r="7" stroke-width="1.5"/>')
        out.append(f'<rect x="{x-10}" y="{center-13}" width="20" height="26" fill="var(--soft, var(--nve-sys-layer-container-background))"/><text class="network-ellipsis" x="{x}" y="{center+6}" text-anchor="middle" font-size="20">⋮</text>')
        foot = ("输入归一化" if zh else "Normalize") if i == 0 else ("ELU" if i < 4 else ("关节动作" if zh else "Joint actions"))
        out.append(f'<text class="network-foot" x="{x}" y="408" text-anchor="middle" font-size="14">{foot}</text></g>')
    return "".join(out) + "</svg>"


def adaptive_flow_svg(b, compact=False):
    """Fit every node and connector, including four-node teaching flows."""
    nodes = b["nodes"]
    cols = 1 if compact else (2 if len(nodes) == 4 else 3)
    canvas_width = 360 if compact else 760
    width = (canvas_width - 48 - 56 * (cols - 1)) / cols
    positions = []
    wrapped = [
        (lines(t, int(width / 8)), lines(body, int(width / 7))) for t, body in nodes
    ]
    card_height = max(32 + len(t) * 22 + len(body) * 21 for t, body in wrapped)
    rows = (len(nodes) + cols - 1) // cols
    height = 48 + rows * card_height + (rows - 1) * 60
    uid = (
        "flow-"
        + hashlib.sha1(b["label"].encode()).hexdigest()[:9]
        + ("-mobile" if compact else "-desktop")
    )
    layout_class = "flow-mobile" if compact else "flow-desktop"
    for i in range(len(nodes)):
        row, col = divmod(i, cols)
        if row % 2:
            col = cols - 1 - col
        positions.append((24 + col * (width + 56), 24 + row * (card_height + 60)))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg {layout_class}" viewBox="0 0 {canvas_width} {height}" role="img" aria-labelledby="{uid}-title {uid}-desc"><title id="{uid}-title">{e(b["label"])}</title><desc id="{uid}-desc">'
        + e(" → ".join(t + ": " + body for t, body in nodes))
        + "</desc>",
        f'<defs><marker id="{uid}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L6 3.5L0 7" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.3"/></marker></defs>',
    ]
    paths = []
    for (x, y), (nx, ny) in zip(positions, positions[1:]):
        if y == ny:
            start, end = (x + width, nx - 6) if nx > x else (x, nx + width + 6)
            paths.append(f"M{start} {y + card_height / 2}H{end}")
        else:
            paths.append(f"M{x + width / 2} {y + card_height}V{ny - 6}")
    if b.get("loop") and len(nodes) == 4:
        x, y = positions[-1]
        paths.append(f"M{x} {y + card_height / 2}H8V{24 + card_height / 2}H18")
    for path in paths:
        out.append(
            f'<path d="{path}" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.5" marker-end="url(#{uid}-arrow)"/>'
        )
    colors = [
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-muted-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-support-success-color)"),
        ("var(--nve-sys-layer-container-accent-background)", "var(--nve-sys-text-muted-color)"),
    ]
    for i, ((title, body), (x, y)) in enumerate(zip(wrapped, positions)):
        fill, stroke = colors[i % len(colors)]
        out.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{card_height}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )
        for j, line in enumerate(title):
            out.append(
                f'<text x="{x + width / 2}" y="{y + 28 + j * 22}" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="16" font-weight="600">{e(line)}</text>'
            )
        for j, line in enumerate(body):
            out.append(
                f'<text x="{x + width / 2}" y="{y + 42 + len(title) * 22 + j * 21}" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">{e(line)}</text>'
            )
    return "".join(out) + "</svg>"


def vla_joint_svg(lang):
    """A VLA that outputs the 14 servo targets directly (Section 4.4), drawn like the
    walking-actor network: columns of nodes joined by bands of connections.

    RGB images and the instruction enter the vision-language model (VLM); its features
    and the separately encoded robot state condition the action head, whose output column
    has the 14 actual servo targets in interface order (left leg 5, neck/head 4, right
    leg 5). Only the output column is drawn to its true size: the other columns are a few
    sample nodes and an ellipsis, without layer counts or feature widths, because no such
    model was trained for the course. Every column, output group and band is a focusable
    part; course-widgets.js shows its explanation below the figure.
    """
    zh = lang == "zh"
    info = vla_joint_parts(lang)
    T = (dict(title="直接输出 14 个舵机目标的 VLA（示意）", img="RGB 图像", text="语言指令", state="机器人状态",
              vlm="视觉语言模型", head="动作头", out="舵机目标", vlm_foot="编码图像与指令", head_foot="flow matching",
              out_foot="每次生成未来多步", groups=["左腿", "颈部与头部", "右腿"]) if zh else
         dict(title="A VLA that outputs the 14 servo targets directly (schematic)", img="RGB images", text="Instruction",
              state="Robot state", vlm="Vision-language model", head="Action head", out="Servo targets",
              vlm_foot="encodes images and text", head_foot="flow matching", out_foot="several future steps per call",
              groups=["Left leg", "Neck and head", "Right leg"]))
    x_in, x_vlm, x_head, x_out = 70, 260, 450, 620
    r, step = 7, 22

    def column(top, slots, gap=step):
        ys = [top + j * gap for j in range(slots)]
        return ys, [y for j, y in enumerate(ys) if j != slots // 2]

    # Input groups: label 16 px above the first node.
    img_slots, img = column(96, 5, 20)
    text_slots, text = column(222, 5, 20)
    state_slots, state = column(348, 5, 20)
    vlm_slots, vlm = column(96, 7)
    head_slots, head = column(150, 13)
    # 14 output nodes in three groups with a small gap between groups.
    out, out_groups, y = [], [], 96
    for count in (5, 4, 5):
        group = [y + j * step for j in range(count)]
        out += group
        out_groups.append(group)
        y = group[-1] + step + 14
    # State connectors pass below the VLM column without touching it.
    vlm_bottom = vlm_slots[-1] + r
    for y1 in state:
        for y2 in head:
            assert y1 + (y2 - y1) * (x_vlm - x_in) / (x_head - x_in) > vlm_bottom + 12

    def edges(xa, ya, xb, yb):
        return "".join(f'<path d="M{xa + r} {a}L{xb - r} {b}"/>' for a in ya for b in yb)

    def band(part, hit, paths):
        x, y0, w, h = hit
        return (f'<g class="network-part network-band" data-part="{part}" data-info="{e(info[part])}" role="button" tabindex="0" aria-label="{e(info[part])}">'
                f'<rect class="network-hit" x="{x}" y="{y0}" width="{w}" height="{h}"/><g class="network-edges">{paths}</g></g>')

    def nodes(ys, x, slots=None):
        out_ = "".join(f'<circle class="network-neuron" cx="{x}" cy="{y}" r="{r}" stroke-width="1.5"/>' for y in ys)
        if slots:
            c, half_gap = slots[len(slots) // 2], slots[1] - slots[0] - r - 1
            out_ += (f'<rect x="{x - 10}" y="{c - half_gap}" width="20" height="{2 * half_gap}" fill="var(--soft, var(--nve-sys-layer-container-background))"/>'
                     f'<text class="network-ellipsis" x="{x}" y="{c + 7}" text-anchor="middle" font-size="20">⋮</text>')
        return out_

    def layer(part, hit, inner):
        x, y0, w, h = hit
        return (f'<g class="network-part network-layer" data-part="{part}" data-info="{e(info[part])}" role="button" tabindex="0" aria-label="{e(info[part])}">'
                f'<rect class="network-hit" x="{x}" y="{y0}" width="{w}" height="{h}"/>{inner}</g>')

    def name(x, y, text, anchor="middle", size=15):
        return f'<text class="network-name" x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}">{e(text)}</text>'

    def sub(x, text):
        return f'<text class="network-foot" x="{x}" y="48" text-anchor="middle" font-size="13">{e(text)}</text>'

    out_svg = [f'<svg xmlns="http://www.w3.org/2000/svg" class="network-svg vla-svg" viewBox="0 0 760 460" role="group" aria-labelledby="vla-joint-title" font-family="Open Sans, Noto Sans SC, sans-serif">'
               f'<title id="vla-joint-title">{e(T["title"])}</title>']
    half = 30
    out_svg.append(band("img-vlm", (x_in + half, 70, x_vlm - x_in - 2 * half, 124), edges(x_in, img, x_vlm, vlm)))
    out_svg.append(band("text-vlm", (x_in + half, 194, x_vlm - x_in - 2 * half, 124), edges(x_in, text, x_vlm, vlm)))
    out_svg.append(band("state-head", (x_in + half, 318, x_head - x_in - 2 * half, 124), edges(x_in, state, x_head, head)))
    out_svg.append(band("vlm-head", (x_vlm + half, 70, x_head - x_vlm - 2 * half, 248), edges(x_vlm, vlm, x_head, head)))
    out_svg.append(band("head-out", (x_head + half, 70, x_out - x_head - 2 * half, 372), edges(x_head, head, x_out, out)))
    for part, ys, slots in (("img", img, img_slots), ("text", text, text_slots), ("state", state, state_slots)):
        out_svg.append(layer(part, (x_in - half - 30, slots[0] - 32, 2 * half + 60, slots[-1] - slots[0] + 44),
                             name(x_in, slots[0] - 16, T[part], size=14) + nodes(ys, x_in, slots)))
    out_svg.append(layer("vlm", (x_vlm - half, 10, 2 * half, vlm_slots[-1] + 2 * r), name(x_vlm, 26, T["vlm"]) + sub(x_vlm, T["vlm_foot"]) + nodes(vlm, x_vlm, vlm_slots)))
    out_svg.append(layer("head", (x_head - half, 10, 2 * half, head_slots[-1] + 2 * r), name(x_head, 26, T["head"]) + sub(x_head, T["head_foot"]) + nodes(head, x_head, head_slots)))
    out_svg.append(name(x_out, 26, T["out"]) + f'<text class="network-width" x="{x_out}" y="56" text-anchor="middle" font-size="23" font-weight="600">14</text>'
                   + f'<text class="network-foot" x="{x_out + 40}" y="{out[-1] + 40}" text-anchor="middle" font-size="13">{e(T["out_foot"])}</text>')
    joints = yaml.safe_load((Path(__file__).resolve().parents[2] / "_data" / "network-interface.yaml").read_text(encoding="utf-8"))["joints"]
    first = 1
    for g, (group, label) in enumerate(zip(out_groups, T["groups"])):
        mid = (group[0] + group[-1]) / 2
        last = first + len(group) - 1
        # Each output node is one actual servo target, named by its joint.
        named = "".join(f'<circle class="network-neuron" cx="{x_out}" cy="{y}" r="{r}" stroke-width="1.5"><title>{first + j} {e(joints[first + j - 1][1 if zh else 2])}</title></circle>'
                        for j, y in enumerate(group))
        out_svg.append(layer(f"out-{g}", (x_out - 20, group[0] - 11, 160, group[-1] - group[0] + 22),
                             named + name(x_out + 18, mid - 2, label, "start", 14)
                             + f'<text class="network-foot" x="{x_out + 18}" y="{mid + 15}" font-size="12">{first}–{last}</text>'))
        first = last + 1
    return "".join(out_svg) + "</svg>"


def vla_joint_parts(lang):
    """Explanations shown below the direct-joint VLA figure for each column, group and band."""
    data = yaml.safe_load((Path(__file__).resolve().parents[2] / "_data" / "network-interface.yaml").read_text(encoding="utf-8"))
    joints, ranges = data["joints"], data["output_groups"]
    zh = lang == "zh"
    sep = "、" if zh else ", "
    names = [sep.join(joints[i - 1][1 if zh else 2] for i in range(a, b + 1)) for a, b in ranges]
    if zh:
        groups = ["左腿", "颈部与头部", "右腿"]
        parts = dict(
            img="RGB 图像：机载相机的画面，可以包含过去几帧。模型要从中看出球、球门和周围环境的位置关系。",
            text="语言指令：一句描述任务的话，例如“走向足球，用右脚把球踢进球门”。它和图像一起送入视觉语言模型。",
            state="机器人状态：关节角度与速度、机身角速度与重力方向、上一步发出的动作。直接输出关节时，模型要靠它判断身体此刻是否平衡。",
            vlm="视觉语言模型：把图像和指令编码成特征，例如 GR00T 中的 Cosmos-Reason2-2B。它本身不输出关节动作。",
            head="动作头：结合视觉语言特征和机器人状态，用 flow matching 从噪声逐步生成一段动作。直接控制关节时，这段动作的每一步都是 14 个舵机目标。",
            **{"img-vlm": "图像被编码成特征后进入视觉语言模型。连线只表示数据流向，不代表具体的连接方式或层数。",
               "text-vlm": "指令被切分、编码后与图像特征一起进入视觉语言模型。",
               "vlm-head": "视觉语言模型的特征作为条件送入动作头；动作头不直接读取原始像素。",
               "state-head": "机器人状态不经过视觉语言模型，单独编码后直接送入动作头，与视觉语言特征一起决定动作。",
               "head-out": "在这种设计中，动作头每一步给出 14 个数，每个数是一个舵机的目标角度；一次调用生成未来多步，每步都是这样 14 个数。"})
        for g, ((a, b), label, joint_names) in enumerate(zip(ranges, groups, names)):
            parts[f"out-{g}"] = f"{label}：第 {a}–{b} 个输出，{b - a + 1} 个舵机目标，依次是{joint_names}。"
    else:
        groups = ["Left leg", "Neck and head", "Right leg"]
        parts = dict(
            img="RGB images: views from the onboard camera, possibly including past frames. From them the model has to work out where the ball, the goal and the surroundings are.",
            text="Instruction: one sentence describing the task, such as “walk to the football and kick it into the goal with the right foot”. It enters the vision-language model together with the images.",
            state="Robot state: joint angles and velocities, body angular velocity and gravity direction, and the last action sent. With direct joint output, the model relies on it to judge whether the body is balanced right now.",
            vlm="Vision-language model: encodes the images and the instruction into features, for example Cosmos-Reason2-2B in GR00T. It does not output joint actions itself.",
            head="Action head: combines the vision-language features with the robot state and uses flow matching to turn noise step by step into a sequence of actions. With direct joint control, every step of that sequence is 14 servo targets.",
            **{"img-vlm": "The images are encoded into features and enter the vision-language model. The lines show where data flows, not the actual connection pattern or the number of layers.",
               "text-vlm": "The instruction is split into tokens, encoded and enters the vision-language model together with the image features.",
               "vlm-head": "The vision-language features condition the action head; the action head does not read the raw pixels.",
               "state-head": "The robot state bypasses the vision-language model: it is encoded separately and goes straight into the action head, where it decides the action together with the vision-language features.",
               "head-out": "In this design the action head gives 14 numbers per step, each the target angle of one servo; one call generates several future steps, each made of these 14 numbers."})
        for g, ((a, b), label, joint_names) in enumerate(zip(ranges, groups, names)):
            parts[f"out-{g}"] = f"{label}: outputs {a}–{b}, {b - a + 1} servo targets, in order: {joint_names}."
    return parts
