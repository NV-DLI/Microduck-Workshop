"""Generate accessible SVG diagrams with explicit ports and orthogonal connectors."""

from html import escape as e
import hashlib
import unicodedata


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


def platform_svg():
    return """<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg" viewBox="0 0 760 384" role="img" aria-labelledby="platform-title platform-desc"><title id="platform-title">Isaac Lab、Newton 与 Isaac Sim 的职责</title><desc id="platform-desc">Isaac Lab 定义机器人学习任务，Newton 在 GPU 上推进物理状态。需要图像或交互界面时，由 Isaac Sim 提供渲染和传感器能力。Newton 物理执行不必依赖 Kit。</desc><defs><marker id="platform-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0 0L6 3.5L0 7" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.3"/></marker></defs>
<rect x="16" y="16" width="728" height="352" rx="10" fill="var(--nve-sys-layer-container-background)" stroke="var(--nve-sys-layer-container-accent-background)" stroke-dasharray="4 4"/>
<text x="36" y="42" fill="var(--nve-sys-text-muted-color)" font-size="13">任务接口、物理后端与渲染能力分工</text>
<path marker-start="url(#platform-arrow)" d="M310 150V176Q310 184 302 184H180Q172 184 172 192V226" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<path d="M588 232V192Q588 184 580 184H458Q450 184 450 176V156" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<text x="176" y="174" fill="var(--nve-sys-text-muted-color)" font-size="13">动作 ⇄ 物理状态</text><text x="488" y="174" fill="var(--nve-sys-text-muted-color)" font-size="13">按需接入图像与界面</text>
<rect x="240" y="62" width="280" height="88" rx="7" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-text-color)"/><text x="380" y="95" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="20" font-weight="600">Isaac Lab</text><text x="380" y="124" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">任务 · 观测 · 奖励 · 并行环境</text>
<rect x="40" y="232" width="264" height="94" rx="7" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-support-success-color)"/><text x="172" y="264" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="19" font-weight="600">Newton</text><text x="172" y="292" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">Warp / CUDA · 动力学与接触</text><text x="172" y="348" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="13">物理执行可不依赖 Kit</text>
<rect x="456" y="232" width="264" height="94" rx="7" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-text-muted-color)"/><text x="588" y="264" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="19" font-weight="600">Isaac Sim</text><text x="588" y="292" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">OpenUSD · RTX · 相机与交互</text><text x="588" y="348" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="13">本课程可视化任务使用此路径</text></svg>"""


def network_svg(lang):
    """Actual actor widths, square-root-compressed heights and sampled neurons."""
    zh = lang == "zh"
    names = (["观测输入", "隐藏层 1", "隐藏层 2", "隐藏层 3", "动作输出"]
             if zh else ["Observation", "Hidden 1", "Hidden 2", "Hidden 3", "Action"])
    widths = [61, 512, 256, 128, 14]
    xs = [70, 225, 380, 535, 690]
    counts = [5, 11, 9, 7, 3]
    heights = [280 * (width / 512) ** .5 for width in widths]
    center = 230
    # The center slot is an ellipsis; all remaining slots are representative units.
    positions = [[center - height / 2 + j * height / (count - 1)
                  for j in range(count) if j != count // 2]
                 for height, count in zip(heights, counts)]
    title = "行走策略网络：61 → 512 → 256 → 128 → 14" if zh else "Walking actor: 61 → 512 → 256 → 128 → 14"
    description = ("层高按维度的平方根缩放，圆点仅示意部分单元。数字为真实维度。这是底层行走策略，不是 GR00T 的网络结构。"
                   if zh else "Layer height scales with the square root of width; circles show selected units. Numbers are exact. This is the walking actor, not the GR00T architecture.")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" class="network-svg" viewBox="0 0 760 470" role="img" aria-labelledby="actor-network-title actor-network-desc" font-family="Open Sans, Noto Sans SC, sans-serif"><title id="actor-network-title">{e(title)}</title><desc id="actor-network-desc">{e(description)}</desc>']
    for i in range(4):
        for y1 in positions[i]:
            for y2 in positions[i + 1]:
                out.append(f'<path d="M{xs[i]+7} {y1:.2f}L{xs[i+1]-7} {y2:.2f}" stroke="var(--nve-sys-text-muted-color)" stroke-opacity=".17" fill="none"/>')
    for i, (x, width, name, height) in enumerate(zip(xs, widths, names, heights)):
        out.append(f'<g class="network-layer" data-dimension="{width}" data-span="{height:.3f}"><text x="{x}" y="26" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="15">{name}</text><text x="{x}" y="56" text-anchor="middle" fill="var(--nve-sys-text-color)" font-size="23" font-weight="600">{width}</text>')
        for y in positions[i]:
            fill = "var(--nve-sys-layer-container-accent-background)" if i in (0, 4) else "var(--nve-sys-layer-container-accent-background)"
            out.append(f'<circle cx="{x}" cy="{y:.2f}" r="7" stroke="var(--nve-sys-text-muted-color)" fill="{fill}" stroke-width="1.5"/>')
        out.append(f'<rect x="{x-10}" y="{center-13}" width="20" height="26" fill="var(--nve-sys-layer-container-background)"/><text x="{x}" y="{center+6}" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="20">⋮</text>')
        foot = ("输入归一化" if zh else "Normalize") if i == 0 else ("ELU" if i < 4 else ("关节动作" if zh else "Joint actions"))
        out.append(f'<text x="{x}" y="408" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="14">{foot}</text></g>')
    notes = (["层高 ∝ √维度 · 圆点为抽样示意，数字为实际维度", "底层行走策略 · 每 20 ms 推理一次 · 不输入相机图像"]
             if zh else ["Height ∝ √width · sampled circles, exact dimension labels", "Walking actor · inference every 20 ms · no camera images"])
    for y, text in zip([438, 460], notes):
        out.append(f'<text x="380" y="{y}" text-anchor="middle" fill="var(--nve-sys-text-muted-color)" font-size="13">{e(text)}</text>')
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
