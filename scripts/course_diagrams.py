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
        ("#dfeaf5", "#0969da"),
        ("#e6f3d4", "#4a7a00"),
        ("#d6f1ee", "#0d8078"),
        ("#eadfff", "#6f42c1"),
        ("#e6f3d4", "#4a7a00"),
        ("#f7e9cf", "#9a6700"),
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg" viewBox="0 0 760 {height}" role="img" aria-labelledby="{uid}-title {uid}-desc"><title id="{uid}-title">{e(b["label"])}</title><desc id="{uid}-desc">'
        + e("；".join(a + "：" + c for a, c in nodes))
        + "</desc>",
        f'<defs><marker id="{uid}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L6 3.5L0 7" fill="none" stroke="#57606a" stroke-width="1.3"/></marker></defs>',
    ]
    groups = b.get("groups", [])
    if six:
        for y, text in [
            (20, groups[0] if groups else "任务执行"),
            (224, groups[1] if len(groups) > 1 else "结果与反馈"),
        ]:
            out.append(
                f'<rect x="8" y="{y}" width="744" height="176" rx="10" fill="#f6f8fa" stroke="#d0d7de" stroke-dasharray="4 4"/><text x="24" y="{y+25}" fill="#57606a" font-size="13">{e(text)}</text>'
            )
    else:
        out.append(
            f'<text x="24" y="32" fill="#57606a" font-size="13">{e(b["label"])}</text>'
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
            f'<path d="{path}" fill="none" stroke="#57606a" stroke-width="1.5" stroke-linejoin="round" marker-end="url(#{uid}-arrow)"/>'
        )
        if i < len(labels) and labels[i]:
            out.append(
                f'<text x="{lx}" y="{ly}" text-anchor="middle" fill="#57606a" font-size="12">{e(labels[i])}</text>'
            )
    if loop and not six:
        out.append(
            f'<path d="M636 176V214Q636 222 628 222H132Q124 222 124 214V182" fill="none" stroke="#57606a" stroke-width="1.5" marker-end="url(#{uid}-arrow)"/><rect x="290" y="209" width="180" height="24" fill="white"/><text x="380" y="226" text-anchor="middle" fill="#57606a" font-size="13">{e(b.get("loop_label","下一次观测"))}</text>'
        )
    for i, ((title, body), (x, y)) in enumerate(zip(nodes, positions)):
        fill, stroke = colors[i]
        out.append(
            f'<rect x="{x}" y="{y}" width="200" height="112" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )
        for j, t in enumerate(lines(title, 23)):
            out.append(
                f'<text x="{x+100}" y="{y+29+j*21}" text-anchor="middle" fill="#24292f" font-size="16" font-weight="600">{e(t)}</text>'
            )
        for j, t in enumerate(sum((lines(part, 26) for part in body.split("，")), [])):
            out.append(
                f'<text x="{x+100}" y="{y+66+j*20}" text-anchor="middle" fill="#424a53" font-size="13">{e(t)}</text>'
            )
    return "".join(out) + "</svg>"


def platform_svg():
    return """<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg" viewBox="0 0 760 384" role="img" aria-labelledby="platform-title platform-desc"><title id="platform-title">Isaac Lab、Newton 与 Isaac Sim 的职责</title><desc id="platform-desc">Isaac Lab 定义机器人学习任务，Newton 在 GPU 上推进物理状态。需要图像或交互界面时，由 Isaac Sim 提供渲染和传感器能力。Newton 物理执行不必依赖 Kit。</desc><defs><marker id="platform-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0 0L6 3.5L0 7" fill="none" stroke="#57606a" stroke-width="1.3"/></marker></defs>
<rect x="16" y="16" width="728" height="352" rx="10" fill="#f6f8fa" stroke="#d0d7de" stroke-dasharray="4 4"/>
<text x="36" y="42" fill="#57606a" font-size="13">任务接口、物理后端与渲染能力分工</text>
<path marker-start="url(#platform-arrow)" d="M310 150V176Q310 184 302 184H180Q172 184 172 192V226" fill="none" stroke="#57606a" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<path d="M588 232V192Q588 184 580 184H458Q450 184 450 176V156" fill="none" stroke="#57606a" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<text x="176" y="174" fill="#57606a" font-size="13">动作 ⇄ 物理状态</text><text x="488" y="174" fill="#57606a" font-size="13">按需接入图像与界面</text>
<rect x="240" y="62" width="280" height="88" rx="7" fill="#e6f3d4" stroke="#4a7a00"/><text x="380" y="95" text-anchor="middle" fill="#24292f" font-size="20" font-weight="600">Isaac Lab</text><text x="380" y="124" text-anchor="middle" fill="#424a53" font-size="14">任务 · 观测 · 奖励 · 并行环境</text>
<rect x="40" y="232" width="264" height="94" rx="7" fill="#d6f1ee" stroke="#0d8078"/><text x="172" y="264" text-anchor="middle" fill="#24292f" font-size="19" font-weight="600">Newton</text><text x="172" y="292" text-anchor="middle" fill="#424a53" font-size="14">Warp / CUDA · 动力学与接触</text><text x="172" y="348" text-anchor="middle" fill="#57606a" font-size="13">物理执行可不依赖 Kit</text>
<rect x="456" y="232" width="264" height="94" rx="7" fill="#dfeaf5" stroke="#0969da"/><text x="588" y="264" text-anchor="middle" fill="#24292f" font-size="19" font-weight="600">Isaac Sim</text><text x="588" y="292" text-anchor="middle" fill="#424a53" font-size="14">OpenUSD · RTX · 相机与交互</text><text x="588" y="348" text-anchor="middle" fill="#57606a" font-size="13">本课程可视化任务使用此路径</text></svg>"""


def network_svg(lang):
    """Show the actual ONNX actor dimensions with representative neurons."""
    zh = lang == "zh"
    names = (
        ["观测输入", "隐藏层 1", "隐藏层 2", "隐藏层 3", "动作输出"]
        if zh
        else ["Observation", "Hidden 1", "Hidden 2", "Hidden 3", "Action"]
    )
    widths = [61, 512, 256, 128, 14]
    xs = [70, 225, 380, 535, 690]
    ys = [95, 130, 165, 220, 255]
    out = [
        '<svg class="network-svg" viewBox="0 0 760 350" role="img" aria-label="61 → 512 → 256 → 128 → 14; ELU" font-family="Open Sans, Noto Sans SC, sans-serif">'
    ]
    for left, right in zip(xs, xs[1:]):
        for y1 in ys:
            for y2 in ys:
                out.append(
                    f'<path d="M{left+9} {y1}L{right-9} {y2}" stroke="#8ca5b7" stroke-opacity=".24" fill="none"/>'
                )
    for i, (x, width, name) in enumerate(zip(xs, widths, names)):
        out.append(
            f'<text x="{x}" y="26" text-anchor="middle" fill="#405367" font-size="15">{name}</text><text x="{x}" y="56" text-anchor="middle" fill="#172d40" font-size="23" font-weight="600">{width}</text>'
        )
        for y in ys:
            out.append(
                f'<circle cx="{x}" cy="{y}" r="9" stroke="#366c91" fill="{("#dcebd1" if i in (0,4) else "#e0edf8")}" stroke-width="1.5"/>'
            )
        out.append(
            f'<text x="{x}" y="198" text-anchor="middle" fill="#526779" font-size="20">⋮</text>'
        )
        foot = (
            ("输入归一化" if zh else "Normalize")
            if i == 0
            else ("ELU" if i < 4 else ("关节动作" if zh else "Joint actions"))
        )
        out.append(
            f'<text x="{x}" y="296" text-anchor="middle" fill="#405367" font-size="14">{foot}</text>'
        )
    out.append(
        '<text x="380" y="335" text-anchor="middle" fill="#526779" font-size="13">'
        + (
            "全连接层 · 每 20 ms 推理一次 · 不输入相机图像"
            if zh
            else "Fully connected · inference every 20 ms · no camera images"
        )
        + "</text></svg>"
    )
    return "".join(out)


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
        f'<defs><marker id="{uid}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L6 3.5L0 7" fill="none" stroke="#57606a" stroke-width="1.3"/></marker></defs>',
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
            f'<path d="{path}" fill="none" stroke="#57606a" stroke-width="1.5" marker-end="url(#{uid}-arrow)"/>'
        )
    colors = [
        ("#dfeaf5", "#0969da"),
        ("#e6f3d4", "#4a7a00"),
        ("#d6f1ee", "#0d8078"),
        ("#eadfff", "#6f42c1"),
    ]
    for i, ((title, body), (x, y)) in enumerate(zip(wrapped, positions)):
        fill, stroke = colors[i % len(colors)]
        out.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{card_height}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )
        for j, line in enumerate(title):
            out.append(
                f'<text x="{x + width / 2}" y="{y + 28 + j * 22}" text-anchor="middle" fill="#24292f" font-size="16" font-weight="600">{e(line)}</text>'
            )
        for j, line in enumerate(body):
            out.append(
                f'<text x="{x + width / 2}" y="{y + 42 + len(title) * 22 + j * 21}" text-anchor="middle" fill="#424a53" font-size="14">{e(line)}</text>'
            )
    return "".join(out) + "</svg>"
