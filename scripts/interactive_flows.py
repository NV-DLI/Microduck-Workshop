"""Native learning diagrams: selectable nodes, explicit data paths and explanations.

Interaction design informed by the Apache-2.0 SO-101 Newton workshop;
this renderer and Microduck content are new. See THIRD_PARTY_NOTICES.md.
"""

from html import escape as e
import hashlib
from course_diagrams import lines


def learning_flow(b, lang):
    d = {**b, **(b.get("english_diagram", {}) if lang == "en" else {})}
    nodes = d["nodes"]
    details = d.get("details", [text for _, text in nodes])
    uid = "learn-" + hashlib.sha1((lang + b["label"]).encode()).hexdigest()[:10]
    n = len(nodes)
    architecture = b.get("layout") == "groot"
    if architecture:
        positions = [(24, 52), (300, 52), (576, 52), (576, 280), (300, 280), (24, 280)]
        links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 2), (3, 5), (5, 0)]
    elif n <= 3:
        positions = [(24 + 276 * i, 48) for i in range(n)]
        links = [(i, i + 1) for i in range(n - 1)]
    else:
        positions = [(24, 48), (300, 48), (576, 48), (576, 270), (300, 270), (24, 270)][
            :n
        ]
        links = [(i, i + 1) for i in range(n - 1)]
    w, h = 228, 142
    height = 480 if n > 3 else (276 if b.get("loop") else 228)
    if b.get("loop") and not architecture:
        links.append((n - 1, b.get("loop_to", 0)))
    paths = []
    for a, z in links:
        x, y = positions[a]
        xx, yy = positions[z]
        if architecture and (a, z) == (4, 2):
            path = "M414 280V236H630V200"
        elif architecture and (a, z) == (3, 5):
            path = "M690 422V456H138V428"
        elif (
            (a, z) == (n - 1, b.get("loop_to", 0))
            and b.get("loop")
            and not architecture
        ):
            path = f"M{x+114} {y+h}V{height-22}H{xx-12}V{yy+71}H{xx-6}"
        elif yy == y:
            path = f"M{x+w if xx>x else x} {y+71}H{xx-6 if xx>x else xx+w+6}"
        elif xx == x:
            path = f"M{x+114} {y+h if yy>y else y}V{yy-6 if yy>y else yy+h+6}"
        else:
            path = f"M{x+114} {y+h}V{y+h+38}H{xx+114}V{yy-6}"
        paths.append(
            f'<path class="learning-edge" data-from="{a}" data-to="{z}" d="{path}" marker-end="url(#{uid}-arrow)"/>'
        )
    cards = []
    for i, ((title, subtitle), (x, y)) in enumerate(zip(nodes, positions)):
        title_lines = lines(title, 26)
        body_lines = lines(subtitle, 29)
        if len(title_lines) > 2 or len(body_lines) > 3:
            raise ValueError(f"Diagram text too long: {title!r} / {subtitle!r}")
        texts = "".join(
            f'<text x="{x+114}" y="{y+35+j*23}" class="learning-node-title">{e(t)}</text>'
            for j, t in enumerate(title_lines)
        )
        texts += "".join(
            f'<text x="{x+114}" y="{y+84+j*19}" class="learning-node-subtitle">{e(t)}</text>'
            for j, t in enumerate(body_lines)
        )
        cards.append(
            f'<g class="learning-node" data-node="{i}" role="button" tabindex="0" aria-label="{e(title)}" aria-controls="{uid}-detail-{i}" aria-expanded="false"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>{texts}</g>'
        )
    prompt = (
        "点击图中的模块，查看模块作用与数据流。也可按 Tab 选择，按 Enter 确认。"
        if lang == "zh"
        else "Select a module to explore its role and data flow. You can also use Tab and Enter."
    )
    scroll_hint = (
        "左右滑动查看完整图示"
        if lang == "zh"
        else "Swipe horizontally to see the full diagram"
    )
    reset = "关闭说明" if lang == "zh" else "Close explanation"
    articles = "".join(
        f'<article id="{uid}-detail-{i}" data-node-detail="{i}" class="learning-definition"><h3>{e(nodes[i][0])}</h3><p>{detail}</p></article>'
        for i, detail in enumerate(details)
    )
    return f"""<figure class="learning-figure" data-learning-flow="{uid}"><p class="learning-help">{prompt}<span class="learning-scroll-hint">{scroll_hint}</span></p><div class="diagram-scroll" tabindex="0" role="region" aria-label="{e(d['label'])}"><svg class="learning-svg" viewBox="0 0 828 {height}" role="group" aria-labelledby="{uid}-title"><title id="{uid}-title">{e(d['label'])}</title><defs><marker id="{uid}-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L7 4L0 8" fill="none" stroke="currentColor" stroke-width="1.4"/></marker></defs>{''.join(paths)}{''.join(cards)}</svg></div><div class="learning-details"><p class="learning-empty">{e(d['label'])}</p><button type="button" class="learning-reset" aria-label="{reset}" hidden>×</button>{articles}</div><span class="sr-only learning-status" role="status"></span><figcaption>{b.get(lang, '')}</figcaption></figure>"""
