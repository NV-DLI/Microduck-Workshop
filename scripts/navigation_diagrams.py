"""Bilingual navigation teaching diagrams; illustrative geometry, not simulation."""

from html import escape


def navigation_diagram(kind, lang, caption):
    zh = lang == "zh"
    label = (
        (
            "球后站位的交互几何图"
            if zh
            else "Interactive geometry of a stance behind the ball"
        )
        if kind == "navigation-stance"
        else ("代价地图与路径示意" if zh else "Illustrative costmap and path")
    )
    prefix = (
        f'<figure class="concept-figure navigation-figure" data-diagram="{kind}">'
        '<p class="diagram-scroll-hint">'
        + (
            "左右滑动查看完整图示"
            if zh
            else "Scroll horizontally to view the full diagram"
        )
        + '</p><div class="diagram-scroll"><svg class="navigation-svg" viewBox="0 0 720 430" '
        f'role="img" aria-label="{label}"><defs><marker id="{kind}-arrow" '
        'markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0 0L7 4L0 8" fill="none" stroke="var(--nve-sys-text-muted-color)"/></marker></defs>'
    )
    controls = ""
    if kind == "navigation-stance":
        svg = """<rect x="88" y="40" width="544" height="340" rx="5" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-text-muted-color)"/>
<path d="M88 210H632M360 40V380" stroke="var(--nve-sys-support-muted-color)" stroke-dasharray="4 5"/>
<path d="M112 350H172M112 350V290" fill="none" stroke="var(--nve-sys-text-muted-color)"/><text x="181" y="355">+x</text><text x="102" y="279">+y</text>
<path class="nav-shot" stroke="var(--nve-sys-text-color)" stroke-width="2" fill="none"/>
<path class="nav-backoff" stroke="var(--nve-sys-text-muted-color)" stroke-width="3" fill="none"/>
<path class="nav-lateral" stroke="var(--nve-sys-text-muted-color)" stroke-width="3" stroke-dasharray="4 3" fill="none"/>
<path class="nav-heading" stroke="var(--nve-sys-text-muted-color)" stroke-width="2" fill="none" marker-end="url(#navigation-stance-arrow)"/>
<path class="nav-goal-frame" stroke="var(--nve-sys-text-muted-color)" stroke-width="4" fill="none"/>
<circle class="nav-ball" r="9" fill="var(--nve-sys-text-muted-color)" stroke="white" stroke-width="2"/>
<circle class="nav-stance" r="10" fill="var(--nve-sys-text-muted-color)" stroke="white" stroke-width="2"/>
<circle class="nav-goal" r="4" fill="var(--nve-sys-text-muted-color)"/>
<text class="nav-ball-label">B</text><text class="nav-stance-label">P*</text><text class="nav-goal-label">G</text>
<text class="nav-distance-label" fill="var(--nve-sys-text-muted-color)">d</text>
<text class="nav-offset-label" fill="var(--nve-sys-text-muted-color)">ℓ</text><text class="nav-direction-label" fill="var(--nve-sys-text-color)">u</text><text class="nav-heading-label" fill="var(--nve-sys-text-muted-color)">θ*</text>
<text x="360" y="414" text-anchor="middle">1.6 m × 1.0 m</text>"""
        specs = [
            (
                "nav-distance",
                "准备距离 d / m",
                "Stand-off d / m",
                ".18",
                ".40",
                ".01",
                ".22",
            ),
            (
                "nav-offset",
                "左向偏移 ℓ / m",
                "Left offset ℓ / m",
                "-.08",
                ".08",
                ".005",
                ".035",
            ),
            ("nav-ball-y", "足球 B 的 y / m", "Ball B: y / m", "-.2", ".2", ".01", "0"),
            ("nav-goal-y", "球门 G 的 y / m", "Goal G: y / m", "-.2", ".2", ".01", "0"),
        ]
        controls = (
            '<div class="concept-controls navigation-controls">'
            + "".join(
                f'<label>{cn if zh else en}<input class="{cls}" type="range" min="{lo}" max="{hi}" step="{step}" value="{value}"></label>'
                for cls, cn, en, lo, hi, step, value in specs
            )
            + '</div><p class="concept-output" aria-live="polite"></p>'
        )
        legend = (
            "蓝点 P*：目标站位；橙点 B：足球；灰点 G：球门中心。棕线表示后退距离 d，蓝虚线表示横向偏移 ℓ，蓝箭头表示目标朝向。"
            if zh
            else "Blue P*: target stance; orange B: ball; gray G: goal center. Brown: stand-off d; dashed blue: lateral offset ℓ; blue arrow: target heading."
        )
    else:
        svg = '<rect x="108" y="64" width="504" height="280" fill="var(--nve-sys-layer-container-background)"/>'
        svg += '<rect x="556" y="64" width="56" height="280" fill="var(--nve-sys-support-muted-color)"/>'
        svg += '<rect x="304" y="148" width="84" height="112" fill="var(--nve-sys-layer-container-accent-background)"/>'
        svg += '<rect x="332" y="176" width="28" height="56" fill="var(--nve-sys-text-muted-color)"/>'
        svg += "".join(
            f'<path d="M{108+i*28} 64V344" stroke="var(--nve-sys-layer-container-accent-background)"/>' for i in range(19)
        )
        svg += "".join(
            f'<path d="M108 {64+i*28}H612" stroke="var(--nve-sys-layer-container-accent-background)"/>' for i in range(11)
        )
        svg += """<path d="M178 246H262L290 134H402L486 190" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="4"/>
<circle cx="178" cy="246" r="23" fill="var(--nve-sys-text-muted-color)" fill-opacity=".12" stroke="var(--nve-sys-text-muted-color)" stroke-dasharray="4 3"/>
<circle cx="178" cy="246" r="6" fill="var(--nve-sys-text-muted-color)"/><circle cx="486" cy="190" r="7" fill="var(--nve-sys-text-color)"/>
<text x="167" y="287">R</text><text x="493" y="177">P*</text>"""
        svg += f'<text x="360" y="34" text-anchor="middle">{("教学示意：假设存在一个障碍" if zh else "Teaching example: a hypothetical obstacle")}</text>'
        svg += f'<text x="360" y="394" text-anchor="middle">{("蓝线：规划路径；虚线圆：机器人外形的近似" if zh else "Blue: planned path; dashed circle: approximate robot footprint")}</text>'
        legend = (
            "白色：已知空地；深灰：障碍；浅棕：靠近障碍的较高代价区域；灰色：未知区域。R 为起点，P* 为目标。"
            if zh
            else "White: known free space; dark gray: obstacle; tan: higher cost near the obstacle; gray: unknown. R is the start and P* the goal."
        )
    return (
        prefix
        + svg
        + '</svg></div><p class="navigation-legend">'
        + escape(legend)
        + "</p>"
        + controls
        + '<figcaption class="diagram-caption">'
        + caption
        + "</figcaption></figure>"
    )
