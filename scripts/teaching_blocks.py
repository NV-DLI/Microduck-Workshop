"""Accessible, local course diagrams, mathematical notation and seekable media."""

from html import escape as e
from course_diagrams import flow_svg, platform_svg, network_svg
from navigation_diagrams import navigation_diagram
from interactive_flows import learning_flow


def render_teaching_block(b, lang):
    k = b["type"]
    zh = lang == "zh"
    caption = b.get(lang, "")
    if k == "demonstration":
        labels = (
            ["前进请求", "转向请求", "视线俯仰（rad）", "视线权重（0–1）"]
            if zh
            else [
                "Forward request",
                "Yaw request",
                "Gaze pitch (rad)",
                "Gaze weight (0–1)",
            ]
        )
        prefix = "../assets/media/" + e(b["file"])
        return (
            '<figure class="media demonstration" data-demonstration="'
            + prefix
            + '.json"><video controls playsinline preload="metadata" poster="'
            + prefix
            + '.jpg" aria-label="'
            + ("训练示范的相机画面" if zh else "Demonstration camera recording")
            + '"><source src="'
            + prefix
            + '.mp4" type="video/mp4"></video>'
            + '<p class="demonstration-time">'
            + ("加载样本…" if zh else "Loading samples…")
            + '</p><div class="table-scroll"><table><thead><tr>'
            + "".join("<th>" + label + "</th>" for label in labels)
            + "</tr></thead><tbody><tr>"
            + '<td class="demonstration-action">-</td>' * 4
            + "</tr></tbody></table></div><details><summary>"
            + (
                "查看当前样本的身体状态与动作"
                if zh
                else "Inspect this sample’s body state and action"
            )
            + '</summary><pre class="demonstration-sample"></pre></details><figcaption>'
            + caption
            + "</figcaption></figure>"
        )
    if k == "network":
        return (
            '<figure class="concept-figure network-figure"><div class="diagram-scroll">'
            + network_svg(lang)
            + "</div><figcaption>"
            + caption
            + "</figcaption></figure>"
        )
    if k == "equation":
        return (
            '<div class="formula"><div class="formula-label">'
            + e(b.get("label_" + lang, b.get("label", "")))
            + "</div><div>\\["
            + e(b["tex"])
            + "\\]</div></div>"
        )
    if k == "sequence":
        cards = "".join(
            '<div><a class="still-link" href="../assets/'
            + e(x["file"])
            + '"><img loading="lazy" src="../assets/'
            + e(x["file"])
            + '" alt="'
            + e(x["label"])
            + '"></a><p>'
            + e(x["label"])
            + "</p></div>"
            for x in b["items"]
        )
        return (
            '<figure class="media"><div class="pose-sequence" aria-label="'
            + e(b["label"])
            + '">'
            + cards
            + "</div><figcaption>"
            + caption
            + "</figcaption></figure>"
        )
    if k == "image":
        return (
            '<figure class="media"><a class="still-link" href="../assets/'
            + e(b["file"])
            + '" aria-label="'
            + ("放大图片" if zh else "Enlarge image")
            + '"><img loading="lazy" src="../assets/'
            + e(b["file"])
            + '" alt="'
            + e(b.get("alt", {}).get(lang, caption))
            + '"></a><figcaption>'
            + caption
            + "</figcaption></figure>"
        )
    if k == "media":
        f = b["file"]
        if "exploded-cycle" in f:
            return ""
        if f == "detections.gif":
            return render_teaching_block(
                dict(
                    type="image",
                    file="media/detection-contact-hd.webp",
                    alt={
                        "zh": "固定检测器在八张图像上的检测框与置信度",
                        "en": "Fixed detector boxes and scores on eight separate images",
                    },
                    zh="固定检测器在八张保留测试图像上的实际输出，框旁标有类别与置信度。场地背景不同，是因为采集时更换了地面样式，帮助检验检测器能否在背景变化后仍找到球和球门；这些格子不是同一次射门的连续画面。点击图片可放大，对比远近、视角和目标大小对检测结果的影响；这组示例不用于估计整体准确率。",
                    en="Eight actual fixed-detector outputs on held-out test images, with class labels and confidence scores. Ground styles vary to check recognition under different backgrounds; these are not consecutive frames from one attempt. Click to enlarge. These examples do not estimate overall accuracy.",
                ),
                lang,
            )
        name = f.removesuffix(".gif") + "-hd"
        return (
            '<figure class="media"><video controls playsinline preload="metadata" poster="../assets/media/'
            + name
            + '.jpg" aria-label="'
            + e(b["alt"][lang])
            + '"><source src="../assets/media/'
            + name
            + '.mp4" type="video/mp4"></video><figcaption>'
            + caption
            + "</figcaption></figure>"
        )
    if k == "platform":
        return (
            '<figure class="concept-figure"><p class="diagram-scroll-hint">'
            + (
                "左右滑动查看完整图示"
                if zh
                else "Scroll horizontally to view the full diagram"
            )
            + '</p><div class="diagram-scroll">'
            + platform_svg()
            + '</div><figcaption class="diagram-caption">'
            + caption
            + "</figcaption></figure>"
        )
    if k == "flow" and b.get("interactive"):
        return learning_flow(b, lang)
    if k == "flow":
        return (
            '<figure class="concept-figure"><p class="diagram-scroll-hint">'
            + (
                "左右滑动查看完整图示"
                if zh
                else "Scroll horizontally to view the full diagram"
            )
            + '</p><div class="diagram-scroll">'
            + flow_svg({**b, **(b.get("english_diagram", {}) if not zh else {})})
            + '</div><figcaption class="diagram-caption">'
            + caption
            + "</figcaption></figure>"
        )
    if k == "diagram":
        name = b["kind"]
        if name in ("navigation-stance", "navigation-costmap"):
            return navigation_diagram(name, lang, caption)
        controls = ""
        svg = ""
        start = (
            '<svg viewBox="0 0 640 320" role="img" aria-label="'
            + e(b["label"])
            + '" font-family="Open Sans, Noto Sans SC, sans-serif" font-size="15"><defs><marker id="arrow-'
            + name
            + '" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="var(--nve-sys-text-muted-color)"/></marker></defs>'
        )
        if name == "ppo":
            svg = (
                start
                + """<rect x="233" y="30" width="164" height="230" fill="var(--nve-sys-layer-container-accent-background)" class="clip-band"/><path d="M64 30V260H574" stroke="var(--nve-sys-text-muted-color)" fill="none"/><path class="ppo-original" d="" stroke="var(--nve-sys-text-muted-color)" stroke-width="2" stroke-dasharray="5 5" fill="none"/><path class="ppo-clipped" d="" stroke="var(--nve-sys-support-success-color)" stroke-width="3" fill="none"/><circle class="ppo-point" r="6" fill="var(--nve-sys-text-muted-color)" stroke="white" stroke-width="2"/><text x="65" y="18">单个样本的代理目标</text><text x="58" y="281">0.5</text><text x="312" y="281">1.0</text><text x="565" y="281">1.5</text><text x="257" y="308">新旧动作概率之比 ρ</text><text class="ppo-low" x="205" y="50" fill="var(--nve-sys-text-muted-color)">0.8</text><text class="ppo-high" x="404" y="50" fill="var(--nve-sys-text-muted-color)">1.2</text><text class="ppo-ymax" x="54" y="38" text-anchor="end">1.6</text><text class="ppo-ymin" x="54" y="263" text-anchor="end">0.4</text></svg>"""
            )
            controls = '<div class="concept-controls"><label>优势 <select class="advantage"><option value="1">A = +1（好于预期）</option><option value="-1">A = −1（差于预期）</option></select></label><label>概率比 ρ <input class="prob-ratio" type="range" min="0.5" max="1.5" step="0.01" value="1.3"></label><label>裁剪范围 ε <input class="clip-epsilon" type="range" min="0.05" max="0.4" step="0.05" value="0.2"></label></div><p class="concept-output" aria-live="polite"></p>'
        elif name == "timing":
            svg = (
                start
                + """<text x="36" y="43">策略：每 20 ms 输出一次动作（50 Hz）</text><rect x="60" y="70" width="520" height="52" rx="5" fill="var(--nve-sys-layer-container-accent-background)" stroke="var(--nve-sys-text-muted-color)"/><text x="270" y="102">aₜ：关节目标</text><path d="M320 123V169" stroke="var(--nve-sys-text-muted-color)" marker-end="url(#arrow-timing)"/><text x="36" y="156">物理：每 5 ms 推进一步（200 Hz）</text>"""
                + "".join(
                    f'<rect x="{60+i*130}" y="182" width="124" height="48" rx="4" fill="var(--nve-sys-layer-container-background)" stroke="var(--nve-sys-text-muted-color)"/><text x="{85+i*130}" y="213">第 {i+1} 步</text><text x="{60+i*130}" y="255">{i*5} ms</text>'
                    for i in range(4)
                )
                + """<text x="550" y="255">20 ms</text><text x="90" y="298">动作保持不变；位置、速度与接触力仍在逐步变化</text></svg>"""
            )
        elif name == "projection":
            svg = (
                start
                + """<path d="M45 265H610" stroke="var(--nve-sys-text-muted-color)" stroke-width="2"/><text x="495" y="292">地面 z = 0</text><circle cx="102" cy="65" r="8" fill="var(--nve-sys-support-success-color)"/><text x="28" y="40">相机原点 o</text><path d="M102 65L475 265" stroke="var(--nve-sys-support-success-color)" stroke-width="2" marker-end="url(#arrow-projection)"/><path d="M205 65V210" stroke="var(--nve-sys-text-muted-color)" stroke-width="3"/><text x="180" y="237">图像平面</text><circle cx="205" cy="120" r="6" fill="var(--nve-sys-text-muted-color)"/><text x="230" y="105">检测像素 (u, v)</text><circle cx="475" cy="265" r="8" fill="var(--nve-sys-text-muted-color)"/><text x="410" y="235">地面交点 p</text><path d="M102 75V265H460" stroke="var(--nve-sys-support-muted-color)" stroke-dasharray="5 5" fill="none"/><text x="320" y="155">沿射线 d 延伸</text><text x="25" y="285">地面投影</text></svg>"""
            )
        elif name == "stance":
            svg = (
                start
                + """<path d="M60 270H570M60 270V35" stroke="var(--nve-sys-text-muted-color)" fill="none"/><text x="579" y="275">x</text><text x="53" y="25">y</text><path d="M234 196L545 72" stroke="var(--nve-sys-support-success-color)" stroke-width="2" marker-end="url(#arrow-stance)"/><rect x="526" y="37" width="22" height="76" rx="2" stroke="var(--nve-sys-text-muted-color)" fill="none" stroke-width="3"/><text x="550" y="65">球门 g</text><circle cx="353" cy="149" r="12" fill="var(--nve-sys-text-muted-color)"/><text x="363" y="182">足球 b</text><circle class="stance-robot" cx="234" cy="224" r="14" fill="var(--nve-sys-text-muted-color)"/><text class="stance-label" x="117" y="246">出脚位置 p*</text><path class="stance-offset" d="M234 196L234 224" stroke="var(--nve-sys-text-muted-color)" stroke-dasharray="4 4"/><text x="180" y="162">沿射门方向后退 d</text><text x="305" y="110">单位方向 u</text><text x="88" y="303">俯视示意：先留出出脚距离，再调整横向偏移</text></svg>"""
            )
            controls = '<div class="concept-controls"><label>后退距离 d / m <input class="stance-distance" type="range" min="0.12" max="0.5" step="0.01" value="0.26"></label><label>横向偏移 ℓ / m <input class="stance-lateral" type="range" min="-0.14" max="0.14" step="0.01" value="0.04"></label></div><p class="concept-output" aria-live="polite"></p>'
        elif name == "terrain":
            labels = (
                ["平地 0.35 m", "坡道水平长度 1 m", "落脚平台", "坡度 α / °"]
                if zh
                else ["Flat: 0.35 m", "Ramp run: 1 m", "Landing", "Slope α / °"]
            )
            svg = (
                start
                + f"""<path d="M60 165H585" stroke="var(--nve-sys-text-muted-color)" stroke-dasharray="5 5"/><path class="terrain-fill" fill="var(--nve-sys-layer-container-accent-background)" stroke="none"/><path class="terrain-surface" fill="none" stroke="var(--nve-sys-text-muted-color)" stroke-width="4"/><path d="M60 280H165M165 274V286M60 274V286M165 280H465M465 274V286" fill="none" stroke="var(--nve-sys-text-muted-color)"/><text x="110" y="305" text-anchor="middle" font-size="13">{labels[0]}</text><text x="315" y="305" text-anchor="middle">{labels[1]}</text><text class="terrain-landing" x="520" y="100" text-anchor="middle" font-size="13">{labels[2]}</text><circle cx="90" cy="148" r="10" fill="var(--nve-sys-text-muted-color)"/><path d="M85 157L82 165M95 157L98 165" stroke="var(--nve-sys-text-muted-color)" stroke-width="3"/><path class="terrain-height" stroke="var(--nve-sys-text-color)" stroke-width="2"/><text class="terrain-height-label" x="485" y="140" fill="var(--nve-sys-text-color)" font-size="13"/><text x="24" y="28">h(x) = tan(α) · clip(x − 0.35, 0, 1)</text></svg>"""
            )
            controls = (
                '<div class="concept-controls"><label>'
                + labels[3]
                + ' <input class="terrain-angle" type="range" min="-15" max="15" step="1" value="5"></label></div><p class="concept-output" aria-live="polite"></p>'
            )
        else:
            raise ValueError("Unknown diagram " + name)
        return (
            '<figure class="concept-figure" data-diagram="'
            + name
            + '">'
            + svg
            + controls
            + '<figcaption class="diagram-caption">'
            + caption
            + "</figcaption></figure>"
        )
    return None
