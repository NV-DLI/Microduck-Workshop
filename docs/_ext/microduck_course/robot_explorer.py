"""``robot-explorer``: the illustrated "Inside the little duck" 3D explorer.

A cel-shaded, outlined view of the original Microduck CAD in three beats (the whole
duck, an exploded view, how motors drive the joints). It is the lesson's only anatomy
figure and independent of the archived ``robot-anatomy`` viewer: its own markup (``dx-`` classes), stylesheet
(``_static/robot-explorer.css``), loader (``_static/robot-explorer-loader.js``) and module
(``web/assets/robot-explorer.js``). Both assets are added only to pages that use it.

Browsers may keep an earlier module or layout file while the page itself is new, so the page
names exact versions: one content fingerprint over the module, its two data files, the stylesheet
and the loader (``data-dx-version``), and a separate one for the large model (``data-dx-model-version``)
so that it is downloaded again only when it changes. The loader imports the module with that
version and the module adds it to every file it fetches.
"""

from __future__ import annotations

import hashlib
from html import escape as e
from pathlib import Path

from sphinx.util.docutils import SphinxDirective
from docutils import nodes

TEXT = {
    "zh": dict(
        chapters=("整只小鸭", "拆解图", "电机怎样驱动关节"),
        views="选择观察方式", reset="从头看", back="返回",
        retry="重新加载", loading="正在加载 3D 小鸭",
        regions="选择部位", joints="选择动作", angle="关节角", controls="关节控制",
        legend=("转动舵机", "跟随部分", "驱动方向"),
        error="3D 小鸭未能加载，请检查网络后重试；下方说明仍可阅读。",
        nojs="3D 小鸭需要启用 JavaScript；下方说明仍可阅读。",
    ),
    "en": dict(
        chapters=("The whole duck", "Exploded view", "How motors drive joints"),
        views="Choose a view", reset="Start over", back="Back",
        retry="Try again", loading="Loading the 3D duck",
        regions="Choose a part", joints="Choose a motion", angle="Joint angle", controls="Joint controls",
        legend=("Rotating servo", "Following parts", "Drive direction"),
        error="The 3D duck did not load. Check the connection and try again; the notes below remain available.",
        nojs="The 3D duck needs JavaScript; the notes below remain available.",
    ),
}

DUCK_ICON = ('<svg class="dx-open-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
             'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">'
             '<path d="M12 2.8 4.2 7.2v9.6L12 21.2l7.8-4.4V7.2L12 2.8Z"/><path d="M4.2 7.2 12 11.6l7.8-4.4M12 11.6v9.6"/></svg>')
# A curved paper ribbon with an arrowhead at each end, like the drive-direction arrow in the 3D view.
RIBBON_PATH = ("M4.47 11.54L3.54 6.90L4.75 7.71L5.30 6.98L5.92 6.31L6.61 5.70L7.36 5.18L8.16 4.74L9.01 4.38L9.88 4.11"
               "L10.78 3.94L11.69 3.86L12.61 3.87L13.52 3.98L14.41 4.19L15.28 4.49L16.11 4.87L16.89 5.35L17.62 5.90"
               "L18.29 6.52L18.90 7.21L19.25 7.71L20.46 6.90L19.53 11.54L16.15 9.80L17.35 8.99L16.94 8.45L16.48 7.96"
               "L15.97 7.52L15.42 7.13L14.83 6.80L14.21 6.54L13.56 6.34L12.90 6.21L12.23 6.15L11.55 6.17L10.88 6.25"
               "L10.22 6.40L9.58 6.62L8.97 6.90L8.39 7.25L7.85 7.66L7.36 8.12L6.92 8.63L6.65 8.99L7.85 9.80Z")
ARC_ICON = ('<svg class="dx-legend-arc" viewBox="0 0 24 14" width="24" height="14" aria-hidden="true" focusable="false">'
            f'<path d="{RIBBON_PATH}"/></svg>')


STATIC = Path(__file__).resolve().parents[2] / "_static"
# Everything the module and its stylesheet must agree on; versioned together.
BUNDLE = ("robot-explorer.js", "robot/explorer-layout.json", "robot/components.json")
STYLE = (STATIC / "robot-explorer.css", STATIC / "robot-explorer-loader.js")
MODEL = "robot/microduck.glb"

_digests: dict[tuple, str] = {}


def digest(*paths: Path) -> str:
    """Short SHA-256 over the files' names and contents (memoised per size and mtime)."""
    key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    if key not in _digests:
        h = hashlib.sha256()
        for p in paths:
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
        _digests[key] = h.hexdigest()[:16]
    return _digests[key]


def explorer_html(lang: str, assets_url: str, version: str, model_version: str) -> str:
    t = TEXT[lang]
    chapters = "".join(
        f'<button type="button" class="dx-chapter" aria-pressed="{str(i == 0).lower()}">{e(name)}</button>'
        for i, name in enumerate(t["chapters"])
    )
    legend = "".join(
        f'<li class="dx-legend-{role}">{ARC_ICON if role == "joint" else '<span class="dx-chip" aria-hidden="true"></span>'}{e(text)}</li>'
        for role, text in zip(("actuator", "link", "joint"), t["legend"])
    )
    return (
        '<div class="duck-explorer" data-duck-explorer data-chapter="0" data-state="pending" '
        f'data-dx-module="{assets_url}/{BUNDLE[0]}?v={version}" data-dx-version="{version}" '
        f'data-dx-model-version="{model_version}">'
        f'<div class="dx-chapters" hidden><div class="dx-tabs" role="group" aria-label="{e(t["views"])}">{chapters}</div>'
        f'<button type="button" class="dx-reset">{e(t["reset"])}</button></div>'
        '<div class="dx-body"><div class="dx-stage">'
        '<svg class="dx-leaders" aria-hidden="true"></svg><svg class="dx-gizmo" aria-hidden="true" hidden></svg>'
        '<div class="dx-callouts"></div>'
        '<div class="dx-handle" role="slider" tabindex="0" aria-orientation="horizontal" hidden></div>'
        f'<button type="button" class="dx-back" hidden>{e(t["back"])}</button>'
        # The stage stays empty and transparent until the 3D duck is ready: the loader starts at
        # once, and loading is only announced off screen. After a failure, the status message and
        # one "try again" button appear; no still image stands in for the model.
        '<div class="dx-launch" hidden>'
        f'<button type="button" class="dx-open">{DUCK_ICON}<span class="dx-open-label">{e(t["retry"])}</span></button></div>'
        f'<p class="dx-status" role="status" aria-live="polite" data-loading="{e(t["loading"])}…" data-error="{e(t["error"])}"></p>'
        f'<noscript><p class="dx-status">{e(t["nojs"])}</p></noscript></div>'
        '<div class="dx-panel" hidden><div class="dx-card" aria-live="polite">'
        '<p class="dx-title"></p><p class="dx-fact"></p><p class="dx-part" hidden></p></div>'
        f'<div class="dx-drive" role="group" aria-label="{e(t["controls"])}" hidden>'
        f'<ul class="dx-legend">{legend}</ul>'
        f'<div class="dx-field"><p class="dx-field-label">{e(t["regions"])}</p>'
        f'<div class="dx-regions" role="group" aria-label="{e(t["regions"])}"></div></div>'
        f'<div class="dx-field"><p class="dx-field-label">{e(t["joints"])}</p>'
        f'<div class="dx-joints" role="group" aria-label="{e(t["joints"])}"></div><p class="dx-fixed" hidden></p></div>'
        f'<div class="dx-readout"><p class="dx-field-label">{e(t["angle"])} <code class="dx-q-id"></code></p>'
        '<p class="dx-q"></p><p class="dx-q-detail"></p></div>'
        '<div class="dx-actions"><button type="button" class="dx-play" aria-pressed="false"></button>'
        '<button type="button" class="dx-zero"></button></div>'
        '</div></div></div></div>'
    )


class RobotExplorer(SphinxDirective):
    has_content = True

    def run(self):
        lang = self.config.course_lang
        root = Path(self.config.course_assets_dir) / "robot"
        assets = root.parent
        bundle = [assets / p for p in BUNDLE] + list(STYLE)
        # Re-read this page whenever any versioned input, or this renderer, changes.
        for p in [*bundle, assets / MODEL, Path(__file__).resolve()]:
            self.env.note_dependency(str(p))
        html = explorer_html(lang, self.config.course_assets_url.rstrip("/"), digest(*bundle), digest(assets / MODEL))
        self.env.metadata[self.env.docname]["robot_explorer"] = True
        figure = nodes.container(classes=["duck-explorer-figure"])
        figure += nodes.raw("", html, format="html")
        if self.content:
            caption = nodes.container(classes=["dx-caption"])
            self.state.nested_parse(self.content, self.content_offset, caption)
            figure += caption
        return [figure]


def add_page_assets(app, pagename, templatename, context, doctree):
    if doctree is not None and app.env.metadata.get(pagename, {}).get("robot_explorer"):
        app.add_css_file("robot-explorer.css")
        app.add_js_file("robot-explorer-loader.js", defer="defer")


def setup(app):
    app.add_directive("robot-explorer", RobotExplorer)
    app.connect("html-page-context", add_page_assets)
