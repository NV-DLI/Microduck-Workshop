"""Sphinx directives for the bilingual Microduck course.

Lesson prose lives in MyST Markdown (``docs/zh`` and ``docs/en``). These
directives embed the interactive figures (flows, charts, 3D anatomy, video
with stage captions) whose markup the browser scripts in ``web/assets`` expect.
Captions given as directive content are parsed as MyST; structured widget
data is YAML whose strings accept inline Markdown (code, links, bold).
"""

from __future__ import annotations

import json
import re
from html import escape as e
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

from . import robot_explorer, symbols, widgets

logger = logging.getLogger(__name__)

UI = {
    "zh": dict(check="检查理解", answer="查看解析", zoom="放大图片", comparison=("上：官方策略", "下：坡地策略"),
               demo_columns=("前进请求", "转向请求", "视线俯仰（rad）", "视线权重（0–1）")),
    "en": dict(check="Check your understanding", answer="Show explanation", zoom="Enlarge image",
               comparison=("Top: official policy", "Bottom: terrain policy"),
               demo_columns=("Forward request", "Yaw request", "Gaze pitch (rad)", "Gaze weight (0–1)")),
}


class wrapper(nodes.Element):
    """An element whose opening and closing HTML surround parsed MyST children."""


def visit_wrapper(self, node):
    self.body.append(node["open"])


def depart_wrapper(self, node):
    self.body.append(node["close"])


class benchmark_placeholder(nodes.General, nodes.Element):
    """Replaced by a bar chart computed from the table that follows it."""


def _skip(self, node):
    raise nodes.SkipNode


# ---------------------------------------------------------------- helpers


class CourseDirective(SphinxDirective):
    @property
    def lang(self):
        return self.config.course_lang

    @property
    def ui(self):
        return UI[self.lang]

    def asset(self, path: str, optional: bool = False) -> str | None:
        """URL of a shared asset in web/assets, with a build-time existence check.

        An optional asset that is not there yet returns None instead of failing the build;
        the page stays a dependency of that path, so it is rebuilt once the file arrives.
        """
        path = path.strip()
        file = Path(self.config.course_assets_dir) / path
        if optional:
            self.env.note_dependency(str(file))
        if not file.exists():
            if optional:
                logger.info("pending course asset (not shown yet): %s", path, location=self.get_location())
                return None
            logger.warning("course asset not found: %s", path, location=self.get_location())
        return self.config.course_assets_url.rstrip("/") + "/" + path

    def data(self):
        try:
            value = yaml.safe_load("\n".join(self.content))
        except yaml.YAMLError as error:
            raise self.error(f"Invalid YAML in {self.name}: {error}")
        if not isinstance(value, dict):
            raise self.error(f"{self.name} expects a YAML mapping")
        return value

    def raw(self, html):
        return nodes.raw("", html, format="html")

    # Explained symbols (see symbols.py): one table per page, filled by {symbols}.
    def symbol_table(self):
        return self.env.temp_data.setdefault("course_symbols", {})

    def resolve_math(self, tex):
        used = self.env.temp_data.setdefault("course_symbols_used", [])
        start = len(used)
        try:
            out = symbols.resolve(tex, self.symbol_table(), used)
        except symbols.SymbolError as error:
            raise self.error(str(error))
        return out, used[start:]

    def render_widget(self, render, *args):
        """Render widget HTML; `$...$` in its strings may use \\sym marks."""
        widgets.MATH_RESOLVER = lambda tex: self.resolve_math(tex)[0]
        try:
            return render(*args)
        finally:
            widgets.MATH_RESOLVER = None

    def wrap(self, open_html, close_html, parse_content=True):
        node = wrapper(open=open_html, close=close_html)
        if parse_content and self.content:
            node += self.parse_content_to_nodes()
        return node


# ---------------------------------------------------------------- page structure


class LessonIntro(CourseDirective):
    """Lead paragraph, lesson metadata and the optional home hero image.

    ``hero`` is a still; with ``hero-animation`` the page shows only the transparent animated
    image, which autoplays and loops from its own file (also with reduced motion; the course
    asks for this) with no control and no script. The still is then only checked, not emitted.
    ``hero-size`` (``WxH``, default ``1200x500``) gives the intrinsic size; with
    ``hero-optional`` a hero whose files are not there yet is left out instead of failing the
    build.
    """

    has_content = True
    option_spec = {"meta": directives.unchanged, "hero": directives.unchanged, "hero-alt": directives.unchanged,
                   "hero-animation": directives.unchanged, "hero-size": directives.unchanged,
                   "hero-optional": directives.flag}

    def hero_figure(self, hero, alt):
        optional = "hero-optional" in self.options
        still = self.asset(hero, optional)
        animation = self.options.get("hero-animation")
        anim = self.asset(animation, optional) if animation else None
        if still is None or (animation and anim is None):
            return ""
        width, height = (int(v) for v in self.options.get("hero-size", "1200x500").lower().split("x"))
        size = f'width="{width}" height="{height}" style="aspect-ratio: {width} / {height}"'
        if not animation:
            return f'<figure class="hero-image"><img fetchpriority="high" src="{e(still)}" alt="{e(alt)}" {size}></figure>'
        return (f'<figure class="hero-image hero-animation">'
                f'<img class="dark-light" fetchpriority="high" src="{e(anim)}" alt="{e(alt)}" {size}></figure>')

    def run(self):
        hero = self.options.get("hero")
        figure = self.hero_figure(hero, self.options.get("hero-alt", "")) if hero else ""
        home = " is-home" if figure else ""
        if figure and self.options.get("hero-animation"):
            home += " has-hero-animation"
        node = wrapper(open=f'<div class="lesson-heading{home}"><div class="lesson-heading-text">', close="")
        lead = wrapper(open='<div class="lead">', close="</div>")
        lead += self.parse_content_to_nodes()
        node += lead
        tail = "</div>"
        if self.options.get("meta"):
            node += self.raw('<div class="lesson-meta">' + widgets.inline_md(self.options["meta"]) + "</div>")
        node["close"] = tail + figure + "</div>"
        return [node]


class Quiz(CourseDirective):
    has_content = True

    def run(self):
        node = self.wrap(f'<div class="quiz"><h3>{e(self.ui["check"])}</h3>', "</div>")
        return [node]


class Answer(CourseDirective):
    has_content = True

    def run(self):
        return [self.wrap(f'<details><summary>{e(self.ui["answer"])}</summary>', "</details>")]


class Enrichment(CourseDirective):
    """Optional deeper material, collapsed by default."""

    has_content = True
    required_arguments = 1
    final_argument_whitespace = True

    def run(self):
        title = widgets.inline_md(self.arguments[0])
        return [self.wrap(f'<details class="enrichment"><summary>{title}</summary><div class="enrichment-body">', "</div></details>")]


class Formula(CourseDirective):
    """A labelled display equation rendered in the browser by local KaTeX."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True

    def run(self):
        label = self.arguments[0] if self.arguments else ""
        tex, keys = self.resolve_math("\n".join(self.content).strip())
        legend = symbols.legend(keys, self.symbol_table(), self.lang)
        return [self.raw(f'<div class="formula"><div class="formula-label">{e(label)}</div><div class="formula-body">\\[{e(tex)}\\]</div>{legend}</div>')]


class Symbols(CourseDirective):
    """Page symbol table: ``key: {tex: ..., text: ...}`` used by \\sym{key}."""

    has_content = True

    def run(self):
        try:
            table = symbols.parse_table(self.data())
        except symbols.SymbolError as error:
            raise self.error(str(error))
        page = self.symbol_table()
        for key in table:
            if key in page:
                raise self.error(f"symbol key {key!r} defined twice on this page")
        page.update(table)
        return [self.raw(symbols.json_script(table))]


# ---------------------------------------------------------------- media


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a JPEG's start-of-frame marker, or None."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    i = 2
    while i + 9 < len(data) and data[i] == 0xFF:
        marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + length
    return None


class CourseVideo(CourseDirective):
    """Seekable MP4 with poster; optional stage captions and comparison labels.

    A solitary clip (no stages or comparison) no wider than 16:9 gets ``media-solo`` and its
    poster's size, so CSS can inset it from the column edges at its own ratio. Wider clips,
    such as multi-panel sensor recordings, keep the full column width.
    """

    has_content = True
    required_arguments = 1
    option_spec = {"alt": directives.unchanged_required, "stages": directives.unchanged, "comparison": directives.flag}

    def run(self):
        base = self.arguments[0]
        stages = self.asset(self.options["stages"]) if self.options.get("stages") else None
        comparison = self.ui["comparison"] if "comparison" in self.options else None
        html = widgets.video_open(self.asset(base + ".mp4"), self.asset(base + ".jpg"), self.options.get("alt", ""), stages, comparison)
        size = jpeg_size(Path(self.config.course_assets_dir) / (base.strip() + ".jpg"))
        if not stages and not comparison and size and size[0] * 9 <= size[1] * 16:
            html = html.replace('<figure class="media">',
                                f'<figure class="media media-solo" style="--media-w: {size[0]}; --media-h: {size[1]}">', 1)
        return [self.wrap(html, "</figcaption></figure>")]


class CourseImage(CourseDirective):
    """Still image that opens enlarged in a dialog.

    ``prefer`` names a replacement image used once it exists (the argument is shown until
    then); ``if-present`` leaves out an image whose file has not been supplied yet.
    """

    has_content = True
    required_arguments = 1
    option_spec = {"alt": directives.unchanged, "no-zoom-label": directives.flag,
                   "prefer": directives.unchanged, "if-present": directives.flag}

    def run(self):
        label = None if "no-zoom-label" in self.options else self.ui["zoom"]
        src = self.asset(self.options["prefer"], optional=True) if self.options.get("prefer") else None
        src = src or self.asset(self.arguments[0], optional="if-present" in self.options)
        if src is None:
            return []
        html = widgets.image_open(src, self.options.get("alt", ""), label)
        return [self.wrap(html, "</figcaption></figure>")]


class SceneCards(CourseDirective):
    """Grid of labelled stills; ``:cards:`` lines are ``path | label``."""

    has_content = True
    option_spec = {"cards": directives.unchanged_required}

    def run(self):
        cards = []
        for line in self.options["cards"].splitlines():
            if line.strip():
                path, label = (x.strip() for x in line.split("|", 1))
                plain = re.sub(r"^\w+ · ", "", label)
                cards.append((self.asset(path), label, plain))
        return [self.wrap(widgets.scene_cards_open(cards), "</figcaption></figure>")]


class Demonstration(CourseDirective):
    """Demonstration video synchronized with its recorded action samples."""

    has_content = True
    required_arguments = 1

    def run(self):
        prefix = self.asset(self.arguments[0] + ".json")[: -len(".json")]
        return [self.wrap(widgets.demonstration_open(prefix, self.lang, self.ui["demo_columns"]), "</figcaption></figure>")]


class RobotAnatomy(CourseDirective):
    def run(self):
        root = Path(self.config.course_assets_dir) / "robot"
        # Transparent exploded-view poster when rendered; the studio still otherwise.
        poster = "robot/anatomy-poster.webp" if (root / "anatomy-poster.webp").exists() else "robot/studio-hd.webp"
        self.asset(poster)
        self.asset("robot/anatomy-layout.json")
        parts = len(json.loads((root / "components.json").read_text())["parts"])
        megabytes = (root / "microduck.glb").stat().st_size / 1e6
        return [self.raw(widgets.anatomy(self.lang, parts, megabytes, poster))]


# ---------------------------------------------------------------- diagrams and charts


class NetworkDiagram(CourseDirective):
    has_content = True

    def run(self):
        # The input/output tables come from a data file; rebuild pages when it changes.
        self.env.note_dependency(str(Path(self.env.srcdir).parent / "_data" / "network-interface.yaml"))
        return [self.wrap(widgets.network_figure_open(self.lang), "</figcaption></figure>")]


class VlaJointDiagram(CourseDirective):
    """A VLA that outputs the 14 servo targets directly (concept figure, Section 5.4)."""

    has_content = True

    def run(self):
        self.env.note_dependency(str(Path(self.env.srcdir).parent / "_data" / "network-interface.yaml"))
        return [self.wrap(widgets.vla_joint_figure_open(self.lang), "</figcaption></figure>")]


class PlatformDiagram(CourseDirective):
    has_content = True

    def run(self):
        return [self.wrap(widgets.platform_figure_open(self.lang), "</figcaption></figure>")]


class YamlWidget(CourseDirective):
    has_content = True
    render = None

    def run(self):
        try:
            html = self.render_widget(type(self).render, self.data(), self.lang)
        except (KeyError, ValueError, TypeError) as error:
            raise self.error(f"{self.name}: {error}")
        return [self.raw(html)]


class LearningFlow(YamlWidget):
    render = staticmethod(widgets.learning_flow)


class ConceptDiagram(YamlWidget):
    render = staticmethod(widgets.concept_diagram)


class ControlTiming(YamlWidget):
    render = staticmethod(widgets.control_timing)


class CodeStory(YamlWidget):
    render = staticmethod(widgets.code_story)


class ModelCard(YamlWidget):
    render = staticmethod(lambda d, lang: widgets.model_card(d))


class ScenarioMatrix(YamlWidget):
    render = staticmethod(lambda d, lang: widgets.scenario_matrix(d))


class TrainingChart(CourseDirective):
    has_content = True

    def run(self):
        data_dir = Path(self.env.srcdir).parent / "_data"
        self.env.note_dependency(str(data_dir / self.data()["data"]))
        return [self.raw(self.render_widget(widgets.training_chart, self.data(), self.lang, data_dir))]


class RecordedRun(CourseDirective):
    """A real, previously executed command with its collapsed output (no copy control)."""

    has_content = True

    def run(self):
        d = self.data()
        image = self.asset(d["image"]) if d.get("image") else None
        return [self.raw(self.render_widget(widgets.recorded_run, d, self.lang, image))]


class BenchmarkChart(CourseDirective):
    """Bar chart of ``passes / total`` cells read from the next table."""

    has_content = True

    def run(self):
        node = benchmark_placeholder()
        node["data"] = self.data()
        self.set_source_info(node)
        return [node]


def resolve_benchmarks(app, doctree):
    for node in list(doctree.findall(benchmark_placeholder)):
        parent = node.parent
        table = None
        for sibling in parent.children[parent.index(node) + 1:]:
            found = list(sibling.findall(nodes.table))
            if found:
                table = found[0]
                break
        if table is None:
            logger.warning("benchmark-chart has no following table", location=node)
            node.replace_self([])
            continue
        head = table.next_node(nodes.thead)
        body = table.next_node(nodes.tbody)
        headers = [entry.astext() for entry in head.findall(nodes.entry)]
        rows = [[entry.astext() for entry in row.findall(nodes.entry)] for row in body.findall(nodes.row)]
        node.replace_self(nodes.raw("", widgets.benchmark_chart(node["data"], headers, rows), format="html"))


def mark_short_cells(app, doctree):
    """Keep short table cells (labels, numbers, units) on one line; long cells still wrap.

    Browsers size auto-layout columns from their content, so a column of short labels next to
    a paragraph column otherwise breaks after every word or character. The limit counts CJK
    characters as two columns; cells with inline code are left free to wrap.
    """
    import unicodedata

    def width(text):
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)

    for table in doctree.findall(nodes.table):
        for entry in table.findall(nodes.entry):
            text = " ".join(entry.astext().split())
            if text and width(text) <= 14 and not any(entry.findall(nodes.literal)):
                entry["classes"].append("cell-short")


class LocalSectionAnchors(SphinxTransform):
    """Section ids such as ``#scene`` are page anchors, not global Sphinx labels.

    Several lessons reuse short section ids (``scene``, ``references``); keeping
    them out of the label registry preserves the old URLs without duplicate-label
    warnings. Link between lessons with ``page.md`` or ``page.html#id``.
    """

    default_priority = 840  # before SphinxDomains (850) collects labels

    def apply(self):
        doc = self.document
        for name, explicit in list(doc.nametypes.items()):
            node = doc.ids.get(doc.nameids.get(name) or "")
            if explicit and isinstance(node, nodes.section):
                doc.nametypes[name] = False


class ResolveSymbols(SphinxTransform):
    """Resolve \\sym marks in MyST inline/display math; report unknown and unused keys."""

    default_priority = 830

    def apply(self):
        temp = self.env.temp_data
        table = temp.get("course_symbols", {})
        used = temp.setdefault("course_symbols_used", [])
        for node in list(self.document.findall(lambda n: isinstance(n, (nodes.math, nodes.math_block)))):
            tex = node.astext()
            if "\\sym" not in tex:
                continue
            try:
                resolved = symbols.resolve(tex, table, used)
            except symbols.SymbolError as error:
                logger.warning("%s", error, location=node)
                continue
            node.children = []
            node += nodes.Text(resolved)
        for key in table:
            if key not in used:
                logger.warning("symbol %r is defined but never used on this page", key, location=(self.env.docname, None))


# ---------------------------------------------------------------- math


def visit_inline_math(self, node):
    self.body.append('<span class="math notranslate nohighlight">\\(' + self.encode(node.astext()) + "\\)</span>")
    raise nodes.SkipNode


def visit_block_math(self, node):
    self.body.append('<div class="math notranslate nohighlight">\\[' + self.encode(node.astext()) + "\\]</div>")
    raise nodes.SkipNode


# ---------------------------------------------------------------- page context, search, redirects


def page_context(app, pagename, templatename, context, doctree):
    context["course_lang"] = app.config.course_lang


def write_redirects(app, exception):
    """Keep old lesson URLs working with small refresh pages and remove stale pages.

    Only top-level ``*.html`` files in this language's output directory are
    considered: anything that is not a current page, a Sphinx page or a redirect
    (for example a lesson removed from the course) is deleted, so it can no
    longer be opened or found.
    """
    if exception or app.builder.format != "html":
        return
    out = Path(app.outdir)
    keep = set(app.env.found_docs) | {"genindex", "search", "py-modindex"} | set(app.config.course_redirects)
    for page in out.glob("*.html"):
        if page.stem not in keep:
            page.unlink()
            logger.info("removed stale page %s", page.name)
    # Search index of the pre-Sphinx generator; it still held removed lessons.
    (out / "search.json").unlink(missing_ok=True)
    lang = app.config.course_lang
    label = "查看课程章节" if lang == "zh" else "Open lesson"
    title = app.config.project
    for old, new in app.config.course_redirects.items():
        (out / (old + ".html")).write_text(
            f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={new}.html">'
            f'<title>{title}</title></head><body><a href="{new}.html">{label}</a></body></html>', encoding="utf-8")


def setup(app):
    app.add_config_value("course_lang", "zh", "env")
    app.add_config_value("course_assets_dir", "", "env")
    app.add_config_value("course_assets_url", "../assets", "env")
    app.add_config_value("course_redirects", {}, "html")

    app.add_node(wrapper, html=(visit_wrapper, depart_wrapper), text=(lambda s, n: None, lambda s, n: None))
    app.add_node(benchmark_placeholder, html=(_skip, None))
    for name, cls in {
        "lesson-intro": LessonIntro,
        "quiz": Quiz,
        "answer": Answer,
        "enrichment": Enrichment,
        "formula": Formula,
        "symbols": Symbols,
        "course-video": CourseVideo,
        "course-image": CourseImage,
        "scene-cards": SceneCards,
        "demonstration": Demonstration,
        "robot-anatomy": RobotAnatomy,
        "network-diagram": NetworkDiagram,
        "vla-joint-diagram": VlaJointDiagram,
        "platform-diagram": PlatformDiagram,
        "learning-flow": LearningFlow,
        "concept-diagram": ConceptDiagram,
        "control-timing": ControlTiming,
        "code-story": CodeStory,
        "model-card": ModelCard,
        "scenario-matrix": ScenarioMatrix,
        "training-chart": TrainingChart,
        "benchmark-chart": BenchmarkChart,
        "recorded-run": RecordedRun,
    }.items():
        app.add_directive(name, cls)
    robot_explorer.setup(app)
    app.add_transform(LocalSectionAnchors)
    app.add_transform(ResolveSymbols)
    app.connect("doctree-read", resolve_benchmarks)
    app.connect("doctree-read", mark_short_cells)
    app.add_html_math_renderer("course_katex", (visit_inline_math, None), (visit_block_math, None))
    app.connect("html-page-context", page_context)
    app.connect("build-finished", write_redirects)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
