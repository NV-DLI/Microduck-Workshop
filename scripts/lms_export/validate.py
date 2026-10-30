"""Validate an exported LMS archive: OLX bindings, assets policy and static dependencies.

The JavaScript checks use the same limited lexer as the exporter: they confirm the references
the exporter knows how to express (``@lms/`` imports, ``__lmsAsset("...")`` calls, bare
specifiers covered by each page's import map) but cannot prove arbitrary JavaScript loads
nothing else. Browser checks under the asset-key route remain necessary.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

from bs4 import BeautifulSoup

from . import jsscan
from .exporter import mime_for
from .manifest import Manifest, text_for
from .olx import read_archive

CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
DEV_URL_RE = re.compile(r"(?i)\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0)\b|file://|/home/[^/\s]+/|/Users/[^/\s]+/")
BOOT_CONFIG_RE = re.compile(r"\}\)\(window,document,(\{.*\})\);\s*$", re.DOTALL)
WRITE_RE = re.compile(r"^__lms\.write\((.*)\)$", re.DOTALL)
ASSET_CALL_RE = re.compile(r"__lmsAsset\(\s*(\"(?:[^\"\\]|\\.)*\")\s*\)")


class Result:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.counts: dict[str, int] = {}

    def error(self, msg: str) -> None:
        if len(self.errors) < 400:
            self.errors.append(msg)

    def warn(self, msg: str) -> None:
        if len(self.warnings) < 400:
            self.warnings.append(msg)


def _xml(files: dict[str, bytes], path: str, res: Result) -> ET.Element | None:
    data = files.get(path)
    if data is None:
        res.error(f"missing {path}")
        return None
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        res.error(f"{path}: {exc}")
        return None


def _local(value: str) -> bool:
    return bool(value) and not jsscan.is_external(value)


def _name(value: str) -> str:
    return jsscan.split_url(value)[0]


def validate(archive: Path, manifest: Manifest | None = None, edition_id: str | None = None,
             forbid: list[str] | None = None, forbid_archives: list[Path] | None = None) -> dict[str, Any]:
    res = Result()
    files_all = read_archive(archive)
    bad_top = sorted({p.split("/")[0] for p in files_all} - {"course"})
    if bad_top:
        res.error(f"archive has entries outside course/: {bad_top}")
    files = {p[len("course/"):]: d for p, d in files_all.items() if p.startswith("course/")}
    static = {p[len("static/"):]: d for p, d in files.items() if p.startswith("static/")}
    for name in static:
        if "/" in name:
            res.error(f"nested static path {name} (assets must be flat)")

    # ------------------------------------------------------------------ OLX
    root = _xml(files, "course.xml", res)
    org = course = run = ""
    display = language = ""
    order: list[dict[str, str]] = []
    referenced: set[str] = {"course.xml"}
    if root is not None:
        org, course, run = root.get("org", ""), root.get("course", ""), root.get("url_name", "")
        cpath = f"course/{run}.xml"
        referenced.add(cpath)
        cel = _xml(files, cpath, res)
        if cel is not None:
            display, language = cel.get("display_name", ""), cel.get("language", "")
            if cel.get("cert_html_view_enabled", "false") != "false":
                res.error("certificates are enabled in course xml")
            for ch_ref in cel.findall("chapter"):
                ch_path = f"chapter/{ch_ref.get('url_name')}.xml"
                referenced.add(ch_path)
                ch = _xml(files, ch_path, res)
                if ch is None:
                    continue
                for seq_ref in ch.findall("sequential"):
                    seq_path = f"sequential/{seq_ref.get('url_name')}.xml"
                    referenced.add(seq_path)
                    seq = _xml(files, seq_path, res)
                    if seq is None:
                        continue
                    if seq.get("graded") or seq.get("format"):
                        res.error(f"{seq_path} is graded/formatted; exporter adds no assessments")
                    for v_ref in seq.findall("vertical"):
                        v_path = f"vertical/{v_ref.get('url_name')}.xml"
                        referenced.add(v_path)
                        v = _xml(files, v_path, res)
                        if v is None:
                            continue
                        children = list(v)
                        if len(children) != 1 or children[0].tag != "html":
                            res.error(f"{v_path}: expected exactly one html child")
                            continue
                        hid = children[0].get("url_name", "")
                        h_path = f"html/{hid}.xml"
                        referenced.update({h_path, f"html/{hid}.html"})
                        h = _xml(files, h_path, res)
                        if h is None:
                            continue
                        if h.get("filename") != hid or h.get("display_name") != "Raw HTML" or h.get("editor") != "raw":
                            res.error(f"{h_path}: descriptor must be filename={hid} display_name='Raw HTML' editor='raw'")
                        body = files.get(f"html/{hid}.html")
                        if body is None:
                            res.error(f"missing html/{hid}.html")
                            continue
                        frames = BeautifulSoup(body.decode("utf-8"), "html.parser").find_all("iframe")
                        if len(frames) != 1:
                            res.error(f"html/{hid}.html: expected one iframe")
                            continue
                        fr = frames[0]
                        src = fr.get("src", "")
                        style = re.sub(r"\s+", "", fr.get("style", "")).lower()
                        if not src.startswith("/static/") or "/" in src[len("/static/"):]:
                            res.error(f"html/{hid}.html: iframe src must be /static/<flat>.html, got {src}")
                        elif src[len("/static/"):] not in static:
                            res.error(f"html/{hid}.html: iframe target {src} not in static/")
                        for need in ("width:100%", "height:90vh", "border:0"):
                            if need not in style:
                                res.error(f"html/{hid}.html: iframe style lacks {need}")
                        if not fr.get("title"):
                            res.error(f"html/{hid}.html: iframe needs a title")
                        order.append({
                            "chapter": ch.get("display_name", ""), "section": seq.get("display_name", ""),
                            "title": v.get("display_name", ""), "page": src[len("/static/"):],
                        })
    for d in ("problem", "credentials", "drafts", "library_content"):
        if any(p.startswith(d + "/") for p in files):
            res.error(f"unexpected {d}/ content")
    for path in files:
        top = path.split("/")[0]
        if top in ("chapter", "sequential", "vertical", "html") and path not in referenced:
            res.error(f"orphan OLX file {path}")

    # ------------------------------------------------------------------ policies
    pol_path = f"policies/{run}/policy.json"
    policy = json.loads(files.get(pol_path, b"{}") or b"{}").get(f"course/{run}", {})
    if not policy:
        res.error(f"{pol_path} missing course/{run}")
    else:
        if policy.get("display_name") != display:
            res.error("policy display_name differs from course xml")
        if policy.get("language") != language:
            res.error("policy language differs from course xml")
        if policy.get("cert_html_view_enabled") or policy.get("certificates"):
            res.error("policy enables certificates")
        if policy.get("advanced_modules"):
            res.error(f"policy lists advanced_modules {policy['advanced_modules']}")
        if not policy.get("start"):
            res.warn("policy has no start date; Open edX treats such courses as not yet started")
    if f"policies/{run}/grading_policy.json" not in files:
        res.error("grading_policy.json missing")
    about_title = files.get("about/title.html", b"").decode("utf-8").strip()
    if about_title != display:
        res.error("about/title.html differs from course display_name")
    if "about/overview.html" not in files:
        res.error("about/overview.html missing")
    assets = json.loads(files.get("policies/assets.json", b"{}"))
    if set(assets) != set(static):
        res.error(f"assets.json/static mismatch: missing {sorted(set(static) - set(assets))[:10]}, "
                  f"extra {sorted(set(assets) - set(static))[:10]}")
    for name, entry in assets.items():
        son = entry.get("content_son", {})
        if (son.get("org"), son.get("course"), son.get("run"), son.get("name")) != (org, course, run, name):
            res.error(f"assets.json {name}: identity differs from course key")
        if entry.get("filename") != f"asset-v1:{org}+{course}+{run}+type@asset+block@{name}":
            res.error(f"assets.json {name}: filename does not match")
        expected = mime_for(name)
        if expected and entry.get("contentType") != expected:
            res.error(f"assets.json {name}: contentType {entry.get('contentType')} != {expected}")
        if entry.get("import_path") != name or entry.get("locked"):
            res.error(f"assets.json {name}: import_path/locked")

    # ------------------------------------------------------------------ static dependencies
    def need(name: str, where: str, what: str) -> None:
        base = _name(name)
        if base not in static:
            res.error(f"{where}: {what} {name!r} is not in static/")

    def path_hygiene(value: str, where: str) -> None:
        if "_static/" in value or "_images/" in value:
            res.error(f"{where}: leftover Sphinx path {value!r}")
        if DEV_URL_RE.search(value):
            res.error(f"{where}: development/private URL {value!r}")
        if "asset-v1:" in value or "/c4x/" in value:
            res.error(f"{where}: hard-coded course asset route {value!r}")

    module_bare: dict[str, set[str]] = {}
    module_imports: dict[str, set[str]] = {}
    for name, data in sorted(static.items()):
        kind = PurePosixPath(name).suffix.lower()
        if kind in (".js", ".mjs"):
            text = data.decode("utf-8", "replace")
            try:
                refs, _ = jsscan.js_references(text)
            except jsscan.LexError as exc:
                res.warn(f"{name}: cannot lex ({exc})")
                continue
            bare, internal = set(), set()
            toks = None
            for ref in refs:
                if ref.kind == "import":
                    if ref.value.startswith("@lms/"):
                        need(ref.value[5:], name, "import")
                        internal.add(ref.value)
                    elif ref.value.startswith(("./", "../", "/")):
                        res.error(f"{name}: relative import {ref.value!r} breaks under asset-key routes")
                    elif not jsscan.is_external(ref.value):
                        bare.add(ref.value)
                elif ref.kind == "module-url" and not jsscan.is_external(ref.value):
                    res.error(f"{name}: new URL({ref.value!r}, import.meta.url) breaks under asset-key routes")
                elif jsscan.looks_like_path(ref.value):
                    path_hygiene(ref.value, name)
            for m in ASSET_CALL_RE.finditer(text):
                need(json.loads(m.group(1)), name, "__lmsAsset")
            module_bare[name] = bare
            module_imports[name] = internal
        elif kind == ".json":
            try:
                refs, _ = jsscan.js_references(data.decode("utf-8", "replace"))
            except jsscan.LexError:
                refs = []
            for ref in refs:
                if ref.kind != "object-key" and _name(ref.value) in static and _name(ref.value) != name:
                    res.warn(f"{name}: names asset {ref.value!r}; its consumer must resolve it with "
                             "__lmsAsset() or it fails under the asset-v1 block@ route")
        elif kind == ".css":
            text = data.decode("utf-8", "replace")
            for ref in jsscan.css_references(text):
                if _local(ref.value) and not ref.value.startswith("data:"):
                    res.error(f"{name}: relative CSS reference {ref.value!r} breaks under asset-key routes")

    ed = manifest.edition(edition_id) if manifest and edition_id else None
    page_names = sorted(n for n in static if n.endswith(".html"))
    iframe_pages = {o["page"] for o in order}
    switch_targets: set[str] = set()
    for name in page_names:
        soup = BeautifulSoup(static[name].decode("utf-8"), "html.parser")
        boot = soup.find("script", attrs={"data-lms-export": "bootstrap"})
        if boot is None:
            res.error(f"{name}: missing LMS bootstrap")
            continue
        first_scripts = soup.head.find_all("script") if soup.head else []
        if not first_scripts or first_scripts[0] is not boot:
            res.error(f"{name}: bootstrap must be the first script in <head>")
        m = BOOT_CONFIG_RE.search(boot.string or "")
        config = json.loads(m.group(1)) if m else {}
        imports = config.get("imports", {})
        for spec, target in imports.items():
            need(target, name, f"import map {spec}")
        if soup.find("script", attrs={"type": "importmap"}):
            res.error(f"{name}: static import map left in page (must be written by the bootstrap)")
        # Module closure of this page must have every bare specifier mapped.
        mods = {t for t in imports.values() if t in module_bare}
        for mod in sorted(mods):
            for spec in module_bare[mod]:
                if spec not in imports:
                    res.error(f"{name}: module {mod} imports {spec!r} but the page import map lacks it")
            for spec in module_imports[mod]:
                if spec not in imports:
                    res.error(f"{name}: module {mod} imports {spec!r} but the page import map lacks it")
        for script in soup.find_all("script"):
            if script is boot:
                continue
            text = (script.string or "").strip()
            if script.get("src"):
                res.error(f"{name}: <script src> left static ({script['src']}); must go through __lms.write")
            wm = WRITE_RE.match(text)
            if wm:
                tag, attrs, urls = json.loads("[" + wm.group(1) + "]")
                for k, v in attrs:
                    if k in urls:
                        need(v, name, f"{tag} {k}")
                        path_hygiene(v, name)
                continue
            for am in ASSET_CALL_RE.finditer(text):
                need(json.loads(am.group(1)), name, "__lmsAsset")
            if (script.get("type") or "") == "module" or "import" in text:
                try:
                    refs, _ = jsscan.js_references(text)
                except jsscan.LexError:
                    refs = []
                for ref in refs:
                    if ref.kind == "import" and ref.value.startswith("@lms/") and ref.value not in imports:
                        res.error(f"{name}: inline import {ref.value} not in import map")
                    elif ref.kind == "import" and ref.value.startswith(("./", "../", "/")):
                        res.error(f"{name}: inline relative import {ref.value!r}")
        for el in soup.find_all(True):
            if el.name in ("script", "style"):
                continue
            lms = el.get("data-lms-url")
            mapped = json.loads(lms) if lms else {}
            for k, v in mapped.items():
                vals = [x.strip().split()[0] for x in v.split(",")] if k in ("srcset", "imagesrcset") else [v]
                for x in vals:
                    if _local(x):
                        need(x, name, f"<{el.name} {k}>")
                        path_hygiene(x, name)
            for attr in ("src", "srcset", "poster", "data", "xlink:href"):
                v = el.get(attr)
                if isinstance(v, str) and _local(v) and attr not in mapped:
                    res.error(f"{name}: <{el.name} {attr}={v!r}> is relative and not routed")
            if el.name == "link" and el.get("href") and _local(el["href"]):
                res.error(f"{name}: <link href={el['href']!r}> left static")
            href = el.get("href")
            if el.name in ("a", "area") and isinstance(href, str) and _local(href):
                if "href" not in mapped:
                    res.error(f"{name}: link {href!r} is relative and not routed")
                need(href, name, "link")
            if el.name not in ("a", "area") and isinstance(href, str) and _local(href) and "href" not in mapped:
                res.error(f"{name}: <{el.name} href={href!r}> is relative and not routed")
            for attr, v in el.attrs.items():
                if attr.startswith("data-") and attr != "data-lms-url" and isinstance(v, str) and jsscan.looks_like_path(v):
                    if ("_static/" in v or "_images/" in v) or (attr not in mapped and jsscan.has_asset_extension(v) and "/" in v):
                        res.warn(f"{name}: {attr}={v!r} looks like an unrouted path")
        switch = soup.select(".lms-lang-switch, [data-lms-lang]")
        lang_cfg = config.get("lang")
        if lang_cfg:
            switch_targets.update(lang_cfg.get("pages", {}).values())
        if ed is not None:
            if len(ed.languages) == 1 and (switch or lang_cfg):
                res.error(f"{name}: single-language edition must not offer a language switch")
            if len(ed.languages) > 1 and ed.switch and not switch:
                res.error(f"{name}: language switch missing")
            if lang_cfg:
                for lang, target in lang_cfg.get("pages", {}).items():
                    if lang not in ed.languages:
                        res.error(f"{name}: language switch offers {lang}")
                    need(target, name, "language switch")
        if ed is not None and len(ed.languages) == 1:
            body_text = soup.body.get_text(" ") if soup.body else ""
            for s in soup.find_all(["script", "style"]):
                body_text = body_text.replace(s.get_text(), "")
            cjk = len(CJK_RE.findall(body_text))
            if cjk:
                res.warn(f"{name}: {cjk} CJK characters in visible text of a single-language edition")

    for name in page_names:
        # Language alternates are reached through the switch rather than an iframe unit.
        if name not in iframe_pages and name not in switch_targets:
            res.warn(f"static {name} is neither an iframe unit page nor a language alternate")

    # ------------------------------------------------------------------ manifest order and identity
    if ed is not None and manifest is not None:
        c = ed.course
        if (org, course, run) != (c["org"], c["course"], c["run"]):
            res.error(f"course key {org}+{course}+{run} != manifest {ed.course_key}")
        if display != text_for(c["title"], ed.default, "course title"):
            res.error("course display_name differs from manifest title")
        if language != c["language"]:
            res.error("course language differs from manifest")
        expected = [
            {"chapter": text_for(ch.titles, ed.default, "chapter"), "section": text_for(sec.titles, ed.default, "section"),
             "title": p.title(ed.default), "page": f"{p.id}{ed.filename_suffix[ed.default]}.html"}
            for ch in manifest.chapters for sec in ch.sections for p in sec.pages
        ]
        if expected != order:
            res.error("OLX unit order/titles/pages differ from the manifest")
        allowed_pages = {f"{p.id}{ed.filename_suffix[l]}.html" for _, _, p in manifest.pages() for l in ed.languages}
        other = {f"{p.id}{suffix}.html" for _, _, p in manifest.pages()
                 for suffix in {f"-{l}" for l in manifest.languages if l not in ed.languages}}
        for name in page_names:
            if name in other or name not in allowed_pages:
                res.error(f"static page {name} does not belong to edition {ed.id}")

    texts = {p: d.decode("utf-8", "replace") for p, d in files.items()
             if PurePosixPath(p).suffix.lower() in (".xml", ".html", ".json", ".js", ".css", ".svg", ".txt", ".mjs")}
    for needle in forbid or []:
        for p, t in texts.items():
            if needle in t:
                res.error(f"forbidden string {needle!r} in {p}")
    for ref_archive in forbid_archives or []:
        ref = read_archive(ref_archive)
        ref_files = {p.split("/", 1)[1]: d for p, d in ref.items() if p.startswith("course/") and "/" in p}
        rroot = ET.fromstring(ref_files["course.xml"]) if "course.xml" in ref_files else None
        needles = set()
        if rroot is not None:
            r_org, r_course, r_run = rroot.get("org"), rroot.get("course"), rroot.get("url_name")
            needles |= {f"{r_org}+{r_course}+{r_run}", f"{r_org}.{r_course}.{r_run}", f"{r_course}+{r_run}"}
            if (r_org, r_course) == (org, course):
                res.error(f"course org/number {org}/{course} reuses the reference archive's identity")
            rc = ET.fromstring(ref_files.get(f"course/{r_run}.xml", b"<course/>"))
            if rc.get("display_name"):
                needles.add(rc.get("display_name"))
        for p in ref_files:
            parts = p.split("/")
            if parts[0] in ("chapter", "sequential", "vertical", "html", "problem") and len(parts) == 2:
                needles.add(PurePosixPath(parts[1]).stem)
        for needle in sorted(n for n in needles if n and len(n) >= 6):
            for p, t in texts.items():
                if needle in t or needle in p:
                    res.error(f"reference identity {needle!r} found in {p}")
                    break

    res.counts = {
        "chapters": sum(1 for p in files if p.startswith("chapter/")),
        "sequentials": sum(1 for p in files if p.startswith("sequential/")),
        "verticals": sum(1 for p in files if p.startswith("vertical/")),
        "html_components": sum(1 for p in files if p.startswith("html/") and p.endswith(".xml")),
        "static_files": len(static),
        "static_pages": len(page_names),
        "static_bytes": sum(len(d) for d in static.values()),
        "modules": len(module_bare),
    }
    return {
        "archive": str(archive),
        "sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        "course_key": f"{org}+{course}+{run}",
        "display_name": display,
        "language": language,
        "counts": res.counts,
        "units": order,
        "errors": res.errors,
        "warnings": res.warnings,
        "ok": not res.errors,
        "limits": "Static checks cover the reference forms the exporter emits; computed JavaScript URLs need a browser check under an asset-key route.",
    }
