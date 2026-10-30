"""Turn selected pages of a built Sphinx site into flat LMS static files."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import mimetypes
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from bs4 import BeautifulSoup, Comment, Tag

from . import jsscan, runtime
from .manifest import Edition, Manifest, text_for

MIME_TYPES = {
    ".js": "application/javascript", ".mjs": "application/javascript", ".cjs": "application/javascript",
    ".css": "text/css", ".html": "text/html", ".htm": "text/html", ".json": "application/json",
    ".geojson": "application/geo+json", ".gltf": "model/gltf+json", ".glb": "model/gltf-binary",
    ".bin": "application/octet-stream", ".wasm": "application/wasm", ".woff2": "font/woff2",
    ".woff": "font/woff", ".ttf": "font/ttf", ".otf": "font/otf", ".eot": "application/vnd.ms-fontobject",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".avif": "image/avif", ".ico": "image/x-icon",
    ".bmp": "image/bmp", ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
    ".ogv": "video/ogg", ".mov": "video/quicktime", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".wav": "audio/wav", ".flac": "audio/flac", ".vtt": "text/vtt",
    ".srt": "text/plain", ".pdf": "application/pdf", ".zip": "application/zip", ".txt": "text/plain",
    ".csv": "text/csv", ".tsv": "text/tab-separated-values", ".md": "text/markdown",
    ".py": "text/x-python", ".yaml": "text/yaml", ".yml": "text/yaml", ".xml": "application/xml",
    ".ipynb": "application/json", ".usd": "application/octet-stream", ".usda": "text/plain",
    ".usdz": "model/vnd.usdz+zip", ".hdr": "image/vnd.radiance", ".exr": "image/x-exr",
    ".ktx2": "image/ktx2",
}

JS_TYPES = {"", "text/javascript", "application/javascript", "module", "text/ecmascript"}
JSON_SCRIPT_TYPES = {"application/json", "application/ld+json"}
FETCHING_ATTRS = {"src", "srcset", "imagesrcset", "poster", "data", "xlink:href"}
URL_ATTRS = ("src", "srcset", "imagesrcset", "poster", "data", "href", "xlink:href")

PYDATA_CSS = """
html, body { background: #fff !important; color-scheme: light; }
html { scroll-padding-top: 0 !important; }
.bd-page-width { max-width: none !important; }
.bd-container, .bd-container__inner, .bd-main, .bd-content { max-width: none !important; width: 100% !important; }
.bd-main { flex-grow: 1 !important; }
.bd-main .bd-content .bd-article-container { max-width: none !important; width: 100% !important; }
.bd-article-container { padding-left: clamp(1rem, 3vw, 2rem); padding-right: clamp(1rem, 3vw, 2rem); }
"""

SWITCH_CSS = """
.lms-lang-switch { display: flex; justify-content: flex-end; gap: .25rem; margin: 0; padding: .5rem clamp(1rem, 3vw, 2rem) 0; font: 500 .8125rem/1.4 system-ui, sans-serif; }
.lms-lang-switch a { padding: .125rem .5rem; border: 1px solid #d0d0d0; border-radius: .25rem; color: #333 !important; text-decoration: none !important; background: #fff; }
.lms-lang-switch a[aria-current="page"] { border-color: #76b900; color: #000 !important; font-weight: 700; }
.lms-lang-switch a:hover, .lms-lang-switch a:focus-visible { border-color: #76b900; }
"""

PROFILES: dict[str, dict[str, Any]] = {
    "pydata": {
        "content_selectors": ["article.bd-article"],
        "remove_selectors": [
            ".bd-header-article", "footer.prev-next-footer", ".prev-next-area", ".bd-sidebar-secondary",
            ".bd-sidebar-primary", "#pst-primary-sidebar-modal", "#pst-secondary-sidebar-modal",
            "header.bd-header", "footer.bd-footer", "footer.bd-footer-content", "#pst-back-to-top",
            "#pst-search-dialog", "#pst-skip-link", ".bd-header-announcement", "#bd-header-version-warning",
            ".pst-async-banner-revealer",
        ],
        "drop_head_selectors": [
            'link[rel~="index"]', 'link[rel~="search"]', 'link[rel~="next"]', 'link[rel~="prev"]',
            'link[rel~="icon"]', 'link[rel~="canonical"]', 'link[rel~="alternate"]', 'meta[name^="docsearch:"]',
        ],
        "drop_script_patterns": [
            r"(^|/)searchtools\.js", r"(^|/)language_data\.js", r"(^|/)searchindex\.js",
            r"(^|/)sphinx_highlight\.js", r"(^|/)doctools\.js", r"(^|/)translations\.js",
            r"(^|/)scripts/pydata-sphinx-theme\.js", r"(^|/)scripts/bootstrap\.js",
        ],
        "drop_inline_script_patterns": [
            r"localStorage\.(?:get|set)Item\(\s*['\"](?:mode|theme)['\"]",
            r"dataset\.(?:mode|theme)\s*=",
        ],
        "html_attributes": {"data-theme": "light", "data-mode": "light"},
        "body_attributes": {},
        "css": PYDATA_CSS,
    },
    "basic": {
        "content_selectors": ['div[role="main"]', "main", "article"],
        "remove_selectors": ["div.related", "div.sphinxsidebar", "div.footer", "nav"],
        "drop_head_selectors": [
            'link[rel~="index"]', 'link[rel~="search"]', 'link[rel~="next"]', 'link[rel~="prev"]',
            'link[rel~="icon"]',
        ],
        "drop_script_patterns": [r"(^|/)searchtools\.js", r"(^|/)language_data\.js", r"(^|/)searchindex\.js"],
        "drop_inline_script_patterns": [],
        "html_attributes": {},
        "body_attributes": {},
        "css": "html, body { background: #fff !important; color-scheme: light; }",
    },
}


def resolve_profile(extract: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    profile = dict(PROFILES[extract.get("profile", "pydata")])
    for key in ("content_selectors", "remove_selectors", "drop_head_selectors", "drop_script_patterns",
                "drop_inline_script_patterns", "html_attributes", "body_attributes"):
        if key in extract:
            profile[key] = extract[key]
        extra = extract.get("extra_" + key)
        if extra:
            profile[key] = ({**profile[key], **extra} if isinstance(extra, dict) else [*profile[key], *extra])
    css = profile["css"]
    if extract.get("css") is not None:
        css = extract["css"]
    extra_css = extract.get("extra_css")
    if extra_css:
        if extra_css.endswith(".css") and "\n" not in extra_css:
            extra_css = (manifest_dir / extra_css).read_text(encoding="utf-8")
        css += "\n" + extra_css
    profile["css"] = css
    profile["page_title"] = extract.get("page_title", "{page} - {course}")
    profile["inline_css_max_bytes"] = int(extract.get("inline_css_max_bytes", 16 * 1024 * 1024))
    profile["prune_font_formats"] = bool(extract.get("prune_font_formats", True))
    return profile


def mime_for(name: str) -> str | None:
    ext = posixpath.splitext(name)[1].lower()
    return MIME_TYPES.get(ext) or mimetypes.guess_type(name)[0]


def kind_for(rel: str) -> str:
    ext = posixpath.splitext(rel)[1].lower()
    if ext in (".js", ".mjs", ".cjs"):
        return "js"
    if ext == ".css":
        return "css"
    if ext in (".json", ".gltf", ".geojson"):
        return "json"
    if ext == ".svg":
        return "svg"
    if ext in (".html", ".htm"):
        return "html"
    return "bin"


class ExportError(RuntimeError):
    pass


class Site:
    """Path resolution confined to the built site root."""

    def __init__(self, root: Path, allow_external_symlinks: bool = False):
        self.root = Path(os.path.abspath(root))
        if not self.root.is_dir():
            raise ExportError(f"site root is not a directory: {root}")
        self.real = self.root.resolve()
        self.allow_external_symlinks = allow_external_symlinks
        self.external_symlinks: set[str] = set()

    def path(self, rel: str) -> Path:
        return self.root / rel

    def check(self, rel: str) -> str | None:
        """Return None when rel is a regular file inside the root; otherwise a reason."""
        if rel == ".." or rel.startswith("../") or rel.startswith("/"):
            return "outside-root"
        p = self.root / rel
        if not p.exists():
            return "missing"
        real = p.resolve()
        try:
            real.relative_to(self.real)
        except ValueError:
            if not self.allow_external_symlinks:
                return "symlink-outside-root"
            self.external_symlinks.add(rel)
        if not real.is_file():
            return "not-a-file"
        return None

    def join(self, base_dir: str, url_path: str) -> str:
        path = unquote(url_path)
        if path.startswith("/"):
            joined = path.lstrip("/")
        else:
            joined = posixpath.join(base_dir, path) if base_dir else path
        norm = posixpath.normpath(joined)
        return "" if norm == "." else norm

    def resolve(self, base_dir: str, value: str) -> tuple[str | None, str, str]:
        """Return (rel or None, suffix, reason)."""
        if jsscan.is_external(value):
            return None, "", "external"
        path, suffix = jsscan.split_url(value.strip())
        if not path:
            return None, suffix, "self"
        rel = self.join(base_dir, path)
        if rel and (self.root / rel).is_dir():
            rel = posixpath.join(rel, "index.html")
        reason = self.check(rel) if rel else "missing"
        return (rel if reason is None else None), suffix, (reason or "ok")


@dataclass
class Node:
    rel: str
    kind: str
    data: bytes
    text: str | None = None
    js_refs: list[jsscan.JsRef] = field(default_factory=list)
    css_refs: list[jsscan.CssRef] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)
    pages: set[str] = field(default_factory=set)
    ref_dirs: set[str] = field(default_factory=set)
    resolved: dict[int, set[str]] = field(default_factory=dict)
    bare: dict[int, dict[str, str | None]] = field(default_factory=dict)  # ref -> page -> target
    lex_error: str | None = None

    @property
    def dir(self) -> str:
        return posixpath.dirname(self.rel)


@dataclass
class PageState:
    key: str  # "<lang>:<page id>"
    page_id: str
    lang: str
    rel: str
    flat: str
    soup: BeautifulSoup
    importmap_exact: dict[str, str] = field(default_factory=dict)
    importmap_prefix: list[tuple[str, str]] = field(default_factory=list)
    targets: set[str] = field(default_factory=set)  # copied assets referenced directly
    inline_bare: set[str] = field(default_factory=set)
    embeds: set[str] = field(default_factory=set)  # embedded as data: URIs, never copied
    actions: list[Any] = field(default_factory=list)

    @property
    def dir(self) -> str:
        return posixpath.dirname(self.rel)


class EditionExport:
    def __init__(self, site: Site, manifest: Manifest, edition: Edition):
        self.site = site
        self.m = manifest
        self.ed = edition
        self.profile = resolve_profile(manifest.extract, manifest.path.parent)
        self.nodes: dict[str, Node] = {}
        self.pages: dict[str, PageState] = {}
        self.page_by_rel: dict[str, PageState] = {}
        self.report: dict[str, Any] = {
            "edition": edition.id, "course_key": edition.course_key, "unresolved": [], "ambiguous": [],
            "unmapped_page_links": [], "dropped_cross_language_links": [], "dynamic_resource_sites": [],
            "object_key_literals": [], "json_relative_refs": [], "js_literal_rewrites": {},
            "unmapped_bare_specifiers": [], "lex_errors": [], "large_inline": [], "notes": [],
            "rewrites_applied": {}, "unknown_mime": [],
        }
        self._names: dict[str, str] = {}
        self._keys: dict[str, str] = {}
        self._render_cache: dict[str, bytes] = {}

    # ------------------------------------------------------------------ helpers
    def _issue(self, bucket: str, **info: Any) -> None:
        entries = self.report[bucket]
        if info not in entries and len(entries) < 500:
            entries.append(info)

    def _node(self, rel: str) -> Node:
        node = self.nodes.get(rel)
        if node:
            return node
        data = self.site.path(rel).read_bytes()
        node = Node(rel, kind_for(rel), data)
        if node.kind in ("js", "json", "css", "svg"):
            try:
                node.text = data.decode("utf-8")
            except UnicodeDecodeError:
                node.kind = "bin"
        if node.kind == "css" and node.text is not None:
            if self.profile["prune_font_formats"]:
                node.text = jsscan.prune_font_formats(node.text)
            node.css_refs = jsscan.css_references(node.text)
        elif node.kind == "svg" and node.text is not None:
            node.css_refs = jsscan.svg_references(node.text)
        elif node.kind in ("js", "json") and node.text is not None:
            try:
                node.js_refs, node.dynamic = jsscan.js_references(node.text)
            except jsscan.LexError as exc:
                node.lex_error = str(exc)
                self._issue("lex_errors", file=rel, error=str(exc))
            for raw in node.dynamic:
                self._issue("dynamic_resource_sites", file=rel, expression=raw)
        self.nodes[rel] = node
        return node

    def _page_rels(self) -> set[str]:
        return set(self.page_by_rel)

    def _resolve_bare(self, page: PageState, spec: str) -> str | None:
        if spec in page.importmap_exact:
            return page.importmap_exact[spec]
        best = None
        for prefix, target_dir in page.importmap_prefix:
            if spec.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
                best = (prefix, target_dir)
        if best is None:
            return None
        rel = self.site.join(best[1], spec[len(best[0]):])
        return rel if self.site.check(rel) is None else None

    def _literal_targets(self, value: str, bases: list[str]) -> set[str]:
        if not jsscan.looks_like_path(value):
            return set()
        found = set()
        for base in bases:
            rel, _, reason = self.site.resolve(base, value)
            if rel and rel not in self.page_by_rel and kind_for(rel) != "html":
                found.add(rel)
        return found

    def _apply_manifest_rewrites(self, node: Node, index: int, value: str) -> str | None:
        for i, rule in enumerate(self.m.rewrites):
            if not fnmatch.fnmatch(node.rel, rule.get("files", "*")):
                continue
            if rule.get("match") == value and "asset" in rule:
                self.report["rewrites_applied"][str(i)] = self.report["rewrites_applied"].get(str(i), 0) + 1
                return rule["asset"]
        return None

    # ------------------------------------------------------------------ graph
    def _resolve_node(self, node: Node) -> None:
        if node.kind in ("css", "svg"):
            for i, ref in enumerate(node.css_refs):
                if i in node.resolved:
                    continue
                rel, _, reason = self.site.resolve(node.dir, ref.value)
                node.resolved[i] = {rel} if rel else set()
                if not rel and reason not in ("external", "self"):
                    self._issue("unresolved", file=node.rel, ref=ref.value, reason=reason)
            return
        bases = [node.dir] + sorted({self.pages[k].dir for k in node.pages}) + sorted(node.ref_dirs)
        for i, ref in enumerate(node.js_refs):
            forced = self._apply_manifest_rewrites(node, i, ref.value)
            if forced is not None:
                reason = self.site.check(forced)
                if reason:
                    raise ExportError(f"rewrite target {forced!r} for {node.rel}: {reason}")
                node.resolved[i] = {forced}
                continue
            if ref.kind == "import" and node.kind == "js":
                value = ref.value
                if value.startswith(("./", "../", "/")):
                    rel, _, reason = self.site.resolve(node.dir, value)
                    node.resolved[i] = {rel} if rel else set()
                    if not rel:
                        self._issue("unresolved", file=node.rel, ref=value, reason=reason)
                elif not jsscan.is_external(value):
                    per_page = node.bare.setdefault(i, {})
                    for key in node.pages:
                        if key not in per_page:
                            per_page[key] = self._resolve_bare(self.pages[key], value)
                            if per_page[key] is None:
                                self._issue("unmapped_bare_specifiers", file=node.rel, specifier=value,
                                            page=self.pages[key].rel)
                continue
            if ref.kind == "module-url":
                rel, _, reason = self.site.resolve(node.dir, ref.value)
                node.resolved[i] = {rel} if rel else set()
                if not rel and reason not in ("external", "self"):
                    self._issue("unresolved", file=node.rel, ref=ref.value, reason=reason)
                continue
            node.resolved[i] = self._literal_targets(ref.value, bases)

    def _edges(self, node: Node) -> set[str]:
        out: set[str] = set()
        for targets in node.resolved.values():
            out |= targets
        for per_page in node.bare.values():
            out |= {t for t in per_page.values() if t}
        return out

    def _walk(self, roots: list[tuple[str, str | None, str | None]]) -> None:
        queue = list(roots)
        while queue:
            rel, page_key, ref_dir = queue.pop()
            node = self._node(rel)
            changed = not node.resolved and not node.bare
            if page_key and page_key not in node.pages:
                node.pages.add(page_key)
                changed = True
            if ref_dir is not None and ref_dir not in node.ref_dirs and ref_dir != node.dir:
                node.ref_dirs.add(ref_dir)
                changed = True
            if not changed:
                continue
            if node.kind in ("js", "json"):
                node.resolved = {}
            self._resolve_node(node)
            for target in self._edges(node):
                for key in sorted(node.pages) or [None]:
                    queue.append((target, key, node.dir))

    # ------------------------------------------------------------------ pages
    def _strip_chrome(self, soup: BeautifulSoup, page_rel: str) -> Tag:
        content = None
        for selector in self.profile["content_selectors"]:
            content = soup.select_one(selector)
            if content:
                break
        if content is None:
            raise ExportError(f"{page_rel}: no element matches {self.profile['content_selectors']}")
        body = soup.body
        if body is None:
            raise ExportError(f"{page_rel}: page has no <body>")
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()
        # Keep the content's ancestor chain (theme CSS relies on it) but nothing beside it,
        # except scripts and JSON data blocks.
        chain = [content, *[p for p in content.parents if p is not body and p.name != "[document]"]]
        chain = [el for el in chain if body in el.parents or el is body]
        for el in chain:
            parent = el.parent
            if parent is None:
                continue
            for sibling in list(parent.children):
                if sibling is el or not isinstance(sibling, Tag):
                    continue
                if sibling.name in ("script", "template", "style"):
                    continue
                sibling.decompose()
        for selector in self.profile["remove_selectors"]:
            for el in soup.select(selector):
                if el is not content and content not in el.descendants:
                    el.decompose()
        return chain[-1]

    def _drop_scripts(self, soup: BeautifulSoup) -> None:
        src_patterns = [re.compile(p) for p in self.profile["drop_script_patterns"]]
        inline_patterns = [re.compile(p) for p in self.profile["drop_inline_script_patterns"]]
        for selector in self.profile["drop_head_selectors"]:
            for el in soup.select(selector):
                el.decompose()
        for el in soup.find_all(["script", "link"]):
            url = el.get("src") if el.name == "script" else el.get("href")
            if url:
                path = jsscan.split_url(url)[0]
                if any(p.search(path) for p in src_patterns):
                    el.decompose()
            elif el.name == "script" and el.string and any(p.search(el.string) for p in inline_patterns):
                el.decompose()

    def _page_flat(self, page_id: str, lang: str) -> str:
        return f"{page_id}{self.ed.filename_suffix[lang]}.html"

    def _scan_inline_js(self, page: PageState, el: Tag, is_json: bool) -> None:
        text = el.string or ""
        try:
            refs, dynamic = jsscan.js_references(text)
        except jsscan.LexError as exc:
            self._issue("lex_errors", file=page.rel, error=f"inline script: {exc}")
            return
        for raw in dynamic:
            self._issue("dynamic_resource_sites", file=page.rel, expression=raw)
        replacements: list[tuple[jsscan.JsRef, str, str]] = []  # ref, target, mode
        for ref in refs:
            if ref.kind == "import" and not is_json:
                if ref.value.startswith(("./", "../", "/")):
                    rel, _, reason = self.site.resolve(page.dir, ref.value)
                    if rel:
                        replacements.append((ref, rel, "import"))
                    else:
                        self._issue("unresolved", file=page.rel, ref=ref.value, reason=reason)
                elif not jsscan.is_external(ref.value):
                    page.inline_bare.add(ref.value)
                continue
            targets = self._literal_targets(ref.value, [page.dir])
            if len(targets) == 1:
                target = targets.pop()
                if ref.kind == "object-key":
                    self._issue("object_key_literals", file=page.rel, ref=ref.value)
                    continue
                replacements.append((ref, target, "json" if is_json else "literal"))
        for _, target, _ in replacements:
            page.targets.add(target)
        page.actions.append(("inline-js", el, text, replacements))

    def _scan_css_text(self, page: PageState, text: str) -> list[tuple[jsscan.CssRef, str]]:
        found = []
        for ref in jsscan.css_references(text):
            rel, _, reason = self.site.resolve(page.dir, ref.value)
            if rel:
                found.append((ref, rel))
                page.embeds.add(rel)
            elif reason not in ("external", "self"):
                self._issue("unresolved", file=page.rel, ref=ref.value, reason=reason)
        return found

    def _link_target(self, page: PageState, href: str) -> tuple[str | None, str]:
        """Map an href to a flat page name; returns (flat or None, status)."""
        path, suffix = jsscan.split_url(href)
        rel = self.site.join(page.dir, path)
        if rel and (self.site.root / rel).is_dir():
            rel = posixpath.join(rel, "index.html")
        rel = self.m.page_aliases.get(rel, rel)
        hit = self.page_by_rel.get(rel)
        if hit:
            return hit.flat + suffix, "page"
        for _, _, p in self.m.pages():
            for lang, prel in p.paths.items():
                if prel == rel:
                    return None, "other-language"
        return None, "unmapped"

    def _scan_page(self, page: PageState) -> None:
        soup = page.soup
        for el in list(soup.find_all(True)):
            if el.parent is None and el.name != "html":
                continue
            name = el.name
            if name == "script":
                stype = (el.get("type") or "").strip().lower()
                if el.get("src"):
                    rel, _, reason = self.site.resolve(page.dir, el["src"])
                    if rel:
                        page.targets.add(rel)
                        page.actions.append(("write", el, {"src": rel}))
                    elif reason != "external":
                        self._issue("unresolved", file=page.rel, ref=el["src"], reason=reason)
                    continue
                if stype == "importmap":
                    try:
                        imap = json.loads(el.string or "{}")
                    except ValueError as exc:
                        raise ExportError(f"{page.rel}: invalid importmap: {exc}") from exc
                    if imap.get("scopes"):
                        self.report["notes"].append(f"{page.rel}: importmap scopes are not supported")
                    for spec, target in (imap.get("imports") or {}).items():
                        tpath = self.site.join(page.dir, jsscan.split_url(target)[0])
                        if spec.endswith("/"):
                            page.importmap_prefix.append((spec, tpath))
                        elif self.site.check(tpath) is None:
                            page.importmap_exact[spec] = tpath
                        else:
                            self._issue("unresolved", file=page.rel, ref=target, reason="importmap target")
                    el.decompose()
                    continue
                if stype in JS_TYPES:
                    self._scan_inline_js(page, el, is_json=False)
                elif stype in JSON_SCRIPT_TYPES:
                    self._scan_inline_js(page, el, is_json=True)
                continue
            if name == "style":
                refs = self._scan_css_text(page, el.string or "")
                if refs:
                    page.actions.append(("inline-css", el, None, refs))
                continue
            if el.get("style"):
                refs = self._scan_css_text(page, el["style"])
                if refs:
                    page.actions.append(("style-attr", el, None, refs))
            if name == "link":
                href = el.get("href")
                if href:
                    rel, _, reason = self.site.resolve(page.dir, href)
                    if rel:
                        page.targets.add(rel)
                        page.actions.append(("write", el, {"href": rel}))
                    elif reason != "external":
                        self._issue("unresolved", file=page.rel, ref=href, reason=reason)
                continue
            updates: dict[str, tuple[str, str]] = {}
            for attr in list(el.attrs):
                value = el.attrs[attr]
                if isinstance(value, list):
                    continue
                if attr in ("srcset", "imagesrcset"):
                    parts = []
                    ok = True
                    for entry in [e.strip() for e in value.split(",") if e.strip()]:
                        bits = entry.split(None, 1)
                        rel, suffix, reason = self.site.resolve(page.dir, bits[0])
                        if rel:
                            page.targets.add(rel)
                            parts.append((rel, suffix, bits[1] if len(bits) > 1 else ""))
                        elif reason == "external":
                            parts.append((None, bits[0], bits[1] if len(bits) > 1 else ""))
                        else:
                            ok = False
                            self._issue("unresolved", file=page.rel, ref=bits[0], reason=reason)
                    if ok and any(p[0] for p in parts):
                        updates[attr] = ("srcset", json.dumps(parts))
                    continue
                is_data = attr.startswith("data-") and not attr.startswith("data-lms-")
                if attr not in URL_ATTRS and not is_data:
                    continue
                if name in ("a", "area") and attr == "href":
                    if jsscan.is_external(value):
                        if self.m.external_link_target and re.match(r"^https?://", value, re.I) and not el.get("target"):
                            el["target"] = self.m.external_link_target
                            rel_attr = set((el.get("rel") or []) if isinstance(el.get("rel"), list) else str(el.get("rel") or "").split())
                            el["rel"] = " ".join(sorted(rel_attr | {"noopener", "noreferrer"}))
                        continue
                    path = jsscan.split_url(value)[0]
                    if not path:
                        continue
                    if path.lower().endswith((".html", ".htm")) or path.endswith("/"):
                        flat, status = self._link_target(page, value)
                        if flat:
                            updates[attr] = ("page", flat)
                        else:
                            bucket = "dropped_cross_language_links" if status == "other-language" else "unmapped_page_links"
                            self._issue(bucket, file=page.rel, href=value)
                            if self.m.unmapped_page_links == "error":
                                raise ExportError(f"{page.rel}: link to {value} is not in edition {self.ed.id}")
                            if self.m.unmapped_page_links == "unlink" or status == "other-language":
                                del el.attrs["href"]
                        continue
                if is_data and not jsscan.looks_like_path(value):
                    continue
                rel, suffix, reason = self.site.resolve(page.dir, value)
                if rel:
                    if rel in self.page_by_rel:
                        updates[attr] = ("page", self.page_by_rel[rel].flat + suffix)
                    else:
                        page.targets.add(rel)
                        updates[attr] = ("asset", json.dumps([rel, suffix]))
                elif reason not in ("external", "self") and (not is_data or jsscan.has_asset_extension(value)):
                    self._issue("unresolved", file=page.rel, ref=value, reason=reason)
            if updates:
                page.actions.append(("attrs", el, updates))

    # ------------------------------------------------------------------ naming / rendering
    def _key(self, rel: str, visiting: frozenset[str] = frozenset()) -> str:
        if rel in self._keys:
            return self._keys[rel]
        if rel in visiting:
            return hashlib.sha256(("cycle:" + rel).encode()).hexdigest()
        node = self.nodes[rel]
        h = hashlib.sha256(node.data)
        h.update(node.kind.encode())
        visiting = visiting | {rel}
        for i in sorted(node.resolved):
            targets = node.resolved[i]
            if len(targets) == 1:
                h.update(f"|{i}:".encode() + self._key(next(iter(targets)), visiting).encode())
        if node.text is not None and node.kind == "css" and self.profile["prune_font_formats"]:
            h.update(b"|pruned-fonts")
        key = h.hexdigest()
        self._keys[rel] = key
        return key

    def name(self, rel: str) -> str:
        if rel in self._names:
            return self._names[rel]
        base = posixpath.basename(rel)
        stem, ext = posixpath.splitext(base)
        if any(fnmatch.fnmatch(rel, g) for g in self.m.keep_basename):
            flat = base
        else:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "asset"
            flat = f"{safe}-{self._key(rel)[:10]}{ext.lower()}"
        self._names[rel] = flat
        return flat

    def _data_uri(self, rel: str) -> str:
        data = self.render(rel)
        mime = mime_for(rel) or "application/octet-stream"
        if len(data) > 512 * 1024:
            self._issue("large_inline", file=rel, bytes=len(data))
        if len(data) > self.profile["inline_css_max_bytes"]:
            raise ExportError(f"{rel} is too large to embed ({len(data)} bytes); raise inline_css_max_bytes")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _css_replace(self, text: str, refs: list[tuple[jsscan.CssRef, str]]) -> str:
        out, pos = [], 0
        for ref, target in sorted(refs, key=lambda r: r[0].start):
            out.append(text[pos:ref.start])
            uri = self._data_uri(target)
            # A base64 data: URL has no quotes, spaces or parentheses, so it is valid unquoted
            # inside url(...) and inside quoted HTML/SVG attributes alike.
            out.append(uri if ref.kind == "href" else f"url({uri})")
            pos = ref.end
        out.append(text[pos:])
        return "".join(out)

    def _js_replace(self, text: str, items: list[tuple[jsscan.JsRef, str, str]], where: str) -> str:
        out, pos = [], 0
        count = 0
        for ref, target, mode in sorted(items, key=lambda r: r[0].start):
            flat = self.name(target)
            suffix = jsscan.split_url(ref.value)[1]
            if mode == "import":
                new = ref.quote + "@lms/" + flat + ref.quote
            elif mode == "json":
                new = json.dumps(flat + suffix)
                self._issue("json_relative_refs", file=where, ref=ref.value)
            else:
                new = "__lmsAsset(" + json.dumps(flat + suffix) + ")"
                count += 1
            out.append(text[pos:ref.start])
            out.append(new)
            pos = ref.end
        out.append(text[pos:])
        if count:
            self.report["js_literal_rewrites"][where] = self.report["js_literal_rewrites"].get(where, 0) + count
        return "".join(out)

    def render(self, rel: str) -> bytes:
        if rel in self._render_cache:
            return self._render_cache[rel]
        self._render_cache[rel] = self.nodes[rel].data  # cycle guard
        node = self.nodes[rel]
        if node.text is None or node.kind == "bin":
            result = node.data
        elif node.kind in ("css", "svg"):
            refs = [(ref, next(iter(node.resolved[i]))) for i, ref in enumerate(node.css_refs)
                    if len(node.resolved.get(i, ())) == 1]
            text = self._css_replace(node.text, refs)
            result = (jsscan.strip_source_maps(text) if node.kind == "css" else text).encode("utf-8")
        else:
            items = []
            for i, ref in enumerate(node.js_refs):
                targets = node.resolved.get(i, set())
                if len(targets) > 1:
                    self._issue("ambiguous", file=rel, ref=ref.value, targets=sorted(targets))
                    continue
                if len(targets) != 1:
                    continue
                target = next(iter(targets))
                if ref.kind == "object-key":
                    self._issue("object_key_literals", file=rel, ref=ref.value)
                    continue
                if node.kind == "json":
                    mode = "json"
                elif ref.kind == "import":
                    mode = "import"
                else:
                    mode = "literal"
                items.append((ref, target, mode))
            text = self._js_replace(node.text, items, rel)
            result = (jsscan.strip_source_maps(text) if node.kind == "js" else text).encode("utf-8")
        self._render_cache[rel] = result
        return result

    def _copy_edges(self, node: Node, page_key: str | None) -> set[str]:
        """Assets that must exist as separate files (not embedded) for this node."""
        if node.kind in ("css", "svg"):
            return set()
        out = set()
        for i, targets in node.resolved.items():
            if len(targets) == 1 and node.js_refs[i].kind != "object-key":
                out |= targets
        for per_page in node.bare.values():
            if page_key is None:
                out |= {t for t in per_page.values() if t}
            elif per_page.get(page_key):
                out.add(per_page[page_key])
        return out

    def _closure(self, roots: set[str], page_key: str | None) -> set[str]:
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            rel = stack.pop()
            if rel in seen:
                continue
            seen.add(rel)
            stack.extend(self._copy_edges(self.nodes[rel], page_key) - seen)
        return seen

    # ------------------------------------------------------------------ main
    def run(self) -> tuple[dict[str, bytes], dict[str, Any]]:
        ed = self.ed
        for _, section, page in self.m.pages():
            for lang in ed.languages:
                rel = page.paths[lang]
                reason = self.site.check(rel)
                if reason:
                    raise ExportError(f"page {page.id} [{lang}] {rel}: {reason}")
                key = f"{lang}:{page.id}"
                html = self.site.path(rel).read_text(encoding="utf-8")
                state = PageState(key, page.id, lang, rel, self._page_flat(page.id, lang),
                                  BeautifulSoup(html, "html.parser"))
                self.pages[key] = state
                self.page_by_rel[rel] = state

        roots: list[tuple[str, str | None, str | None]] = []
        for state in self.pages.values():
            self._drop_scripts(state.soup)
            state.chain_top = self._strip_chrome(state.soup, state.rel)
            self._scan_page(state)
            for spec in sorted(state.inline_bare):
                target = self._resolve_bare(state, spec)
                if target:
                    state.targets.add(target)
                else:
                    self._issue("unmapped_bare_specifiers", file=state.rel, specifier=spec, page=state.rel)
            roots.extend((t, state.key, None) for t in sorted(state.targets | state.embeds))
        extra = []
        for pattern in self.m.extra_assets:
            matches = [p.relative_to(self.site.root).as_posix() for p in sorted(self.site.root.glob(pattern)) if p.is_file()]
            if not matches:
                raise ExportError(f"extra_assets entry matched nothing: {pattern}")
            for rel in matches:
                reason = self.site.check(rel)
                if reason:
                    raise ExportError(f"extra asset {rel}: {reason}")
                extra.append(rel)
                for key in self.pages:
                    roots.append((rel, key, None))
        self._walk(roots)

        files: dict[str, bytes] = {}
        sources: dict[str, set[str]] = {}
        all_assets: set[str] = set(extra)
        page_closures: dict[str, set[str]] = {}
        for state in self.pages.values():
            closure = self._closure(state.targets | set(extra), state.key)
            page_closures[state.key] = closure
            all_assets |= closure
        page_names = {s.flat for s in self.pages.values()}
        for rel in sorted(all_assets):
            flat = self.name(rel)
            if flat in page_names:
                raise ExportError(f"asset name {flat} collides with a page")
            data = self.render(rel)
            if flat in files and files[flat] != data:
                raise ExportError(f"two different assets map to {flat}: {sorted(sources[flat])} and {rel}")
            files[flat] = data
            sources.setdefault(flat, set()).add(rel)
            if not mime_for(flat):
                self._issue("unknown_mime", file=flat)

        page_meta = []
        for state in self.pages.values():
            html = self._render_page(state, page_closures[state.key])
            files[state.flat] = html.encode("utf-8")
            sources.setdefault(state.flat, set()).add(state.rel)
            page_meta.append({"key": state.key, "flat": state.flat, "source": state.rel})

        for name in files:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
                raise ExportError(f"unsafe static file name {name!r}")
        self.report["pages"] = page_meta
        self.report["assets"] = {flat: sorted(src) for flat, src in sorted(sources.items())}
        self.report["external_symlinks"] = sorted(self.site.external_symlinks)
        for i, rule in enumerate(self.m.rewrites):
            if str(i) not in self.report["rewrites_applied"]:
                self.report["notes"].append(f"rewrite rule {i} matched nothing in this edition")
        return files, self.report

    def _render_page(self, page: PageState, closure: set[str]) -> str:
        soup = page.soup
        ed = self.ed
        lang_spec = self.m.languages[page.lang]
        page_obj = next(p for _, _, p in self.m.pages() if p.id == page.page_id)
        course_title = text_for(ed.course["title"], page.lang, "course title")
        page_title = page_obj.title(page.lang)

        for action in page.actions:
            kind, el = action[0], action[1]
            if kind == "write":
                attr, rel = next(iter(action[2].items()))
                suffix = jsscan.split_url(el[attr])[1]
                attrs: list[tuple[str, str | None]] = []
                for k, v in el.attrs.items():
                    v = " ".join(v) if isinstance(v, list) else v
                    if k == attr:
                        v = self.name(rel) + suffix
                    attrs.append((k, v))
                script = soup.new_tag("script")
                script.string = runtime.write_call(el.name, attrs, [attr])
                el.replace_with(script)
            elif kind == "inline-js":
                _, _, text, replacements = action
                el.string = self._js_replace(text, replacements, page.rel)
            elif kind == "inline-css":
                el.string = self._css_replace(el.string or "", action[3])
            elif kind == "style-attr":
                el["style"] = self._css_replace(el["style"], action[3])
            elif kind == "attrs":
                lms: dict[str, str] = {}
                for attr, (mode, payload) in action[2].items():
                    if mode == "page":
                        value = payload
                    elif mode == "asset":
                        rel, suffix = json.loads(payload)
                        value = self.name(rel) + suffix
                    else:
                        value = ", ".join(
                            " ".join(x for x in ((self.name(r) + s) if r else s, d) if x)
                            for r, s, d in json.loads(payload)
                        )
                    lms[attr] = value
                    if attr in FETCHING_ATTRS or (attr == "href" and el.name not in ("a", "area")):
                        del el.attrs[attr]
                    else:
                        el[attr] = value
                el["data-lms-url"] = json.dumps(lms, ensure_ascii=False, separators=(",", ":"))

        html_el = soup.html
        html_lang = lang_spec.get("html_lang")
        if html_lang:
            html_el["lang"] = html_lang
        for k, v in self.profile["html_attributes"].items():
            html_el[k] = v
        body = soup.body
        for k, v in self.profile["body_attributes"].items():
            body[k] = v
        head = soup.head
        if head is None:
            head = soup.new_tag("head")
            html_el.insert(0, head)
        seen_viewport = False
        for meta in head.find_all("meta"):
            if meta.get("name") == "viewport":
                if seen_viewport:
                    meta.decompose()
                seen_viewport = True
            elif meta.get("name") == "color-scheme":
                meta["content"] = "light"
        title = head.find("title") or soup.new_tag("title")
        title.string = self.profile["page_title"].format(page=page_title, course=course_title)
        if title.parent is None:
            head.append(title)

        # Runtime: import map entries for every module the page can reach.
        imports: dict[str, str] = {}
        for rel in sorted(closure):
            if self.nodes[rel].kind == "js":
                imports["@lms/" + self.name(rel)] = self.name(rel)
        for rel in sorted(closure):
            node = self.nodes[rel]
            for i, per_page in node.bare.items():
                target = per_page.get(page.key)
                if target:
                    imports[node.js_refs[i].value] = self.name(target)
        for spec in sorted(page.inline_bare):
            target = self._resolve_bare(page, spec)
            if target:
                imports[spec] = self.name(target)
        config: dict[str, Any] = {"imports": imports}
        if ed.switch:
            config["lang"] = {
                "key": ed.course_key, "cur": page.lang, "isDefault": page.lang == ed.default,
                "remember": ed.remember,
                "pages": {lang: self._page_flat(page.page_id, lang) for lang in ed.languages},
            }
        boot = soup.new_tag("script")
        boot["data-lms-export"] = "bootstrap"
        boot.string = runtime.bootstrap(config)
        charset = head.find("meta", attrs={"charset": True})
        if charset:
            charset.insert_after(boot)
        else:
            head.insert(0, boot)
        style = soup.new_tag("style")
        style["data-lms-export"] = "layout"
        style.string = self.profile["css"] + (SWITCH_CSS if ed.switch else "")
        head.append(style)

        if ed.switch:
            nav = soup.new_tag("nav", attrs={"class": "lms-lang-switch", "aria-label": "Language"})
            for lang in ed.languages:
                flat = self._page_flat(page.page_id, lang)
                spec = self.m.languages[lang]
                a = soup.new_tag("a", href=flat)
                a["data-lms-url"] = json.dumps({"href": flat})
                a["data-lms-lang"] = lang
                if spec.get("html_lang"):
                    a["lang"] = spec["html_lang"]
                    a["hreflang"] = spec["html_lang"]
                if lang == page.lang:
                    a["aria-current"] = "page"
                a.string = spec["label"]
                nav.append(a)
            body.insert(0, nav)

        fix = soup.new_tag("script")
        fix["data-lms-export"] = "fix"
        fix.string = runtime.FIX_CALL
        page.chain_top.insert_after(fix)
        out = str(soup)
        if not out.lstrip().lower().startswith("<!doctype"):
            out = "<!DOCTYPE html>\n" + out
        return out
