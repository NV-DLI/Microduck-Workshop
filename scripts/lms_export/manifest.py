"""Load and check an LMS export manifest (see the skill's references/manifest.md)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SUFFIX_RE = re.compile(r"^[A-Za-z0-9_.-]*$")
KEY_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ManifestError(ValueError):
    pass


def _text(value: Any, lang: str, where: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get(lang), str):
        return value[lang]
    raise ManifestError(f"{where}: needs a string or a {{language: string}} object with {lang!r}")


@dataclass
class Page:
    id: str
    paths: dict[str, str]
    titles: dict[str, Any]

    def title(self, lang: str) -> str:
        return _text(self.titles, lang, f"page {self.id} title")


@dataclass
class Section:
    id: str
    titles: Any
    pages: list[Page]


@dataclass
class Chapter:
    id: str
    titles: Any
    sections: list[Section]


@dataclass
class Edition:
    id: str
    archive: str
    languages: list[str]
    default: str
    switch: bool
    remember: bool
    course: dict[str, Any]
    filename_suffix: dict[str, str]

    @property
    def course_key(self) -> str:
        return f"{self.course['org']}+{self.course['course']}+{self.course['run']}"


@dataclass
class Manifest:
    path: Path
    raw: dict[str, Any]
    languages: dict[str, dict[str, Any]]
    chapters: list[Chapter]
    editions: list[Edition]
    extract: dict[str, Any]
    extra_assets: list[str] = field(default_factory=list)
    rewrites: list[dict[str, Any]] = field(default_factory=list)
    keep_basename: list[str] = field(default_factory=list)
    page_aliases: dict[str, str] = field(default_factory=dict)
    unmapped_page_links: str = "unlink"
    external_link_target: str | None = "_blank"
    allow_external_symlinks: bool = False

    def pages(self) -> list[tuple[Chapter, Section, Page]]:
        return [(c, s, p) for c in self.chapters for s in c.sections for p in s.pages]

    def edition(self, edition_id: str) -> Edition:
        for ed in self.editions:
            if ed.id == edition_id:
                return ed
        raise ManifestError(f"unknown edition {edition_id!r}")


def _require(obj: dict[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise ManifestError(f"{where}: missing {key!r}")
    return obj[key]


def _check_rel(path: str, where: str) -> str:
    if not isinstance(path, str) or not path or path.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", path):
        raise ManifestError(f"{where}: path must be relative to the site root: {path!r}")
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        raise ManifestError(f"{where}: path may not contain '..': {path!r}")
    return "/".join(p for p in parts if p not in ("", "."))


def load_manifest(path: Path) -> Manifest:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be a JSON object")
    languages = _require(raw, "languages", "manifest")
    if not isinstance(languages, dict) or not languages:
        raise ManifestError("languages must be a non-empty object")
    for code, spec in languages.items():
        if not ID_RE.match(code) or not isinstance(spec, dict) or not isinstance(spec.get("label"), str):
            raise ManifestError(f"language {code!r} needs a label")

    chapters: list[Chapter] = []
    seen_pages: set[str] = set()
    seen_ids: set[str] = set()
    chapter_ids: set[str] = set()
    outline = _require(raw, "outline", "manifest")
    if isinstance(outline, dict):
        outline = [outline]
    for ci, ch in enumerate(outline):
        cid = _require(ch, "id", f"outline[{ci}]")
        if not isinstance(cid, str) or not ID_RE.match(cid) or cid in chapter_ids:
            raise ManifestError(f"chapter id {cid!r} must be unique and match {ID_RE.pattern}")
        chapter_ids.add(cid)
        sections = []
        for si, sec in enumerate(_require(ch, "sections", f"chapter {cid}")):
            sid = _require(sec, "id", f"chapter {cid} section {si}")
            pages = []
            for pi, pg in enumerate(_require(sec, "pages", f"section {sid}")):
                pid = _require(pg, "id", f"section {sid} page {pi}")
                if not ID_RE.match(pid):
                    raise ManifestError(f"page id {pid!r} must match {ID_RE.pattern}")
                if pid in seen_pages:
                    raise ManifestError(f"duplicate page id {pid!r}")
                seen_pages.add(pid)
                paths = {
                    lang: _check_rel(p, f"page {pid}")
                    for lang, p in _require(pg, "paths", f"page {pid}").items()
                }
                for lang in paths:
                    if lang not in languages:
                        raise ManifestError(f"page {pid}: unknown language {lang!r}")
                pages.append(Page(pid, paths, _require(pg, "title", f"page {pid}")))
            for ident in (sid,):
                if ident in seen_ids:
                    raise ManifestError(f"duplicate section id {ident!r}")
                seen_ids.add(ident)
            sections.append(Section(sid, _require(sec, "title", f"section {sid}"), pages))
        chapters.append(Chapter(cid, _require(ch, "title", f"chapter {cid}"), sections))

    editions: list[Edition] = []
    keys: set[str] = set()
    edition_ids: set[str] = set()
    archives: set[str] = set()
    for ei, ed in enumerate(_require(raw, "editions", "manifest")):
        eid = _require(ed, "id", f"editions[{ei}]")
        if not isinstance(eid, str) or not ID_RE.match(eid) or eid in edition_ids:
            raise ManifestError(f"edition id {eid!r} must be unique and match {ID_RE.pattern}")
        edition_ids.add(eid)
        langs = _require(ed, "languages", f"edition {eid}")
        default = ed.get("default", langs[0])
        if default not in langs or any(l not in languages for l in langs):
            raise ManifestError(f"edition {eid}: languages/default must be declared languages")
        course = dict(_require(ed, "course", f"edition {eid}"))
        for k in ("org", "course", "run"):
            v = _require(course, k, f"edition {eid} course")
            if not isinstance(v, str) or not KEY_PART_RE.match(v):
                raise ManifestError(f"edition {eid}: course.{k} must match {KEY_PART_RE.pattern}")
        _require(course, "title", f"edition {eid} course")
        _require(course, "language", f"edition {eid} course")
        key = f"{course['org']}+{course['course']}+{course['run']}"
        if key in keys:
            raise ManifestError(f"editions share course key {key}; each LMS course needs its own")
        keys.add(key)
        archive = ed.get("archive", f"{eid}.tar.gz")
        if "/" in archive or "\\" in archive or not archive.endswith(".tar.gz"):
            raise ManifestError(f"edition {eid}: archive must be a plain *.tar.gz file name")
        suffix = {l: ("" if l == default else f"-{l}") for l in langs}
        suffix.update(ed.get("filename_suffix", {}))
        for l, sfx in suffix.items():
            if not isinstance(sfx, str) or not SUFFIX_RE.match(sfx):
                raise ManifestError(f"edition {eid}: filename_suffix[{l!r}]={sfx!r} must match {SUFFIX_RE.pattern}")
        if archive in archives:
            raise ManifestError(f"edition {eid}: archive name {archive} is used twice")
        archives.add(archive)
        if len(set(suffix[l] for l in langs)) != len(langs):
            raise ManifestError(f"edition {eid}: filename_suffix values must differ per language")
        for _, _, page in [(c, s, p) for c in chapters for s in c.sections for p in s.pages]:
            for lang in langs:
                if lang not in page.paths:
                    raise ManifestError(f"edition {eid}: page {page.id} has no {lang!r} path")
        editions.append(
            Edition(
                id=eid,
                archive=archive,
                languages=list(langs),
                default=default,
                switch=bool(ed.get("switch", len(langs) > 1)) and len(langs) > 1,
                remember=bool(ed.get("remember_language", False)),
                course=course,
                filename_suffix=suffix,
            )
        )

    return Manifest(
        path=path,
        raw=raw,
        languages=languages,
        chapters=chapters,
        editions=editions,
        extract=dict(raw.get("extract", {})),
        extra_assets=[_check_rel(p, "extra_assets") for p in raw.get("extra_assets", [])],
        rewrites=list(raw.get("rewrites", [])),
        keep_basename=list(raw.get("keep_basename", [])),
        page_aliases={
            _check_rel(k, "page_aliases"): _check_rel(v, "page_aliases")
            for k, v in raw.get("page_aliases", {}).items()
        },
        unmapped_page_links=raw.get("unmapped_page_links", "unlink"),
        external_link_target=raw.get("external_link_target", "_blank"),
        allow_external_symlinks=bool(raw.get("allow_external_symlinks", False)),
    )


def text_for(value: Any, lang: str, where: str) -> str:
    return _text(value, lang, where)
