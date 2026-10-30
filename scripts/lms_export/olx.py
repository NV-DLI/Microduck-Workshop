"""Write a new Open edX OLX course tree and archive; read archives safely."""

from __future__ import annotations

import gzip
import html
import io
import json
import os
import stat
import tarfile
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

from .exporter import mime_for
from .manifest import Edition, Manifest, text_for

# Namespace for this exporter's deterministic block ids (uuid5 of course key + kind + id).
ID_NAMESPACE = uuid.UUID("6f1d3a52-7c0e-5b8e-9a41-2d9e3c5b7a10")

POLICY_BLOCKLIST = {
    "certificates", "cert_html_view_enabled", "cert_html_view_overrides", "certificates_display_behavior",
    "minimum_grade_credit", "credit_requirements", "advanced_modules", "entrance_exam_enabled",
    "entrance_exam_id", "is_proctored_enabled", "enable_proctored_exams", "enable_timed_exams",
}


def block_id(edition: Edition, kind: str, ident: str) -> str:
    return uuid.uuid5(ID_NAMESPACE, f"{edition.course_key}|{kind}|{ident}").hex


def _xml(element: ET.Element) -> bytes:
    ET.indent(element, space="  ")
    return ET.tostring(element, encoding="utf-8", xml_declaration=False) + b"\n"


def iframe_html(page_flat: str, title: str) -> str:
    return (
        f'<iframe src="/static/{html.escape(page_flat)}"\n'
        '        style="width:100%; height:90vh; border:0; display:block;"\n'
        '        loading="lazy"\n'
        f'        title="{html.escape(title)}"></iframe>\n'
    )


def asset_entry(edition: Edition, name: str) -> dict[str, Any]:
    c = edition.course
    return {
        "contentType": mime_for(name) or "application/octet-stream",
        "content_son": {
            "category": "asset", "course": c["course"], "name": name, "org": c["org"],
            "revision": None, "run": c["run"], "tag": "c4x",
        },
        "displayname": name,
        "filename": f"asset-v1:{c['org']}+{c['course']}+{c['run']}+type@asset+block@{name}",
        "import_path": name,
        "locked": False,
        "thumbnail_location": None,
    }


def build_tree(manifest: Manifest, edition: Edition, static: dict[str, bytes],
               page_flat: dict[tuple[str, str], str]) -> dict[str, bytes]:
    """Return {path under course/: bytes} for a complete OLX course."""
    c = edition.course
    lang = edition.default
    run = c["run"]
    title = text_for(c["title"], lang, "course title")
    files: dict[str, bytes] = {}

    root = ET.Element("course", {"url_name": run, "org": c["org"], "course": c["course"]})
    files["course.xml"] = _xml(root)

    course_attrs = {
        "display_name": title,
        "language": c["language"],
        "cert_html_view_enabled": "false",
    }
    course_el = ET.Element("course", course_attrs)

    for chapter in manifest.chapters:
        ch_id = block_id(edition, "chapter", chapter.id)
        ET.SubElement(course_el, "chapter", {"url_name": ch_id})
        ch_el = ET.Element("chapter", {"display_name": text_for(chapter.titles, lang, f"chapter {chapter.id}")})
        for section in chapter.sections:
            seq_id = block_id(edition, "sequential", section.id)
            ET.SubElement(ch_el, "sequential", {"url_name": seq_id})
            seq_el = ET.Element("sequential", {"display_name": text_for(section.titles, lang, f"section {section.id}")})
            for page in section.pages:
                v_id = block_id(edition, "vertical", page.id)
                h_id = block_id(edition, "html", page.id)
                ET.SubElement(seq_el, "vertical", {"url_name": v_id})
                page_title = page.title(lang)
                v_el = ET.Element("vertical", {"display_name": page_title})
                ET.SubElement(v_el, "html", {"url_name": h_id})
                files[f"vertical/{v_id}.xml"] = _xml(v_el)
                files[f"html/{h_id}.xml"] = _xml(
                    ET.Element("html", {"filename": h_id, "display_name": "Raw HTML", "editor": "raw"})
                )
                files[f"html/{h_id}.html"] = iframe_html(page_flat[(page.id, lang)], page_title).encode("utf-8")
            files[f"sequential/{seq_id}.xml"] = _xml(seq_el)
        files[f"chapter/{ch_id}.xml"] = _xml(ch_el)
    files[f"course/{run}.xml"] = _xml(course_el)

    policy_extra = dict(c.get("policy", {}))
    bad = sorted(set(policy_extra) & POLICY_BLOCKLIST)
    if bad:
        raise ValueError(f"edition {edition.id}: course.policy may not set {bad}")
    policy: dict[str, Any] = {
        "advanced_modules": [],
        "cert_html_view_enabled": False,
        "display_name": title,
        "language": c["language"],
        "tabs": [
            {"course_staff_only": False, "name": "Course", "type": "courseware"},
            {"course_staff_only": False, "name": "Progress", "type": "progress"},
            {"course_staff_only": False, "name": "Dates", "type": "dates"},
        ],
        "xml_attributes": {"filename": [f"course/{run}.xml", f"course/{run}.xml"]},
    }
    if c.get("start"):
        policy["start"] = c["start"]
    policy.update(policy_extra)
    files[f"policies/{run}/policy.json"] = (json.dumps({f"course/{run}": policy}, ensure_ascii=False, indent=4, sort_keys=True) + "\n").encode("utf-8")
    grading = c.get("grading", {"GRADER": [], "GRADE_CUTOFFS": {"Pass": 0.5}})
    files[f"policies/{run}/grading_policy.json"] = (json.dumps(grading, indent=4, sort_keys=True) + "\n").encode("utf-8")
    assets = {name: asset_entry(edition, name) for name in sorted(static)}
    files["policies/assets.json"] = (json.dumps(assets, ensure_ascii=False, indent=4, sort_keys=True) + "\n").encode("utf-8")
    files["assets/assets.xml"] = b"<assets/>\n"

    about = c.get("about", {})
    short = about.get("short_description", "")
    short = text_for(short, lang, "about.short_description") if short else ""
    overview = about.get("overview_html")
    if overview:
        overview = text_for(overview, lang, "about.overview_html")
        if overview.endswith(".html") and "\n" not in overview:
            overview = (manifest.path.parent / overview).read_text(encoding="utf-8")
    else:
        overview = (
            '<section class="about">\n'
            f"  <h2>{html.escape(title)}</h2>\n"
            + (f"  <p>{html.escape(short)}</p>\n" if short else "")
            + "</section>\n"
        )
    files["about/title.html"] = (title + "\n").encode("utf-8")
    files["about/overview.html"] = overview.encode("utf-8")
    files["about/short_description.html"] = short.encode("utf-8")
    for key in ("effort", "duration"):
        if about.get(key):
            files[f"about/{key}.html"] = str(about[key]).encode("utf-8")
    files["info/updates.html"] = b"<ol></ol>\n"
    for name, data in static.items():
        files[f"static/{name}"] = data
    return files


def write_archive(files: dict[str, bytes], output: Path, mtime: int = 0) -> None:
    """Deterministic tar.gz with a single top-level course/ directory."""
    dirs = set()
    for path in files:
        parts = PurePosixPath(path).parts[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        def add_dir(name: str) -> None:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.mtime = mtime
            tf.addfile(info)

        add_dir("course")
        for d in sorted(dirs):
            add_dir(f"course/{d}")
        for path in sorted(files):
            info = tarfile.TarInfo(f"course/{path}")
            info.size = len(files[path])
            info.mode = 0o644
            info.mtime = mtime
            tf.addfile(info, io.BytesIO(files[path]))
    # A uniquely named temporary file in the target directory (created with O_EXCL, so an
    # existing file or symlink is never followed), then an atomic rename. Only that file is
    # removed on failure.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=mtime, compresslevel=9) as gz:
                gz.write(buf.getvalue())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, output)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


class UnsafeArchive(ValueError):
    pass


def read_archive(path: Path, max_bytes: int = 8 * 1024**3) -> dict[str, bytes]:
    """Read regular files from a tar archive without extracting; reject unsafe members."""
    files: dict[str, bytes] = {}
    total = 0
    with tarfile.open(path, "r:*") as tf:
        for member in tf:
            name = member.name
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or (p.parts and ":" in p.parts[0]):
                raise UnsafeArchive(f"unsafe path in archive: {name!r}")
            if member.issym() or member.islnk():
                raise UnsafeArchive(f"link member in archive: {name!r}")
            if member.isdir():
                continue
            if not member.isfile():
                raise UnsafeArchive(f"special member in archive: {name!r}")
            if member.mode & (stat.S_ISUID | stat.S_ISGID):
                raise UnsafeArchive(f"setuid/setgid member: {name!r}")
            total += member.size
            if total > max_bytes:
                raise UnsafeArchive("archive is larger than the configured limit")
            norm = "/".join(x for x in p.parts if x not in ("", "."))
            if norm in files:
                raise UnsafeArchive(f"duplicate member: {name!r}")
            files[norm] = tf.extractfile(member).read()
    return files
