#!/usr/bin/env python3
"""Exporter regression tests. Usage: regression_tests.py SCRIPTS_DIR [RESULTS_JSON]

SCRIPTS_DIR contains the lms_export package (repository scripts/ or the skill's scripts/).
Every case runs in its own temporary directory; nothing outside it is touched.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))

from lms_export import jsscan  # noqa: E402
from lms_export.exporter import EditionExport, Site  # noqa: E402
from lms_export.manifest import ManifestError, load_manifest  # noqa: E402
from lms_export.olx import build_tree, write_archive  # noqa: E402
from lms_export.validate import validate  # noqa: E402

BASE = {
    "languages": {"en": {"label": "English", "html_lang": "en"}},
    "extract": {"profile": "basic"},
    "outline": [{"id": "ch", "title": "Ch", "sections": [{"id": "sec", "title": "Sec", "pages": [
        {"id": "page", "title": "Page", "paths": {"en": "en/page.html"}}]}]}],
    "editions": [{"id": "en", "languages": ["en"], "course": {
        "org": "ORG", "course": "COURSE", "run": "RUN", "title": "Course", "language": "en",
        "start": "2026-01-01T00:00:00Z"}}],
}
results = []


def site(td: Path, body: str, files: dict, raw: dict | None = None):
    (td / "en").mkdir(exist_ok=True)
    (td / "en/page.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"></head><body data-lang="en"><main>'
        + body + "</main></body></html>")
    for rel, data in files.items():
        p = td / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
    (td / "m.json").write_text(json.dumps(raw or BASE))
    m = load_manifest(td / "m.json")
    return m, EditionExport(Site(td), m, m.editions[0])


def check(name: str, ok: bool, detail: object = "") -> None:
    results.append({"case": name, "pass": bool(ok), "detail": detail})


# 1. Asset referenced only from inline style/style= is embedded, not missing.
with tempfile.TemporaryDirectory(prefix="lmsreg-") as d:
    td = Path(d)
    m, ex = site(td, "<p style=\"background-image:url('../img.png')\">x</p><style>.a{background:url(../img2.png)}</style>",
                 {"img.png": b"\x89PNGfake", "img2.png": b"\x89PNGtwo"})
    static, report = ex.run()
    html = static["page.html"].decode()
    check("inline-style-only asset embedded", "data:image/png;base64," in html and html.count("data:image/png") == 2
          and not any(k.startswith("img") for k in static), sorted(static))

# 2. SVG <image href> receives a raw data URI (valid SVG), CSS url() inside SVG stays url().
with tempfile.TemporaryDirectory(prefix="lmsreg-") as d:
    td = Path(d)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
           '<image href="img.png"/><image xlink:href="img.png"/><rect style="fill:url(#g)"/>'
           '<style>.b{background:url(img.png)}</style></svg>')
    m, ex = site(td, '<img src="../drawing.svg">', {"drawing.svg": svg, "img.png": b"\x89PNGfake"})
    static, _ = ex.run()
    out = next(v.decode() for k, v in static.items() if k.endswith(".svg"))
    import xml.etree.ElementTree as ET
    ET.fromstring(out)
    check("svg image href raw data URI", 'href="data:image/png;base64,' in out and 'xlink:href="data:image/png' in out
          and "url(data:image/png" in out and 'href="url(' not in out and "url(#g)" in out, out[:300])

# 3. Duplicate chapter / edition ids, unsafe suffix and duplicate archive names are rejected.
for label, mutate in (
    ("duplicate chapter id", lambda r: r["outline"].append(
        {"id": "ch", "title": "Two", "sections": [{"id": "sec2", "title": "S", "pages": [
            {"id": "page2", "title": "P", "paths": {"en": "en/page.html"}}]}]})),
    ("duplicate edition id", lambda r: r["editions"].append(
        dict(r["editions"][0], course=dict(r["editions"][0]["course"], run="RUN2")))),
    ("unsafe filename suffix", lambda r: r["editions"][0].update(filename_suffix={"en": "/../x"})),
    ("duplicate archive name", lambda r: r["editions"].extend([
        dict(r["editions"].pop(0), archive="same.tar.gz"),
        dict(r["editions"][0] if r["editions"] else BASE["editions"][0], id="en2", archive="same.tar.gz",
             course=dict(BASE["editions"][0]["course"], run="RUN2"))])),
):
    with tempfile.TemporaryDirectory(prefix="lmsreg-") as d:
        raw = json.loads(json.dumps(BASE))
        mutate(raw)
        Path(d, "m.json").write_text(json.dumps(raw))
        try:
            load_manifest(Path(d, "m.json"))
            check(label + " rejected", False, "accepted")
        except ManifestError as exc:
            check(label + " rejected", True, str(exc))

# 4. A pre-existing symlink named like a temp file is never followed; the sentinel survives.
with tempfile.TemporaryDirectory(prefix="lmsreg-") as d:
    td = Path(d)
    victim = td / "existing-user-file.txt"
    victim.write_text("KEEP ORIGINAL")
    target = td / "candidate.tar.gz"
    trap = target.with_name(target.name + ".partial")
    trap.symlink_to(victim)
    out_link = td / "linked.tar.gz"
    other = td / "other.txt"
    other.write_text("OTHER")
    out_link.symlink_to(other)
    write_archive({"course.xml": b"<course/>"}, target)
    write_archive({"course.xml": b"<course/>"}, out_link)
    leftovers = sorted(p.name for p in td.iterdir() if p.name.endswith(".partial") and p != trap)
    check("archive write ignores symlink traps",
          victim.read_text() == "KEEP ORIGINAL" and trap.is_symlink() and not target.is_symlink()
          and target.read_bytes()[:2] == b"\x1f\x8b" and other.read_text() == "OTHER"
          and not out_link.is_symlink() and not leftovers,
          {"victim": victim.read_text(), "leftovers": leftovers})

# 5. Display template strings are not dynamic-resource reports; loader sites are.
_, dyn = jsscan.js_references('console.log(`three.js r${R}`); fetch(`${base}/a.json`); x.src = "./m/" + n;')
check("dynamic site classifier", dyn == ["`${base}/a.json`", '"./m/" + ...'], dyn)

# 6. JSON naming an asset is reported by exporter and validator (asset-key route limitation).
with tempfile.TemporaryDirectory(prefix="lmsreg-") as d:
    td = Path(d)
    m, ex = site(td, '<script>fetch("../meta.json").then(r=>r.json()).then(d=>fetch(d.image));</script>',
                 {"meta.json": '{"image":"../icon.png"}', "icon.png": b"\x89PNG"})
    static, report = ex.run()
    page_flat = {("page", "en"): "page.html"}
    out = td / "out.tar.gz"
    write_archive(build_tree(m, m.editions[0], static, page_flat), out)
    v = validate(out, m, "en")
    check("json asset names reported", bool(report["json_relative_refs"])
          and any("names asset" in w for w in v["warnings"]) and v["ok"], v["warnings"])
    with tempfile.TemporaryDirectory(prefix="lmsreg-out-") as out_dir:
        cli = Path(sys.argv[1]).resolve() / "export_lms.py"
        run = subprocess.run([sys.executable, str(cli), "export", "--site", str(td),
                              "--manifest", str(td / "m.json"), "--out", out_dir, "--strict"],
                             capture_output=True, text=True)
        check("strict rejects unresolved JSON asset routing", run.returncode == 1, run.stdout + run.stderr)


# 7. Bilingual edition: alternates are not warned about; EN-only has no switch and no zh pages.
with tempfile.TemporaryDirectory(prefix="lmsreg-") as d:
    td = Path(d)
    for lang in ("en", "zh"):
        (td / lang).mkdir()
        (td / lang / "a.html").write_text(
            f'<html lang="{lang}"><head><meta charset="utf-8"></head><body data-lang="{lang}"><main>'
            '<a href="../zh/a.html">ZH</a></main></body></html>')
    raw = {"languages": {"en": {"label": "English", "html_lang": "en"}, "zh": {"label": "中文", "html_lang": "zh-CN"}},
           "extract": {"profile": "basic"},
           "outline": [{"id": "c", "title": {"en": "C", "zh": "C"}, "sections": [{"id": "s", "title": {"en": "S", "zh": "S"},
                        "pages": [{"id": "a", "title": {"en": "A", "zh": "甲"}, "paths": {"en": "en/a.html", "zh": "zh/a.html"}}]}]}],
           "editions": [
               {"id": "zh", "languages": ["zh", "en"], "default": "zh", "course": {
                   "org": "O", "course": "C", "run": "ZH", "title": {"zh": "课程", "en": "Course"}, "language": "zh_HANS",
                   "start": "2026-01-01T00:00:00Z"}},
               {"id": "en", "languages": ["en"], "switch": True, "course": {
                   "org": "O", "course": "C", "run": "EN", "title": "Course", "language": "en", "start": "2026-01-01T00:00:00Z"}}]}
    (td / "m.json").write_text(json.dumps(raw, ensure_ascii=False))
    m = load_manifest(td / "m.json")
    summary = {}
    for ed in m.editions:
        static, report = EditionExport(Site(td), m, ed).run()
        flat = {(p.id, l): f"{p.id}{ed.filename_suffix[l]}.html" for _, _, p in m.pages() for l in ed.languages}
        out = td / f"{ed.id}.tar.gz"
        write_archive(build_tree(m, ed, static, flat), out)
        v = validate(out, m, ed.id)
        summary[ed.id] = {"ok": v["ok"], "warnings": v["warnings"], "pages": sorted(k for k in static if k.endswith(".html")),
                          "switch": "lms-lang-switch" in static["a.html"].decode(),
                          "dropped": report["dropped_cross_language_links"]}
    check("edition language handling",
          summary["zh"]["ok"] and not summary["zh"]["warnings"] and summary["zh"]["switch"]
          and summary["zh"]["pages"] == ["a-en.html", "a.html"]
          and summary["en"]["ok"] and summary["en"]["pages"] == ["a.html"] and not summary["en"]["switch"]
          and summary["en"]["dropped"], summary)

for r in results:
    print("PASS" if r["pass"] else "FAIL", r["case"])
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
sys.exit(0 if all(r["pass"] for r in results) else 1)
