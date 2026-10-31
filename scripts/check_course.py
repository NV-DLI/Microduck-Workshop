#!/usr/bin/env python3
"""Check the course source (docs/), the built site (web/), lab commands and immutable models.

  python3 scripts/check_course.py              # everything
  python3 scripts/check_course.py --site-only  # Markdown source and built pages only

Site and source problems, model identity problems and runtime-bundle template
mismatches are reported in separate groups; any of them fails the check.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse
import os
import ast
import hashlib
import json
import re
import subprocess

R = Path(__file__).resolve().parents[1]
W = R / "web"
D = R / "docs"
LANGS = ("zh", "en")
# Course structure: page -> section number, grouped into six numbered parts.
STRUCTURE = [
    ("1", ["index", "setup"]),
    ("2", ["01-play", "02-anatomy"]),
    ("3", ["03-learning", "03-walk-training", "04-terrain", "05-skills"]),
    ("4", ["06-perception", "06-onboard"]),
    ("5", ["07-groot", "08-groot-interface", "08-groot-lab", "09-worlds"]),
    ("6", ["models", "troubleshooting", "references"]),
]
NUMBERS = {page: f"{part}.{i + 1}" for part, pages in STRUCTURE for i, page in enumerate(pages)}
LESSONS = ["01-play", "02-anatomy", "03-learning", "03-walk-training", "04-terrain", "05-skills",
           "06-onboard", "06-perception", "07-groot", "08-groot-interface", "08-groot-lab", "09-worlds"]
REMOVED = ["06-autonomy", "08-aspire", "how-to-learn"]  # removed from the course; old URLs redirect
VISUALS = ("course-video", "robot-anatomy", "course-image", "learning-flow", "concept-diagram", "network-diagram",
           "platform-diagram", "scene-cards", "demonstration", "training-chart", "benchmark-chart", "control-timing", "{.pipeline}")
INTERNAL = ["/home/yuki/", "@COURSE@", "@REPO@", "localhost:8018", "mark-complete", "local-progress",
            "migration-20261019", "lab/frozen/", "brand-mark", "sphinx-migration-2026"]


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for key in ("href", "src", "poster", "data-stages", "data-demonstration"):
            if a.get(key):
                self.links.append(a[key])


def check_source(errors):
    """Markdown is the authoritative lesson source; both languages share page names."""
    pages = {lang: sorted(p.stem for p in (D / lang).glob("*.md")) for lang in LANGS}
    if pages["zh"] != pages["en"]:
        errors.append(f"Language page sets differ: {sorted(set(pages['zh']) ^ set(pages['en']))}")
    if pages["zh"] != sorted(NUMBERS):
        errors.append(f"Course pages differ from the course structure: {sorted(set(pages['zh']) ^ set(NUMBERS))}")
    chapters = sorted(p for p in pages["zh"] if re.match(r"\d\d-", p))
    if chapters != LESSONS:
        errors.append(f"Lesson pages are not {LESSONS}: {chapters}")
    for lang in LANGS:
        index = (D / lang / "index.md").read_text()
        order = [line.strip() for block in re.findall(r"^```\{toctree\}\n(.*?)^```", index, re.S | re.M)
                 for line in block.split("\n\n", 1)[-1].splitlines() if line.strip()]
        expected = ["self" if p == "index" else p for _, group in STRUCTURE for p in group]
        if order != expected:
            errors.append(f"{lang} navigation order differs: {order}")
        captions = re.findall(r"^:caption: (\d) ", index, re.M)
        if captions != [part for part, _ in STRUCTURE]:
            errors.append(f"{lang} navigation parts are not numbered 1–{len(STRUCTURE)}: {captions}")
    commands = 0
    for lang in LANGS:
        for name in pages[lang]:
            text = (D / lang / f"{name}.md").read_text()
            if re.search(r"^[`:]{3,}\{raw\}", text, re.M):
                errors.append(f"Raw HTML block in lesson source {lang}/{name}.md")
            title = re.search(r"^# (.+)$", text, re.M)
            if not title:
                errors.append(f"Missing page title {lang}/{name}.md")
            elif name in NUMBERS and not title.group(1).startswith(NUMBERS[name] + " "):
                errors.append(f"Title of {lang}/{name}.md should start with {NUMBERS[name]}")
            for removed in REMOVED:
                if re.search(rf"\]\({removed}(\.md|\.html)?(#[^)]*)?\)", text):
                    errors.append(f"Link to removed page {removed} in {lang}/{name}.md")
            if name in chapters and not any(v in text for v in VISUALS):
                errors.append(f"Missing visual explanation {lang}/{name}.md")
            for block in re.finditer(r"^(`{3,})\{code-block\} bash\n((?::[^\n]*\n)*)\n(.*?)\n\1$", text, re.S | re.M):
                if ":class: terminal" not in block.group(2):
                    continue
                commands += 1
                code = block.group(3)
                if subprocess.run(["bash", "-n"], input=code, capture_output=True, text=True).returncode:
                    errors.append(f"Invalid shell block {lang}/{name}.md")
                for script in re.findall(r"python3 (scripts/[\w.-]+\.py)", code):
                    if not (R / script).is_file():
                        errors.append(f"Missing command {script} in {lang}/{name}.md")
            if "\u2014" in text:
                errors.append(f"Em dash in lesson source {lang}/{name}.md (use a colon, comma or ASCII hyphen)")
            if "workshop.py" in text:
                errors.append(f"Lesson {lang}/{name}.md names scripts/workshop.py; the lab entry is scripts/run.py")
    return dict(pages=len(pages["zh"]) * 2, chapters=len(chapters), terminal_blocks=commands)


def pending_assets():
    """Optional media named by lessons (hero-optional, if-present, prefer) that are not there yet."""
    missing = set()
    for md in sorted(D.glob("*/*.md")):
        text = md.read_text()
        optional = []
        for block in re.findall(r"^:{3,}\{(?:lesson-intro|course-image)\}[^\n]*\n((?::[^\n]*\n)+)", text, re.M):
            options = dict(re.findall(r"^:([\w-]+):[ ]?(.*)$", block, re.M))
            if "hero-optional" in options:
                optional += [options.get("hero", ""), options.get("hero-animation", "")]
            if options.get("prefer"):
                optional.append(options["prefer"])
        for path in re.findall(r"^:{3,}\{course-image\} (\S+)\n(?::[^\n]*\n)*?:if-present:", text, re.M):
            optional.append(path)
        missing |= {p for p in optional if p and not (W / "assets" / p).exists()}
    return sorted(missing)


def check_site(errors, pages):
    parsed = {}
    for f in sorted(W.rglob("*.html")):
        if {"assets", "_static"} & set(f.relative_to(W).parts):
            continue
        p = Page()
        text = f.read_text()
        p.feed(text)
        parsed[f] = p
        if len(p.ids) != len(set(p.ids)):
            errors.append(f"Duplicate IDs: {f.relative_to(R)}")
        for pattern in INTERNAL:
            if pattern in text:
                errors.append(f"Internal reference {pattern!r} in page: {f.relative_to(R)}")
        visible = re.sub(r"<script\b.*?</script>", "", text, flags=re.S)
        if "\u2014" in visible or "&#8212;" in visible or "&mdash;" in visible:
            errors.append(f"Em dash in learner-visible page: {f.relative_to(W)}")
    count = 0
    for f, p in parsed.items():
        for link in p.links:
            u = urlsplit(link)
            if u.scheme or u.netloc:
                prefix = "/NV-DLI/Microduck-Workshop/blob/main/"
                if u.netloc == "github.com" and u.path.startswith(prefix) and not (R / unquote(u.path[len(prefix):])).is_file():
                    errors.append("Missing public source: " + link)
                continue
            if link.startswith(("javascript:", "mailto:")):
                continue
            # normpath keeps a staging build's assets symlink inside its root.
            target = Path(os.path.normpath(f.parent / unquote(u.path))) if u.path else f
            if not target.is_relative_to(W):
                errors.append(f"Outside web root: {f.name} {link}")
            elif not target.exists():
                errors.append(f"Missing: {f.relative_to(W)} {link}")
            elif u.fragment and target in parsed and unquote(u.fragment) not in parsed[target].ids:
                errors.append(f"Missing fragment: {f.relative_to(W)} {link}")
            count += 1
    for lang in LANGS:
        for md in (D / lang).glob("*.md"):
            if not (W / lang / f"{md.stem}.html").exists():
                errors.append(f"Missing built page {lang}/{md.stem}.html (run scripts/build_site.py)")
            elif (W / lang / f"{md.stem}.html").stat().st_mtime < md.stat().st_mtime:
                errors.append(f"Built page older than source: {lang}/{md.stem}")
        if not (W / lang / "searchindex.js").exists():
            errors.append(f"Missing search index for {lang}")
        else:
            index_js = (W / lang / "searchindex.js").read_text()
            for removed in REMOVED:
                if f'"{removed}"' in index_js:
                    errors.append(f"Removed page {removed} is still in the {lang} search index")
        for removed in REMOVED:
            page = W / lang / f"{removed}.html"
            if page.exists() and "http-equiv=\"refresh\"" not in page.read_text()[:600]:
                errors.append(f"Removed lesson {lang}/{removed}.html is still built as a page")
    meta = json.loads((W / "assets/robot/components.json").read_text())
    ids = [p["id"] for p in meta["parts"]]
    if len(ids) != 70 or len(set(ids)) != 70:
        errors.append("Robot components must be 70 unique parts")
    return dict(html_files=len(parsed), local_links=count)


def check_models(errors):
    for item in json.loads((R / "models/manifest.json").read_text())["files"]:
        f = R / "models" / item["file"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != item["sha256"]:
            errors.append("Model mismatch " + f.name)
    indoor = json.loads((R / "models/indoor-detector/deployment.json").read_text())
    for key in ("weights", "onnx"):
        f = R / "models/indoor-detector" / indoor[key]
        if hashlib.sha256(f.read_bytes()).hexdigest() != indoor[key + "_sha256"]:
            errors.append("Indoor model mismatch " + f.name)
    for folder in ["scripts", "lab", "docs"]:
        for f in (R / folder).rglob("*.py"):
            try:
                ast.parse(f.read_text(), filename=str(f))
            except SyntaxError:
                errors.append("Python syntax " + str(f.relative_to(R)))
    for f in (R / "lab/runtime").rglob("*"):
        if f.is_file() and re.search(r"migration-\d+|[_-]v\d+(?=[_.-]|$)|candidate|experimental|development|kick_runtime_early", str(f.relative_to(R))):
            errors.append("Non-teaching source name: " + str(f.relative_to(R)))
    return dict(models=len(json.loads((R / "models/manifest.json").read_text())["files"]), additional_indoor_models=2)


def check_bundle(errors):
    items = json.loads((R / "lab/bundle-manifest.json").read_text())["files"]
    for item in items:
        f = R / "lab/runtime" / item["path"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != item["template_sha256"]:
            errors.append("Bundle mismatch " + str(f.relative_to(R)))
    return dict(bundle_files=len(items))


def main():
    global W
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site-only", action="store_true", help="check only docs/ source and web/ output")
    parser.add_argument("--web", type=Path, default=W, help="built site to check (default: web)")
    args = parser.parse_args()
    W = args.web.resolve()
    groups = {"site": [], "models": [], "runtime_bundle": []}
    summary = check_source(groups["site"])
    summary.update(check_site(groups["site"], summary))
    # Optional media still to be supplied: reported, not a failure (the pages build without it).
    summary["pending_assets"] = pending_assets()
    if not args.site_only:
        summary.update(check_models(groups["models"]))
        summary.update(check_bundle(groups["runtime_bundle"]))
    failed = {k: v for k, v in groups.items() if v}
    for name, problems in failed.items():
        print(f"[{name}] {len(problems)} problem(s):")
        for problem in problems:
            print("  " + problem)
    print(json.dumps(dict(summary, passed=not failed, failed_groups=sorted(failed)), ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
