#!/usr/bin/env python3
"""Check published links, bilingual lessons, lab commands, assets and immutable weights."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import ast
import hashlib
import json
import re
import subprocess

R = Path(__file__).resolve().parents[1]
W = R / "web"
errors = []
count = 0


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for key in ("href", "src", "poster"):
            if a.get(key):
                self.links.append(a[key])


parsed = {}
for f in W.rglob("*.html"):
    p = Page()
    p.feed(f.read_text())
    parsed[f.resolve()] = p
    if len(p.ids) != len(set(p.ids)):
        errors.append(f"Duplicate IDs: {f}")
    for pattern in [
        "/home/yuki/",
        "@COURSE@",
        "@REPO@",
        "localhost:8018",
        "mark-complete",
        "local-progress",
        "migration-20261019",
        "lab/frozen/",
        "brand-mark",
    ]:
        if pattern in f.read_text():
            errors.append(f"Internal path in page: {f}")
for f, p in parsed.items():
    for link in p.links:
        u = urlsplit(link)
        if u.scheme or u.netloc:
            prefix = "/NV-DLI/Microduck-Workshop/blob/main/"
            if (
                u.netloc == "github.com"
                and u.path.startswith(prefix)
                and not (R / unquote(u.path[len(prefix) :])).is_file()
            ):
                errors.append("Missing public source: " + link)
            continue
        target = (f.parent / unquote(u.path)).resolve() if u.path else f
        if not target.is_relative_to(W):
            errors.append(f"Outside web root: {f.name} {link}")
        elif not target.exists():
            errors.append(f"Missing: {f.name} {link}")
        elif (
            u.fragment
            and target in parsed
            and unquote(u.fragment) not in parsed[target].ids
        ):
            errors.append(f"Missing fragment: {f.name} {link}")
        count += 1
pages = json.loads((R / "content/course.json").read_text())
chapters = [p for p in pages if p.get("number")]
assert [int(p["number"]) for p in chapters] == list(range(1, 10))
for p in pages:
    for lang in ["zh", "en"]:
        if not (W / lang / (p["id"] + ".html")).exists():
            errors.append("Missing language page " + p["id"])
    if p.get("number") and not any(
        b["type"] in ("media", "anatomy", "image", "flow", "diagram", "pipeline")
        for s in p["sections"]
        for b in s["blocks"]
    ):
        errors.append("Missing visual explanation " + p["id"])
    for s in p["sections"] + json.loads((R / "content/zh-depth.json").read_text()).get(
        p["id"], []
    ):
        for b in s["blocks"]:
            if b["type"] == "code" and b.get("label", "").startswith("Terminal"):
                result = subprocess.run(
                    ["bash", "-n"], input=b["code"], capture_output=True, text=True
                )
                if result.returncode:
                    errors.append("Invalid shell block " + p["id"])
                for script in re.findall(r"python3 (scripts/[\w.-]+\.py)", b["code"]):
                    if not (R / script).is_file():
                        errors.append("Missing command " + script)
for item in json.loads((R / "models/manifest.json").read_text())["files"]:
    f = R / "models" / item["file"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != item["sha256"]:
        errors.append("Model mismatch " + f.name)
indoor = json.loads((R / "models/indoor-detector/deployment.json").read_text())
for key in ("weights", "onnx"):
    f = R / "models/indoor-detector" / indoor[key]
    if hashlib.sha256(f.read_bytes()).hexdigest() != indoor[key + "_sha256"]:
        errors.append("Indoor model mismatch " + f.name)
for folder in ["scripts", "lab"]:
    for f in (R / folder).rglob("*.py"):
        try:
            ast.parse(f.read_text(), filename=str(f))
        except SyntaxError:
            errors.append("Python syntax " + str(f))
for item in json.loads((R / "lab/bundle-manifest.json").read_text())["files"]:
    f = R / "lab/runtime" / item["path"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != item["template_sha256"]:
        errors.append("Bundle mismatch " + str(f))
meta = json.loads((W / "assets/robot/components.json").read_text())
assert len(meta["parts"]) == 70
ids = [p["id"] for p in meta["parts"]]
assert len(set(ids)) == 70
if errors:
    raise SystemExit("\n".join(errors))
for f in (R / "lab/runtime").rglob("*"):
    if f.is_file() and re.search(
        r"migration-\d+|[_-]v\d+(?=[_.-]|$)|candidate|experimental|development|kick_runtime_early",
        str(f.relative_to(R)),
    ):
        errors.append("Non-teaching source name: " + str(f))
if errors:
    raise SystemExit("\n".join(errors))
print(
    json.dumps(
        dict(
            pages=len(parsed),
            local_links=count,
            chapters=len(chapters),
            models=len(json.loads((R / "models/manifest.json").read_text())["files"]),
            additional_indoor_models=2,
            components=70,
            passed=True,
        )
    )
)
