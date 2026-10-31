#!/usr/bin/env python3
"""Build the bilingual course site with Sphinx.

Lesson source: MyST Markdown in docs/zh and docs/en (edit those files directly).
Output: <out>/zh and <out>/en (default: web/). Shared media, the 3D robot,
fonts and vendor scripts stay in web/assets and are referenced as ../assets.
content/course.json is a read-only archive of the pre-Sphinx site and is not read.

  python3 scripts/build_site.py                 # build into web/
  python3 scripts/build_site.py --out /tmp/site # staging build (links assets)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LANGS = ("zh", "en")


def sphinx_python():
    """Prefer the documented docs environment, else the current interpreter."""
    venv = ROOT / ".local/docs-venv/bin/python"
    for python in (os.environ.get("COURSE_DOCS_PYTHON"), str(venv) if venv.exists() else None, sys.executable):
        if python and subprocess.run([python, "-c", "import sphinx, myst_parser, nvidia_sphinx_theme"], capture_output=True).returncode == 0:
            return python
    raise SystemExit("Sphinx is not installed. Run:\n  python3 -m venv .local/docs-venv && .local/docs-venv/bin/pip install -r docs/requirements.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "web", help="output directory (default: web)")
    parser.add_argument("--clean", action="store_true", help="discard cached doctrees and rebuild every page")
    args = parser.parse_args()
    out = args.out.resolve()
    python = sphinx_python()
    cache = ROOT / ".local/docs-build"
    if args.clean and cache.exists():
        shutil.rmtree(cache)
    out.mkdir(parents=True, exist_ok=True)
    if out != (ROOT / "web").resolve() and not (out / "assets").exists():
        (out / "assets").symlink_to(ROOT / "web/assets", target_is_directory=True)
    for lang in LANGS:
        command = [python, "-m", "sphinx", "-b", "html", "-W", "--keep-going", "-q",
                   "-d", str(cache / f"doctrees-{lang}"), str(DOCS / lang), str(out / lang)]
        if args.clean:
            command.insert(3, "-E")
        print("$", " ".join(command), flush=True)
        if subprocess.run(command).returncode:
            raise SystemExit(f"Sphinx build failed for {lang}")
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=zh/index.html">'
        '<title>Train Microduck in Simulation with NVIDIA Isaac</title></head><body><a href="zh/index.html">使用 NVIDIA Isaac 在仿真环境训练 Microduck</a> · <a href="en/index.html">Train Microduck in Simulation with NVIDIA Isaac</a></body></html>'
    )
    (out / ".nojekyll").touch()
    print(f"Built {sum(1 for lang in LANGS for _ in (DOCS / lang).glob('*.md'))} lesson pages into {out}")


if __name__ == "__main__":
    main()
