"""Command line: export, validate, serve (route preview) and fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import __version__
from .exporter import EditionExport, ExportError, Site
from .manifest import ManifestError, load_manifest
from .olx import build_tree, read_archive, write_archive
from .validate import validate

PACKAGE_DIR = Path(__file__).resolve().parent


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def cmd_export(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    site = Site(args.site, allow_external_symlinks=manifest.allow_external_symlinks)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    protect = {Path(p).resolve() for p in args.protect} | {Path(args.manifest).resolve()}
    editions = [manifest.edition(e) for e in args.edition] if args.edition else manifest.editions
    mtime = int(os.environ.get("SOURCE_DATE_EPOCH", manifest.raw.get("source_date_epoch", 0)))
    report: dict = {"tool": "lms_export", "version": __version__, "fingerprint": fingerprint(),
                    "site": str(site.root), "manifest": str(Path(args.manifest).resolve()), "editions": []}
    status = 0
    for ed in editions:
        output = out_dir / ed.archive
        real_out = output.resolve()
        if real_out in protect:
            raise ExportError(f"refusing to overwrite protected file {real_out}")
        if _inside(real_out, site.real):
            raise ExportError(f"refusing to write the archive inside the site root: {real_out}")
        if output.exists() and not args.force:
            raise ExportError(f"{output} exists; pass --force to replace it")
        exporter = EditionExport(site, manifest, ed)
        static, ed_report = exporter.run()
        page_flat = {(p.id, lang): f"{p.id}{ed.filename_suffix[lang]}.html"
                     for _, _, p in manifest.pages() for lang in ed.languages}
        tree = build_tree(manifest, ed, static, page_flat)
        write_archive(tree, output, mtime=mtime)
        result = validate(output, manifest, ed.id, forbid=args.forbid,
                          forbid_archives=[Path(p) for p in args.forbid_from_archive])
        ed_report["archive"] = str(output)
        ed_report["validation"] = result
        report["editions"].append(ed_report)
        problems = sum(len(ed_report[k]) for k in (
            "unresolved", "ambiguous", "unmapped_bare_specifiers", "lex_errors",
            "dynamic_resource_sites", "json_relative_refs",
        ))
        print(f"{ed.id}: {output} sha256={result['sha256']} static={result['counts']['static_files']} "
              f"bytes={result['counts']['static_bytes']} errors={len(result['errors'])} "
              f"warnings={len(result['warnings'])} export-issues={problems}")
        if not result["ok"]:
            status = 1
        if args.strict and problems:
            status = 1
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest) if args.manifest else None
    result = validate(Path(args.archive), manifest, args.edition, forbid=args.forbid,
                      forbid_archives=[Path(p) for p in args.forbid_from_archive])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("course_key", "sha256", "counts", "ok")}, ensure_ascii=False))
    for e in result["errors"][:50]:
        print("ERROR", e)
    for w in result["warnings"][:20]:
        print("WARN ", w)
    return 0 if result["ok"] else 1


class _Preview(BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    assets: dict = {}
    key: tuple[str, str, str] = ("", "", "")
    units: list[tuple[str, str]] = []
    misses: list[str] = []
    lock = threading.Lock()

    def log_message(self, fmt: str, *a: object) -> None:  # keep stdout for 404s only
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _asset(self, name: str) -> None:
        data = self.files.get(f"static/{name}")
        if data is None:
            self._miss()
            return
        ctype = self.assets.get(name, {}).get("contentType") or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self._send(200, data, ctype)

    def _miss(self) -> None:
        with self.lock:
            self.misses.append(self.path)
        sys.stderr.write(f"404 {self.path}\n")
        self._send(404, b"not found", "text/plain")

    def do_GET(self) -> None:  # noqa: N802
        org, course, run = self.key
        path = unquote(urlsplit(self.path).path)
        query = urlsplit(self.path).query
        prefixes = {
            "static": "/static/",
            "c4x": f"/c4x/{org}/{course}/asset/",
            "asset-slash": f"/asset-v1:{org}+{course}+{run}+type@asset+block/",
            "asset-key": f"/asset-v1:{org}+{course}+{run}+type@asset+block@",
        }
        path = re.sub(r"^/assets/courseware/v\d+/[0-9a-f]+(?=/asset-v1:)", "", path)
        for prefix in prefixes.values():
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                self._asset(path[len(prefix):])
                return
        if path == "/__misses":
            with self.lock:
                body = json.dumps(self.misses).encode()
            self._send(200, body, "application/json")
            return
        m = re.fullmatch(r"/unit/(\d+)", path)
        if m and int(m.group(1)) < len(self.units):
            route = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("route", "asset-key")
            title, body = self.units[int(m.group(1))]
            body = body.replace('src="/static/', f'src="{prefixes.get(route, prefixes["asset-key"])}')
            page = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head>"
                    f"<body style='margin:0;background:#f4f4f4'><h1 style='font:16px sans-serif;margin:8px'>"
                    f"{html.escape(title)} ({html.escape(route)})</h1>{body}</body></html>")
            self._send(200, page.encode(), "text/html; charset=utf-8")
            return
        if path == "/":
            rows = "".join(
                f"<li>{html.escape(t)}: " + " ".join(f"<a href='/unit/{i}?route={r}'>{r}</a>" for r in prefixes) + "</li>"
                for i, (t, _) in enumerate(self.units))
            self._send(200, f"<!DOCTYPE html><meta charset='utf-8'><ol>{rows}</ol>".encode(), "text/html; charset=utf-8")
            return
        self._miss()


def cmd_serve(args: argparse.Namespace) -> int:
    import xml.etree.ElementTree as ET

    raw = read_archive(Path(args.archive))
    files = {p[len("course/"):]: d for p, d in raw.items() if p.startswith("course/")}
    root = ET.fromstring(files["course.xml"])
    key = (root.get("org"), root.get("course"), root.get("url_name"))
    units = []
    course_el = ET.fromstring(files[f"course/{key[2]}.xml"])
    for ch in course_el.findall("chapter"):
        for seq in ET.fromstring(files[f"chapter/{ch.get('url_name')}.xml"]).findall("sequential"):
            for v in ET.fromstring(files[f"sequential/{seq.get('url_name')}.xml"]).findall("vertical"):
                vel = ET.fromstring(files[f"vertical/{v.get('url_name')}.xml"])
                for h in vel.findall("html"):
                    units.append((vel.get("display_name", ""), files[f"html/{h.get('url_name')}.html"].decode()))
    _Preview.files = files
    _Preview.assets = json.loads(files.get("policies/assets.json", b"{}"))
    _Preview.key = key
    _Preview.units = units
    server = ThreadingHTTPServer((args.host, args.port), _Preview)
    print(f"serving {args.archive} as {'+'.join(key)} on http://{args.host}:{server.server_port}/ "
          "(routes: static, c4x, asset-slash, asset-key; /__misses lists 404s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_lms.py", description="Export a built Sphinx site to Open edX OLX archives.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="build one archive per manifest edition")
    p.add_argument("--site", required=True, type=Path, help="built HTML site root")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="directory for the .tar.gz archives")
    p.add_argument("--edition", action="append", default=[], help="edition id (repeatable; default all)")
    p.add_argument("--report", type=Path)
    p.add_argument("--force", action="store_true", help="replace existing output archives")
    p.add_argument("--protect", action="append", default=[], help="file that must never be overwritten")
    p.add_argument("--forbid", action="append", default=[], help="string that must not appear in the output")
    p.add_argument("--forbid-from-archive", action="append", default=[],
                   help="reference OLX archive whose course key, block ids and title must not appear")
    p.add_argument("--strict", action="store_true", help="fail on unresolved/ambiguous references too")
    p.set_defaults(func=cmd_export)
    v = sub.add_parser("validate", help="validate an exported archive")
    v.add_argument("archive")
    v.add_argument("--manifest", type=Path)
    v.add_argument("--edition")
    v.add_argument("--report", type=Path)
    v.add_argument("--forbid", action="append", default=[])
    v.add_argument("--forbid-from-archive", action="append", default=[])
    v.set_defaults(func=cmd_validate)
    s = sub.add_parser("serve", help="serve an archive under static/c4x/asset-v1 routes for browser checks")
    s.add_argument("archive")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)
    f = sub.add_parser("fingerprint", help="hash of this package (compare skill and repo copies)")
    f.set_defaults(func=lambda a: print(fingerprint()) or 0)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ExportError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
