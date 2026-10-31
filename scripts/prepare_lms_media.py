#!/usr/bin/env python3
"""Stage a built site for LMS export with smaller PNG/JPEG/WebP files (LMS archives only).

The normal site and web/assets are never modified. `stage` asks the LMS exporter which files
each edition reads, copies exactly those into a NEW directory as regular files, re-encodes the
PNG/JPEG/WebP copies in their own format (same name, same extension, same pixel size) and
keeps a copy only when it is smaller and passes the checks below. The exporter then hashes
the new bytes into new flat names, so every HTML/CSS/JS reference stays consistent.

  .local/docs-venv/bin/python scripts/prepare_lms_media.py stage \
      --site web --manifest docs/lms/microduck.json --out /tmp/microduck-lms/site
  .local/docs-venv/bin/python scripts/export_lms.py export \
      --site /tmp/microduck-lms/site --manifest docs/lms/microduck.json --out DIR

Checks per optimized file: same dimensions, mode family and alpha channel bit for bit; for
animated WebP also frame count, frame order, per-frame durations and loop count; lossy
results need a minimum PSNR of the image composited on white and on black (what a viewer sees
in either theme). SVG, GLB, fonts, scripts, styles, video and GIF are copied unchanged.

`stage` runs with the docs venv (BeautifulSoup for the exporter); image work runs in
`--image-python` (default /usr/bin/python3, which has Pillow with WebP animation and
libimagequant) as at most `--workers` CPU processes. `probe` optimizes copies of single files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
DEFAULTS = {"webp_quality": 82, "jpeg_quality": 85, "min_psnr": 38.0, "min_psnr_png": 40.0,
            "min_gain": 0.10}
ISSUE_BUCKETS = ("unresolved", "ambiguous", "unmapped_page_links", "dropped_cross_language_links",
                 "dynamic_resource_sites", "object_key_literals", "json_relative_refs",
                 "unmapped_bare_specifiers", "lex_errors", "unknown_mime")


class StageError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ safe output paths


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _real_target(path: Path) -> Path:
    """Real path of a not-yet-existing path: resolve the nearest existing ancestor."""
    existing = path
    rest: list[str] = []
    while not existing.exists():
        rest.insert(0, existing.name)
        existing = existing.parent
    return existing.resolve().joinpath(*rest)


def check_output(out: Path, protected: list[Path]) -> Path:
    """Return the absolute output path, or refuse an unsafe one.

    The output must not exist yet, its existing ancestors must contain no symlinks, and it may
    neither lie inside nor contain any protected path (source site, its real path, asset
    directories, the real directories of copied files).
    """
    out_abs = Path(os.path.abspath(out))
    if os.path.lexists(out_abs):
        raise StageError(f"output already exists (staging needs a new directory): {out_abs}")
    real = _real_target(out_abs)
    if real != out_abs:
        raise StageError(f"output path goes through a symlink: {out_abs} -> {real}")
    for p in protected:
        for q in {Path(os.path.abspath(p)), Path(p).resolve()}:
            if _inside(out_abs, q) or _inside(q, out_abs):
                raise StageError(f"output {out_abs} overlaps protected source path {q}")
    return out_abs


def _make_dirs(root: Path, rel_dir: str) -> None:
    path = root
    for part in [p for p in rel_dir.split("/") if p]:
        path = path / part
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            raise StageError(f"staging directory is not a plain directory: {path}")


def write_new(path: Path, data: bytes) -> None:
    """Create a new regular file; never follows or replaces a symlink or existing file."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def replace_regular(target: Path, data: bytes) -> None:
    """Replace a staged regular file through a new sibling file (atomic rename)."""
    st = os.lstat(target)
    if not stat.S_ISREG(st.st_mode):
        raise StageError(f"refusing to replace non-regular staged file {target}")
    tmp = target.with_name(f".{target.name}.optimized.tmp")
    write_new(tmp, data)
    os.replace(tmp, target)


# ------------------------------------------------------------------ image work (Pillow)


def _pil():
    from PIL import Image, ImageChops, ImageStat  # noqa: F401
    return Image, ImageChops, ImageStat


def _composite_psnr(a, b) -> float:
    """PSNR of two RGBA images composited on white and on black; the worse of the two."""
    Image, ImageChops, ImageStat = _pil()
    worst = 99.0
    for bg in ((255, 255, 255, 255), (0, 0, 0, 255)):
        x = Image.alpha_composite(Image.new("RGBA", a.size, bg), a).convert("RGB")
        y = Image.alpha_composite(Image.new("RGBA", b.size, bg), b).convert("RGB")
        rms = ImageStat.Stat(ImageChops.difference(x, y)).rms
        mse = sum(v * v for v in rms) / 3
        worst = min(worst, 99.0 if mse == 0 else 10 * math.log10(255 * 255 / mse))
    return worst


def _alpha_diff(a, b) -> int:
    _, ImageChops, _ = _pil()
    return ImageChops.difference(a.getchannel("A"), b.getchannel("A")).getextrema()[1]


def _frames(data: bytes):
    Image, _, _ = _pil()
    im = Image.open(io.BytesIO(data))
    frames, durations = [], []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        im.load()
        frames.append(im.convert("RGBA").copy())
        durations.append(im.info.get("duration"))
    info = {"format": im.format, "mode": im.mode, "size": list(im.size), "frames": len(frames),
            "loop": im.info.get("loop"), "background": im.info.get("background"),
            "icc_profile": im.info.get("icc_profile"), "exif": im.info.get("exif")}
    return frames, durations, info


def webp_codec(data: bytes) -> str:
    """'lossless', 'lossy' or 'mixed' from the RIFF chunk types (VP8L vs VP8)."""
    lossless, lossy = data.count(b"VP8L"), data.count(b"VP8 ")
    return "mixed" if lossless and lossy else ("lossless" if lossless else "lossy")


def _compare(src_frames, src_durations, src_info, out: bytes) -> dict:
    """Check an encoded candidate against the decoded source, frame by frame."""
    frames, durations, info = _frames(out)
    res = {"size": info["size"], "frames": info["frames"], "loop": info["loop"]}
    res["same_size"] = info["size"] == src_info["size"]
    res["same_frames"] = info["frames"] == src_info["frames"]
    res["same_durations"] = durations == src_durations
    res["same_loop"] = info["loop"] == src_info["loop"]
    if not (res["same_size"] and res["same_frames"]):
        res.update(alpha_max_diff=None, psnr_min=None, psnr_mean=None)
        return res
    alpha = 0
    psnrs = []
    for a, b in zip(src_frames, frames):
        alpha = max(alpha, _alpha_diff(a, b))
        psnrs.append(_composite_psnr(a, b))
    res["alpha_max_diff"] = alpha
    res["psnr_min"] = round(min(psnrs), 2)
    res["psnr_mean"] = round(sum(psnrs) / len(psnrs), 2)
    return res


def _candidates(fmt: str, data: bytes, frames, durations, info, opts: dict):
    """Yield (method, encoded bytes, min_psnr or None for lossless)."""
    Image, _, _ = _pil()
    first = frames[0]
    extra = {k: info[k] for k in ("icc_profile", "exif") if info.get(k)}
    if fmt == "webp" and info["frames"] > 1:
        buf = io.BytesIO()
        first.save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=durations,
                   loop=info["loop"] if info["loop"] is not None else 0,
                   background=info["background"] or (0, 0, 0, 0), lossless=False,
                   quality=opts["webp_quality"], alpha_quality=100, method=6, kmax=0,
                   allow_mixed=False, **extra)
        yield f"webp-animated-q{opts['webp_quality']}", buf.getvalue(), opts["min_psnr"]
    elif fmt == "webp":
        img = first if info["mode"] in ("RGBA", "LA", "P") else first.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=True, quality=100, method=6, exact=True, **extra)
        yield "webp-lossless-m6", buf.getvalue(), None
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=False, quality=opts["webp_quality"], alpha_quality=100,
                 method=6, **extra)
        yield f"webp-q{opts['webp_quality']}", buf.getvalue(), opts["min_psnr"]
    elif fmt == "png":
        src = Image.open(io.BytesIO(data))
        buf = io.BytesIO()
        src.save(buf, "PNG", optimize=True, **{k: v for k, v in extra.items() if k == "icc_profile"})
        yield "png-lossless", buf.getvalue(), None
        if info["mode"] in ("RGB", "RGBA"):
            opaque = first.getchannel("A").getextrema() == (255, 255)
            base = first.convert("RGB") if opaque else first
            pal = base.quantize(256, method=Image.Quantize.LIBIMAGEQUANT, dither=Image.Dither.FLOYDSTEINBERG)
            buf = io.BytesIO()
            pal.save(buf, "PNG", optimize=True, **{k: v for k, v in extra.items() if k == "icc_profile"})
            yield "png-palette256", buf.getvalue(), opts["min_psnr_png"]
    elif fmt == "jpeg":
        from PIL import JpegImagePlugin
        src = Image.open(io.BytesIO(data))
        sampling = JpegImagePlugin.get_sampling(src)
        buf = io.BytesIO()
        src.save(buf, "JPEG", quality=opts["jpeg_quality"], optimize=True, progressive=True,
                 subsampling=sampling if sampling >= 0 else 2, **extra)
        yield f"jpeg-q{opts['jpeg_quality']}", buf.getvalue(), opts["min_psnr"]


def optimize_bytes(data: bytes, fmt: str, opts: dict) -> tuple[bytes | None, dict]:
    """Return (new bytes or None to keep the original, per-file report)."""
    opts = {**DEFAULTS, **(opts or {})}
    frames, durations, info = _frames(data)
    rep = {"format": fmt, "mode": info["mode"], "size": info["size"], "frames": info["frames"],
           "loop": info["loop"], "bytes_in": len(data), "sha256_in": sha256(data),
           "has_alpha": any(f.getchannel("A").getextrema() != (255, 255) for f in frames),
           "candidates": []}
    if info["frames"] > 1:
        rep["durations_ms"] = sorted(set(d for d in durations if d is not None))
        rep["duration_total_ms"] = sum(d or 0 for d in durations)
    if fmt == "webp":
        rep["codec_in"] = webp_codec(data)
    if fmt != "webp" and info["frames"] > 1:
        rep.update(method="unchanged", reason=f"animated {fmt} is not optimized", bytes_out=len(data))
        return None, rep
    best = None
    for method, out, min_psnr in _candidates(fmt, data, frames, durations, info, opts):
        cmp = _compare(frames, durations, info, out)
        lossless_ok = min_psnr is None and cmp.get("psnr_min") == 99.0 and cmp["alpha_max_diff"] == 0
        ok = (cmp["same_size"] and cmp["same_frames"] and cmp["same_durations"] and cmp["same_loop"]
              and cmp["alpha_max_diff"] == 0
              and (lossless_ok or (min_psnr is not None and cmp["psnr_min"] >= min_psnr)))
        gain = 1 - len(out) / len(data)
        # Re-encoding an already lossy source costs quality, so it must buy a real reduction.
        lossy_again = min_psnr is not None and (fmt == "jpeg" or rep.get("codec_in") == "lossy")
        reason = None
        if not ok:
            reason = "checks failed"
        elif len(out) >= len(data):
            reason = "not smaller"
        elif lossy_again and gain < opts["min_gain"]:
            reason = f"gain {gain:.1%} below {opts['min_gain']:.0%} for a lossy re-encode"
        rep["candidates"].append({"method": method, "bytes": len(out), "accepted": reason is None,
                                  "reason": reason, **cmp})
        if reason is None and (best is None or len(out) < len(best[1])):
            best = (method, out, cmp)
    if best is None:
        rep.update(method="unchanged", reason="no candidate passed and was smaller", bytes_out=len(data),
                   sha256_out=rep["sha256_in"])
        return None, rep
    method, out, cmp = best
    rep.update(method=method, bytes_out=len(out), sha256_out=sha256(out), alpha_max_diff=cmp["alpha_max_diff"],
               psnr_min=cmp["psnr_min"], psnr_mean=cmp["psnr_mean"], same_durations=cmp["same_durations"],
               same_loop=cmp["same_loop"], same_frames=cmp["same_frames"], same_size=cmp["same_size"])
    return out, rep


def _worker_main() -> int:
    """Optimize one staged file: job JSON on stdin, report JSON on stdout."""
    job = json.load(sys.stdin)
    path = Path(job["path"])
    data = path.read_bytes()
    t = time.time()
    try:
        out, rep = optimize_bytes(data, job["format"], job["options"])
    except Exception as exc:  # report and keep the original
        out, rep = None, {"format": job["format"], "bytes_in": len(data), "bytes_out": len(data),
                          "sha256_in": sha256(data), "method": "unchanged", "reason": f"error: {exc!r}"}
    rep["seconds"] = round(time.time() - t, 1)
    if out is not None:
        write_new(Path(job["candidate"]), out)
    json.dump(rep, sys.stdout)
    return 0


def run_jobs(jobs: list[dict], image_python: str, workers: int) -> list[dict]:
    """Run optimize jobs, each in its own CPU process, at most `workers` at a time."""
    def one(job):
        proc = subprocess.run([image_python, "-I", str(Path(__file__).resolve()), "_worker"],
                              input=json.dumps(job), capture_output=True, text=True)
        if proc.returncode != 0:
            raise StageError(f"image worker failed for {job['path']}: {proc.stderr.strip()[-800:]}")
        return json.loads(proc.stdout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, jobs))


def check_image_python(image_python: str) -> None:
    code = ("import PIL.features as f, sys; "
            "sys.exit(0 if f.check('webp') and f.check('webp_anim') and f.check_feature('libimagequant') else 3)")
    proc = subprocess.run([image_python, "-I", "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise StageError(f"{image_python} needs Pillow with WebP animation and libimagequant "
                         f"(exit {proc.returncode}): {proc.stderr.strip()[-400:]}")


# ------------------------------------------------------------------ exporter closure


def _exporter():
    sys.path.insert(0, str(ROOT / "scripts"))
    from lms_export.exporter import EditionExport, Site
    from lms_export.manifest import load_manifest
    return EditionExport, Site, load_manifest


def export_in_memory(site_root: Path, manifest_path: Path) -> dict:
    """Run every edition's export in memory; return files, reports and every file read."""
    EditionExport, Site, load_manifest = _exporter()
    manifest = load_manifest(manifest_path)
    site = Site(site_root, allow_external_symlinks=manifest.allow_external_symlinks)
    result = {"editions": {}, "read": {}, "manifest": manifest}
    for ed in manifest.editions:
        ex = EditionExport(site, manifest, ed)
        files, report = ex.run()
        names = {rel: ex.name(rel) for rel in ex.nodes}
        result["editions"][ed.id] = {"files": files, "report": report, "names": names, "edition": ed}
        rels = set(ex.nodes) | set(ex.page_by_rel)
        for page in ex.pages.values():
            rels |= set(page.importmap_exact.values())
        for rel in rels:
            entry = result["read"].setdefault(rel, {"editions": set(), "pages": set()})
            entry["editions"].add(ed.id)
            node = ex.nodes.get(rel)
            if node is not None:
                entry["pages"] |= {f"{ed.id}/{k}" for k in node.pages}
            elif rel in ex.page_by_rel:
                entry["pages"].add(f"{ed.id}/{ex.page_by_rel[rel].key}")
    return result


def archive_bytes(manifest, edition, files: dict[str, bytes]) -> int:
    """Size of the .tar.gz the exporter would write for these static files."""
    from lms_export.olx import build_tree, write_archive
    page_flat = {(p.id, lang): f"{p.id}{edition.filename_suffix[lang]}.html"
                 for _, _, p in manifest.pages() for lang in edition.languages}
    tree = build_tree(manifest, edition, files, page_flat)
    with tempfile.TemporaryDirectory(prefix="lms-size-") as tmp:
        path = Path(tmp) / edition.archive
        write_archive(tree, path, mtime=0)
        return path.stat().st_size


def verify_staging(src: dict, stg: dict, optimized: set[str]) -> dict:
    """The staged site must export to the same pages and sources with only new media bytes."""
    out = {}
    for ed_id, a in src["editions"].items():
        b = stg["editions"][ed_id]
        ra, rb = a["report"], b["report"]
        problems = []
        for bucket in ISSUE_BUCKETS:
            if ra.get(bucket) != rb.get(bucket):
                problems.append(f"{bucket} differs")
        if [p["key"] for p in ra["pages"]] != [p["key"] for p in rb["pages"]]:
            problems.append("page list differs")
        srcs_a = sorted(s for v in ra["assets"].values() for s in v)
        srcs_b = sorted(s for v in rb["assets"].values() for s in v)
        if srcs_a != srcs_b:
            problems.append("asset source set differs")
        if len(a["files"]) != len(b["files"]):
            problems.append("static file count differs")
        if set(a["names"]) != set(b["names"]):
            problems.append("set of files read by the exporter differs")
        texts = {flat: data for flat, data in b["files"].items()
                 if os.path.splitext(flat)[1].lower() in (".html", ".css", ".js", ".mjs", ".json", ".svg")}
        for rel in sorted(optimized & set(a["names"])):
            old, new = a["names"][rel], b["names"][rel]
            if old == new:
                problems.append(f"{rel}: flat name did not change")
            if old in a["files"] and new not in b["files"]:
                problems.append(f"{rel}: optimized file {new} is not in the export")
            problems += [f"{flat} still names {old}" for flat, data in texts.items() if old.encode() in data]
        # Files that embed or import an optimized file get new names too; listed for review.
        renamed = sorted(a["names"][r] for r in set(a["names"]) & set(b["names"])
                         if r not in optimized and a["names"][r] != b["names"][r] and a["names"][r] in a["files"])
        out[ed_id] = {"ok": not problems, "problems": problems[:50], "renamed_dependents": renamed,
                      "static_files": len(b["files"]),
                      "static_bytes_before": sum(map(len, a["files"].values())),
                      "static_bytes_after": sum(map(len, b["files"].values()))}
    return out


# ------------------------------------------------------------------ commands


def _options(args) -> dict:
    return {"webp_quality": args.webp_quality, "jpeg_quality": args.jpeg_quality,
            "min_psnr": args.min_psnr, "min_psnr_png": args.min_psnr_png, "min_gain": args.min_gain}


def cmd_stage(args) -> int:
    t0 = time.time()
    site_root = Path(os.path.abspath(args.site))
    manifest_path = Path(args.manifest).resolve()
    if not args.no_optimize:
        check_image_python(args.image_python)
    print(f"reading the export closure of {site_root} ...", flush=True)
    src = export_in_memory(site_root, manifest_path)
    rels = sorted(src["read"])
    protected = [site_root, site_root.resolve(), ROOT / "web/assets", (ROOT / "web/assets").resolve(),
                 manifest_path.parent]
    protected += sorted({(site_root / rel).resolve().parent for rel in rels})
    out = check_output(Path(args.out), protected)
    report_json = Path(args.report or f"{out}.report.json").resolve()
    for r in (report_json, report_json.with_suffix(".md")):
        if os.path.lexists(r):
            raise StageError(f"report already exists: {r}")
        if any(_inside(r, p.resolve()) for p in protected) or _inside(r, out):
            raise StageError(f"report path overlaps the source or the staging site: {r}")

    source_sha = {}
    out.mkdir(parents=True)
    for rel in rels:
        data = (site_root / rel).read_bytes()
        source_sha[rel] = sha256(data)
        _make_dirs(out, os.path.dirname(rel))
        write_new(out / rel, data)
    print(f"staged {len(rels)} files ({sum(os.path.getsize(out / r) for r in rels):,} bytes) in {out}", flush=True)

    jobs = []
    for rel in rels:
        fmt = IMAGE_EXTS.get(os.path.splitext(rel)[1].lower())
        if fmt and not args.no_optimize and os.path.getsize(out / rel) >= args.min_bytes:
            jobs.append({"rel": rel, "path": str(out / rel), "format": fmt, "options": _options(args),
                         "candidate": str(out / os.path.dirname(rel) / f".{os.path.basename(rel)}.candidate")})
    print(f"optimizing {len(jobs)} images with {args.workers} CPU worker(s) ...", flush=True)
    results = run_jobs(jobs, args.image_python, args.workers) if jobs else []
    optimized = set()
    assets = []
    for job, rep in zip(jobs, results):
        cand = Path(job["candidate"])
        if rep.get("bytes_out", rep["bytes_in"]) < rep["bytes_in"] and rep["method"] != "unchanged":
            data = cand.read_bytes()
            if sha256(data) != rep["sha256_out"] or len(data) >= rep["bytes_in"]:
                raise StageError(f"candidate for {job['rel']} does not match its report")
            replace_regular(out / job["rel"], data)
            cand.unlink()
            optimized.add(job["rel"])
        elif os.path.lexists(cand):
            cand.unlink()
        assets.append({"rel": job["rel"], "source": str((site_root / job["rel"]).resolve()),
                       "editions": sorted(src["read"][job["rel"]]["editions"]),
                       "pages": sorted(src["read"][job["rel"]]["pages"]), **rep})
    leftovers = [p for p in out.rglob(".*") if p.is_file()]
    if leftovers:
        raise StageError(f"unexpected temporary files in staging: {leftovers[:5]}")

    print("re-exporting the staged site in memory to verify references ...", flush=True)
    stg = export_in_memory(out, manifest_path)
    verification = verify_staging(src, stg, optimized)
    sizes = {}
    if args.measure_archives:
        for ed_id in src["editions"]:
            ed = src["editions"][ed_id]["edition"]
            sizes[ed_id] = {"before": archive_bytes(src["manifest"], ed, src["editions"][ed_id]["files"]),
                            "after": archive_bytes(stg["manifest"], ed, stg["editions"][ed_id]["files"])}
    changed = [rel for rel in rels if sha256((site_root / rel).read_bytes()) != source_sha[rel]]
    report = {
        "tool": "prepare_lms_media", "site": str(site_root), "manifest": str(manifest_path), "staging": str(out),
        "options": _options(args), "workers": args.workers, "image_python": args.image_python,
        "staged_files": len(rels), "optimized_files": len(optimized),
        "bytes_images_before": sum(a["bytes_in"] for a in assets),
        "bytes_images_after": sum(a.get("bytes_out", a["bytes_in"]) for a in assets),
        "source_unchanged": not changed, "source_changed": changed,
        "source_sha256": source_sha, "verification": verification, "archives": sizes,
        "seconds": round(time.time() - t0, 1), "assets": assets,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_json.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    ok = report["source_unchanged"] and all(v["ok"] for v in verification.values())
    print(f"images {report['bytes_images_before']:,} -> {report['bytes_images_after']:,} bytes; "
          f"source unchanged: {report['source_unchanged']}; verification ok: {ok}")
    for ed_id, s in sizes.items():
        print(f"archive {ed_id}: {s['before']:,} -> {s['after']:,} bytes")
    print(f"report: {report_json} (+ .md)")
    return 0 if ok else 1


def markdown_report(rep: dict) -> str:
    lines = ["# LMS media staging report", "",
             f"- Site: `{rep['site']}`", f"- Staging: `{rep['staging']}`",
             f"- Options: `{json.dumps(rep['options'])}`, workers {rep.get('workers', 1)}",
             f"- Files staged: {rep.get('staged_files', len(rep['assets'])):,}; optimized: {rep['optimized_files']:,}",
             f"- Image bytes: {rep['bytes_images_before']:,} -> {rep['bytes_images_after']:,} "
             f"({1 - rep['bytes_images_after'] / max(1, rep['bytes_images_before']):.1%} smaller)",
             f"- Source files unchanged (SHA-256 before = after): {rep['source_unchanged']}"]
    for ed_id, s in (rep.get("archives") or {}).items():
        lines.append(f"- Archive `{ed_id}`: {s['before']:,} -> {s['after']:,} bytes "
                     f"({1 - s['after'] / s['before']:.1%} smaller)")
    for ed_id, v in (rep.get("verification") or {}).items():
        lines.append(f"- Export verification `{ed_id}`: {'ok' if v['ok'] else 'FAILED'} "
                     f"({v['static_files']} static files, {v['static_bytes_before']:,} -> {v['static_bytes_after']:,} bytes)"
                     + ("" if v["ok"] else "; " + "; ".join(v["problems"][:5])))
    lines += ["", "| File | Method | Bytes in | Bytes out | Size | Frames | Alpha Δ | PSNR min/mean (dB) | Durations/loop |",
              "|---|---|---:|---:|---|---:|---:|---|---|"]
    for a in sorted(rep["assets"], key=lambda a: -a["bytes_in"]):
        anim = ""
        if (a.get("frames") or 1) > 1:
            anim = f"same: {a.get('same_durations', '-')}, loop {a.get('loop')}"
        psnr = "-" if a.get("psnr_min") is None else (
            "lossless" if a["psnr_min"] == 99.0 else f"{a['psnr_min']}/{a['psnr_mean']}")
        size = "x".join(map(str, a.get("size") or [])) or "-"
        lines.append(f"| `{a['rel']}` | {a['method']}{'' if a['method'] != 'unchanged' else ' (' + str(a.get('reason')) + ')'} "
                     f"| {a['bytes_in']:,} | {a.get('bytes_out', a['bytes_in']):,} | {size} | {a.get('frames', '-')} "
                     f"| {a.get('alpha_max_diff', '-')} | {psnr} | {anim} |")
    return "\n".join(lines) + "\n"


def cmd_probe(args) -> int:
    """Optimize copies of single files into a new directory and report (no exporter needed)."""
    check_image_python(args.image_python)
    files = [Path(os.path.abspath(f)) for f in args.files]
    out = check_output(Path(args.out), [f.parent for f in files] + [f.resolve().parent for f in files]
                       + [ROOT / "web/assets", (ROOT / "web/assets").resolve()])
    out.mkdir(parents=True)
    before = {str(f): sha256(f.read_bytes()) for f in files}
    jobs = []
    for f in files:
        fmt = IMAGE_EXTS.get(f.suffix.lower())
        if not fmt:
            raise StageError(f"not a PNG/JPEG/WebP file: {f}")
        write_new(out / f.name, f.read_bytes())
        jobs.append({"rel": f.name, "path": str(out / f.name), "format": fmt, "options": _options(args),
                     "candidate": str(out / f".{f.name}.candidate")})
    results = run_jobs(jobs, args.image_python, args.workers)
    assets = []
    for f, job, rep in zip(files, jobs, results):
        cand = Path(job["candidate"])
        if os.path.lexists(cand):
            replace_regular(out / f.name, cand.read_bytes())
            cand.unlink()
        assets.append({"rel": f.name, "source": str(f), **rep})
    report = {"tool": "prepare_lms_media probe", "site": "-", "staging": str(out), "options": _options(args),
              "workers": args.workers, "optimized_files": sum(a["method"] != "unchanged" for a in assets),
              "bytes_images_before": sum(a["bytes_in"] for a in assets),
              "bytes_images_after": sum(a.get("bytes_out", a["bytes_in"]) for a in assets),
              "source_unchanged": all(sha256(Path(k).read_bytes()) == v for k, v in before.items()),
              "assets": assets}
    (out / "probe-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "probe-report.md").write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    return 0 if report["source_unchanged"] else 1


def main(argv: list[str] | None = None) -> int:
    if (argv if argv is not None else sys.argv[1:])[:1] == ["_worker"]:
        return _worker_main()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--image-python", default="/usr/bin/python3",
                       help="Python with Pillow (WebP animation, libimagequant) for the image work")
        p.add_argument("--workers", type=int, default=3, help="parallel CPU image processes (default 3)")
        p.add_argument("--webp-quality", type=int, default=DEFAULTS["webp_quality"])
        p.add_argument("--jpeg-quality", type=int, default=DEFAULTS["jpeg_quality"])
        p.add_argument("--min-psnr", type=float, default=DEFAULTS["min_psnr"],
                       help="minimum composite PSNR for lossy WebP/JPEG re-encodes (dB)")
        p.add_argument("--min-psnr-png", type=float, default=DEFAULTS["min_psnr_png"],
                       help="minimum composite PSNR for 256-color PNG palettes (dB)")
        p.add_argument("--min-gain", type=float, default=DEFAULTS["min_gain"],
                       help="minimum size reduction to re-encode an already lossy file")

    s = sub.add_parser("stage", help="copy the export closure into a new directory and optimize its images")
    s.add_argument("--site", required=True, type=Path, help="built site root (normally web)")
    s.add_argument("--manifest", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path, help="NEW staging directory (must not exist)")
    s.add_argument("--report", type=Path, help="report JSON path (default <out>.report.json; .md beside it)")
    s.add_argument("--min-bytes", type=int, default=16 * 1024, help="leave smaller images as they are")
    s.add_argument("--no-optimize", action="store_true", help="only stage the closure")
    s.add_argument("--measure-archives", action="store_true",
                   help="also report .tar.gz sizes before/after (written to a temporary directory)")
    common(s)
    s.set_defaults(func=cmd_stage)
    p = sub.add_parser("probe", help="optimize copies of single image files into a NEW directory")
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True, type=Path)
    common(p)
    p.set_defaults(func=cmd_probe)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if type(exc).__name__ not in ("StageError", "ExportError", "ManifestError"):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
