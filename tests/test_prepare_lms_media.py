"""CPU tests of the LMS media staging utility (scripts/prepare_lms_media.py).

Image checks need Pillow with WebP animation and libimagequant; the staging test needs
BeautifulSoup (the exporter) and runs the image work in /usr/bin/python3. Run both:
  /usr/bin/python3 -m unittest tests/test_prepare_lms_media.py
  .local/docs-venv/bin/python -m unittest tests/test_prepare_lms_media.py
Every case works in its own temporary directory; nothing in the repository is written.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_lms_media as plm  # noqa: E402

try:
    from PIL import Image, ImageDraw
    import PIL.features
    HAVE_PIL = PIL.features.check("webp_anim") and PIL.features.check_feature("libimagequant")
except ImportError:
    HAVE_PIL = False
try:
    import bs4  # noqa: F401
    HAVE_BS4 = True
except ImportError:
    HAVE_BS4 = False
IMAGE_PYTHON = "/usr/bin/python3"


def raw_png(width, height, pixel):
    """An uncompressed RGB PNG written without Pillow (stored deflate blocks)."""
    rows = b"".join(b"\0" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 0)) + chunk(b"IEND", b""))


@unittest.skipUnless(HAVE_PIL, "needs Pillow with WebP animation and libimagequant")
class ImageTests(unittest.TestCase):
    def animated_source(self):
        # Smooth photographic-like content (gradients) on a transparent canvas, a moving
        # antialiased disc and a semi-transparent band, as in the course hero.
        frames, durations = [], []
        shade = Image.merge("RGB", (Image.linear_gradient("L").resize((320, 192)),
                                    Image.radial_gradient("L").resize((320, 192)),
                                    Image.linear_gradient("L").rotate(90).resize((320, 192))))
        for i in range(8):
            im = Image.new("RGBA", (320, 192), (0, 0, 0, 0))
            mask = Image.new("L", (1280, 768), 0)
            ImageDraw.Draw(mask).ellipse((80 + 96 * i, 120, 560 + 96 * i, 600), fill=255)
            im.paste(shade.rotate(8 * i), (0, 0), mask.resize((320, 192), Image.LANCZOS))
            band = Image.new("RGBA", (320, 24), (30, 120, 30, 128))
            im.alpha_composite(band, (0, 168))
            frames.append(im)
            durations.append(66 if i % 3 else 67)
        buf = io.BytesIO()
        frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=durations, loop=0,
                       lossless=True)
        return buf.getvalue(), frames, durations

    def test_animated_webp_keeps_alpha_timing_loop_and_order(self):
        data, frames, durations = self.animated_source()
        out, rep = plm.optimize_bytes(data, "webp", {})
        self.assertIsNotNone(out, rep)
        self.assertLess(len(out), len(data))
        self.assertEqual(rep["alpha_max_diff"], 0)
        self.assertTrue(rep["same_durations"] and rep["same_loop"] and rep["same_frames"] and rep["same_size"])
        im = Image.open(io.BytesIO(out))
        self.assertEqual((im.n_frames, im.info["loop"], im.size), (8, 0, (320, 192)))
        for i in range(im.n_frames):
            im.seek(i)
            im.load()
            self.assertEqual(im.info["duration"], durations[i])
            got = im.convert("RGBA")
            self.assertEqual(got.getchannel("A").tobytes(), frames[i].getchannel("A").tobytes())
            # Frame order: each frame is closest to its own source frame.
            own = plm._composite_psnr(frames[i], got)
            for j in (i - 1, i + 1):
                if 0 <= j < len(frames):
                    self.assertGreater(own, plm._composite_psnr(frames[j], got))
        self.assertGreaterEqual(rep["psnr_min"], plm.DEFAULTS["min_psnr"])

    def test_png_palette_or_lossless_keeps_pixels_and_alpha(self):
        im = Image.new("RGBA", (300, 200), (255, 255, 255, 0))
        d = ImageDraw.Draw(im)
        d.rectangle((20, 20, 280, 180), fill=(40, 40, 40, 255))
        d.text((40, 40), "Isaac Sim - Newton", fill=(118, 185, 0, 255))
        d.ellipse((150, 90, 260, 170), fill=(200, 60, 60, 140))
        buf = io.BytesIO()
        im.save(buf, "PNG", compress_level=0)
        out, rep = plm.optimize_bytes(buf.getvalue(), "png", {})
        self.assertIsNotNone(out)
        self.assertEqual(rep["alpha_max_diff"], 0)
        back = Image.open(io.BytesIO(out)).convert("RGBA")
        self.assertEqual(back.size, im.size)
        self.assertEqual(back.getchannel("A").tobytes(), im.getchannel("A").tobytes())
        self.assertGreaterEqual(rep["psnr_min"], plm.DEFAULTS["min_psnr_png"])
        self.assertEqual(Image.open(io.BytesIO(out)).format, "PNG")

    def test_jpeg_reencode_needs_gain_and_quality(self):
        im = Image.new("RGB", (320, 240))
        im.putdata([((x * 3) % 256, (y * 5) % 256, (x * y) % 256) for y in range(240) for x in range(320)])
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=98)
        out, rep = plm.optimize_bytes(buf.getvalue(), "jpeg", {})
        if out is not None:
            self.assertLess(len(out), len(buf.getvalue()) * (1 - plm.DEFAULTS["min_gain"]))
            self.assertGreaterEqual(rep["psnr_min"], plm.DEFAULTS["min_psnr"])
            self.assertEqual(Image.open(io.BytesIO(out)).format, "JPEG")
        else:
            self.assertEqual(rep["method"], "unchanged")

    def test_already_small_file_is_kept(self):
        im = Image.new("P", (16, 16))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        out, rep = plm.optimize_bytes(buf.getvalue(), "png", {})
        self.assertIsNone(out)
        self.assertEqual(rep["method"], "unchanged")


class SafetyTests(unittest.TestCase):
    def test_refuses_existing_symlinked_and_overlapping_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            src = tmp / "site"
            (src / "assets").mkdir(parents=True)
            with self.assertRaises(plm.StageError):
                plm.check_output(src, [src])  # exists
            with self.assertRaises(plm.StageError):
                plm.check_output(src / "stage", [src])  # inside the source
            with self.assertRaises(plm.StageError):
                plm.check_output(tmp / "x", [tmp / "x" / "y"])  # contains a protected path
            (tmp / "link").symlink_to(tmp / "elsewhere", target_is_directory=True)
            (tmp / "elsewhere").mkdir()
            with self.assertRaises(plm.StageError):
                plm.check_output(tmp / "link" / "stage", [src])  # through a symlink
            with self.assertRaises(plm.StageError):
                plm.check_output(tmp / "link", [src])  # the output itself is a symlink
            self.assertEqual(plm.check_output(tmp / "new" / "stage", [src]), tmp / "new" / "stage")

    def test_writes_never_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            victim = tmp / "victim.png"
            victim.write_bytes(b"original")
            (tmp / "staged.png").symlink_to(victim)
            with self.assertRaises(FileExistsError):
                plm.write_new(tmp / "staged.png", b"new")
            with self.assertRaises(plm.StageError):
                plm.replace_regular(tmp / "staged.png", b"new")
            self.assertEqual(victim.read_bytes(), b"original")


@unittest.skipUnless(HAVE_BS4 and os.path.exists(IMAGE_PYTHON), "needs bs4 and /usr/bin/python3 with Pillow")
class StageTests(unittest.TestCase):
    def test_stage_copies_closure_optimizes_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            site = tmp / "site"
            (site / "en").mkdir(parents=True)
            (site / "media").mkdir()
            shot = raw_png(240, 160, lambda x, y: (40 if x < 120 else 200, 120, 60 if y < 80 else 220))
            (site / "media" / "shot.png").write_bytes(shot)
            (site / "media" / "unused.png").write_bytes(shot + b"")  # never referenced
            (site / "media" / "bg.png").write_bytes(raw_png(64, 64, lambda x, y: (x * 4, y * 4, 0)))
            (site / "media" / "alias.png").symlink_to("shot.png")  # in-site symlink becomes a file
            (site / "en" / "style.css").write_text(".x{background:url(../media/bg.png)}")
            (site / "en" / "page.html").write_text(
                '<!DOCTYPE html><html><head><meta charset="utf-8"><link rel="stylesheet" href="style.css"></head>'
                '<body><div role="main"><img src="../media/shot.png" alt="a"><img src="../media/alias.png" alt="b">'
                "</div></body></html>")
            (tmp / "lms").mkdir()
            manifest = tmp / "lms" / "manifest.json"
            manifest.write_text(json.dumps({
                "languages": {"en": {"label": "English", "html_lang": "en"}},
                "extract": {"profile": "basic"},
                "outline": [{"id": "ch", "title": "Ch", "sections": [{"id": "sec", "title": "Sec", "pages": [
                    {"id": "page", "title": "Page", "paths": {"en": "en/page.html"}}]}]}],
                "editions": [{"id": "en", "languages": ["en"], "course": {
                    "org": "ORG", "course": "COURSE", "run": "RUN", "title": "Course", "language": "en",
                    "start": "2026-01-01T00:00:00Z"}}]}))
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in site.rglob("*") if p.is_file()}
            out = tmp / "stage" / "site"
            code = plm.main(["stage", "--site", str(site), "--manifest", str(manifest), "--out", str(out),
                             "--min-bytes", "1", "--workers", "2", "--image-python", IMAGE_PYTHON])
            self.assertEqual(code, 0)
            after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in site.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            staged = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if not p.is_dir())
            self.assertEqual(staged, ["en/page.html", "en/style.css", "media/alias.png", "media/bg.png",
                                      "media/shot.png"])
            for rel in staged:
                self.assertTrue((out / rel).is_file() and not (out / rel).is_symlink(), rel)
            report = json.loads((tmp / "stage" / "site.report.json").read_text())
            self.assertTrue(report["source_unchanged"])
            self.assertTrue(report["verification"]["en"]["ok"], report["verification"])
            shot_rep = next(a for a in report["assets"] if a["rel"] == "media/shot.png")
            self.assertNotEqual(shot_rep["method"], "unchanged")
            self.assertLess((out / "media/shot.png").stat().st_size, len(shot))
            self.assertTrue((tmp / "stage" / "site.report.md").exists())
            # A second run into the same directory is refused.
            self.assertEqual(plm.main(["stage", "--site", str(site), "--manifest", str(manifest),
                                       "--out", str(out), "--image-python", IMAGE_PYTHON]), 2)


if __name__ == "__main__":
    unittest.main()
