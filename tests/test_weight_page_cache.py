"""CPU tests of the page-cache advice after hashing GR00T weights (no GPU, no course weights).

Run with any Python 3.11+:
  python3 -m unittest tests/test_weight_page_cache.py

Covered with temporary sparse files and mocks:
- only large .safetensors files read through a read-only descriptor are advised, with
  (fd, 0, 0, POSIX_FADV_DONTNEED) on the hashed descriptor, at the size threshold;
- small or non-weight files, writable descriptors, missing platform support, the opt-out
  variable and an OSError from the kernel are skipped without raising;
- digests and check_package(full=True) results stay unchanged;
- the inference server advises inside the hashing `with`, after the digest.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab/groot_walking"))
import weight_page_cache as wpc


def make_file(folder, name, size, head=b"microduck"):
    path = Path(folder) / name
    with path.open("wb") as stream:
        stream.write(head)
        stream.truncate(size)
    return path


class ReleaseAfterHash(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("MICRODUCK_KEEP_WEIGHT_CACHE", None)

    def advised(self, path, mode="rb", threshold=4096):
        with mock.patch.object(wpc, "LARGE_BYTES", threshold), \
                mock.patch.object(os, "posix_fadvise", create=True) as advise, \
                path.open(mode) as stream:
            given = wpc.release_after_hash(stream)
            return given, advise, stream.fileno()

    def test_threshold_boundary(self):
        below = make_file(self.folder.name, "below.safetensors", 4095)
        at = make_file(self.folder.name, "at.safetensors", 4096)
        given, advise, _ = self.advised(below)
        self.assertFalse(given)
        advise.assert_not_called()
        given, advise, fd = self.advised(at)
        self.assertTrue(given)
        advise.assert_called_once_with(fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def test_digest_unchanged(self):
        path = make_file(self.folder.name, "model.safetensors", 3 << 20, os.urandom(4096))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        with mock.patch.object(wpc, "LARGE_BYTES", 1 << 20), path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            # The real advice on a temporary file (Linux); harmless where unsupported.
            wpc.release_after_hash(stream)
        self.assertEqual(digest, expected)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_small_and_nonweight_files_are_not_advised(self):
        # Course processor statistics (19,640 bytes) under the real threshold; others above it.
        for name, size, threshold in (("stats.safetensors", 19640, wpc.LARGE_BYTES),
                                      ("config.json", 8192, 4096),
                                      ("file-000.parquet", 8192, 4096),
                                      ("model.safetensors.bak", 8192, 4096)):
            given, advise, _ = self.advised(make_file(self.folder.name, name, size), threshold=threshold)
            self.assertFalse(given, name)
            advise.assert_not_called()

    def test_only_readonly_descriptor(self):
        path = make_file(self.folder.name, "model.safetensors", 8192)
        for mode in ("r+b", "ab", "wb"):
            given, advise, _ = self.advised(path, mode)
            self.assertFalse(given, mode)
            advise.assert_not_called()

    def test_directory_is_not_advised(self):
        folder = Path(self.folder.name) / "weights.safetensors"
        folder.mkdir()
        fd = os.open(folder, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        stream = SimpleNamespace(name=str(folder), readable=lambda: True, writable=lambda: False,
                                 fileno=lambda: fd)
        with mock.patch.object(os, "posix_fadvise", create=True) as advise, \
                mock.patch.object(wpc, "LARGE_BYTES", 0):
            self.assertFalse(wpc.release_after_hash(stream))
        advise.assert_not_called()

    def test_unsupported_platform(self):
        path = make_file(self.folder.name, "model.safetensors", 8192)
        bare = SimpleNamespace(environ=os.environ, fstat=os.fstat)
        with mock.patch.object(wpc, "os", bare), mock.patch.object(wpc, "LARGE_BYTES", 4096), \
                path.open("rb") as stream:
            self.assertFalse(wpc.release_after_hash(stream))

    def test_refused_advice_is_harmless(self):
        path = make_file(self.folder.name, "model.safetensors", 8192)
        with mock.patch.object(os, "posix_fadvise", create=True,
                               side_effect=OSError(22, "Invalid argument")), \
                mock.patch.object(wpc, "LARGE_BYTES", 4096), path.open("rb") as stream:
            self.assertFalse(wpc.release_after_hash(stream))
            self.assertFalse(stream.closed)

    def test_opt_out(self):
        os.environ["MICRODUCK_KEEP_WEIGHT_CACHE"] = "1"
        given, advise, _ = self.advised(make_file(self.folder.name, "model.safetensors", 8192))
        self.assertFalse(given)
        advise.assert_not_called()


class Callers(unittest.TestCase):
    def test_check_package_full_hash(self):
        spec = importlib.util.spec_from_file_location(
            "groot_walking_cli", ROOT / "scripts/groot_walking.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            files = [make_file(data, "model.safetensors", 8192, b"weights"),
                     make_file(data, "stats.safetensors", 100, b"stats"),
                     make_file(data, "config.json", 50, b"{}")]
            manifest = dict(files=[dict(path=p.name, bytes=p.stat().st_size,
                                        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                   for p in files])
            (data / "manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(cli, "DATA", data), \
                    mock.patch.object(wpc, "LARGE_BYTES", 4096), \
                    mock.patch.object(os, "posix_fadvise", create=True) as advise:
                self.assertEqual(cli.check_package(full=True), manifest)
                self.assertEqual(advise.call_count, 1)
                self.assertEqual(advise.call_args.args[1:], (0, 0, os.POSIX_FADV_DONTNEED))
                with mock.patch.object(cli, "release_after_hash") as release:
                    cli.check_package(full=False)
                release.assert_not_called()
                manifest["files"][0]["sha256"] = "0" * 64
                (data / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(SystemExit):
                    cli.check_package(full=True)

    def test_server_advises_after_digest_inside_hashing_block(self):
        source = (ROOT / "lab/groot_walking/rollout/walking_history_server.py").read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.With) and "file_digest" in ast.unparse(node):
                body = [ast.unparse(statement) for statement in node.body]
                break
        else:
            self.fail("hashing block not found")
        self.assertEqual(body, [
            "digest = hashlib.file_digest(stream, 'sha256').hexdigest()",
            "release_after_hash(stream)",
        ])


if __name__ == "__main__":
    unittest.main()
