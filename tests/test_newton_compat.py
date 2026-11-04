"""The course adapter must never weaken an explicit restricted import request."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "lab/course_labs/newton_compat.py"
spec = spec_from_file_location("course_newton_compat", PATH)
compat = module_from_spec(spec)
spec.loader.exec_module(compat)


class ImportCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls

        class Builder:
            def add_usd(self, source, *, root_path=None):
                calls.append((source, root_path))
                return "loaded"

        self.builder = Builder
        self.source = SimpleNamespace(read_text=lambda _: json.dumps({
            "vcs_info": {"commit_id": compat.NEWTON_COMMIT}
        }))
        self.modules = patch.dict(sys.modules, {"newton": SimpleNamespace(ModelBuilder=Builder)})
        self.metadata = patch.object(compat, "distribution", return_value=self.source)
        self.modules.start(); self.metadata.start()
        self.addCleanup(self.modules.stop); self.addCleanup(self.metadata.stop)

    def test_explicit_permission_and_ordinary_call_keep_arguments_and_result(self):
        compat.install()
        self.assertEqual(self.builder().add_usd("fixed.usd", root_path="/World",
                                               allow_unsafe_torch_checkpoints=True), "loaded")
        self.assertEqual(self.builder().add_usd("plain.usd"), "loaded")
        self.assertEqual(self.calls, [("fixed.usd", "/World"), ("plain.usd", None)])

    def test_restricted_or_non_boolean_permission_fails_before_loading(self):
        compat.install()
        for value in [False, None, 1, "true"]:
            with self.assertRaises(ValueError):
                self.builder().add_usd("asset.usd", allow_unsafe_torch_checkpoints=value)
        self.assertEqual(self.calls, [])

    def test_unknown_keywords_remain_errors(self):
        compat.install()
        with self.assertRaises(TypeError):
            self.builder().add_usd("asset.usd", unrelated_new_option=True)
        self.assertEqual(self.calls, [])

    def test_install_is_idempotent(self):
        compat.install(); first = self.builder.add_usd
        compat.install()
        self.assertIs(first, self.builder.add_usd)

    def test_other_revision_is_rejected(self):
        self.source.read_text = lambda _: '{"vcs_info":{"commit_id":"different"}}'
        with self.assertRaises(RuntimeError):
            compat.install()

    def test_unexpected_import_signature_is_rejected(self):
        self.builder.add_usd = lambda self, source, **kwargs: None
        with self.assertRaises(RuntimeError):
            compat.install()


if __name__ == "__main__":
    unittest.main()
