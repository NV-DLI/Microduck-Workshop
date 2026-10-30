"""Sphinx configuration for the English course; shared settings live in docs/conf_common.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conf_common import configure  # noqa: E402

configure(globals(), "en")
