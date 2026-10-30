#!/usr/bin/env python3
"""Export a built Sphinx site to Open edX OLX archives (see lms_export/__init__.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lms_export.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
