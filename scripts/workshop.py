#!/usr/bin/env python3
"""Compatibility entry for older commands; the course entry is scripts/run.py.

`python3 scripts/workshop.py <command>` runs exactly `python3 scripts/run.py <command>`,
and modules that still import from `workshop` receive the same objects.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import *  # noqa: E402,F401,F403  re-exports ROOT, LOCAL, config, environment, fresh, ...
from run import cli  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).with_name("run.py"))
    cli()
