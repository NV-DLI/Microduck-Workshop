"""Paths: Microduck simulation and control."""

from pathlib import Path

C = Path("@COURSE@")
M = C / "simulation/core"
O = C / "configuration/core"
INSTALL_DIR = M / "evaluation/football"
BASE_DIR = M / "evaluation/baseline"
K10_DEPLOYMENT = O / "kick_policy/export/deployment.json"


def require_installed(entry_dir):
    """Production entry check: only the root-installed copy may run."""
    if Path(entry_dir).resolve() != INSTALL_DIR.resolve():
        raise RuntimeError(
            "v121 runs only from " + str(INSTALL_DIR) + ", not " + str(entry_dir)
        )
    return INSTALL_DIR
