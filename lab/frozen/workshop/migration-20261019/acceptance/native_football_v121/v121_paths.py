"""Explicit v121 roots. No path is inferred from where a copy happens to live.

The candidate is prepared under outputs/ and is only runnable after root copies
it to INSTALL_DIR and freezes it. execute_media.py refuses any other location.
"""
from pathlib import Path

C = Path('@COURSE@')
M = C / 'workshop/migration-20261019'
O = C / 'outputs/migration-20261019'
INSTALL_DIR = M / 'acceptance/native_football_v121'
BASE_DIR = M / 'acceptance/native_football_v117'
K10_DEPLOYMENT = O / 'kick-native-v10-fresh512x500-s980702/export/deployment.json'


def require_installed(entry_dir):
    """Production entry check: only the root-installed copy may run."""
    if Path(entry_dir).resolve() != INSTALL_DIR.resolve():
        raise RuntimeError('v121 runs only from ' + str(INSTALL_DIR) + ', not ' + str(entry_dir))
    return INSTALL_DIR
