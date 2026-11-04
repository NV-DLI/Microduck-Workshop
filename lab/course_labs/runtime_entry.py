"""Start a student physics entry with the course's pinned import adapter."""
from pathlib import Path
import runpy
import sys

from newton_compat import install


def main():
    entry = Path(sys.argv[1]).resolve()
    install()
    sys.argv = [str(entry), *sys.argv[2:]]
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
