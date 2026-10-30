"""Export a built (multi-language) Sphinx site to Open edX OLX iframe-course archives.

Public API:
    load_manifest(path) -> Manifest
    Site(root), EditionExport(site, manifest, edition).run() -> (static files, report)
    build_tree(manifest, edition, static, page_flat), write_archive(files, output)
    validate(archive, manifest=None, edition_id=None, forbid=(), forbid_archives=())
Command line: scripts/export_lms.py {export,validate,serve,fingerprint}
Dependencies: Python 3.10+, beautifulsoup4.
"""

__version__ = "1.0.0"

from .manifest import load_manifest  # noqa: E402,F401
