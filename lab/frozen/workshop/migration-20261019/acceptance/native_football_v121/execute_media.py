#!/usr/bin/env python3
"""Explicit v121 controller plus independently frozen no-drop media executor.

Refuses unless installed at the explicit v121 location and frozen (release.json).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from v121_paths import M, require_installed  # noqa: E402
MEDIA = M / 'football/media_archive_v1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify():
    require_installed(D)
    if not (D / 'release.json').is_file():
        raise RuntimeError('v121 is not frozen: physical calibration/near-ball gates and root release pending')
    release = json.loads((D / 'release.json').read_text())
    if release.get('schema') != 'microduck.v121-retreat-recovery-release.v1' or release.get('test_only') is not False:
        raise RuntimeError('v121 release schema or test-only release refused')
    for path, expected in release['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('v121 source mismatch: ' + path)
    return release


def main():
    release = verify()
    helper = load('v121_media_frozen_helper', MEDIA / 'execute.py')
    media_release = helper.verify()
    if sys.argv[1:] == ['--verify-only']:
        print(json.dumps({'verified': True, 'control_files': len(release['source_sha256']),
                          'media_files': len(media_release['sourceFiles'])}))
        return
    if 'isaac_execute' in sys.modules:
        raise RuntimeError('Fresh process required for explicit executor binding')
    sys.path.insert(0, str(MEDIA))
    entry = helper.load('isaac_execute', D / 'executor.py')
    assert entry.ROOT == M / 'football' and entry.M == M
    controller = helper.load('v121_media_controller', D / 'execute.py')
    assert controller.entry is entry and controller.base_run is entry.run
    try:
        controller.main()
    finally:
        helper.verify()
        verify()


if __name__ == '__main__':
    main()
