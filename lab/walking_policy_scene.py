"""Execute a trained walking GR00T policy with the preserved motor skills."""
from pathlib import Path
import hashlib
import json
import sys

import football_scene


def main():
    if '--walking-policy-server' not in sys.argv:
        raise ValueError('Provide --walking-policy-server for a trained walking checkpoint')
    here=Path(__file__).resolve().parent
    source=here/'runtime/simulation/core/evaluation/football/executor.py'
    # Keep the installed traditional release intact. Only this opt-in entry
    # selects the executor with the high-level walking policy interface.
    original_load=football_scene.load
    def load(name,path):
        return original_load(name, source if name=='isaac_execute' else path)
    football_scene.load=load
    output=Path(sys.argv[sys.argv.index('--output')+1])
    files=[Path(__file__),source,here/'learning/walking.py',here/'learning/walking_runtime.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    record=output.with_name(output.name+'-walking-policy-source.json')
    with record.open('x') as f:json.dump(hashes,f,indent=2)
    football_scene.main()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError('Walking source changed during execution')


if __name__=='__main__':main()
