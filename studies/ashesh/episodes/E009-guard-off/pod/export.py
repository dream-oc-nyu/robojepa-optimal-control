"""Read-only pod export of the two E009 units; SHA256-verified and resumable."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = '/workspace/robojepa-E001-20261112T003859Z'
KUBE = ['kubectl', '--context', 'hsrn', '-n', 'wireless']
MAX_BYTES = 40_000_000
NAMES = {f'inst{i}-seed0/unit.json' for i in (100, 107)}


def validate_entry(row):
    if (row['path'] not in NAMES or type(row['size']) is not int or
            not 0 < row['size'] <= MAX_BYTES or
            not re.fullmatch('[a-f0-9]{64}', row['sha256'])):
        raise ValueError(f'Unsafe export entry: {row}')
    return Path(row['path'])


def matches(path, row):
    return (path.is_file() and path.stat().st_size == row['size'] and
            hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'])


def copy_verified(pod, remote, out, row):
    relative = validate_entry(row)
    out = Path(out).resolve()
    target = out / relative
    if not target.resolve().is_relative_to(out):
        raise ValueError('Export target escapes output directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    if matches(target, row):
        return False
    partial = target.with_suffix('.json.part')
    try:
        for attempt in range(3):
            partial.unlink(missing_ok=True)
            proc = subprocess.run(KUBE + ['cp', '-c', 'dev', f'{pod}:{remote}/{relative}', str(partial)])
            if proc.returncode == 0 and matches(partial, row):
                partial.replace(target)
                if not matches(target, row):
                    raise RuntimeError(f'Published checksum mismatch: {target}')
                return True
            if attempt < 2:
                time.sleep(2)
        raise RuntimeError(f'Export verification failed after three attempts: {relative}')
    finally:
        partial.unlink(missing_ok=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not Path('bench/harness.py').is_file():
        raise SystemExit('Run from the repository root')
    pod = subprocess.check_output(KUBE + ['get', 'pods', '-l', 'app=spatial-reasoning-gpu',
        '-o', 'jsonpath={.items[0].metadata.name}'], text=True).strip()
    if not pod:
        raise RuntimeError('No matching pod')
    remote = ROOT + '/e009/runs/cem_release_guardoff'
    # The remote operation only reads two known files; it never stages a script.
    code = f'''import pathlib,json,hashlib
root=pathlib.Path({remote!r})
rows=[]
for instance in (100,107):
    relative=f'inst{{instance}}-seed0/unit.json'
    path=root/relative
    if not path.is_file():
        raise RuntimeError('Missing guard-off unit: '+str(path))
    if not 0 < path.stat().st_size <= {MAX_BYTES}:
        raise ValueError('Unit exceeds 40 MB cap: '+str(path))
    raw=path.read_bytes()
    unit=json.loads(raw)
    if (unit.get('complete') is not True or len(unit.get('ticks',[])) != 5 or
        unit.get('instance') != instance or unit.get('seed') != 0 or
        unit.get('planner_label') != 'cem_release_guardoff' or
        unit.get('reach_v0_unit') is not False):
        raise ValueError('Incomplete/incompatible unit: '+str(path))
    rows.append(dict(path=relative,size=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(rows))
'''
    manifest = json.loads(subprocess.check_output(KUBE + ['exec', pod, '-c', 'dev', '--',
        'bash', '-lc', 'set -e; source ' + ROOT + '/session-env.sh; python -B -c ' +
        shlex.quote(code)], text=True))
    names = [str(validate_entry(row)) for row in manifest]
    if len(names) != 2 or set(names) != NAMES:
        raise ValueError('Expected exactly the two E009 units')
    out = Path('runs/E009-guard-off/cem_release_guardoff')
    copied = sum(copy_verified(pod, remote, out, row) for row in manifest)
    print(json.dumps(dict(verified_files=2, copied=copied, already_verified=2-copied,
                          output=str(out), files=manifest)))


if __name__ == '__main__':
    main()
