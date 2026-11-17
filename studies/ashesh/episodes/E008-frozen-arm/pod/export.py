"""Read-only pod export of E008 JSON receipts, SHA256 checked before rename."""
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


def validate_entry(row):
    path = Path(row['path'])
    if (path.is_absolute() or len(path.parts) != 1 or path.name != row['path'] or
        not re.fullmatch(r'(?:replay|inst10[0-7]-seed[0-2])\.json', row['path']) or
        not isinstance(row['size'], int) or not 0 <= row['size'] <= MAX_BYTES or
        not re.fullmatch('[a-f0-9]{64}', row['sha256'])):
        raise ValueError(f'Unsafe export entry {row}')
    return path


def matches(path, row):
    return (path.is_file() and path.stat().st_size == row['size'] and
            hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'])


def copy_verified(pod, remote, out, row):
    relative = validate_entry(row)
    target = out / relative
    if matches(target, row):
        return False
    partial = target.with_suffix('.json.part')
    try:
        for attempt in range(3):
            # Remove prior partials so a failed transfer cannot reuse old bytes.
            partial.unlink(missing_ok=True)
            proc = subprocess.run(KUBE + ['cp', '-c', 'dev', f'{pod}:{remote}/{relative}', str(partial)])
            if proc.returncode == 0 and matches(partial, row):
                partial.replace(target)
                return True
            if attempt < 2:
                time.sleep(2)
        raise RuntimeError(f'Export verification failed after three attempts: {relative}')
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('planner', nargs='?', default='cem_release')
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9_]+', args.planner):
        parser.error('Invalid planner name')
    if not Path('bench/harness.py').exists():
        raise SystemExit('Run from the repository root')
    pod = subprocess.check_output(KUBE + ['get', 'pods', '-l', 'app=spatial-reasoning-gpu',
        '-o', 'jsonpath={.items[0].metadata.name}'], text=True).strip()
    if not pod:
        raise RuntimeError('No matching pod')
    remote = ROOT + '/e008/out/' + args.planner
    code = ("import pathlib,json,hashlib; r=pathlib.Path(" + repr(remote) + "); "
            "print(json.dumps([{'path':p.name,'size':p.stat().st_size,"
            "'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} "
            "for p in sorted(r.glob('*.json')) if p.is_file()]))")
    manifest = json.loads(subprocess.check_output(KUBE + ['exec', pod, '-c', 'dev', '--',
        'bash', '-lc', 'source ' + ROOT + '/session-env.sh; python -B -c ' + shlex.quote(code)], text=True))
    if not manifest:
        raise RuntimeError('No E008 receipts found; nothing exported')
    names = [str(validate_entry(r)) for r in manifest]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate export manifest entries')
    out = Path('runs/E008-frozen-arm') / args.planner
    out.mkdir(parents=True, exist_ok=True)
    copied = sum(copy_verified(pod, remote, out, row) for row in manifest)
    receipt = dict(planner=args.planner, remote=remote, files=manifest,
                   verified_files=len(manifest), copied=copied, already_verified=len(manifest)-copied)
    partial = out / 'export-manifest.json.part'
    partial.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    partial.replace(out / 'export-manifest.json')
    print(json.dumps(dict(verified_files=len(manifest), copied=copied,
                         already_verified=len(manifest)-copied, output=str(out))))


if __name__ == '__main__':
    main()
