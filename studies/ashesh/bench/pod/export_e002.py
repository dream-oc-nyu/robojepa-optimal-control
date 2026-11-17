"""Copy immutable small E002 receipts in individually resumable pieces."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = '/workspace/robojepa-E001-20261112T003859Z'
KUBE = ['kubectl', '--context', 'hsrn', '-n', 'wireless']
OUT = Path('runs/E002-planner-loop')
MAX_BYTES = 2_000_000
SIDECAR = 'planner_log.json'      # optional per-unit planner search log written by run_planner.py


def manifest_code(root):
    """Read-only remote listing: every unit.json, the planner log next to it if any, budget, smoke.

    A file above the size cap is listed with its size but not read (sha256 null).
    """
    return ("import pathlib,json,hashlib; r=pathlib.Path(" + repr(root) + "); "
        "units=sorted((r/'runs').rglob('unit.json')); "
        "files=units+[p.with_name(" + repr(SIDECAR) + ") for p in units]+[r/'budget.json',r/'smoke.json']; "
        "print(json.dumps([{'path':str(p.relative_to(r)), 'size':p.stat().st_size, "
        "'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size<=" + str(MAX_BYTES) + " else None} "
        "for p in files if p.is_file()]))")


def select_exports(manifest):
    """Split manifest rows into (rows to export, oversize planner logs to skip); raise on anything unsafe.

    A planner log is accepted only beside a unit.json of the same manifest. Above the per-file cap a
    planner log is skipped (the unit itself still exports); any other oversize file stays an error.
    """
    units = {str(Path(row['path']).parent) for row in manifest if Path(row['path']).name == 'unit.json'}
    selected, oversize = [], []
    for row in manifest:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.json':
            raise ValueError(f'Unsafe export entry {row}')
        sidecar = relative.name == SIDECAR
        if sidecar and str(relative.parent) not in units:
            raise ValueError(f'Unsafe export entry {row}')
        if row['size'] > MAX_BYTES:
            if not sidecar:
                raise ValueError(f'Unsafe export entry {row}')
            oversize.append(row)
            continue
        selected.append(row)
    return selected, oversize


def main():
    if not Path('bench/interface.py').exists():
        raise SystemExit('Run from the repository root')
    pod = subprocess.check_output(KUBE + ['get', 'pods', '-l', 'app=spatial-reasoning-gpu',
        '-o', 'jsonpath={.items[0].metadata.name}'], text=True).strip()
    # Nothing is written remotely; only complete, atomic receipts are exported.
    code = manifest_code(ROOT+'/e002')
    manifest = json.loads(subprocess.check_output(KUBE + ['exec', pod, '-c', 'dev', '--',
        'bash', '-lc', 'source '+ROOT+'/session-env.sh; python -c '+__import__('shlex').quote(code)], text=True))
    manifest, oversize = select_exports(manifest)
    for row in oversize:
        print(f"WARNING planner log not exported, {row['size']} bytes exceeds the {MAX_BYTES}-byte cap: {row['path']}",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for row in manifest:
        relative = Path(row['path'])
        target = OUT/(Path(*relative.parts[1:]) if relative.parts[0] == 'runs' else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        def matches(p):
            return p.exists() and p.stat().st_size == row['size'] and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
        if matches(target):
            skipped += 1; continue
        partial = target.with_suffix('.json.part')
        for attempt in range(3):
            result = subprocess.run(KUBE + ['cp', '-c', 'dev', f'{pod}:{ROOT}/e002/{relative}', str(partial)])
            if result.returncode == 0 and matches(partial):
                partial.replace(target); copied += 1; break
            time.sleep(2)
        else:
            raise RuntimeError(f'Export verification failed: {relative}')
    (OUT/'export-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(verified_files=len(manifest), copied=copied, already_verified=skipped,
        complete_units=sum(r['path'].endswith('/unit.json') for r in manifest),
        planner_logs=sum(Path(r['path']).name == SIDECAR for r in manifest),
        planner_logs_skipped_oversize=len(oversize), output=str(OUT))))

if __name__ == '__main__':
    main()
