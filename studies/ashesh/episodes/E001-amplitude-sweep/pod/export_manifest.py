"""Snapshot finished instances and live logs without deleting session contents."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
root=Path(__file__).resolve().parent
assert str(root)=='/workspace/robojepa-E001-20261112T003859Z'
with (root/'pids.log').open('a') as f: f.write(f'{time.time()} export-manifest {os.getpid()}\n')
snapshot=root/'export-snapshots'/str(time.time_ns()); snapshot.mkdir(parents=True)
(snapshot/'results').mkdir()
finished=[]
for p in sorted((root/'results').glob('instance-*')):
    # The collector atomically renames the complete directory only after validation.
    if p.is_dir() and (p/'meta.json').exists():
        shutil.copytree(p,snapshot/'results'/p.name,copy_function=os.link)
        finished.append(p.name)
for folder in ['logs','artifacts','markers']:
    shutil.copytree(root/folder,snapshot/folder)
for name in ['current-stage','driver.pid','driver.log','pids.log']:
    if (root/name).exists(): shutil.copy2(root/name,snapshot/name)
files=[]
for p in sorted(snapshot.rglob('*')):
    if p.is_file():
        with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        files.append({'path':str(p.relative_to(snapshot)),'bytes':p.stat().st_size,'sha256':sha})
manifest={'session_root':str(root),'snapshot':str(snapshot),'completed_instances':finished,
    'files':files,'total_files':len(files),'total_bytes':sum(x['bytes'] for x in files)}
path=snapshot/'manifest.json'; path.write_text(json.dumps(manifest,indent=2))
with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps({'snapshot':str(snapshot),'manifest_sha256':sha}))
