"""No-argument, idempotent local-side E001 export; never cleans the pod."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT='/workspace/robojepa-E001-20261112T003859Z'
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'runs/E001-amplitude-sweep'; OUT.mkdir(parents=True,exist_ok=True)
K=['kubectl','--context','hsrn','-n','wireless']
def run(argv): return subprocess.check_output(argv,text=True)
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def cp(remote,local):
    subprocess.run(K+['cp',POD+':'+remote,str(local),'-c','dev','--retries=3'],check=True,stdout=subprocess.DEVNULL)
# Serialize overlapping cron/coordinator exports locally, not the collector.
import fcntl
lock=(OUT/'export.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX)
POD=run(K+['get','pods','-l','app=spatial-reasoning-gpu','-o','jsonpath={.items[0].metadata.name}'])
# Safe to refresh this read-only helper even while collection is running.
subprocess.run(K+['cp',str(Path(__file__).with_name('export_manifest.py')),POD+':'+ROOT+'/export_manifest.py','-c','dev'],check=True,stdout=subprocess.DEVNULL)
receipt=json.loads(run(K+['exec',POD,'-c','dev','--','bash','-lc',
    f'source {ROOT}/session-env.sh; python {ROOT}/export_manifest.py']))
snapshot=receipt['snapshot']; assert snapshot.startswith(ROOT+'/export-snapshots/')
manifest_path=OUT/'manifest.next.json'; cp(snapshot+'/manifest.json',manifest_path)
assert sha(manifest_path)==receipt['manifest_sha256'],'Manifest transport hash mismatch'
manifest=json.loads(manifest_path.read_text())
assert len(manifest['files'])==manifest['total_files']
assert sum(x['bytes'] for x in manifest['files'])==manifest['total_bytes']
missing=[]
for row in manifest['files']:
    path=OUT/row['path']; assert path.resolve().is_relative_to(OUT)
    if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
        missing.append(row['path'])
if missing:
    request=OUT/'export-files.null'; request.write_bytes(b'\0'.join(x.encode() for x in missing)+b'\0')
    subprocess.run(K+['cp',str(request),POD+':'+snapshot+'/export-files.null','-c','dev'],check=True,stdout=subprocess.DEVNULL)
    archive=OUT/'export-delta.tar'
    with archive.open('wb') as f:
        subprocess.run(K+['exec',POD,'-c','dev','--','tar','-cf','-','-C',snapshot,'--null','-T',snapshot+'/export-files.null'],check=True,stdout=f)
    with tarfile.open(archive) as tf:
        members=tf.getmembers()
        assert all((OUT/m.name).resolve().is_relative_to(OUT) and m.isfile() for m in members)
        assert {m.name for m in members}==set(missing)
        tf.extractall(OUT,filter='data')
for row in manifest['files']:
    path=OUT/row['path']
    assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row
from validate_raw import validate
import numpy as np
for name in manifest['completed_instances']:
    folder=OUT/'results'/name; validate(folder)
    meta=json.loads((folder/'meta.json').read_text())
    with np.load(folder/meta['archive'],allow_pickle=False) as data:
        for j,row in enumerate(meta['trials']):
            seal=folder/f'sealed-{j:02d}.npy'
            assert sha(seal)==row['seal_sha256']
            np.testing.assert_array_equal(np.load(seal,allow_pickle=False),data['z_pred_cmd'][j])
manifest_path.replace(OUT/'manifest.json')
verified={'verified':True,'session_root':ROOT,'remote_root_retained':True,
    'instances':manifest['completed_instances'],'files':manifest['total_files'],
    'bytes':manifest['total_bytes'],'manifest_sha256':receipt['manifest_sha256'],
    'transferred_files':len(missing),'skipped_verified_files':manifest['total_files']-len(missing),
    'verified_unix_seconds':time.time()}
(OUT/'export-verification.json').write_text(json.dumps(verified,indent=2))
# Only local transfer scratch is removed; no deletion on the pod.
for name in ['export-delta.tar','export-files.null']:
    p=OUT/name
    if p.exists(): p.unlink()
print(f"E001 export verified: {len(verified['instances'])}/8 instances, {verified['files']} files, {verified['bytes']} bytes; {verified['transferred_files']} copied, {verified['skipped_verified_files']} skipped; pod root retained.")
