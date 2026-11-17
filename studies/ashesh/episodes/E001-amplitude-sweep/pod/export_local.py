"""Resume-safe export transport and independent local integrity validation.

Run from the local repository. Does not clean the remote root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--out',required=True)
a=p.parse_args(); root=a.root; out=Path(a.out).resolve()
assert root.startswith('/workspace/robojepa-E001-') and '..' not in root
out.mkdir(parents=True,exist_ok=True)
k=['kubectl','--context','hsrn','-n','wireless']
pod=subprocess.check_output(k+['get','pods','-l','app=spatial-reasoning-gpu','-o','jsonpath={.items[0].metadata.name}'],text=True)
parts=out/'transport'; parts.mkdir(exist_ok=True)
def copy(remote,local):
    for attempt in range(4):
        result=subprocess.run(k+['cp',pod+':'+remote,str(local),'-c','dev','--retries=3'])
        if result.returncode==0: return
        time.sleep(3)
    raise RuntimeError('kubectl cp failed: '+remote)
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
copy(root+'/export-parts/parts.json',parts/'parts.json')
items=json.loads((parts/'parts.json').read_text())
for row in items:
    path=parts/row['name']
    if not path.exists() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
        copy(root+'/export-parts/'+row['name'],path)
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row
    print('verified transport',row['name'],flush=True)
archive=out/'export.tar'
with archive.open('wb') as target:
    for row in items:
        with (parts/row['name']).open('rb') as source:
            while block:=source.read(8*1024*1024): target.write(block)
with tarfile.open(archive) as tf:
    assert all((out/m.name).resolve().is_relative_to(out) and not m.issym() and not m.islnk() for m in tf.getmembers())
    tf.extractall(out,filter='data')
remote=out/'remote'; manifest=json.loads((remote/'manifest.json').read_text())
assert len(manifest['files'])==manifest['total_files']
assert sum(r['bytes'] for r in manifest['files'])==manifest['total_bytes']
for row in manifest['files']:
    path=remote/row['path']; assert path.is_file(),row
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row
# Exercise the exact schema, float32/float64 types, finiteness and seal files.
from validate_raw import validate
import numpy as np
summaries=[]
for folder in sorted((remote/'results').glob('instance-*')):
    summaries.append(validate(folder))
    meta=json.loads((folder/'meta.json').read_text())
    with np.load(folder/meta['archive'],allow_pickle=False) as d:
        for j,row in enumerate(meta['trials']):
            seal=folder/f'sealed-{j:02d}.npy'
            assert sha(seal)==row['seal_sha256']
            np.testing.assert_array_equal(np.load(seal,allow_pickle=False),d['z_pred_cmd'][j])
assert len(summaries)==len(manifest['completed_instances'])
receipt={'verified':True,'files':manifest['total_files'],'bytes':manifest['total_bytes'],
 'manifest_sha256':sha(remote/'manifest.json'),'instances':summaries,'remote_root':root}
(out/'export-verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
