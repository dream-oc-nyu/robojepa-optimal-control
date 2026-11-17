"""Build a compact immutable export, excluding env/assets/weights/caches."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root=Path(__file__).resolve().parent
assert str(root).startswith('/workspace/robojepa-E001-')
assert (root/'markers/collect.done').exists(), 'Collection driver must have stopped'
export=root/'export'
assert not export.exists(), 'Export already exists; validate it rather than overwrite'
export.mkdir()
for folder in ['results','logs','artifacts']:
    shutil.copytree(root/folder,export/folder)
for name in ['pids.log','driver.pid','current-stage','driver.log']:
    if (root/name).exists(): shutil.copy2(root/name,export/name)
(export/'code').mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.py','.sh','.json'}:
        shutil.copy2(path,export/'code'/path.name)
shutil.copy2(root/'weights/config.json',export/'code/model-config.json')
items=[]
for p in sorted(export.rglob('*')):
    if p.is_file():
        with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        items.append({'path':str(p.relative_to(export)),'bytes':p.stat().st_size,'sha256':sha})
manifest={'files':items,'total_files':len(items),'total_bytes':sum(x['bytes'] for x in items),
 'completed_instances':[p.name for p in sorted((export/'results').glob('instance-*')) if (p/'meta.json').exists()]}
(export/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=root/'export.tar'
with tarfile.open(archive,'w') as tf: tf.add(export,arcname='remote')
parts=root/'export-parts'; parts.mkdir()
with archive.open('rb') as f:
    i=0
    while True:
        block=f.read(256*1024*1024)
        if not block: break
        (parts/f'part-{i:03d}').write_bytes(block); i+=1
part_manifest=[]
for p in sorted(parts.iterdir()):
    with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    part_manifest.append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha})
(parts/'parts.json').write_text(json.dumps(part_manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
print('parts',len(part_manifest))
