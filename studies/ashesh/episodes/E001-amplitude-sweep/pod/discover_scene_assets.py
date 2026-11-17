"""Discover only missing official lightwheel categories from bounded scene attempts."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import zipfile
root=Path(os.environ['SESSION_ROOT'])
assets=root/'upstream/robocasa-456174f62b89b8fca99eaaf33949c29fec9cfc2a/robocasa/models/assets'
catalog=json.loads((root/'optional-object-assets.json').read_text())
receipts=[]
for attempt in range(12):
    result=subprocess.run([sys.executable,'reach_scene_probe.py'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    (root/f'artifacts/scene-discovery-{attempt}.log').write_text(result.stdout)
    print('attempt',attempt,'exit',result.returncode, result.stdout[-1300:],flush=True)
    if result.returncode==0:
        break
    missing=re.findall(r"FileNotFoundError:.*?/objects/lightwheel/([^/]+)/",result.stdout)
    if not missing:
        raise RuntimeError('Not an asset-category error; inspect log rather than altering scene')
    spec=next(x for x in catalog if x['path']==f'objects_lightwheel/{missing[-1]}.zip')
    url=f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
    archive=root/'tmp'/Path(spec['path']).name
    urllib.request.urlretrieve(url,archive)
    assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==spec['sha256']
    target=assets/spec['extract_to']
    with zipfile.ZipFile(archive) as z:
        assert all((target/n).resolve().is_relative_to(assets.resolve()) for n in z.namelist())
        z.extractall(target)
    receipts.append(spec)
    (root/'artifacts/optional-object-assets-receipt.json').write_text(json.dumps(receipts,indent=2))
    # Owner requested retaining the staged session, including downloaded packs.
else:
    raise RuntimeError('12-attempt asset-discovery bound reached')
