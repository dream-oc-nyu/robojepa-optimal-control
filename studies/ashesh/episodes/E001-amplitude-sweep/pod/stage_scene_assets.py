"""Stage only official fixture/texture packs, with pinned SHA256s; no datasets."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile
root = Path(os.environ['SESSION_ROOT'])
assets = root/'upstream/robocasa-456174f62b89b8fca99eaaf33949c29fec9cfc2a/robocasa/models/assets'
manifest = json.loads((root/'scene-assets.json').read_text())
receipts = []
for spec in manifest:
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
    archive = root/'tmp'/Path(spec['path']).name
    print('download',url,flush=True)
    urllib.request.urlretrieve(url, archive)
    digest = hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    assert digest == spec['sha256'] and archive.stat().st_size == spec['bytes']
    target = assets/spec['extract_to']
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert (target/name).resolve().is_relative_to(assets.resolve()), name
        z.extractall(target)
    receipts.append({**spec,'url':url,'verified':True})
    (root/'artifacts/scene-assets-receipt.json').write_text(json.dumps(receipts,indent=2))
    # Owner requested retaining the staged session, including downloaded packs.
print('verified extracted packs',len(receipts))
