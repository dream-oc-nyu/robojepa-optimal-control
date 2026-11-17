"""Read-only status, including exact owned PID liveness and recent output."""
import json
import os
from pathlib import Path
ROOT = Path('/workspace/robojepa-E001-20261112T003859Z')
for name in ('e002-calibration', 'e002-cem'):
    job = ROOT/'jobs'/name
    pid = int((job/'pid').read_text()) if (job/'pid').exists() else None
    alive = False
    if pid:
        try: os.kill(pid, 0); alive = True
        except ProcessLookupError: pass
    print(json.dumps(dict(job=str(job), pid=pid, alive=alive,
        state=(job/'state').read_text().strip() if (job/'state').exists() else 'not-started')))
    if (job/'log').exists():
        print('\n'.join((job/'log').read_text(errors='replace').splitlines()[-15:]))
units = sorted((ROOT/'e002/runs').rglob('unit.json'))
print('UNIT_FILES', len(units), [str(p.relative_to(ROOT/'e002/runs')) for p in units])
if (ROOT/'e002/budget.json').exists():
    print('BUDGET', (ROOT/'e002/budget.json').read_text())
