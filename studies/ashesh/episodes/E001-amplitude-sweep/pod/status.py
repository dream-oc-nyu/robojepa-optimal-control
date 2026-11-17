"""Compact read-only status for a session-local stage driver."""
import json
import os
from pathlib import Path
import subprocess
import time
root=Path(__file__).resolve().parent
stage=(root/'current-stage').read_text().strip() if (root/'current-stage').exists() else 'not-started'
print(json.dumps({'root':str(root),'unix_time':time.time(),'stage':stage,
 'driver_pid':(root/'driver.pid').read_text().strip() if (root/'driver.pid').exists() else None,
 'markers':[p.name for p in (root/'markers').glob('*.done')],
 'completed_instances':[p.name for p in (root/'results').glob('instance-*') if (p/'meta.json').exists()],
 'weights':{p.name:p.stat().st_size for p in (root/'weights').glob('*')}},indent=2))
for name in ['pids.log','logs/exit.log','logs/timings.log',f'logs/{stage}.log']:
 p=root/name
 if p.exists():
  with open(p,'rb') as f:
   f.seek(max(0,p.stat().st_size-1800)); text=f.read().decode(errors='replace')
  print('\n'+name+'\n'+text)
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True))
