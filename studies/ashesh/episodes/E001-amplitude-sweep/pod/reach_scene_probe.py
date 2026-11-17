"""Smallest released Reach scene feasibility check; no benchmark download."""
import json
import os
from pathlib import Path
import time
import numpy as np
root = Path(os.environ['SESSION_ROOT'])
from scripts.deployment.robots.robocasa.generate_reach_invariants_golden import _build_env
started = time.perf_counter()
env = _build_env(0, 'pretrain')
try:
    obs, history = env.reset(mode='init', seed=0, reach_task_seed=0, reach_episode_idx=0)
    assert isinstance(obs, dict) and isinstance(history, list)
    print('OBS', [(k, getattr(v, 'shape', None)) for k, v in obs.items()], flush=True)
    np.savez_compressed(root/'artifacts/reach-initial-observation.npz', **obs)
    from PIL import Image
    for k,v in obs.items():
        if k.startswith('img_'):
            Image.fromarray(v).save(root/'artifacts'/f'{k}.png')
    (root/'artifacts/reach-scene.json').write_text(json.dumps({'scene':0, 'split':'pretrain', 'elapsed_s':time.perf_counter()-started}))
finally:
    env.close()
