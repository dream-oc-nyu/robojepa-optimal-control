"""Session-contained reach-v0 calibration and dev driver (no held-out mode)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from bench.harness import run_unit, atomic_json
from planners.cem_release import CEMRelease

ROOT = Path(os.environ['SESSION_ROOT'])
SOURCES = {'robo_jepa':'f93106f5f4a451fd526d51778aa8ab7c852eb824',
 'robosuite':'5ce6643f3092639d08f7b0f90ed1c6a84f50552c',
 'robocasa':'456174f62b89b8fca99eaaf33949c29fec9cfc2a',
 'vjepa2':'204698b45b3712590f06245fbfba32d3be539812'}
EXPECTED = {'predictor':'bdd77b212d81e11438fd6bd85525bed72ff1f47a6b3211ac925ddfbc7c8351a6',
 'encoder':'7aae1a3c7a31d258af9c985388b5d2f20587469380f2e11b54d7876ac8cfe58a'}


def load_model():
    from app.robo_jepa.world_model import WorldModel
    torch.set_num_threads(4)
    torch.manual_seed(29)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, filename in [('predictor','predictor.pth.tar'), ('encoder','encoder.pt')]:
        with (ROOT/'weights'/filename).open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != EXPECTED[name]:
            raise ValueError(f'Weight hash mismatch: {name}')
    wm = WorldModel.load_pretrained(str(ROOT/'weights'), weights_filename='predictor.pth.tar',
        load_decoder=False, load_compiled=False, load_encoder=False, device='cuda')
    ckpt = torch.load(ROOT/'weights/predictor.pth.tar', map_location='cpu', weights_only=True)
    wm.transition_model.load_state_dict({k.replace('module.', ''):v for k,v in
        ckpt['predictor' if 'predictor' in ckpt else 'model'].items()}, strict=True)
    del ckpt
    wm.transition_model.eval().requires_grad_(False)
    wm.load_encoder(str(ROOT/'weights/encoder.pt'), use_proj=False, compile=False)
    wm.encoder.eval().requires_grad_(False)
    assert wm.model.embodiment_list[2] == 'franka_omron'
    assert all(p.dtype == torch.float32 for p in wm.model.parameters())
    assert all(p.dtype == torch.float32 for p in wm.encoder.parameters())
    return wm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--dev', action='store_true')
    p.add_argument('--seeds', default='0,1,2', help='Comma-separated planner seeds from 0,1,2')
    args = p.parse_args()
    if args.calibrate == args.dev:
        p.error('Choose --calibrate or --dev')
    seed_tokens = args.seeds.split(',')
    if any(seed not in ('0', '1', '2') for seed in seed_tokens):
        p.error('--seeds must be a comma-separated subset of 0,1,2')
    seeds = [int(seed) for seed in seed_tokens]
    print(f'START pid={os.getpid()} mode={"calibration" if args.calibrate else "dev"}'
          + (f' seeds={seeds}' if args.dev else ''), flush=True)
    start = time.time()
    wm = load_model()
    print(f'MODEL_READY seconds={time.time()-start:.3f}', flush=True)
    provenance = dict(source_commits=SOURCES, weight_sha256=EXPECTED, fp32=True,
                      controller='release ReachBlockingRoboCasaEnv, in-process')
    if args.calibrate:
        unit = run_unit(wm, CEMRelease(), 100, 0, float('inf'),
            ROOT/'e002/runs/_calibration/inst100-seed0/unit.json', provenance, calibration=True)
        values = [t['planning_s'] for t in unit['ticks']]
        b = statistics.median(values)
        atomic_json(ROOT/'e002/budget.json', dict(budget_s=b, planning_s=values,
            calibration_unit='runs/_calibration/inst100-seed0/unit.json',
            aggregation='median of five native CEM plan() wall times, CUDA synchronized',
            embodiment_id=2, horizon_max=1, step_bound_m=.3, **provenance))
        print(f'BUDGET_S={b:.9f}', flush=True)
    else:
        b = json.loads((ROOT/'e002/budget.json').read_text())['budget_s']
        for seed in seeds:
            for instance in range(100, 108):
                run_unit(wm, CEMRelease(), instance, seed, b,
                    ROOT/f'e002/runs/cem_release/inst{instance}-seed{seed}/unit.json', provenance)

if __name__ == '__main__':
    main()
