"""Session-contained registered-planner dev driver (no held-out mode).

The fixed harness never reads ``ctx.extras``. ``LoggedPlanner`` keeps a reference to each tick's
``ctx.extras['planner_log']`` (one list append inside the timed ``plan()``; no I/O, no encoding);
after a unit that actually ran, ``persist_planner_log`` writes ``planner_log.json`` next to its
``unit.json`` and prints one ``PLANNER_LOG`` line per tick. ``unit.json`` is untouched.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

from bench.harness import run_unit
from bench.pod.run_unit import load_model, ROOT, SOURCES, EXPECTED
from planners.cem_clampexec import CEMClampExec
from planners.cem_cap10 import CEMCap10
from planners.gn_cube_a import Planner as GNCubeA
from planners.gn_cube_b import Planner as GNCubeB
from planners.bisect_a import PLANNERS as LENGTH_RULE_A
from planners.bisect_b import PLANNERS as LENGTH_RULE_B

PLANNERS = {'cem_clampexec': CEMClampExec, 'cem_cap10': CEMCap10,
            'gn_cube_a': GNCubeA, 'gn_cube_b': GNCubeB}
for registry in (LENGTH_RULE_A, LENGTH_RULE_B):
    collisions = PLANNERS.keys() & registry.keys()
    assert not collisions, f'Planner registry name collision: {sorted(collisions)}'
    PLANNERS.update(registry)
assert len(PLANNERS) == 22, f'Expected 22 registered planners, got {len(PLANNERS)}'

SIDECAR = 'planner_log.json'      # never named unit.json: scorer/exporter find units by rglob('unit.json')
TICKS = 5                         # ticks per unit in the fixed harness


class LoggedPlanner:
    """Same name/reset/plan as the wrapped planner; remembers (tick, planner_log reference) per plan()."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.records = []

    def reset(self, seed):
        return self.inner.reset(seed)

    def plan(self, ctx):
        plan = self.inner.plan(ctx)
        # Timed by the harness: a reference and a list append only. Both GN planners hand over a
        # dict created inside this plan() call, so no copy is needed.
        self.records.append((ctx.tick, ctx.extras.get('planner_log')))
        return plan


def jsonable(value):
    """Strict-JSON view: non-finite floats -> None, tensors/arrays -> lists or scalars, unknown -> str."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key if isinstance(key, str) else str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    tolist = getattr(value, 'tolist', None)          # torch tensors, numpy arrays and numpy scalars
    if callable(tolist):
        detach = getattr(value, 'detach', None)
        return jsonable(detach().tolist() if callable(detach) else tolist())
    return str(value)


def persist_planner_log(planner, unit_path, instance, seed):
    """Write the sidecar and print the per-tick summary for a unit that ran now. Never raises.

    Nothing happens unless exactly ticks 0..4 were collected (a skipped complete unit collects none,
    so an existing sidecar is left alone) and at least one tick carries a log. Returns the sidecar
    path if it was written, else None.
    """
    written = None
    try:
        records = list(planner.records)
        if [tick for tick, _ in records] != list(range(TICKS)) or all(log is None for _, log in records):
            return None
        ticks = [dict(tick=int(tick), log=jsonable(log)) for tick, log in records]
        try:
            path = Path(unit_path).with_name(SIDECAR)
            temp = path.with_name(SIDECAR + '.tmp')
            try:
                with temp.open('w') as f:
                    json.dump(dict(planner=planner.name, instance=int(instance), seed=int(seed), ticks=ticks),
                              f, allow_nan=False)
                    f.write('\n'); f.flush(); os.fsync(f.fileno())
                os.replace(temp, path)
                written = path
            finally:
                temp.unlink(missing_ok=True)
        except Exception as error:      # the summary lines below are then the only copy
            print(f'PLANNER_LOG_ERROR {error!r}', flush=True)
        for row in ticks:
            log = row['log']
            if isinstance(log, dict):
                log = {key: item for key, item in log.items() if key not in ('iterations', 'config')}
            print('PLANNER_LOG ' + json.dumps(dict(instance=int(instance), seed=int(seed), tick=row['tick'],
                  log=log), allow_nan=False, separators=(',', ':')), flush=True)
    except Exception as error:
        try:
            print(f'PLANNER_LOG_ERROR {error!r}', flush=True)
        except Exception:
            pass
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--planner', required=True, choices=PLANNERS)
    p.add_argument('--seeds', default='0', help='Comma-separated planner seeds from 0,1,2')
    args = p.parse_args()
    seed_tokens = args.seeds.split(',')
    if any(seed not in ('0', '1', '2') for seed in seed_tokens):
        p.error('--seeds must be a comma-separated subset of 0,1,2')
    seeds = [int(seed) for seed in seed_tokens]
    print(f'START pid={os.getpid()} mode=dev planner={args.planner} seeds={seeds}', flush=True)
    start = time.time()
    wm = load_model()
    print(f'MODEL_READY seconds={time.time()-start:.3f}', flush=True)
    provenance = dict(source_commits=SOURCES, weight_sha256=EXPECTED, fp32=True,
                      controller='release ReachBlockingRoboCasaEnv, in-process')
    b = json.loads((ROOT/'e002/budget.json').read_text())['budget_s']
    for seed in seeds:
        for instance in range(100, 108):
            planner = LoggedPlanner(PLANNERS[args.planner]())
            output = ROOT/f'e002/runs/{planner.name}/inst{instance}-seed{seed}/unit.json'
            run_unit(wm, planner, instance, seed, b, output, provenance)
            persist_planner_log(planner, output, instance, seed)


if __name__ == '__main__':
    main()
