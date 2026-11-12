"""Markdown tables from run folders written by scripts/run_task.sh (OUT/<task>/<tag>/). Needs only numpy.

    python planners/summarize.py closed <OUT/<task>> [tag ...]               one row per run folder

A tag "a+b" pools run folders a and b into one row (one configuration split over several jobs). The score is the
paper's normalized cumulative reward (NCR, the mean of r_t over the episode's steps), or with METRIC=final the final
reward r_T (the paper scores ObstacleReach and ObjectPush this way). r_t = max(0, 1 - d_t / d_0), d the distance to
the goal of the end effector (Reach, ObstacleReach) or of the object (ObjectReach, ObjectPush).
"""

import glob
import json
import os
import sys

import numpy as np


def ms(v, f="{:.3f}"):
    v = np.asarray(v, dtype=float)
    return (f + " +- " + f).format(v.mean(), v.std()) if len(v) > 1 else f.format(v.mean())


def per_seed(root, tag):
    """seed -> score, d_T (cm) and the episode's plan-log lines, for run folder tag ("a+b": folders pooled)."""
    final = os.environ.get("METRIC") == "final"
    out = {}
    for d in (os.path.join(root, t) for t in tag.split("+")):
        p = os.path.join(d, "plan_log.jsonl")
        log = [json.loads(line) for line in open(p)] if os.path.exists(p) else []
        for f in sorted(glob.glob(os.path.join(d, "ep_ts*_0.npz"))):
            seed = int(os.path.basename(f)[5:-6])
            z = np.load(f, allow_pickle=True)
            # ObjectPush logs the object's reward and distance under object_push_*, ObjectReach under
            # object_reach_*, the reach tasks the end effector's under reach_*
            k = next((p for p in ("env_info_reach_", "env_info_object_push_", "env_info_object_reach_")
                      if p + "reward" in z.files), "env_info_reach_")
            r = z[k + "reward"]
            out[seed] = dict(score=r[-1] if final else r.mean(), dT=100 * z[k + "distance_to_goal"][-1],
                             calls=[c for c in log if c.get("seed") == seed])
    return out


def closed(root, tags):
    tags = tags or sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    score = "final reward" if os.environ.get("METRIC") == "final" else "NCR"
    print(f"| Run | Seeds | Rollouts / episode | Planning time / episode (s) | J_exec, step 1 | d_T (cm) | {score} |")
    print("|---|---|---|---|---|---|---|")
    for tag in tags:
        runs = per_seed(root, tag)
        if not runs:
            continue
        calls = [r["calls"] for r in runs.values() if r["calls"]]
        roll = [sum(c["rollouts"] for c in ep) for ep in calls]
        secs = [sum(c["seconds"] for c in ep) for ep in calls]
        J1 = [ep[0]["J_exec"] for ep in calls]
        seeds = ",".join(str(s) for s in sorted(runs))
        print(f"| `{tag}` | {seeds} | {ms(roll, '{:,.0f}') if roll else '-'} | {ms(secs, '{:.0f}') if secs else '-'} | "
              f"{ms(J1, '{:.4f}') if J1 else '-'} | {ms([r['dT'] for r in runs.values()], '{:.2f}')} | "
              f"{ms([r['score'] for r in runs.values()])} |")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "closed":
        sys.exit(__doc__)
    closed(sys.argv[2], sys.argv[3:])
