"""Are two closed-loop runs of the same seeds the same episodes? Per seed: every array of the episode .npz (the key
sets must match; dtype, shape and bytes compared) and, per planning call, every plan-log field both runs log except
the bookkeeping (`seconds`, and `call`, a counter that restarts when a run resumes). A check that compares nothing
(no episodes, or a seed with no logged planning calls) is not a pass. Exit status 0 only when bit-identical. Needs
only numpy.

    python planners/compare_runs.py <OUT/<task>> <tag a> <tag b>
"""

import glob
import json
import os
import sys

import numpy as np

SKIP = {"seconds", "call"}  # plan-log fields that are bookkeeping, not part of the episode


def episodes(d):
    return {int(os.path.basename(f)[5:-6]): f for f in glob.glob(os.path.join(d, "ep_ts*_0.npz"))}


def log(d):
    p = os.path.join(d, "plan_log.jsonl")
    return {(r["seed"], r["global_step"]): r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def same_array(x, y):
    if x.dtype != y.dtype or x.shape != y.shape:
        return False
    return repr(x.tolist()) == repr(y.tolist()) if x.dtype == object else x.tobytes() == y.tobytes()


def differing_fields(x, y):
    """Plan-log fields both calls log (one run may log extra diagnostics), bookkeeping left out."""
    return sorted(f for f in (set(x) & set(y)) - SKIP if x[f] != y[f])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    root, a, b = sys.argv[1:4]
    A, B = (os.path.join(root, t) for t in (a, b))
    ea, eb, la, lb = episodes(A), episodes(B), log(A), log(B)
    if not ea and not eb:
        sys.exit(f"{a} vs {b}: no episodes in either folder, nothing compared")
    same = True
    for s in sorted(set(ea) | set(eb)):
        if s not in ea or s not in eb:
            print(f"seed {s}: only in {a if s in ea else b}")
            same = False
            continue
        za, zb = np.load(ea[s], allow_pickle=True), np.load(eb[s], allow_pickle=True)
        # an array only one file has counts as a difference
        diff = [k for k in sorted(set(za.files) | set(zb.files))
                if k not in za.files or k not in zb.files or not same_array(za[k], zb[k])]
        calls = sorted(k for k in set(la) | set(lb) if k[0] == s)
        bad = [k for k in calls if k not in la or k not in lb or differing_fields(la[k], lb[k])]
        secs = [la[k]["seconds"] / lb[k]["seconds"] for k in calls if k in la and k in lb and lb[k]["seconds"]]
        # the task's own reward, picked as summarize.py does: ObjectPush's under object_push_*, ObjectReach's under
        # object_reach_*, the reach tasks' under reach_*
        r = next(p for p in ("env_info_reach_reward", "env_info_object_push_reward", "env_info_object_reach_reward")
                 if p in za.files)
        ncr = za[r].mean(), zb[r].mean() if r in zb.files else float("nan")
        ratio = f"; planning time a/b {np.mean(secs):.2f}" if secs else ""
        print(f"seed {s}: NCR {ncr[0]:.3f} / {ncr[1]:.3f}; npz arrays differing: {diff or 'none'} of "
              f"{len(set(za.files) | set(zb.files))}; planning calls differing: {[k[1] for k in bad] or 'none'} of "
              f"{len(calls)}{ratio}")
        if bad:
            k = bad[0]
            if k not in la or k not in lb:
                print(f"  step {k[1]}: logged only in {a if k in la else b}")
            else:
                print(f"  first differing call, step {k[1]}: {', '.join(differing_fields(la[k], lb[k]))}; "
                      f"action {la[k]['action']} vs {lb[k]['action']}")
        if not calls:
            print(f"  no planning calls logged for seed {s}: the plans were not compared")
        same &= bool(calls) and not diff and not bad
    print(f"{a} vs {b}: {'bit-identical' if same else 'NOT identical'}")
    sys.exit(0 if same else 1)
