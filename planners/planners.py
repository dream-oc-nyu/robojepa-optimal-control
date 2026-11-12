"""Planners that stand in for mpc_utils.cem inside the official policy_client.py: same arguments, same return value
(rewards, means, stds). run_client.py swaps the one named by PLANNER in; the official code stays untouched.

So far the only planner is the official CEM (`cem`), the baseline. To add one, write a function with cem()'s
signature, planner(world_model, wm_state, goal_features, **kw), that returns (rewards, means, stds) as cem() does,
and add it to PLANNERS. `_problem` gives what a planner needs: `problem.features(X)` rolls plans X [N, R, 3] (the
translation of each of R steps) through the predictor, and `problem.score.J(F)` is the cost cem() minimizes: the
view-weighted mean l1 distance between the last predicted features and the goal features (lower is better).

Environment variables:
    PLANNER      the entry of PLANNERS to use (default cem; read by run_client.py)
    PLAN_CHUNK   batch size of the rollouts made here (default 32)
    PLAN_LOG     one JSON line per planning call (set by run_task.sh)
"""

import json
import os
import time

import torch

from app.robo_jepa.mpc import mpc_utils

_cem = mpc_utils.cem  # the original, taken before run_client.py swaps the module attribute

ROLLOUTS = 0  # predictor rollouts spent by the planners in the current call


def _env(name, default):
    return type(default)(os.environ.get(name, default))


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────────


def rollout_features(wm_state, actions, step_func, device, use_cache, repeat=1, count=True):
    """actions [N, R, 7] -> features of the last predicted step, [N, V, D] in fp32. Each action is applied as
    `repeat` world-model steps of action / repeat, as cem() does with action_repeat."""
    global ROLLOUTS
    chunk = _env("PLAN_CHUNK", 32)
    outs = []
    for i in range(0, len(actions), chunk):
        a = actions[i : i + chunk]
        history = wm_state.repeat(len(a)).to(device)  # repeat() already casts the features to fp32
        if history.cache is not None and not torch.is_autocast_enabled("cuda"):
            # the KV cache was filled under fp16 autocast; in an fp32 region its fp16 keys would meet fp32 queries
            c = history.cache
            c.keys, c.values, c.dtype = [k.float() for k in c.keys], [v.float() for v in c.values], torch.float32
        steps = []
        for h in range(a.shape[1]):
            for _ in range(repeat):
                state, history = step_func(
                    a[:, h : h + 1] / repeat, history, no_grad=True, device=device, use_cache=use_cache, batch=100
                )
            steps.append(state.flatten(3, 5)[:, 0].float())
        outs.append(steps[-1])
        del history
    if count:
        ROLLOUTS += len(actions)
    return torch.cat(outs)


class Scorer:
    """J (l1) for one planning problem, as cem() scores its samples."""

    def __init__(self, goal_features, view_weights, device):
        self.goal = goal_features.flatten(3, 5)[:, 0].float().to(device)  # [1, V, D]
        V = self.goal.shape[1]
        w = torch.full((V,), 1.0 / V) if view_weights is None else torch.tensor(view_weights, dtype=torch.float32)
        self.w = w.to(device)

    def J(self, F):  # [N]
        return ((F - self.goal).abs().mean(-1) * self.w).sum(-1)

    def per_view_sims(self, F):  # [N, V], what cem() returns as rewards (higher is better)
        return -(F - self.goal).abs().mean(-1)


def to_actions(X):
    """translation [N, R, 3] -> planner actions [N, R, 7] (rotation and gripper zero, as in translation_only)."""
    return torch.cat([X, torch.zeros(*X.shape[:2], 4, device=X.device)], dim=-1)


class Problem:
    """one planning call: start state, goal, and how to roll actions out."""

    def __init__(self, wm_state, goal_features, view_weights, step_func, device, use_cache, repeat):
        self.wm_state, self.step_func, self.device, self.use_cache, self.repeat = (
            wm_state, step_func, device, use_cache, repeat)
        self.score = Scorer(goal_features, view_weights, device)

    def features(self, X, count=True):  # X [N, R, 3]
        return rollout_features(self.wm_state, to_actions(X), self.step_func, self.device, self.use_cache,
                                self.repeat, count)


def _problem(world_model, wm_state, goal_features, kw):
    if not kw.get("translation_only", True) or kw.get("use_gripper", False):
        raise NotImplementedError("the planners here search translation only (the client's --translation_only)")
    step_func = kw.get("step_func") or world_model.step
    return Problem(wm_state, goal_features, kw.get("view_weights"), step_func, kw.get("device", "cuda"),
                   kw.get("use_cache", True), kw.get("action_repeat", 1))


def _horizon(kw):
    return min(kw.get("rollout", 3), kw.get("total_steps", 100) - kw.get("global_step", 0))


PLANNERS = {"cem": _cem}


# ── the wrapper run_client.py installs as mpc_utils.cem ───────────────────────────────────────────────


def make_planner(name):
    """PLANNERS[name], with one JSON line per planning call appended to PLAN_LOG: setup seed, step, wall-clock time,
    predictor rollouts, J of the executed plan (J_exec) and the executed action.

    The client runs its task files in sorted(os.listdir()) order (policy_client.py), not in the order the seeds
    were listed: ts10.json before ts3.json. TASK_DIR (run_task.sh) recovers each call's seed from that order."""
    plan = PLANNERS[name]
    calls, episode = [0], [-1]
    task_dir = os.environ.get("TASK_DIR")
    seeds = [int(f[2:-5]) for f in sorted(os.listdir(task_dir)) if f.endswith(".json")] if task_dir else None

    def planner(world_model, wm_state, goal_features, **kw):
        global ROLLOUTS
        step = kw.get("global_step", 0)  # 0 at the first call of every episode
        episode[0] += step == 0
        seed = seeds[episode[0]] if seeds and episode[0] < len(seeds) else None  # bookkeeping must never raise
        ROLLOUTS = 0
        torch.cuda.synchronize()
        t = time.time()
        rewards, means, stds = plan(world_model, wm_state, goal_features, **kw)
        torch.cuda.synchronize()
        seconds = time.time() - t
        if name == "cem":
            rollouts = kw["cem_steps"] * kw["samples"] * kw.get("num_batches", 1) * kw.get("world_size", 1)
        else:
            rollouts = ROLLOUTS
        problem = _problem(world_model, wm_state, goal_features, kw)
        x = means[-1][: _horizon(kw), :3].to(problem.device)
        J_exec = problem.score.J(problem.features(x[None], count=False)).item()  # one rollout, not counted
        rec = dict(planner=name, call=calls[0], seed=seed, global_step=step, seconds=round(seconds, 3),
                   rollouts=rollouts, J_exec=J_exec, action=x.flatten().tolist())
        log = os.environ.get("PLAN_LOG")
        if log:
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        calls[0] += 1
        return rewards, means, stds

    return planner
