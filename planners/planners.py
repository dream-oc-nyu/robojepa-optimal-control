"""Planners that stand in for mpc_utils.cem inside the official policy_client.py: same arguments, same return value
(rewards, means, stds). run_client.py swaps the one named by PLANNER in; the official code stays untouched.

Planners: the official CEM (`cem`, the baseline), and `adam` and `lbfgs`, which descend J by gradients through the
predictor (gradient-based shooting, see _grad_planner). To add one, write a function with cem()'s
signature, planner(world_model, wm_state, goal_features, **kw), that returns (rewards, means, stds) as cem() does,
and add it to PLANNERS. `_problem` gives what a planner needs: `problem.features(X)` rolls plans X [N, R, 3] (the
translation of each of R steps) through the predictor, and `problem.score.J(F)` is the cost cem() minimizes: the
view-weighted mean l1 distance between the last predicted features and the goal features (lower is better).

Environment variables:
    PLANNER           the entry of PLANNERS to use (default cem; read by run_client.py)
    PLAN_CHUNK        batch size of the rollouts made here (default 32)
    PLAN_LOG          one JSON line per planning call (set by run_task.sh)
    PLAN_SEED_OFFSET  k (default 0): a different random draw on the same setup
    adam / lbfgs:
    PLAN_K            candidate plans optimized in parallel (default 8): the warm start and K-1 perturbations of it
    PLAN_ITERS        Adam steps (default 20) or L-BFGS iterations (default 10)
    PLAN_LR           step size (default 0.1 Adam, 1.0 L-BFGS)
    PLAN_INIT_STD     std of the perturbations (default the client's robot_init_std)
    PLAN_GRAD_CHUNK   candidates per forward+backward pass (default 4; lower it on out-of-memory)
    PLAN_LOSS_SCALE   J is multiplied by this before the fp16 backward pass (default 4096), so small gradients don't
                      underflow; non-finite gradients are zeroed and counted (nonfinite in PLAN_LOG)
    PLAN_FP32         1: the gradient rollouts run in fp32 instead of the client's fp16 autocast (default 0)

Every planning call of a known setup seed first seeds Python's, numpy's and torch's generators from (seed, step,
PLAN_SEED_OFFSET) (make_planner), so the random samples depend on the call alone, not on what the process ran
before.
"""

import contextlib
import json
import math
import os
import random
import time

import numpy as np
import torch

from app.robo_jepa import world_model as _wm_module
from app.robo_jepa.mpc import mpc_utils
from app.robo_jepa.robo_jepa.models.utils.modules import KeyValueCache

_cem = mpc_utils.cem  # the original, taken before run_client.py swaps the module attribute

ROLLOUTS = 0  # predictor rollouts spent by the planners in the current call
BACKWARDS = 0  # of them, rollouts also differentiated (a backward pass costs about two forward ones)
EXTRA = {}  # planner-specific numbers for the current call's PLAN_LOG line


def _env(name, default):
    return type(default)(os.environ.get(name, default))


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────────


def _rollout(wm_state, a, step_func, device, use_cache, repeat, every=False):
    """actions [n, R, 7] -> features of the last predicted step, [n, V, D] in fp32 (every=True: of every plan step,
    concatenated along the last axis, [n, V, R*D]), in one batch."""
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
    del history
    return torch.cat(steps, -1) if every else steps[-1]


def rollout_features(wm_state, actions, step_func, device, use_cache, repeat=1, count=True, every=False):
    """actions [N, R, 7] -> features of the last predicted step, [N, V, D] in fp32, in batches of PLAN_CHUNK. Each
    action is applied as `repeat` world-model steps of action / repeat, as cem() does with action_repeat. every=True:
    the features of every plan step instead, concatenated along the last axis, [N, V, R*D] (make_planner's per-step
    log)."""
    global ROLLOUTS
    chunk = _env("PLAN_CHUNK", 32)
    outs = [_rollout(wm_state, actions[i : i + chunk], step_func, device, use_cache, repeat, every)
            for i in range(0, len(actions), chunk)]
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

    def features(self, X, count=True, every=False):  # X [N, R, 3]
        return rollout_features(self.wm_state, to_actions(X), self.step_func, self.device, self.use_cache,
                                self.repeat, count, every)


def _problem(world_model, wm_state, goal_features, kw):
    if not kw.get("translation_only", True) or kw.get("use_gripper", False):
        raise NotImplementedError("the planners here search translation only (the client's --translation_only)")
    step_func = kw.get("step_func") or world_model.step
    return Problem(wm_state, goal_features, kw.get("view_weights"), step_func, kw.get("device", "cuda"),
                   kw.get("use_cache", True), kw.get("action_repeat", 1))


def _horizon(kw):
    return min(kw.get("rollout", 3), kw.get("total_steps", 100) - kw.get("global_step", 0))


# ── gradient-based shooting ─────────────────────────────────────────────────────────────────────────────
#
# The official rollout is not differentiable in two places, both patched only while _differentiable() is active:
# - KeyValueCache.update stores the new keys and values detached and returns the cache buffers, so attention would
#   see no gradient through any key or value, not even those of the current frame. _update_with_grad does the same
#   bookkeeping (pointer, eviction, key_start) without in-place writes or detaching. The context frames' keys stay
#   detached, which is right: they don't depend on the plan.
# - compute_new_pose (the proprio fed back to the predictor) goes through numpy. It is run on detached inputs; for
#   the state-delta embodiments (new_state = pose + action) the identity gradient is added back straight-through.
#   For other embodiments the pose gets no gradient (action-only: it is zero anyway).

_update = KeyValueCache.update
_compute_new_pose = _wm_module.compute_new_pose


def _update_with_grad(self, new_k, new_v, layer=0, cache_size=None, update_cache=True):
    k = new_k.view(self.batch, self.num_heads, -1, self.frame_tokens, self.head_dim)
    v = new_v.view(self.batch, self.num_heads, -1, self.frame_tokens, self.head_dim)
    n = k.shape[2]
    cache_size = self.cache_len if cache_size is None else cache_size
    if not update_cache or n >= self.cache_len:
        # these branches already attend over the live new keys; what they store is only context frames
        return _update(self, new_k, new_v, layer=layer, cache_size=cache_size, update_cache=update_cache)
    keys, values = self.keys[layer], self.values[layer]
    for t in range(n):
        kt, vt = k[:, :, t : t + 1].to(keys.dtype), v[:, :, t : t + 1].to(values.dtype)
        ptr = self.pointer[layer]
        if cache_size + t < self.cache_len:  # append at the pointer
            keys = torch.cat([keys[:, :, :ptr], kt, keys[:, :, ptr + 1 :]], dim=2)
            values = torch.cat([values[:, :, :ptr], vt, values[:, :, ptr + 1 :]], dim=2)
            self.pointer[layer] += 1
        else:  # full: drop the oldest frame (the original rolls slot i <- i + 1), write at ptr - 1
            keys = torch.cat([keys[:, :, 1:], keys[:, :, :1]], dim=2)
            values = torch.cat([values[:, :, 1:], values[:, :, :1]], dim=2)
            keys = torch.cat([keys[:, :, : ptr - 1], kt, keys[:, :, ptr:]], dim=2)
            values = torch.cat([values[:, :, : ptr - 1], vt, values[:, :, ptr:]], dim=2)
            self.evicted[layer] += 1
    self.keys[layer], self.values[layer] = keys, values
    m = min(self.cache_len, cache_size + n)
    self.key_start[layer] = self.evicted[layer]
    shape = (self.batch, self.num_heads, m * self.frame_tokens, self.head_dim)
    return keys[:, :, :m].reshape(shape), values[:, :, :m].reshape(shape)


def _pose_with_grad(pose, action, embodiment_ids=None, embodiment_list=None):
    out = _compute_new_pose(pose.detach(), action.detach(), embodiment_ids, embodiment_list)
    if embodiment_ids is not None and embodiment_list is not None:
        if "state_delta" in embodiment_list[int(embodiment_ids[0, 0])]:
            K = out.shape[-1]
            live = pose[..., :K] + action[..., :K]
            out = out + (live - live.detach()).to(out.device, out.dtype)
    return out


@contextlib.contextmanager
def _differentiable(fp32):
    KeyValueCache.update, _wm_module.compute_new_pose = _update_with_grad, _pose_with_grad
    try:
        with torch.enable_grad(), (torch.autocast("cuda", enabled=False) if fp32 else contextlib.nullcontext()):
            yield
    finally:
        KeyValueCache.update, _wm_module.compute_new_pose = _update, _compute_new_pose


def _to_ball(U, r):
    """R^3 -> the open ball of radius r, smooth and one-to-one: the norm bound (cem()'s maxnorm) without projection,
    so that L-BFGS sees an unconstrained problem."""
    return r * U / torch.sqrt(1 + (U * U).sum(-1, keepdim=True))


def _from_ball(X, r):
    y = X / r
    y = y * (0.99 / y.norm(dim=-1, keepdim=True).clamp(min=0.99))  # strictly inside the ball
    return y / torch.sqrt(1 - (y * y).sum(-1, keepdim=True))


def _grad_planner(method):
    """Gradient-based shooting: minimize J over the translations of the plan, X [K, R, 3], by Adam or L-BFGS, through
    the predictor. K candidates start from cem()'s initial mean (the previous plan, shifted, padded with zeros) and K-1
    Gaussian perturbations of it; the per-step norm bound is built in (_to_ball). The plan returned is the lowest-J
    candidate seen at any evaluation. With L-BFGS the K candidates are one problem (J summed), sharing curvature
    estimates and line searches; with Adam they are independent (per-coordinate steps)."""

    def plan(world_model, wm_state, goal_features, **kw):
        if isinstance(goal_features, dict):
            raise NotImplementedError("the gradient planners score the final goal only, not sub-goal dicts")
        if kw.get("objective", mpc_utils.l1) is not mpc_utils.l1:
            raise NotImplementedError("the gradient planners minimize the l1 objective (the client's --objective l1)")
        problem = _problem(world_model, wm_state, goal_features, kw)
        dev, R, r = problem.device, _horizon(kw), kw.get("maxnorm", 0.05)
        K = _env("PLAN_K", 8)
        iters = _env("PLAN_ITERS", 10 if method == "lbfgs" else 20)
        lr = _env("PLAN_LR", 1.0 if method == "lbfgs" else 0.1)
        std0 = _env("PLAN_INIT_STD", float(kw.get("robot_init_std", 0.05)))
        chunk, scale, fp32 = _env("PLAN_GRAD_CHUNK", 4), _env("PLAN_LOSS_SCALE", 4096.0), _env("PLAN_FP32", 0)

        warm = torch.zeros(R, 3)
        prev = kw.get("prev_actions")
        if prev is not None:
            n = min(len(prev), R)
            warm[:n] = prev[:n, :3].float().cpu()
        X0 = warm.expand(K, R, 3).clone()
        X0[1:] += std0 * torch.randn(K - 1, R, 3)  # the generators are seeded per call (make_planner)
        U = _from_ball(X0, r).to(dev).requires_grad_(True)

        best_J, best_X = torch.full((K,), math.inf), X0.clone()
        best_S = torch.zeros(K, problem.score.goal.shape[1])
        stats = dict(evals=0, nonfinite=0, J_warm=None)  # J_warm: J of the warm start (candidate 0) before any step
        trace, t0 = [], time.time()  # per evaluation: (rollouts so far, best J so far, seconds so far)
        precision = (lambda: torch.autocast("cuda", enabled=False)) if fp32 else contextlib.nullcontext

        def keep(i, X, F, J):  # remember each candidate's best point
            J, X = J.detach().float().cpu(), X.detach().float().cpu()
            if stats["J_warm"] is None and i == 0:
                stats["J_warm"] = float(J[0])
            better = J < best_J[i : i + len(J)]
            best_J[i : i + len(J)][better] = J[better]
            best_X[i : i + len(J)][better] = X[better]
            best_S[i : i + len(J)][better] = problem.score.per_view_sims(F.detach()).float().cpu()[better]

        def closure():
            global BACKWARDS
            stats["evals"] += 1
            grad, total = torch.zeros_like(U), 0.0
            for i in range(0, K, chunk):
                with _differentiable(fp32):
                    X = _to_ball(U[i : i + chunk], r)
                    F = problem.features(X)
                    J = problem.score.J(F)
                    # autograd.grad, not backward(): no gradient is computed for (or left on) the model's weights
                    (g,) = torch.autograd.grad(J.sum() * scale, U)
                BACKWARDS += len(J)
                g = g / scale
                bad = ~torch.isfinite(g)
                if bad.any():
                    stats["nonfinite"] += int(bad.any(-1).any(-1).sum())
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                grad += g
                keep(i, X, F, J)
                total += J.sum().item()
                del F, J, g
            U.grad = grad
            trace.append((ROLLOUTS, float(best_J.min()), round(time.time() - t0, 3)))  # keep() synced the GPU
            return torch.tensor(total)

        if method == "adam":
            opt = torch.optim.Adam([U], lr=lr)
            for _ in range(iters):
                opt.zero_grad()
                closure()
                opt.step()
        else:
            opt = torch.optim.LBFGS([U], lr=lr, max_iter=iters, history_size=_env("PLAN_HISTORY", 20),
                                    line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)
            opt.step(closure)
        with torch.no_grad(), precision():  # the last step's point has not been scored yet
            for i in range(0, K, chunk):
                X = _to_ball(U[i : i + chunk], r)
                F = problem.features(X)
                keep(i, X, F, problem.score.J(F))

        k = int(best_J.argmin())
        plan = torch.zeros(R, 7)
        plan[:, :3] = best_X[k]
        trace.append((ROLLOUTS, float(best_J.min()), round(time.time() - t0, 3)))
        EXTRA.update(J_best=float(best_J[k]), trace=trace, **stats)
        return [best_S[k]], [plan], [torch.zeros(R, 7)]

    return plan


PLANNERS = {"cem": _cem, "adam": _grad_planner("adam"), "lbfgs": _grad_planner("lbfgs")}


# ── the wrapper run_client.py installs as mpc_utils.cem ───────────────────────────────────────────────


def make_planner(name):
    """PLANNERS[name], with one JSON line per planning call appended to PLAN_LOG: setup seed, step, the call's random
    seed, wall-clock time, predictor rollouts (forward; backwards: how many of them were also differentiated), the GPU memory allocated when the call starts and its peak during the
    call (PyTorch tensors only, torch.cuda.max_memory_allocated; the difference is what the planner itself needs),
    J of the executed plan (J_exec), the model's J after each of its steps (J_steps), the executed action, and for
    CEM the mean J of the elites in its last iteration (J_elite) and after every iteration (trace: (rollouts so far,
    J)), and for adam / lbfgs their EXTRA: J of the warm start (J_warm), the best J they found (J_best), evaluations
    of all K candidates (evals), candidates with non-finite gradients (nonfinite) and after every evaluation (trace:
    (rollouts so far, best J so far, seconds so far)). t_start is the call's start (Unix time, s): the gaps between
    calls are the time spent outside planning (env steps, encoding).

    The client runs its task files in sorted(os.listdir()) order (policy_client.py), not in the order the seeds
    were listed: ts10.json before ts3.json. TASK_DIR (run_task.sh) recovers each call's seed from that order."""
    plan = PLANNERS[name]
    calls, episode = [0], [-1]
    task_dir = os.environ.get("TASK_DIR")
    seeds = [int(f[2:-5]) for f in sorted(os.listdir(task_dir)) if f.endswith(".json")] if task_dir else None

    def planner(world_model, wm_state, goal_features, **kw):
        global ROLLOUTS, BACKWARDS
        step = kw.get("global_step", 0)  # 0 at the first call of every episode
        episode[0] += step == 0
        seed = seeds[episode[0]] if seeds and episode[0] < len(seeds) else None  # bookkeeping must never raise
        rng_seed = None if seed is None else 1000 * seed + step + 1_000_000 * _env("PLAN_SEED_OFFSET", 0)
        if rng_seed is not None:  # cem() samples from the global generators (np.random, torch.randn)
            random.seed(rng_seed)
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)  # CUDA generators too
        ROLLOUTS, BACKWARDS = 0, 0
        EXTRA.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated()
        t = time.time()
        rewards, means, stds = plan(world_model, wm_state, goal_features, **kw)
        torch.cuda.synchronize()
        seconds = time.time() - t
        mem_peak = torch.cuda.max_memory_allocated()
        if name == "cem":
            rollouts = kw["cem_steps"] * kw["samples"] * kw.get("num_batches", 1) * kw.get("world_size", 1)
        else:
            rollouts = ROLLOUTS
        problem = _problem(world_model, wm_state, goal_features, kw)
        x = means[-1][: _horizon(kw), :3].to(problem.device)
        # J after each step of the executed plan, from one rollout (not counted); J_exec is the last one
        Fs = problem.features(x[None], count=False, every=True)  # [1, V, R*D]
        D = problem.score.goal.shape[-1]
        J_steps = [problem.score.J(Fs[..., h * D : (h + 1) * D]).item() for h in range(Fs.shape[-1] // D)]
        rec = dict(planner=name, call=calls[0], seed=seed, rng_seed=rng_seed, global_step=step,
                   t_start=round(t, 3), seconds=round(seconds, 3), rollouts=rollouts, backwards=BACKWARDS,
                   gpu_mem_start_gb=round(mem_start / 1e9, 3), gpu_mem_peak_gb=round(mem_peak / 1e9, 3), J_exec=J_steps[-1], J_steps=J_steps,
                   action=x.flatten().tolist(), **EXTRA)
        if name == "cem":
            # cem() returns, per iteration, the per-view reward averaged over its elites (higher is better)
            w = problem.score.w.cpu()
            rec["J_elite"] = -(rewards[-1].float() * w).sum().item()
            per_iter = kw["samples"] * kw.get("num_batches", 1) * kw.get("world_size", 1)
            rec["trace"] = [(per_iter * (i + 1), -(r.float().cpu() * w).sum().item()) for i, r in enumerate(rewards)]
        log = os.environ.get("PLAN_LOG")
        if log:
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        calls[0] += 1
        return rewards, means, stds

    return planner
