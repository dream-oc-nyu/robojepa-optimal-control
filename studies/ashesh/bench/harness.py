"""Fixed reach-v0 closed-loop harness. Simulator state never enters PlanContext."""
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from bench.interface import Feedback, PlanContext
from bench.rollout import Rollout


def bounded_plan(plan, ctx):
    plan = torch.as_tensor(plan, device=ctx.goal.device, dtype=ctx.goal.dtype)
    if plan.ndim != 2 or plan.shape[1] != 3 or not 1 <= len(plan) <= ctx.horizon_max:
        raise ValueError('Plan must be (H,3), 1 <= H <= horizon_max')
    if not torch.isfinite(plan).all():
        raise ValueError('Plan contains nonfinite values')
    norm = plan.norm(dim=-1, keepdim=True)
    return plan * (ctx.step_bound_m / norm.clamp_min(1e-30)).clamp(max=1)


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def encode(wm, obs):
    with torch.no_grad():
        # Equal-resolution inputs: release's PIL resize is an identity here.
        z = wm.encode(np.stack([obs['img_0'], obs['img_1']]), size=(240, 320), dtype=torch.float32)
    if not torch.isfinite(z).all():
        raise ValueError('Nonfinite encoder output')
    return z.reshape(1, 1, 2, 15, 20, 1664).to('cuda')


def goal_cost(goal):
    """Native .5/.5 weighted view means (retain release reduction order)."""
    def cost(z):
        per_view = (z - goal).abs().reshape(*z.shape[:-1], 2, -1).mean(-1)
        return (per_view * .5).sum(-1)
    return cost


def build_env():
    from robots.robocasa.robot_env import RoboCasaEnv
    from robots.robocasa.blocking_robot_env import ReachBlockingRoboCasaEnv
    settings = dict(reach_task_seed=0, reach_min_distance=0., reach_max_distance=.30,
                    reach_clear_radius=1., kitchen_split='pretrain')
    inner = RoboCasaEnv(task_name='CloseStandMixerHead',
        camera_names=['robot0_agentview_left', 'robot0_eye_in_hand'],
        camera_width=320, camera_height=240, seed=0, robot='PandaOmron',
        chunk_size=4, wm_embodiment='franka_omron_state_delta', reach_task=True, **settings)
    return ReachBlockingRoboCasaEnv(inner, pos_tol=.001, max_steps=200, **settings)


def make_state(wm, features, observations, embodiment_id=2):
    from app.robo_jepa.world_model import WorldModelState, poses_to_diffs
    from app.robo_jepa.embodiment_preprocessing import get_action_preprocessor, get_state_preprocessor
    name = wm.model.embodiment_list[embodiment_id]
    key = 'proprio_robocasa' if name.endswith('state_delta') else 'pose'
    poses = torch.as_tensor(np.stack([o[key] for o in observations]), dtype=torch.float32, device='cuda')
    if len(observations) > 1:
        raw_actions = (poses[1:] - poses[:-1]) if name.endswith('state_delta') else poses_to_diffs(poses.cpu()).to(poses.device)
        actions = get_action_preprocessor(name)(raw_actions[None], wm.model.action_dim)
    else:
        actions = poses.new_zeros(1, 0, wm.model.action_dim)
    poses = get_state_preprocessor(name)(poses[None], wm.model.state_dim)
    return WorldModelState.from_features(torch.cat(features, 1), actions=actions, states=poses,
        view_map=torch.tensor([[0, 2]], device='cuda'),
        embodiment_ids=torch.tensor([[embodiment_id]], device='cuda'))


def run_unit(wm, planner, instance, seed, budget_s, output, provenance, calibration=False):
    output = Path(output)
    if output.exists():
        old = json.loads(output.read_text())
        if old.get('complete') and len(old.get('ticks', [])) == 5:
            print(f'SKIP {output}', flush=True); return old
        raise ValueError(f'Incomplete published unit: {output}')
    if instance not in range(100, 108) or seed not in (0, 1, 2):
        raise ValueError('Only registered dev units are allowed')
    env = build_env()
    try:
        print(f'RESET instance={instance} seed={seed}', flush=True)
        goal_obs, _ = env.reset(mode='goal', seed=0, reach_task_seed=instance, reach_episode_idx=0)
        goal = encode(wm, goal_obs).reshape(-1).detach()
        obs, _ = env.reset(mode='init', seed=0, reach_task_seed=instance, reach_episode_idx=0)
        features, observations = [encode(wm, obs)], [obs]
        # Two equally sized views; explicit native .5/.5 reduction order.
        cost = goal_cost(goal)
        unit = dict(planner=planner.name, instance=instance, seed=seed, complete=False,
            calibration=calibration, initial_distance_m=float(obs['reach_distance_to_goal']),
            budget_s=None if math.isinf(budget_s) else budget_s, horizon_max=1, step_bound_m=.3,
            embodiment_id=2, view_weights=[.5, .5], scene_seed=0, reach_episode_idx=0,
            **provenance, ticks=[])
        planner.reset(seed)
        feedback = warm = None
        for tick in range(5):
            state = make_state(wm, features, observations)
            rollout = Rollout(wm, state, horizon_max=1)
            rollout.prepare_cache()
            ctx = PlanContext(rollout, cost, goal, 1, .3, budget_s, tick,
                              feedback=feedback, warm_start=warm, seed=seed)
            torch.cuda.synchronize(); start = time.perf_counter()
            plan = planner.plan(ctx)
            torch.cuda.synchronize(); planning_s = time.perf_counter() - start
            plan = bounded_plan(plan, ctx).detach()
            planning_calls, planning_samples = rollout.predictor_calls, rollout.predicted_samples
            commanded = plan[0]
            with torch.no_grad():
                predicted = float(cost(rollout(commanded[None, None]))[0, 0])
                model_before = float(cost(rollout(torch.zeros_like(commanded)[None, None]))[0, 0])
                before = float(cost(features[-1].reshape(-1)))
            # Seal scalar predictions before executing; no feature tensors exported.
            atomic_json(output.parent/'progress.json', dict(tick=tick, stage='predicted',
                commanded=commanded.tolist(), cost_predicted=predicted, cost_model_before=model_before,
                completed_ticks=unit['ticks']))
            new_obs, reward, done, info = env.step(dict(eef_translation=commanded.cpu().numpy(),
                eef_rotation=np.zeros(3), gripper=np.array([-1.])))
            executed = torch.as_tensor(new_obs['proprio_robocasa'][7:10] - obs['proprio_robocasa'][7:10],
                                       dtype=torch.float32, device='cuda')
            z = encode(wm, new_obs)
            after = float(cost(z.reshape(-1)))
            feedback = Feedback(commanded, executed, before, after, predicted, model_before)
            rho = feedback.rho
            row = dict(tick=tick, commanded=commanded.tolist(), executed=executed.tolist(),
                cost_before=before, cost_after=after, cost_predicted=predicted, cost_model_before=model_before,
                rho=rho if rho is not None and math.isfinite(rho) else None,
                distance_m=float(info['reach_distance_to_goal']), release_reward=float(reward),
                release_success=bool(info['success']), planning_s=planning_s,
                budget_overrun_s=max(0., planning_s-budget_s),
                predictor_calls=planning_calls, predicted_samples=planning_samples,
                harness_predictor_calls=rollout.predictor_calls-planning_calls,
                harness_predicted_samples=rollout.predicted_samples-planning_samples,
                controller_sim_steps=info['sim_steps'], blocking_converged=info['blocking_converged'])
            unit['ticks'].append(row)
            print(json.dumps(dict(instance=instance, seed=seed, **row), allow_nan=False), flush=True)
            atomic_json(output.parent/'progress.json', dict(stage='executed', **unit))
            obs = new_obs
            observations = (observations + [obs])[-2:]
            features = (features + [z])[-2:]
            warm = plan[1:]
        unit['complete'] = True
        atomic_json(output, unit)
        print(f'COMPLETE {output}', flush=True)
        return unit
    finally:
        env.close()
