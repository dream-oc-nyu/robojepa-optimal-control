"""Real-model forward/gradient/cache and centimetre-controller smoke checks.

Run only with an idle GPU, after calibration, before the baseline launch.
Writes compact JSON, never tensors. Uses a separate fresh dev-100 reset.
"""
import os
from pathlib import Path
import numpy as np
import torch
from bench.harness import build_env, encode, make_state, atomic_json
from bench.rollout import Rollout
from bench.pod.run_unit import load_model


def main():
    print(f'VERIFY_START pid={os.getpid()}', flush=True)
    wm = load_model()
    env = build_env()
    checks = []
    try:
        obs, _ = env.reset(mode='init', seed=0, reach_task_seed=100, reach_episode_idx=0)
        features, observations = [encode(wm, obs)], [obs]
        initial = float(obs['reach_distance_to_goal'])
        command = np.array([.01, 0., 0.])
        next_obs, _, _, info = env.step(dict(eef_translation=command,
                                            eef_rotation=np.zeros(3), gripper=np.array([-1.])))
        executed = next_obs['proprio_robocasa'][7:10] - obs['proprio_robocasa'][7:10]
        error = float(np.linalg.norm(executed-command))
        assert error < .003, ('centimetre control error', error, executed)
        features.append(encode(wm, next_obs)); observations.append(next_obs)
        for context_len in (1, 2):
            state = make_state(wm, features[-context_len:], observations[-context_len:])
            rollout = Rollout(wm, state, 1)
            u = torch.tensor([[[.001, .002, .003]]], device='cuda', requires_grad=True)
            # Actual full-history predictor derivative; no differentiating cached buffers.
            z = rollout(u)
            derivative = torch.autograd.grad(z.square().mean(), u)[0]
            assert torch.isfinite(derivative).all() and derivative.norm() > 0
            reference = z.detach()
            rollout.prepare_cache()
            with torch.no_grad():
                cached = rollout(u.detach())
                repeated = rollout(u.detach())
            torch.testing.assert_close(cached, reference, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(cached, repeated, rtol=0, atol=0)
            checks.append(dict(context_len=context_len, gradient=derivative.tolist(),
                cache_max_abs=float((cached-reference).abs().max()), cache_repeat_exact=True))
            del z, reference, cached, repeated, rollout
        report = dict(complete=True, commanded=command.tolist(), executed=executed.tolist(),
            execution_error_m=error, initial_distance_m=initial,
            distance_m=float(info['reach_distance_to_goal']), checks=checks)
        atomic_json(Path(os.environ['SESSION_ROOT'])/'e002/smoke.json', report)
        print(report, flush=True)
    finally:
        env.close()

if __name__ == '__main__': main()
