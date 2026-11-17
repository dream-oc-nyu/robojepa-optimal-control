"""Differentiable translation-only released WM rollout, with native CEM cache.

State-delta integration is the exact additive update (world_model.py:140-147),
without its NumPy detach. Action-only integration is zero (161-163).
"""
import torch
from app.robo_jepa.world_model import WorldModelState
from app.robo_jepa.embodiment_preprocessing import (
    get_action_preprocessor, get_state_preprocessor, resolve_embodiment_name,
)


class Rollout:
    def __init__(self, wm, state, horizon_max):
        if state.cache is not None or state.features.shape[0] != 1:
            raise ValueError('Supply one uncached real observation history')
        self.wm, self.state, self.horizon_max = wm, state, horizon_max
        self.predictor_calls = self.predicted_samples = 0
        self.cached = None
        self.name = resolve_embodiment_name(state.embodiment_ids, wm.model.embodiment_list)
        if self.name not in ('franka_omron', 'franka_omron_state_delta'):
            raise ValueError('Unsupported embodiment')

    def prepare_cache(self):
        """Native policy_client.py:1065-1120 cache priming; outside plan timer."""
        s = self.state
        cached = WorldModelState.from_features(s.features[:, :1], states=s.states[:, :1],
                    view_map=s.view_map, embodiment_ids=s.embodiment_ids)
        with torch.no_grad():
            for i in range(s.features.shape[1] - 1):
                _, cached = self.wm.step(s.actions[:, i:i+1], cached, use_cache=True,
                                        device=s.features.device, no_grad=True)
                cached.features = s.features[:, i+1:i+2]
                cached.states = s.states[:, i+1:i+2]
        self.cached = cached.to(s.features.device)

    def __call__(self, actions):
        if actions.ndim != 3 or actions.shape[-1] != 3 or not 1 <= actions.shape[1] <= self.horizon_max:
            raise ValueError('Expected (B,H,3), 1 <= H <= horizon_max')
        if not torch.isfinite(actions).all():
            raise ValueError('Nonfinite action')
        b, h, _ = actions.shape
        use_cache = not torch.is_grad_enabled() and self.cached is not None
        state = (self.cached if use_cache else self.state).repeat(b).to(actions.device)
        predictions = []
        for t in range(h):
            xyz = actions[:, t:t+1]
            raw = torch.cat([xyz, xyz.new_zeros(b, 1, 4)], -1)
            if use_cache:
                out, state = self.wm.step(raw, state, use_cache=True, no_grad=True,
                                           device=actions.device)
                out = out.to(actions.device)
            else:
                action = get_action_preprocessor(self.name)(raw, self.wm.model.action_dim)
                history = state.actions
                if history.shape[-1] != action.shape[-1]:
                    if history.shape[1]:
                        history = get_action_preprocessor(self.name)(history, action.shape[-1])
                    else:
                        history = action[:, :0]
                history = torch.cat([history, action], 1)
                poses = state.states
                if poses is not None:
                    poses = get_state_preprocessor(self.name)(poses, getattr(self.wm.model, 'state_dim', action.shape[-1]))
                camera = state.camera_extrinsics
                out = self.wm.forward(state.features, history, states=poses,
                    camera_extrinsics=None if camera is None else camera.repeat(1, state.features.shape[1], 1),
                    view_map=state.view_map, embodiment_ids=state.embodiment_ids,
                    dt=state.dt, use_cache=False)[0][:, -1:]
                if poses is not None:
                    nxt = poses[:, -1:] + action if self.name.endswith('state_delta') else torch.zeros_like(poses[:, -1:])
                    poses = torch.cat([poses, nxt], 1)
                state = WorldModelState(features=torch.cat([state.features, out], 1),
                    actions=history, states=poses, camera_extrinsics=camera,
                    view_map=state.view_map, embodiment_ids=state.embodiment_ids, dt=state.dt)
            self.predictor_calls += 1
            self.predicted_samples += b
            predictions.append(out.reshape(b, 1, -1))
        return torch.cat(predictions, 1)
