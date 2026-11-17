"""Differentiable translation-only RoboCasa state-delta adapter.

No surrogate dynamics or action rescaling: use the released action preprocessor.
Only additive pose integration replaces the native NumPy round trip.
"""
import torch
from contextlib import contextmanager
from app.robo_jepa.embodiment_preprocessing import franka_omron_state_delta_action_preprocess


def translation_action(xyz, action_dim):
    """Map base-frame metres to canonical ee_pos_rel delta slots 7:10."""
    planner = torch.cat([xyz, xyz.new_zeros(*xyz.shape[:-1], 4)], dim=-1)
    return franka_omron_state_delta_action_preprocess(planner, action_dim)


def advance_state(pose, action):
    """Exact state-delta additive integration, retaining action derivatives."""
    return pose.float() + action.float()


def differentiable_step(wm, xyz, state):
    """One uncached released forward with full, non-detached action/state history.

    Requires canonical padded histories, one homogeneous state_delta embodiment,
    fixed orientation/gripper, and the release's unchanged attention/config.
    """
    from app.robo_jepa.world_model import WorldModelState
    ids = state.embodiment_ids
    expected = wm.model.embodiment_list.index('franka_omron_state_delta')
    if not bool((ids == expected).all()) or state.cache is not None:
        raise ValueError('Only uncached homogeneous RoboCasa state_delta is supported')
    action = translation_action(xyz, wm.model.action_dim)
    actions = torch.cat([state.actions.to(xyz.device), action], dim=1)
    poses = state.states.to(xyz.device)
    features = state.features.to(xyz.device)
    extrinsics = state.camera_extrinsics
    camera = None if extrinsics is None else extrinsics.to(xyz.device).repeat(1, features.shape[1], 1)
    out = wm.forward(features, actions, states=poses, camera_extrinsics=camera,
                     view_map=state.view_map, embodiment_ids=ids, dt=state.dt,
                     use_cache=False)[0][:, -1:]
    updated = WorldModelState(features=torch.cat([features, out], dim=1),
                             actions=actions,
                             states=torch.cat([poses, advance_state(poses[:, -1:], action)], dim=1),
                             camera_extrinsics=extrinsics, view_map=state.view_map,
                             embodiment_ids=ids, dt=state.dt)
    return out, updated


@contextmanager
def math_attention():
    """Temporarily select exact PyTorch math SDPA for vector JVPs.

    Scoped to this process, serial use only. Upstream hard-selects fused kernels
    whose backward has no derivative. No weights/config/masks are changed.
    This is a diagnostic backend, not a claimed release-performance baseline.
    """
    from app.robo_jepa.robo_jepa.models.utils import modules
    from torch.nn.attention import SDPBackend, sdpa_kernel
    original = modules.attention

    def attention(q, k, v, attn_mask, **kwargs):
        if kwargs.get('attn_impl', 'fmha') != 'sdpa' or kwargs.get('use_alibi', False):
            raise ValueError('Math diagnostic supports only the released non-Alibi SDPA path')
        use_gqa = q.shape[1] != k.shape[1]
        if use_gqa and attn_mask is not None:
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            use_gqa = False
        with sdpa_kernel([SDPBackend.MATH]):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=kwargs.get('proj_drop_prob', 0.0),
                is_causal=kwargs.get('is_causal', False),
                enable_gqa=use_gqa, scale=kwargs.get('scale'))

    modules.attention = attention
    try:
        yield
    finally:
        modules.attention = original
