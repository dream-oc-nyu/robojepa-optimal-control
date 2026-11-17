"""Per axis reset (L_i, p_i, f_i) = (L0, 0, False); sat_i := |a_i| >= sat - tol.
Let m := feedback exists and ||e||_2 >= moved, s_i := sign(a_i), and
flip_i := bisect and sat_i and (p_i = -s_i) and m; f_i <- f_i or flip_i.
Halve_i := flip_i (shrink='flip'), or bisect and sat_i and m and f_i
(shrink='saturated', using updated f_i). If Halve_i, L_i <- max(L_i/2, min_step).
Saturated out_i := s_i * (L_i if bisect else L0); unsaturated a_i passes
through bit-exactly. Set p_i <- s_i if sat_i else 0. Only row 0 changes.
"""
from dataclasses import dataclass
from functools import partial

import torch

from bench.interface import PlanContext
from planners.cem_release import CEMRelease
from planners.gn_cube_a import Planner as GNA
from planners.gn_cube_b import Planner as GNB


@dataclass(frozen=True)
class LengthRuleConfig:
    l0_m: float = 0.10
    sat_m: float = 0.05
    sat_tol_m: float = 1e-5
    min_step_m: float = 0.01
    moved_m: float = 1e-3
    bisect: bool = True
    shrink: str = 'flip'

    def __post_init__(self):
        if self.shrink not in ('flip', 'saturated'):
            raise ValueError("shrink must be 'flip' or 'saturated'")


class LengthRule:
    """Wrap a base planner; call reset(seed) before the first plan of each unit."""

    def __init__(self, base, config: LengthRuleConfig = LengthRuleConfig(), *, name=None):
        self.base = base
        self.config = config
        self.name = name or f"{'bisect' if config.bisect else 'sat10'}_{base.name}_a"

    def reset(self, seed: int) -> None:
        self.base.reset(seed)
        self._lengths = [float(self.config.l0_m)] * 3
        self._prev_sat_sign = [0.0, 0.0, 0.0]
        self._flipped = [False, False, False]

    def plan(self, ctx: PlanContext) -> torch.Tensor:
        a = self.base.plan(ctx)
        out = a.clone()
        cfg = self.config
        command = a[0].tolist()
        signs = [float((value > 0) - (value < 0)) for value in command]
        saturated = [abs(value) >= cfg.sat_m - cfg.sat_tol_m for value in command]
        previous = self._prev_sat_sign[:]
        before = self._lengths[:]
        flipped_before = self._flipped[:]
        executed = None if ctx.feedback is None else ctx.feedback.executed.tolist()
        moved = ctx.feedback is not None and float(ctx.feedback.executed.norm()) >= cfg.moved_m
        flips = [False, False, False]
        for i in range(3):
            if not saturated[i]:
                continue
            flips[i] = cfg.bisect and previous[i] == -signs[i] and moved
            self._flipped[i] = self._flipped[i] or flips[i]
            halve = (flips[i] if cfg.shrink == 'flip' else
                     cfg.bisect and moved and self._flipped[i])
            if halve:
                self._lengths[i] = max(self._lengths[i] / 2, cfg.min_step_m)
            length = self._lengths[i] if cfg.bisect else cfg.l0_m
            out[0, i] = signs[i] * length
        self._prev_sat_sign = [s if sat else 0.0 for s, sat in zip(signs, saturated)]
        log = ctx.extras.get('planner_log')
        if not isinstance(log, dict):
            log = {}
        log['length_rule'] = dict(
            base_command=command, saturated=saturated, prev_sat_sign=previous,
            flip=flips, length_before=before, length_after=self._lengths[:],
            executed_prev=executed, out=out[0].tolist(),
            l0_m=float(cfg.l0_m), min_step_m=float(cfg.min_step_m),
            bisect=bool(cfg.bisect), shrink=cfg.shrink,
            flipped_before=flipped_before, flipped_after=self._flipped[:],
            rule=('halve_on_flip' if cfg.shrink == 'flip' else 'halve_on_saturated_after_flip'),
        )
        ctx.extras['planner_log'] = log
        return out


def _make(base_type, name, bisect, shrink):
    return LengthRule(base_type(), LengthRuleConfig(bisect=bisect, shrink=shrink), name=name)


PLANNERS = {
    f'{arm}_{kind}_a': partial(_make, base_type, f'{arm}_{kind}_a', enabled, shrink)
    for kind, base_type in (('cem', CEMRelease), ('gna', GNA), ('gnb', GNB))
    for arm, enabled, shrink in (('sat10', False, 'flip'), ('bisect', True, 'flip'),
                                 ('halvesat', True, 'saturated'))
}
