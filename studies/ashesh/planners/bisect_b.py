"""Per-axis executed-length rule for saturated commands (E011, attempt B).

The model scores each action component only inside +/-5 cm. A base command
a_i with |a_i| >= sat (saturated) therefore carries a sign s_i = sgn(a_i)
and no length; the executed length on that axis is chosen here.

Per axis i the state is a length L_i (L0 at reset), p_i, the saturation
sign of the previous tick's base command (0 at reset or if unsaturated),
and f_i, 'this axis has flipped since reset' (False at reset).
With a = base command row 0 and e = executed translation of the previous
tick (absent on the first tick):

    sat_i  = |a_i| >= sat - tol,   s_i = sgn(a_i)
    moved  = e present and ||e||_2 >= moved
    flip_i = bisect and sat_i and p_i = -s_i and moved
    flip_i :  f_i <- True
    halve_i:  L_i <- max(L_i / 2, min_step)        (before executing)
    sat_i  :  out_i = s_i L_i   (bisect)    out_i = s_i L0   (bisect=False)
    else   :  out_i = a_i                           (unchanged)
    last   :  p_i <- s_i if sat_i else 0

The schedule `shrink` chooses halve_i (f_i read after this tick's flip):

    'flip'      :  halve_i = flip_i
    'saturated' :  halve_i = bisect and sat_i and moved and f_i

'flip' (default) halves only on a flip: L0, L0/2, L0/4, ... down to
min_step, kept through saturated ticks of unchanged sign. 'saturated'
halves on the first flip and then on every later saturated tick of that
axis that follows a moved tick, whatever the sign; the flip tick halves
once. Under both, L_i is kept through unsaturated ticks, never grows, and
returns to L0 only at reset. The flip test compares base command signs;
the executed vector enters only through `moved`, never as a length.
bisect=False is the reference arm 'sat10': every saturated component is
executed at exactly L0, L_i never changes and f_i stays False.

Only row 0 of the base output is transformed. The rule draws no random
numbers and does no I/O; what it did is in ctx.extras['planner_log']
['length_rule'] next to the base planner's own log keys.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from bench.interface import PlanContext
from planners.cem_release import CEMRelease
from planners.gn_cube_a import Planner as GNCubeA
from planners.gn_cube_b import Planner as GNCubeB


RULE_NAMES = {'flip': 'halve_on_flip', 'saturated': 'halve_on_saturated_after_flip'}


@dataclass(frozen=True)
class LengthRuleConfig:
    l0_m: float = 0.10        # L_i at reset; executed length of every saturated component if not bisect
    sat_m: float = 0.05       # the model's per-component clamp
    sat_tol_m: float = 1e-5   # saturated iff |a_i| >= sat_m - sat_tol_m
    min_step_m: float = 0.01  # floor of L_i
    moved_m: float = 1e-3     # the previous tick 'moved' iff ||executed|| >= moved_m
    bisect: bool = True       # False: reference arm sat10, L_i never changes
    shrink: str = 'flip'      # 'flip': halve on a flip; 'saturated': also on later saturated moved ticks

    def __post_init__(self):
        if self.shrink not in RULE_NAMES:
            raise ValueError(f'shrink must be one of {sorted(RULE_NAMES)}, got {self.shrink!r}')


def _finite_or_none(x: float):
    return x if math.isfinite(x) else None


class LengthRule:
    def __init__(self, base, config: LengthRuleConfig = LengthRuleConfig(), name: str = 'length_rule'):
        self.base = base
        self.config = config
        self.name = name
        self._clear()

    def _clear(self) -> None:
        self.length = [self.config.l0_m] * 3   # L_i
        self.prev_sat_sign = [0, 0, 0]         # p_i
        self.flipped = [False, False, False]   # f_i

    def reset(self, seed: int) -> None:
        self.base.reset(seed)
        self._clear()

    def plan(self, ctx: PlanContext) -> torch.Tensor:
        cfg = self.config
        base_out = self.base.plan(ctx)
        a = [float(x) for x in base_out[0].detach().cpu().tolist()]

        executed = None
        if ctx.feedback is not None:
            executed = [float(x) for x in ctx.feedback.executed.detach().cpu().tolist()]
        moved = executed is not None and math.sqrt(sum(x * x for x in executed)) >= cfg.moved_m

        out = base_out.clone()
        prev = list(self.prev_sat_sign)
        before = list(self.length)
        flipped_before = list(self.flipped)
        saturated, flips, out_log = [], [], []
        for i in range(3):
            sat = abs(a[i]) >= cfg.sat_m - cfg.sat_tol_m
            sign = (1 if a[i] > 0 else -1) if sat else 0
            flip = bool(cfg.bisect and sat and prev[i] == -sign and moved)
            if flip:
                self.flipped[i] = True
            if cfg.shrink == 'flip':
                halve = flip
            else:
                halve = bool(cfg.bisect and sat and moved and self.flipped[i])
            if halve:
                self.length[i] = max(self.length[i] / 2, cfg.min_step_m)
            if sat:
                out[0, i] = sign * (self.length[i] if cfg.bisect else cfg.l0_m)
            saturated.append(bool(sat))
            flips.append(flip)
            out_log.append(_finite_or_none(float(out[0, i])))
            self.prev_sat_sign[i] = sign

        left = ctx.extras.get('planner_log')
        log = dict(left) if isinstance(left, dict) else {}
        log['length_rule'] = {
            'rule': RULE_NAMES[cfg.shrink],
            'shrink': cfg.shrink,
            'base_command': [_finite_or_none(x) for x in a],
            'saturated': saturated,
            'prev_sat_sign': prev,
            'flip': flips,
            'flipped_before': flipped_before,
            'flipped_after': list(self.flipped),
            'length_before': before,
            'length_after': list(self.length),
            'executed_prev': None if executed is None else [_finite_or_none(x) for x in executed],
            'out': out_log,
            'l0_m': cfg.l0_m,
            'min_step_m': cfg.min_step_m,
            'bisect': cfg.bisect,
        }
        ctx.extras['planner_log'] = log
        return out


def _factory(name: str, base_cls, bisect: bool, shrink: str):
    return lambda: LengthRule(base_cls(), LengthRuleConfig(bisect=bisect, shrink=shrink), name)


# registry prefix -> (bisect, shrink)
ARMS = {'sat10': (False, 'flip'), 'bisect': (True, 'flip'), 'halvesat': (True, 'saturated')}

PLANNERS = {
    f'{arm}_{tag}_b': _factory(f'{arm}_{tag}_b', base_cls, bisect, shrink)
    for tag, base_cls in (('cem', CEMRelease), ('gna', GNCubeA), ('gnb', GNCubeB))
    for arm, (bisect, shrink) in ARMS.items()
}
