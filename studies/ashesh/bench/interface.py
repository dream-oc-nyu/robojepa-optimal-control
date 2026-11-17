"""Planner contract for the reach benchmark. The harness owns everything here; planners only consume it.

A planner sees the learned model and the goal features. It never sees simulator state, physical
goal position, or realized outcomes of actions it has not yet executed.

Units: actions are commanded end-effector translations in metres, one row per model tick.
Cost: C(z) = mean |z - g| over all feature entries (the release's native goal cost, mean-normalised).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Protocol

if TYPE_CHECKING:  # torch exists only on the pod; the contract stays importable anywhere
    from torch import Tensor


@dataclass(frozen=True)
class Feedback:
    """What the last executed tick revealed. All costs use the same goal features."""

    commanded: "Tensor"        # (3,) action sent
    executed: "Tensor"         # (3,) end-effector translation that happened (from proprioception)
    cost_before: float         # C(E(o_t))      realized, encoder on the observation before the tick
    cost_after: float          # C(E(o_{t+1}))  realized, encoder on the observation after the tick
    cost_predicted: float      # C(F(h_t, u_t)) what the model predicted for that tick
    cost_model_before: float   # C(F(h_t, 0))   model's prediction for a null action from the same history

    @property
    def rho(self) -> Optional[float]:
        """Trust-region ratio: realized cost reduction / predicted cost reduction. None if predicted ~ 0."""
        pred = self.cost_model_before - self.cost_predicted
        return None if abs(pred) < 1e-12 else (self.cost_before - self.cost_after) / pred


@dataclass
class PlanContext:
    rollout: Callable[["Tensor"], "Tensor"]  # (B, H, 3) actions -> (B, H, P) predicted features; differentiable
    cost: Callable[["Tensor"], "Tensor"]     # (..., P) features -> (...) goal cost
    goal: "Tensor"                           # (P,)
    horizon_max: int                         # largest H the harness will roll out
    step_bound_m: float                      # per-tick ||u|| limit enforced by the harness (it clips)
    budget_s: float                          # wall-clock allowed for this plan() call
    tick: int                                # index of the executed tick about to be planned
    feedback: Optional[Feedback] = None      # None on the first tick
    warm_start: Optional["Tensor"] = None    # (H, 3) previous plan shifted by one tick, if any
    seed: int = 0
    extras: dict = field(default_factory=dict)


class Planner(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def plan(self, ctx: PlanContext) -> "Tensor":
        """Return (H, 3) commanded actions, H <= ctx.horizon_max. The harness executes row 0, then replans."""
        ...
