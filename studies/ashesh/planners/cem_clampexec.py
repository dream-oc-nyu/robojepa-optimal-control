"""Execute the action the model was shown, with unmodified native CEM search.

Planner embodiment 2 sees clamp(a / 0.05 m, -1, 1) per axis while the
controller executes the metric command; this planner clamps that command.
"""
from planners.cem_release import CEMRelease

CLAMP_M = 0.05


class CEMClampExec(CEMRelease):
    name = 'cem_clampexec'

    def plan(self, ctx):
        return super().plan(ctx).clamp(-CLAMP_M, CLAMP_M)


Planner = CEMClampExec
