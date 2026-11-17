"""Execute native CEM output as clamp(a, +/-10 cm) per axis; search untouched."""
from planners.cem_release import CEMRelease

CAP_M = 0.10


class CEMCap10(CEMRelease):
    name = 'cem_cap10'

    def plan(self, ctx):
        return super().plan(ctx).clamp(-CAP_M, CAP_M)


Planner = CEMCap10
