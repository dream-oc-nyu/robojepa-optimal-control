"""Predeclared E001 candidates; units are metres in the robot base frame."""
import numpy as np

def candidates(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    axis = np.eye(3)[np.argmin(np.abs(v))]
    u = np.cross(v, axis); u /= np.linalg.norm(u)
    w = np.cross(v, u)
    rows = [('neutral0','neutral',np.zeros(3)),('neutral1','neutral',np.zeros(3))]
    for a in [.001,.003,.01,.03]:
        for s, sign in [(1,'plus'),(-1,'minus')]:
            rows.append((f'onray-{sign}-{a:g}', 'onray', s*a*v))
    for i, direction in enumerate([u,w]):
        for a in [.01,.03]:
            for s, sign in [(1,'plus'),(-1,'minus')]:
                rows.append((f'offray{i}-{sign}-{a:g}', 'offray', s*a*direction))
    rows.append(('goal','goal',.05*v))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows],dtype=np.float64), v, (u,w)
