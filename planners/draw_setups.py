"""Random setup seeds, reproducibly: a random order of the ids in [low, high) from numpy's default generator (PCG64)
with a fixed seed; the first n in that order, printed sorted. A fixed draw keeps a comparison honest: nobody picks the
setups after seeing results, and anyone can draw the same list again.

    python planners/draw_setups.py <generator seed> <how many> [--low 100] [--high 1000] [--exclude <seed>:<n> ...]

--exclude skips the ids of earlier draws (generator seed : how many), e.g. to get fresh setups for a final check
that no earlier choice has seen: the next ids in the same random order take their place.
"""

import argparse

import numpy as np


def order(seed, low=100, high=1000):
    return np.random.default_rng(seed).permutation(np.arange(low, high)).tolist()


def draw(seed, n, low=100, high=1000, exclude=()):
    drop = set()
    for s, k in exclude:
        drop |= set(draw(s, k, low, high))
    return sorted([i for i in order(seed, low, high) if i not in drop][:n])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("n", type=int)
    p.add_argument("--low", type=int, default=100)
    p.add_argument("--high", type=int, default=1000)
    p.add_argument("--exclude", nargs="*", default=[])
    a = p.parse_args()
    ex = [tuple(int(x) for x in e.split(":")) for e in a.exclude]
    print(" ".join(map(str, draw(a.seed, a.n, a.low, a.high, ex))))
