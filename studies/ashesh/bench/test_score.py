"""Score immutable episode receipts, not precomputed headline metrics."""
import unittest
from bench.score import summarize, paired_bootstrap

class ScoreTests(unittest.TestCase):
    def test_raw_distances_determine_progress_and_release_return(self):
        units = [{"planner": "cem_release", "instance": 100, "seed": s,
                  "initial_distance_m": .2,
                  "ticks": [{"distance_m": .1, "release_reward": .5,
                             "release_success": False, "rho": 2., "planning_s": 3.,
                             "budget_overrun_s": 0.}]} for s in range(3)]
        result = summarize(units)
        self.assertEqual(result["cem_release"]["per_instance"], {100: .5})
        self.assertEqual(result["cem_release"]["median_progress"], .5)
        self.assertEqual(result["cem_release"]["mean_release_return"], .5)
        self.assertEqual(result["cem_release"]["units"], 3)

    def test_bootstrap_resamples_paired_instances_not_individual_seeds(self):
        result = paired_bootstrap({100: .8, 101: .4}, {100: .3, 101: -.1}, draws=100)
        self.assertAlmostEqual(result["delta_median"], .5)
        self.assertAlmostEqual(result["ci95"][0], .5)
        self.assertAlmostEqual(result["ci95"][1], .5)
        self.assertEqual(result["paired_instances"], 2)

if __name__ == "__main__":
    unittest.main()
