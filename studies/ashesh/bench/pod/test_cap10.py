"""CPU torch checks for the 10 cm execute-only cap; never run native CEM."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from planners.cem_release import CEMRelease
from planners.cem_cap10 import CEMCap10, Planner


class Cap10Tests(unittest.TestCase):
    def test_elementwise_clamp_preserves_tensor_contract(self):
        cases = [
            ([[0.0293, 0.0307, -0.06], [0.10, -0.10, 0.0]],
             [[0.0293, 0.0307, -0.06], [0.10, -0.10, 0.0]]),
            ([[0.0293, 0.0307, -0.1708]],
             [[0.0293, 0.0307, -0.10]]),
            ([[0.3, -0.2, 0.06], [-0.101, 0.101, -0.3]],
             [[0.10, -0.10, 0.06], [-0.10, 0.10, -0.10]]),
        ]
        for dtype in (torch.float32, torch.float64):
            for values, expected_values in cases:
                with self.subTest(dtype=dtype, values=values):
                    native = torch.tensor(values, dtype=dtype, device='cpu')
                    expected = torch.tensor(expected_values, dtype=dtype, device='cpu')
                    ctx = object()
                    with patch.object(CEMRelease, 'plan', return_value=native) as plan:
                        actual = CEMCap10().plan(ctx)
                    plan.assert_called_once_with(ctx)
                    self.assertEqual(tuple(actual.shape), (len(values), 3))
                    self.assertEqual(actual.dtype, native.dtype)
                    self.assertEqual(actual.device, native.device)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reset_preserves_baseline_rng_state(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                baseline, planner = CEMRelease(), CEMCap10()
                baseline.reset(seed)
                planner.reset(seed)
                self.assertEqual(planner.rng[0], baseline.rng[0])
                np.testing.assert_array_equal(planner.rng[1], baseline.rng[1])
                self.assertEqual(planner.rng[2:], baseline.rng[2:])

    def test_planner_name(self):
        self.assertEqual(CEMCap10().name, 'cem_cap10')

    def test_planner_export(self):
        self.assertIs(Planner, CEMCap10)


if __name__ == '__main__':
    unittest.main()
