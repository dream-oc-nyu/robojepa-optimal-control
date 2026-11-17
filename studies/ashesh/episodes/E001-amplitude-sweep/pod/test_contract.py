"""Independent E001 candidate contract, not a model-quality test."""
import unittest
import numpy as np

class CandidateTest(unittest.TestCase):
    def test_fixed_sweep_geometry_and_repeatable_seed(self):
        from sweep_contract import candidates
        names, kinds, xyz, v, axes = candidates(101)
        self.assertEqual(len(set(names)), 19)
        self.assertEqual(list(kinds).count('neutral'), 2)
        self.assertEqual(list(kinds).count('onray'), 8)
        self.assertEqual(list(kinds).count('offray'), 8)
        self.assertEqual(list(kinds).count('goal'), 1)
        np.testing.assert_array_equal(xyz, candidates(101)[2])
        self.assertFalse(np.array_equal(v, candidates(102)[3]))
        np.testing.assert_allclose(np.stack([v,*axes]) @ np.stack([v,*axes]).T, np.eye(3), atol=1e-14)
        np.testing.assert_array_equal(xyz[:2], np.zeros((2,3)))
        np.testing.assert_allclose(sorted(xyz[kinds=='onray'] @ v), [-.03,-.01,-.003,-.001,.001,.003,.01,.03], atol=1e-14)
        np.testing.assert_allclose(xyz[kinds=='offray'] @ v, 0, atol=1e-14)
        np.testing.assert_allclose(sorted(np.linalg.norm(xyz[kinds=='offray'],axis=1)), [.01]*4+[.03]*4)
        np.testing.assert_allclose(xyz[-1], .05*v, atol=1e-14)
        self.assertEqual(xyz.dtype,np.float64)

if __name__ == '__main__': unittest.main()
