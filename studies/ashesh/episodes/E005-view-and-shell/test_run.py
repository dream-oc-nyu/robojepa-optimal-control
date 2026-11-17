"""Synthetic shell contract: strict ties, class split, file-order argmin."""
import importlib.util
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np


class ShellContract(unittest.TestCase):
    def test_shell_partition_and_tied_selection(self):
        path = Path(__file__).with_name('run.py')
        self.assertTrue(path.exists(), 'E005 implementation does not yet exist')
        spec = importlib.util.spec_from_file_location('e005_test_target', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        inst = dict(names=['onray-plus-0.01', 'onray-minus-0.01'] +
                    [f'offray{k // 2}-{s}-0.01' for k, s in enumerate(['plus', 'minus', 'plus', 'minus'])],
                    classes=['toward', 'away'] + ['lateral'] * 4)
        d = np.array([1., 6., 2., 3., 4., 5.])
        result = module.shell_instance(inst, d, d, 1)
        self.assertEqual((result['concordant'], result['distinct_concordant'], result['latlat_concordant']), (15, 9, 6))
        self.assertTrue(result['toward_selected'])
        tied = module.shell_instance(inst, np.zeros(6), d, 1)
        self.assertEqual((tied['concordant'], tied['total'], tied['distinct_total'], tied['latlat_total']), (0, 15, 9, 6))
        self.assertTrue(tied['toward_selected'])
        reverse = module.shell_instance(inst, -d, d, 1)
        self.assertEqual(reverse['concordant'], 0)
        self.assertFalse(reverse['toward_selected'])


if __name__ == '__main__':
    unittest.main()
