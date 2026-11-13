"""Regression checks for the element-ID representation correction."""
import unittest
from action_metrics import target_indices_equal


class TargetIndexTests(unittest.TestCase):
    def test_integer_strings(self):
        self.assertTrue(target_indices_equal([8117], ['8117']))
        self.assertTrue(target_indices_equal(['0008117', ' +54 '], [8117, 54]))
        self.assertTrue(target_indices_equal([-2], ['-2']))

    def test_mixed_encodings_and_order(self):
        self.assertTrue(target_indices_equal([54, '55'], ['54', 55]))
        self.assertFalse(target_indices_equal([54, 55], ['55', '54']))

    def test_distinct_targets_and_extra_actions(self):
        self.assertFalse(target_indices_equal([493], ['504']))
        self.assertFalse(target_indices_equal([22264], ['22264', '22273']))
        self.assertFalse(target_indices_equal([1, 1], ['1']))

    def test_no_boolean_or_float_coercion(self):
        self.assertFalse(target_indices_equal([True], [1]))
        self.assertFalse(target_indices_equal([8117.0], ['8117']))
        self.assertFalse(target_indices_equal(['8117.0'], [8117]))
        self.assertFalse(target_indices_equal(['8.117e3'], [8117]))

    def test_non_integer_ids_remain_exact(self):
        self.assertTrue(target_indices_equal(['element-1'], ['element-1']))
        self.assertFalse(target_indices_equal(['element-1'], ['1']))

    def test_missing_lists_are_not_matches(self):
        self.assertFalse(target_indices_equal(None, None))
        self.assertFalse(target_indices_equal([], None))
        self.assertTrue(target_indices_equal([], []))


if __name__ == '__main__':
    unittest.main()
