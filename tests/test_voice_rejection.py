"""Regression tests for the calibration-only rejection rule and verification gate."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import voice_model as vm


class VoiceRejectionTests(unittest.TestCase):
    def test_calibration_max_is_rejected_with_inclusive_gate(self):
        scores = np.array([.32, .5, .64, .89])
        threshold = vm.calibrate_rejection_threshold(scores)
        self.assertGreater(threshold, scores.max())
        self.assertFalse(np.any(scores >= threshold))
        self.assertLess(threshold, .9)

    def test_perfect_calibration_confidence_fails_closed(self):
        self.assertGreater(vm.calibrate_rejection_threshold([1.0]), 1.0)

    def test_invalid_calibration_is_not_replaced_by_permissive_default(self):
        for scores in ([], [np.nan], [np.inf], [-.1], [1.1], [[.5]]):
            with self.assertRaises(ValueError):
                vm.calibrate_rejection_threshold(scores)

    def check_gate(self, scores, expected, threshold):
        with patch.object(vm, 'load_model', return_value={'classes':['amazing','gilbert','kate'],'threshold':threshold}), \
             patch.object(vm, '_probabilities', return_value=np.array(scores)):
            return vm.verify_voice('unused.wav', expected)

    def test_rejects_ambiguous_tie(self):
        self.assertFalse(self.check_gate([.5,.5,0], 'amazing', .5)[0])
        self.assertFalse(self.check_gate([.5,.5,0], 'gilbert', .5)[0])

    def test_genuine_claim_approved_but_wrong_claim_denied(self):
        self.assertTrue(self.check_gate([.02,.96,.02], ' Gilbert ', .9)[0])
        self.assertFalse(self.check_gate([.02,.96,.02], 'kate', .9)[0])
        self.assertEqual(self.check_gate([.02,.96,.02], 'stranger', .9), (False,0.0))

    def test_exact_limit_can_pass_but_invalid_override_cannot(self):
        self.assertTrue(self.check_gate([.1,.8,.1], 'gilbert', .8)[0])
        with self.assertRaises(ValueError):
            self.check_gate([.1,.8,.1], 'gilbert', -.1)


if __name__ == '__main__':
    unittest.main()
