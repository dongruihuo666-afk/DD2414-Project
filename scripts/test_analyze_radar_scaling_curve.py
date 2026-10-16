#!/usr/bin/env python3
"""CPU tests for the radar scaling curve analysis."""

import unittest

import numpy as np

from analyze_radar_scaling_curve import (
    bootstrap_interval,
    earliest_persistent_scale,
    weighted_mean,
)


class RadarScalingCurveAnalysisTests(unittest.TestCase):
    def test_weighted_mean_matches_expanded_observations(self):
        values = np.asarray((1.0, 3.0, 8.0))
        weights = np.asarray((2.0, 1.0, 0.0))
        self.assertAlmostEqual(weighted_mean(values, weights), 5.0 / 3.0)

    def test_bootstrap_interval_is_deterministic_and_contains_estimate(self):
        values = np.asarray((0.1, 0.2, 0.4, 0.8))
        weights = np.asarray((1.0, 2.0, 3.0, 4.0))
        indices = np.random.default_rng(125).integers(0, 4, size=(5000, 4))
        first = bootstrap_interval(values, weights, indices)
        second = bootstrap_interval(values, weights, indices)
        self.assertEqual(first, second)
        estimate = weighted_mean(values, weights)
        self.assertLess(first[0], estimate)
        self.assertGreater(first[1], estimate)

    def test_persistent_onset_rejects_earlier_transient_positive_point(self):
        scales = ('64', '256', '1024', '4096', 'full')
        intervals = ((0.01, 0.03), (-0.01, 0.02), (0.02, 0.04),
                     (0.03, 0.05), (0.02, 0.06))
        self.assertEqual(earliest_persistent_scale(scales, intervals), '1024')
        self.assertIsNone(earliest_persistent_scale(
            scales, tuple((-0.1, 0.1) for _ in scales),
        ))

    def test_paired_difference_preserves_weighted_aggregate_difference(self):
        first = np.asarray((0.1, 0.2, 0.5))
        second = np.asarray((0.2, 0.1, 0.8))
        weights = np.asarray((2.0, 3.0, 5.0))
        expected = weighted_mean(second, weights) - weighted_mean(first, weights)
        self.assertAlmostEqual(weighted_mean(second - first, weights), expected)


if __name__ == '__main__':
    unittest.main()
