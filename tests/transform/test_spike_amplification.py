#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import sys
import unittest

import numpy as np
import pandas as pd

from merlion.utils import TimeSeries, UnivariateTimeSeries
from merlion.transform.normalize import SpikeAmplification
from merlion.transform.factory import TransformFactory

logger = logging.getLogger(__name__)


class TestSpikeAmplification(unittest.TestCase):
    """Tests the SpikeAmplification transform."""

    def setUp(self):
        """Create test data with spikes."""
        np.random.seed(42)
        n = 100

        # Create base time series with normal distribution
        timestamps = pd.date_range(start="2020-01-01", periods=n, freq="1h")
        base_values = np.random.randn(n) * 10 + 50

        # Add some spikes (extreme values)
        spike_indices = [10, 25, 40, 60, 80]
        for idx in spike_indices:
            base_values[idx] += np.random.uniform(20, 40)

        self.ts = TimeSeries.from_pd(pd.Series(base_values, index=timestamps, name="value"))

    def test_linear_amplification(self):
        """Test basic linear spike amplification."""
        logger.info("Testing linear spike amplification")

        transform = SpikeAmplification(spike_threshold_quantile=0.75, amplification_factor=1.5, use_power=False)

        # Train the transform
        transform.train(self.ts)

        # Apply transform
        transformed = transform(self.ts)

        # Check that high values are amplified
        orig_values = self.ts.univariates[self.ts.names[0]].np_values
        trans_values = transformed.univariates[transformed.names[0]].np_values

        # Values above threshold should be amplified (increased)
        threshold = transform.threshold[self.ts.names[0]]
        above_threshold = orig_values > threshold

        if above_threshold.any():
            # Mean of amplified values should be higher than original
            self.assertGreater(
                trans_values[above_threshold].mean(),
                orig_values[above_threshold].mean(),
                "High values should be amplified",
            )

        # Test inversion
        inverted = transform.invert(transformed)
        inv_values = inverted.univariates[inverted.names[0]].np_values

        # Inversion should approximately recover original values
        rel_error = np.abs((inv_values - orig_values) / (orig_values + 1e-8)).mean()
        self.assertLess(rel_error, 0.1, f"Inversion error too high: {rel_error}")

    def test_power_amplification(self):
        """Test power-based spike amplification."""
        logger.info("Testing power spike amplification")

        transform = SpikeAmplification(spike_threshold_quantile=0.8, amplification_power=1.3, use_power=True)

        # Train and apply
        transform.train(self.ts)
        transformed = transform(self.ts)

        orig_values = self.ts.univariates[self.ts.names[0]].np_values
        trans_values = transformed.univariates[transformed.names[0]].np_values

        # Check amplification occurred
        threshold = transform.threshold[self.ts.names[0]]
        above_threshold = orig_values > threshold

        if above_threshold.any():
            self.assertGreater(
                trans_values[above_threshold].mean(),
                orig_values[above_threshold].mean(),
                "High values should be amplified with power transform",
            )

    def test_multivariate(self):
        """Test spike amplification on multivariate time series."""
        logger.info("Testing multivariate spike amplification")

        np.random.seed(42)
        n = 100
        timestamps = pd.date_range(start="2020-01-01", periods=n, freq="1h")

        # Create multivariate time series
        values1 = np.random.randn(n) * 10 + 50
        values2 = np.random.randn(n) * 5 + 30

        # Add spikes to both series
        values1[20] += 30
        values1[50] += 40
        values2[30] += 20
        values2[70] += 25

        ts_multi = TimeSeries(
            {
                "var1": UnivariateTimeSeries(timestamps, values1, "var1"),
                "var2": UnivariateTimeSeries(timestamps, values2, "var2"),
            }
        )

        transform = SpikeAmplification(spike_threshold_quantile=0.75, amplification_factor=1.5)

        transform.train(ts_multi)
        transformed = transform(ts_multi)

        # Check both variables are transformed
        self.assertEqual(len(transformed.univariates), 2)
        self.assertIn("var1", transformed.names)
        self.assertIn("var2", transformed.names)

        # Check inversion works
        inverted = transform.invert(transformed)
        for name in ts_multi.names:
            orig = ts_multi.univariates[name].np_values
            inv = inverted.univariates[name].np_values
            rel_error = np.abs((inv - orig) / (orig + 1e-8)).mean()
            self.assertLess(rel_error, 0.1, f"Inversion error too high for {name}: {rel_error}")

    def test_factory_creation(self):
        """Test creating transform via TransformFactory."""
        logger.info("Testing TransformFactory creation")

        transform = TransformFactory.create(
            "SpikeAmplification", spike_threshold_quantile=0.8, amplification_factor=2.0, use_power=False
        )

        self.assertIsInstance(transform, SpikeAmplification)
        self.assertEqual(transform.spike_threshold_quantile, 0.8)
        self.assertEqual(transform.amplification_factor, 2.0)
        self.assertEqual(transform.use_power, False)

        # Test it works
        transform.train(self.ts)
        transformed = transform(self.ts)
        self.assertIsInstance(transformed, TimeSeries)

    def test_serialization(self):
        """Test transform can be serialized and deserialized."""
        logger.info("Testing serialization")

        transform = SpikeAmplification(spike_threshold_quantile=0.75, amplification_factor=1.5, smoothing_window=0.2)

        # Train the transform
        transform.train(self.ts)

        # Convert to dict and back
        state_dict = transform.to_dict()
        self.assertIn("spike_threshold_quantile", state_dict)
        self.assertIn("amplification_factor", state_dict)

        # Recreate from dict
        transform2 = SpikeAmplification.from_dict({k: v for k, v in state_dict.items() if k != "name"})

        # Train the new transform
        transform2.train(self.ts)

        # Should produce same results
        result1 = transform(self.ts)
        result2 = transform2(self.ts)

        values1 = result1.univariates[result1.names[0]].np_values
        values2 = result2.univariates[result2.names[0]].np_values

        np.testing.assert_array_almost_equal(values1, values2, decimal=10)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("Testing edge cases")

        # Test that untrained transform raises error
        transform = SpikeAmplification()
        with self.assertRaises(RuntimeError):
            transform(self.ts)

        with self.assertRaises(RuntimeError):
            transform.invert(self.ts)

        # Test invalid parameters
        with self.assertRaises(AssertionError):
            SpikeAmplification(spike_threshold_quantile=1.5)  # > 1

        with self.assertRaises(AssertionError):
            SpikeAmplification(amplification_factor=0.5)  # < 1

        with self.assertRaises(AssertionError):
            SpikeAmplification(smoothing_window=1.5)  # > 1


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
