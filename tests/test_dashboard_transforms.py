#!/usr/bin/env python
#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Test script for dashboard transform functionality.
This script tests the transform utilities and ensures they work correctly with all available transforms.
Tests include: parameter handling, transform creation, application to data, and model save/load.
"""
import sys
import os
import tempfile
import unittest

sys.path.insert(0, "/home/sriteja/Research/Merlion-SF-IIIT")

from merlion.dashboard.utils.transform_utils import (
    get_available_transforms,
    get_transform_param_info,
    parse_transform_parameters,
    create_transform_from_config,
    create_transform_sequence,
    get_transform_defaults,
)
from merlion.utils import TimeSeries
from merlion.models.forecast.sarima import Sarima, SarimaConfig
import pandas as pd
import numpy as np


class TestDashboardTransforms(unittest.TestCase):
    """Test suite for dashboard transform functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data used across all tests."""
        # Create sample time series with various patterns
        dates = pd.date_range(start="2020-01-01", periods=200, freq="1h")
        values = np.random.randn(200) * 5 + 50
        # Add trend
        values += np.linspace(0, 20, 200)
        # Add some spikes
        values[40] += 30
        values[80] += 35
        values[120] += 40
        values[160] += 32

        cls.ts = TimeSeries.from_pd(pd.Series(values, index=dates, name="value"))
        cls.train_ts = cls.ts[:150]
        cls.test_ts = cls.ts[150:]

    def test_get_available_transforms(self):
        """Test that all expected transforms are available."""
        transforms = get_available_transforms()

        self.assertGreater(len(transforms), 0, "Should have at least one transform")

        # Check for all expected transforms
        expected = [
            "MeanVarNormalize",
            "MinMaxNormalize",
            "DifferenceTransform",
            "MovingAverage",
            "ExponentialMovingAverage",
            "SpikeAmplification",
            "BoxCoxTransform",
            "LowerUpperClip",
        ]

        for transform_name in expected:
            self.assertIn(transform_name, transforms, f"{transform_name} should be in available transforms")

    def test_all_transforms_with_defaults(self):
        """Test creating and applying each transform with default parameters."""
        transforms = get_available_transforms()

        for transform_name in transforms:
            with self.subTest(transform=transform_name):
                # Get defaults
                defaults = get_transform_defaults(transform_name)

                # Create transform
                transform = create_transform_from_config(transform_name, defaults)
                self.assertIsNotNone(transform, f"Failed to create {transform_name}")

                # Train and apply
                transform.train(self.train_ts)
                transformed = transform(self.train_ts)

                self.assertIsNotNone(transformed, f"{transform_name} returned None")
                self.assertGreater(len(transformed), 0, f"{transform_name} returned empty series")

    def test_all_transforms_serialization(self):
        """Test that all transforms can be serialized to dict and back."""
        transforms = get_available_transforms()

        for transform_name in transforms:
            with self.subTest(transform=transform_name):
                defaults = get_transform_defaults(transform_name)
                transform = create_transform_from_config(transform_name, defaults)

                # Train the transform
                transform.train(self.train_ts)

                # Serialize to dict
                state_dict = transform.to_dict()
                self.assertIsInstance(state_dict, dict, f"{transform_name}.to_dict() should return dict")
                self.assertIn("name", state_dict, f"{transform_name} dict should have 'name' key")

                # Try to deserialize - remove 'name' key first as from_dict doesn't expect it
                transform_class = type(transform)
                state_without_name = {k: v for k, v in state_dict.items() if k != "name"}
                restored = transform_class.from_dict(state_without_name)
                self.assertIsNotNone(restored, f"Failed to restore {transform_name} from dict")

    def test_model_save_load_with_all_transforms(self):
        """Test that models with each transform can be saved and loaded."""
        transforms = get_available_transforms()

        for transform_name in transforms:
            with self.subTest(transform=transform_name):
                # Create and train transform
                defaults = get_transform_defaults(transform_name)
                user_transform = create_transform_from_config(transform_name, defaults)
                user_transform.train(self.train_ts)
                transformed_ts = user_transform(self.train_ts)

                # Create model with transform
                config = SarimaConfig(order=[1, 0, 1], target_seq_index=0)
                model = Sarima(config)
                model.transform = user_transform

                # Train model
                try:
                    model.train(transformed_ts)
                except Exception as e:
                    self.fail(f"Training failed for {transform_name}: {e}")

                # Save and load model
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = os.path.join(tmpdir, "model")

                    # Save
                    try:
                        model.save(model_path)
                    except Exception as e:
                        self.fail(f"Save failed for model with {transform_name}: {e}")

                    # Load
                    try:
                        loaded_model = Sarima.load(model_path)
                        self.assertIsNotNone(
                            loaded_model.transform, f"Loaded model should have transform for {transform_name}"
                        )
                    except Exception as e:
                        self.fail(f"Load failed for model with {transform_name}: {e}")

    def test_moving_average_json_serialization(self):
        """Specific test for MovingAverage numpy array serialization fix."""
        import json

        defaults = get_transform_defaults("MovingAverage")
        transform = create_transform_from_config("MovingAverage", defaults)

        # Get state dict
        state_dict = transform.to_dict()

        # Should be JSON serializable now
        try:
            json_str = json.dumps(state_dict)
            self.assertIsInstance(json_str, str, "Should serialize to JSON string")
        except TypeError as e:
            self.fail(f"MovingAverage should be JSON serializable: {e}")

        # Verify weights is a list, not ndarray
        self.assertIsInstance(state_dict["weights"], list, "Weights should be a list in serialized form")

    def test_parameter_parsing(self):
        """Test that parameter parsing handles different types correctly."""
        test_cases = [
            (
                "SpikeAmplification",
                {"spike_threshold_quantile": "0.8", "amplification_factor": "2.0", "use_power": "false"},
                {"spike_threshold_quantile": 0.8, "amplification_factor": 2.0, "use_power": False},
            ),
            ("MovingAverage", {"n_steps": "10"}, {"n_steps": 10}),
            ("LowerUpperClip", {"lower": "-10.5", "upper": "100.5"}, {"lower": -10.5, "upper": 100.5}),
        ]

        for transform_name, input_params, expected_output in test_cases:
            with self.subTest(transform=transform_name):
                parsed = parse_transform_parameters(transform_name, input_params)

                for key, expected_value in expected_output.items():
                    self.assertIn(key, parsed, f"{key} should be in parsed params")
                    self.assertEqual(
                        parsed[key], expected_value, f"{key} should be {expected_value}, got {parsed[key]}"
                    )
                    self.assertIsInstance(
                        parsed[key], type(expected_value), f"{key} should be type {type(expected_value)}"
                    )

    def test_transform_inversion(self):
        """Test that invertible transforms can invert correctly."""
        invertible_transforms = [
            "MeanVarNormalize",
            "MinMaxNormalize",
            "MovingAverage",
            "ExponentialMovingAverage",
            "DifferenceTransform",
        ]

        for transform_name in invertible_transforms:
            with self.subTest(transform=transform_name):
                defaults = get_transform_defaults(transform_name)
                transform = create_transform_from_config(transform_name, defaults)

                # Train and transform
                transform.train(self.train_ts)
                transformed = transform(self.train_ts)

                # Invert
                inverted = transform.invert(transformed)

                self.assertIsNotNone(inverted, f"{transform_name} inversion returned None")
                self.assertEqual(len(inverted), len(self.train_ts), f"{transform_name} inversion changed length")


def run_tests():
    """Run all tests with unittest."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDashboardTransforms)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
