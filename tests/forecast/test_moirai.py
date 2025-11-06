#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Tests for MOIRAI forecaster model.
"""
import logging
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from merlion.models.forecast.moirai import Moirai, MoiraiConfig
from merlion.utils import TimeSeries

logger = logging.getLogger(__name__)


class TestMoiraiConfig(unittest.TestCase):
    """Test MoiraiConfig validation and initialization."""

    def test_valid_config(self):
        """Test creating a valid config."""
        config = MoiraiConfig(
            max_forecast_steps=24,
            model_variant="moirai2",
            model_size="small",
            context_length=168,
        )
        self.assertEqual(config.model_variant, "moirai2")
        self.assertEqual(config.model_size, "small")
        self.assertEqual(config.context_length, 168)
        self.assertEqual(config.max_forecast_steps, 24)

    def test_invalid_variant(self):
        """Test that invalid model_variant raises error."""
        with self.assertRaises(ValueError):
            MoiraiConfig(model_variant="invalid")

    def test_invalid_size(self):
        """Test that invalid model_size raises error."""
        with self.assertRaises(ValueError):
            MoiraiConfig(model_size="invalid")

    def test_pretrained_path_generation(self):
        """Test automatic pretrained model path generation."""
        # Test moirai
        config1 = MoiraiConfig(model_variant="moirai", model_size="small")
        self.assertEqual(config1.pretrained_model_path, "Salesforce/moirai-1.1-R-small")

        # Test moirai-moe
        config2 = MoiraiConfig(model_variant="moirai-moe", model_size="base")
        self.assertEqual(config2.pretrained_model_path, "Salesforce/moirai-moe-1.0-R-base")

        # Test moirai2
        config3 = MoiraiConfig(model_variant="moirai2", model_size="large")
        self.assertEqual(config3.pretrained_model_path, "Salesforce/moirai-2.0-R-large")

    def test_custom_pretrained_path(self):
        """Test custom pretrained model path."""
        config = MoiraiConfig(pretrained_model_path="custom/path")
        self.assertEqual(config.pretrained_model_path, "custom/path")


class TestMoirai(unittest.TestCase):
    """Test MOIRAI forecaster model."""

    def setUp(self):
        """Set up test data."""
        # Create simple synthetic time series
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=200, freq="H")
        values = np.sin(np.arange(200) * 2 * np.pi / 24) + np.random.randn(200) * 0.1
        self.train_data = TimeSeries.from_pd(pd.Series(values[:168], index=dates[:168]))
        self.test_data = TimeSeries.from_pd(pd.Series(values[168:], index=dates[168:]))

    def test_initialization(self):
        """Test model initialization."""
        config = MoiraiConfig(
            max_forecast_steps=24,
            model_variant="moirai2",
            model_size="small",
            context_length=168,
        )
        model = Moirai(config)

        self.assertIsNotNone(model)
        self.assertEqual(model.model_variant, "moirai2")
        self.assertEqual(model.model_size, "small")
        self.assertIsNone(model.model)  # Model not loaded yet

    def test_require_even_sampling(self):
        """Test that MOIRAI doesn't require even sampling."""
        config = MoiraiConfig()
        model = Moirai(config)
        self.assertFalse(model.require_even_sampling)

    def test_require_univariate(self):
        """Test that MOIRAI supports multivariate."""
        config = MoiraiConfig()
        model = Moirai(config)
        self.assertFalse(model.require_univariate)

    @unittest.skip("Requires downloading pretrained model - skip in CI")
    def test_train_and_forecast(self):
        """Test training and forecasting with MOIRAI."""
        config = MoiraiConfig(
            max_forecast_steps=24,
            model_variant="moirai2",
            model_size="small",
            context_length=168,
            device="cpu",  # Force CPU for testing
        )
        model = Moirai(config)

        # Train (loads pretrained model)
        train_pred, train_err = model.train(self.train_data)

        # Check that model is loaded
        self.assertIsNotNone(model.model)

        # Forecast
        forecast, err = model.forecast(time_stamps=24, time_series_prev=self.train_data)

        # Check forecast shape and type
        self.assertIsInstance(forecast, TimeSeries)
        self.assertEqual(len(forecast), 24)

        # Check uncertainty estimates
        self.assertIsInstance(err, TimeSeries)
        self.assertEqual(len(err), 24)


class TestMoiraiFactory(unittest.TestCase):
    """Test MOIRAI model creation via factory."""

    def test_factory_create(self):
        """Test creating MOIRAI via ModelFactory."""
        from merlion.models.factory import ModelFactory

        model = ModelFactory.create(
            "Moirai",
            max_forecast_steps=24,
            model_variant="moirai2",
            model_size="small",
        )

        self.assertIsInstance(model, Moirai)
        self.assertEqual(model.config.max_forecast_steps, 24)
        self.assertEqual(model.config.model_variant, "moirai2")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    unittest.main()
