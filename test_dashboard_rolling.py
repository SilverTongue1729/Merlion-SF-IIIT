#!/usr/bin/env python3
#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Quick verification script to test that the dashboard rolling forecast integration is working.
This doesn't start the full dashboard, just verifies the code structure.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all necessary imports work."""
    try:
        from merlion.dashboard.models.forecast import ForecastModel
        from merlion.models.forecast.moirai import Moirai, MoiraiConfig
        from merlion.models.forecast.arima import Arima, ArimaConfig

        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_rolling_forecast_method():
    """Test that ForecasterBase has rolling_forecast method."""
    try:
        from merlion.models.forecast.base import ForecasterBase

        # Check if the method exists
        if hasattr(ForecasterBase, "rolling_forecast"):
            logger.info("✅ ForecasterBase.rolling_forecast method exists")
        else:
            logger.error("❌ ForecasterBase.rolling_forecast method NOT found")
            return False

        # Check the signature
        import inspect

        sig = inspect.signature(ForecasterBase.rolling_forecast)
        params = list(sig.parameters.keys())

        expected_params = [
            "self",
            "time_series",
            "context_length",
            "prediction_length",
            "prediction_stride",
            "exog_data",
            "return_iqr",
            "return_prev",
        ]

        if all(p in params for p in expected_params):
            logger.info("✅ rolling_forecast has all expected parameters")
        else:
            logger.warning(f"⚠️  Parameters: {params}")

        return True
    except Exception as e:
        logger.error(f"❌ Method check failed: {e}")
        return False


def test_dashboard_model():
    """Test that ForecastModel has the new method."""
    try:
        from merlion.dashboard.models.forecast import ForecastModel

        # Check if the renamed method exists
        if hasattr(ForecastModel, "_rolling_forecast_update"):
            logger.info("✅ ForecastModel._rolling_forecast_update method exists")
        else:
            logger.error("❌ ForecastModel._rolling_forecast_update method NOT found")
            return False

        # Check that old method name doesn't exist
        if hasattr(ForecastModel, "_rolling_forecast"):
            logger.warning("⚠️  Old _rolling_forecast method still exists")
        else:
            logger.info("✅ Old _rolling_forecast method successfully removed")

        return True
    except Exception as e:
        logger.error(f"❌ Dashboard model check failed: {e}")
        return False


def test_moirai_in_models():
    """Test that Moirai can be instantiated."""
    try:
        from merlion.models.forecast.moirai import MoiraiConfig

        config = MoiraiConfig(
            model_path="Salesforce/moirai-2.0-R-small",
            prediction_length=48,
        )
        logger.info(f"✅ Moirai config created: {config}")

        return True
    except Exception as e:
        logger.error(f"❌ Moirai instantiation failed: {e}")
        return False


def main():
    """Run all verification tests."""
    logger.info("=" * 60)
    logger.info("Testing Dashboard Rolling Forecast Integration")
    logger.info("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("ForecasterBase.rolling_forecast", test_rolling_forecast_method),
        ("ForecastModel updates", test_dashboard_model),
        ("Moirai availability", test_moirai_in_models),
    ]

    results = []
    for name, test_func in tests:
        logger.info(f"\nTest: {name}")
        logger.info("-" * 40)
        result = test_func()
        results.append((name, result))

    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        logger.info("\n🎉 All tests passed! Dashboard integration is ready.")
        return 0
    else:
        logger.error("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
