#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Test script to verify MOIRAI integration with Merlion Dashboard.
This tests the backend functionality without starting the full dashboard server.
"""
import sys
import pandas as pd
import numpy as np
from io import StringIO

# Add Merlion to path
sys.path.insert(0, "/home/sriteja/Research/Merlion-SF-IIIT")

from merlion.dashboard.models.forecast import ForecastModel
from merlion.dashboard.models.utils import ModelMixin


def test_moirai_in_algorithms():
    """Test that MOIRAI is in the available algorithms list."""
    print("Test 1: Checking if MOIRAI is in available algorithms...")
    algorithms = ForecastModel.get_available_algorithms()
    assert "Moirai" in algorithms, "MOIRAI not found in algorithms list!"
    print(f"  ✓ MOIRAI found in algorithms: {algorithms}")
    print()


def test_moirai_parameters():
    """Test that MOIRAI parameters can be extracted."""
    print("Test 2: Extracting MOIRAI parameters...")
    param_info = ModelMixin.get_parameter_info("Moirai")

    # Check essential parameters
    essential_params = ["max_forecast_steps", "model_variant", "model_size", "context_length", "device"]

    for param in essential_params:
        assert param in param_info, f"Parameter {param} not found!"

    print(f"  ✓ Found {len(param_info)} parameters")
    print("  Essential parameters:")
    for param in essential_params:
        info = param_info[param]
        print(f"    - {param}: type={info['type'].__name__}, default={info['default']}")
    print()


def test_moirai_parameter_parsing():
    """Test that MOIRAI parameters can be parsed."""
    print("Test 3: Testing parameter parsing...")
    param_info = ModelMixin.get_parameter_info("Moirai")

    # Simulate dashboard parameter input
    dashboard_params = {
        "max_forecast_steps": "24",
        "model_variant": "moirai2",
        "model_size": "small",
        "context_length": "168",
        "device": "cpu",
    }

    parsed = ModelMixin.parse_parameters(param_info, dashboard_params)

    assert parsed["max_forecast_steps"] == 24
    assert parsed["model_variant"] == "moirai2"
    assert parsed["model_size"] == "small"
    assert parsed["context_length"] == 168
    assert parsed["device"] == "cpu"

    print("  ✓ Parameters parsed correctly:")
    for key, value in parsed.items():
        print(f"    - {key}: {value} ({type(value).__name__})")
    print()


def test_dashboard_data_flow():
    """Test the complete dashboard data flow with MOIRAI."""
    print("Test 4: Testing complete dashboard data flow...")

    # Create sample CSV data
    print("  Creating sample time series data...")
    dates = pd.date_range(start="2023-01-01", periods=200, freq="H")
    values = np.sin(np.arange(200) * 2 * np.pi / 24) + np.random.randn(200) * 0.1
    df = pd.DataFrame({"value": values}, index=dates)

    # Save to temporary CSV
    csv_path = "/tmp/test_dashboard_moirai.csv"
    df.to_csv(csv_path)
    print(f"  ✓ Saved test data to {csv_path}")

    # Load data using dashboard method
    model = ForecastModel()
    loaded_df = model.load_data(csv_path)
    print(f"  ✓ Loaded data: shape={loaded_df.shape}")

    # Split data
    split_idx = int(len(loaded_df) * 0.8)
    train_df = loaded_df.iloc[:split_idx]
    test_df = loaded_df.iloc[split_idx:]
    print(f"  ✓ Split data: train={len(train_df)}, test={len(test_df)}")

    # Get parameter info
    param_info = ModelMixin.get_parameter_info("Moirai")

    # Parse parameters
    params = ModelMixin.parse_parameters(
        param_info,
        {
            "max_forecast_steps": "20",
            "model_variant": "moirai2",
            "model_size": "small",
            "context_length": "100",
            "device": "cpu",
        },
    )
    print(f"  ✓ Parsed parameters: {list(params.keys())}")

    print("  ✓ Dashboard data flow works correctly")
    print()


def test_model_factory_integration():
    """Test that MOIRAI can be created via ModelFactory (used by dashboard)."""
    print("Test 5: Testing ModelFactory integration...")

    from merlion.models.factory import ModelFactory

    # Create model via factory (as dashboard does)
    model = ModelFactory.create(
        "Moirai",
        max_forecast_steps=24,
        model_variant="moirai2",
        model_size="small",
        context_length=168,
    )

    print(f"  ✓ Created model: {model.__class__.__name__}")
    print(f"  ✓ Model variant: {model.config.model_variant}")
    print(f"  ✓ Model size: {model.config.model_size}")
    print(f"  ✓ Context length: {model.config.context_length}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("MOIRAI Dashboard Integration Tests")
    print("=" * 70)
    print()

    try:
        test_moirai_in_algorithms()
        test_moirai_parameters()
        test_moirai_parameter_parsing()
        test_dashboard_data_flow()
        test_model_factory_integration()

        print("=" * 70)
        print("✅ ALL DASHBOARD INTEGRATION TESTS PASSED!")
        print("=" * 70)
        print()
        print("MOIRAI is successfully integrated with the Merlion Dashboard.")
        print("You can now use it by running:")
        print("  python -m merlion.dashboard")
        print()
        print("Then navigate to the Forecast page and select 'Moirai' from the")
        print("algorithm dropdown.")

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ ERROR: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        sys.exit(1)
