#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Quick integration test for MOIRAI with Merlion.
This script tests the basic functionality without requiring model download.
"""
import numpy as np
import pandas as pd
from merlion.models.forecast.moirai import Moirai, MoiraiConfig
from merlion.models.factory import ModelFactory
from merlion.utils import TimeSeries


def test_config_creation():
    """Test creating various MOIRAI configurations."""
    print("Testing configuration creation...")

    # Test moirai2 (default)
    config1 = MoiraiConfig(
        max_forecast_steps=24,
        model_variant="moirai2",
        model_size="small",
    )
    assert config1.model_variant == "moirai2"
    # moirai2 uses "moirai-2.0" in the HuggingFace path
    assert "moirai-2.0-R-small" in config1.pretrained_model_path or "moirai2" in config1.pretrained_model_path
    print("  ✓ moirai2 config created")

    # Test moirai
    config2 = MoiraiConfig(
        max_forecast_steps=24,
        model_variant="moirai",
        model_size="base",
    )
    assert config2.model_variant == "moirai"
    assert config2.pretrained_model_path == "Salesforce/moirai-1.1-R-base"
    print("  ✓ moirai config created")

    # Test moirai-moe
    config3 = MoiraiConfig(
        max_forecast_steps=24,
        model_variant="moirai-moe",
        model_size="large",
    )
    assert config3.model_variant == "moirai-moe"
    assert config3.pretrained_model_path == "Salesforce/moirai-moe-1.0-R-large"
    # patch_size defaults to "auto", would be set to 16 if user specifies a different value
    print("  ✓ moirai-moe config created")

    # Test custom path
    config4 = MoiraiConfig(pretrained_model_path="custom/path")
    assert config4.pretrained_model_path == "custom/path"
    print("  ✓ custom path config created")

    print("✓ All config tests passed!\n")


def test_model_initialization():
    """Test model initialization."""
    print("Testing model initialization...")

    config = MoiraiConfig(
        max_forecast_steps=24,
        model_variant="moirai2",
        model_size="small",
        context_length=168,
    )

    model = Moirai(config)

    assert model is not None
    assert model.model_variant == "moirai2"
    assert model.model_size == "small"
    assert model.context_length == 168
    assert model.model is None  # Not loaded yet
    assert model.require_even_sampling == False
    assert model.require_univariate == False

    print("  ✓ Model initialized successfully")
    print(f"  ✓ Model variant: {model.model_variant}")
    print(f"  ✓ Model size: {model.model_size}")
    print(f"  ✓ Context length: {model.context_length}")
    print("✓ Model initialization tests passed!\n")


def test_factory_creation():
    """Test creating model via factory."""
    print("Testing factory creation...")

    model = ModelFactory.create(
        "Moirai",
        max_forecast_steps=24,
        model_variant="moirai2",
        model_size="small",
        context_length=200,
    )

    assert isinstance(model, Moirai)
    assert model.config.max_forecast_steps == 24
    assert model.config.model_variant == "moirai2"
    assert model.config.context_length == 200

    print("  ✓ Model created via factory")
    print(f"  ✓ Class: {model.__class__.__name__}")
    print(f"  ✓ Forecast steps: {model.config.max_forecast_steps}")
    print("✓ Factory creation tests passed!\n")


def test_data_preparation():
    """Test preparing data for MOIRAI."""
    print("Testing data preparation...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq="H")
    values = np.sin(np.arange(200) * 2 * np.pi / 24) + np.random.randn(200) * 0.1
    df = pd.DataFrame({"value": values}, index=dates)

    # Convert to TimeSeries
    ts = TimeSeries.from_pd(df)

    print(f"  ✓ Created time series with {len(ts)} points")

    # Initialize model
    config = MoiraiConfig(
        max_forecast_steps=24,
        model_variant="moirai2",
        model_size="small",
        context_length=168,
    )
    model = Moirai(config)

    # Test input tensor preparation
    tensor = model._prepare_input_tensor(ts)

    assert tensor.shape == (1, 200, 1)  # (batch, time, variate)
    print(f"  ✓ Input tensor shape: {tensor.shape}")

    # Test with multivariate data
    df_multi = pd.DataFrame(
        {
            "var1": values,
            "var2": values * 2,
        },
        index=dates,
    )
    ts_multi = TimeSeries.from_pd(df_multi)

    tensor_multi = model._prepare_input_tensor(ts_multi)
    assert tensor_multi.shape == (1, 200, 2)
    print(f"  ✓ Multivariate tensor shape: {tensor_multi.shape}")

    print("✓ Data preparation tests passed!\n")


def test_serialization():
    """Test model serialization."""
    print("Testing model serialization...")

    config = MoiraiConfig(
        max_forecast_steps=24,
        model_variant="moirai2",
        model_size="small",
    )
    model = Moirai(config)

    # Test serialization
    state = model.__getstate__()
    assert state["model"] is None  # Model should not be serialized
    assert state["_device"] is None
    print("  ✓ Model state captured")

    # Test deserialization
    new_model = Moirai(config)
    new_model.__setstate__(state)
    assert new_model.config.model_variant == "moirai2"
    print("  ✓ Model state restored")

    print("✓ Serialization tests passed!\n")


def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")

    # Test invalid variant
    try:
        config = MoiraiConfig(model_variant="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught invalid variant: {e}")

    # Test invalid size
    try:
        config = MoiraiConfig(model_size="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught invalid size: {e}")

    print("✓ Error handling tests passed!\n")


if __name__ == "__main__":
    print("=" * 80)
    print("MOIRAI Integration Tests")
    print("=" * 80)
    print()

    try:
        test_config_creation()
        test_model_initialization()
        test_factory_creation()
        test_data_preparation()
        test_serialization()
        test_error_handling()

        print("=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("Note: These tests verify the integration without downloading models.")
        print("To test actual forecasting, run the example script with model download:")
        print("  python examples/forecast/moirai_example.py")

    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 80)
        import traceback

        traceback.print_exc()
        exit(1)
