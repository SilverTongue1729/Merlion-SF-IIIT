#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Unit tests for the rolling_forecast feature in ForecasterBase
"""
import numpy as np
import pandas as pd
import pytest
from merlion.models.forecast.moirai import Moirai, MoiraiConfig
from merlion.utils import TimeSeries


class TestRollingForecast:
    """Test suite for the rolling_forecast method"""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing"""
        url = (
            "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
            "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
        )
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df = df[df["item_id"] == "A"].drop(columns=["item_id"])
        return df

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create and train a Moirai model for testing"""
        train_df = sample_data.iloc[:168]
        train_ts = TimeSeries.from_pd(train_df)

        config = MoiraiConfig(
            max_forecast_steps=48,
            model_variant="moirai2",
            model_size="small",
            context_length=168,
            target_seq_index=0,
            device="cpu",
        )

        model = Moirai(config)
        model.train(train_ts)
        return model

    def test_basic_rolling_forecast(self, trained_model, sample_data):
        """Test basic rolling forecast functionality"""
        full_ts = TimeSeries.from_pd(sample_data)

        forecast, stderr = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
            prediction_stride=24,
        )

        # Check that forecast is returned
        assert forecast is not None
        assert len(forecast) > 0

        # Check that predictions are within the expected range
        assert forecast.time_stamps[0] >= full_ts.time_stamps[168]
        assert forecast.time_stamps[-1] <= full_ts.time_stamps[-1]

    def test_overlapping_windows(self, trained_model, sample_data):
        """Test rolling forecast with overlapping windows"""
        full_ts = TimeSeries.from_pd(sample_data)

        # Overlapping windows (50% overlap)
        forecast_overlap, _ = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
            prediction_stride=12,  # Half of prediction_length
        )

        # Non-overlapping windows
        forecast_no_overlap, _ = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
            prediction_stride=24,
        )

        # Overlapping should generate more predictions
        assert len(forecast_overlap) >= len(forecast_no_overlap)

    def test_return_iqr(self, trained_model, sample_data):
        """Test rolling forecast with IQR return"""
        full_ts = TimeSeries.from_pd(sample_data)

        forecast, lb, ub = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
            prediction_stride=24,
            return_iqr=True,
        )

        # Check that all three components are returned
        assert forecast is not None
        assert lb is not None or ub is not None  # Some models may not support bounds

        if lb is not None and ub is not None:
            # Check that bounds make sense
            assert len(lb) == len(forecast)
            assert len(ub) == len(forecast)

    def test_error_on_short_time_series(self, trained_model):
        """Test that error is raised for too-short time series"""
        # Create a very short time series
        short_df = pd.DataFrame(
            {"target": np.random.randn(50)}, index=pd.date_range("2021-01-01", periods=50, freq="H")
        )
        short_ts = TimeSeries.from_pd(short_df)

        with pytest.raises(ValueError, match="too short"):
            trained_model.rolling_forecast(
                time_series=short_ts,
                context_length=100,  # Too large for this time series
                prediction_length=20,
            )

    def test_default_prediction_stride(self, trained_model, sample_data):
        """Test that prediction_stride defaults to prediction_length"""
        full_ts = TimeSeries.from_pd(sample_data)

        # Call without specifying prediction_stride
        forecast1, _ = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
        )

        # Call with explicit prediction_stride = prediction_length
        forecast2, _ = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
            prediction_stride=24,
        )

        # Should produce identical results
        assert len(forecast1) == len(forecast2)
        assert list(forecast1.time_stamps) == list(forecast2.time_stamps)

    def test_no_duplicate_timestamps(self, trained_model, sample_data):
        """Test that duplicate timestamps are removed"""
        full_ts = TimeSeries.from_pd(sample_data)

        # Use overlapping windows which could create duplicates
        forecast, _ = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
            prediction_stride=12,
        )

        # Check for no duplicates
        timestamps = forecast.time_stamps
        assert len(timestamps) == len(set(timestamps)), "Found duplicate timestamps"

    def test_forecast_values_reasonable(self, trained_model, sample_data):
        """Test that forecast values are in a reasonable range"""
        full_ts = TimeSeries.from_pd(sample_data)
        actual_values = sample_data["target"].values

        forecast, _ = trained_model.rolling_forecast(
            time_series=full_ts,
            context_length=168,
            prediction_length=24,
            prediction_stride=24,
        )

        forecast_df = forecast.to_pd()
        forecast_values = forecast_df.values.flatten()

        # Forecast should be in a reasonable range relative to actual data
        actual_min, actual_max = actual_values.min(), actual_values.max()
        actual_range = actual_max - actual_min

        # Allow some buffer beyond the actual range
        assert forecast_values.min() >= actual_min - 2 * actual_range
        assert forecast_values.max() <= actual_max + 2 * actual_range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
