#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
import sys

import pandas as pd

from merlion.models.factory import ModelFactory
from merlion.evaluate.forecast import ForecastEvaluator, ForecastMetric
from merlion.utils.time_series import TimeSeries
from merlion.dashboard.models.utils import ModelMixin, DataMixin
from merlion.dashboard.utils.log import DashLogger

dash_logger = DashLogger(stream=sys.stdout)


class ForecastModel(ModelMixin, DataMixin):
    algorithms = [
        "DefaultForecaster",
        "Arima",
        "LGBMForecaster",
        "ETS",
        "AutoETS",
        "Prophet",
        "AutoProphet",
        "Sarima",
        "VectorAR",
        "RandomForestForecaster",
        "ExtraTreesForecaster",
        "Moirai",
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(dash_logger)

    @staticmethod
    def get_available_algorithms():
        return ForecastModel.algorithms

    @staticmethod
    def _compute_metrics(evaluator, ts, predictions):
        return {
            m: round(evaluator.evaluate(ground_truth=ts, predict=predictions, metric=ForecastMetric[m]), 5)
            for m in ["MAE", "MARRE", "RMSE", "sMAPE", "RMSPE"]
        }

    def train(
        self,
        algorithm,
        train_df,
        test_df,
        target_column,
        feature_columns,
        exog_columns,
        params,
        set_progress,
        forecast_mode="single",
        rolling_window_size=24,
        context_length=168,
        prediction_stride=None,
    ):
        if target_column not in train_df:
            target_column = int(target_column)
        assert target_column in train_df, f"The target variable {target_column} is not in the time series."
        try:
            feature_columns = [int(c) if c not in train_df else c for c in feature_columns]
        except (ValueError, TypeError):
            feature_columns = []
        try:
            exog_columns = (
                [int(c) if c not in train_df else c for c in exog_columns] if exog_columns is not None else []
            )
        except (ValueError, TypeError):
            exog_columns = []
        for exog_column in exog_columns:
            assert exog_column in train_df, f"Exogenous variable {exog_column} is not in the time series."

        # Re-arrange dataframe so that the target column is first, and exogenous columns are last
        columns = [target_column] + feature_columns + exog_columns
        train_df = train_df.loc[:, columns]
        test_df = test_df.loc[:, columns]

        # Get the target_seq_index & initialize the model
        params["target_seq_index"] = columns.index(target_column)

        # For rolling forecast modes, ensure max_forecast_steps is set appropriately
        # to avoid the "padding with last value" issue that creates flat lines
        if forecast_mode in ["rolling_update", "rolling_sliding"]:
            # For rolling modes, max_forecast_steps should match the prediction window
            if "max_forecast_steps" not in params or params["max_forecast_steps"] is None:
                params["max_forecast_steps"] = rolling_window_size
            elif params["max_forecast_steps"] < rolling_window_size:
                self.logger.warning(
                    f"max_forecast_steps ({params['max_forecast_steps']}) is less than "
                    f"rolling_window_size ({rolling_window_size}). Setting to {rolling_window_size}."
                )
                params["max_forecast_steps"] = rolling_window_size

            # CRITICAL: Set context_length in model config to match the rolling forecast context_length
            # This ensures the model uses the correct amount of historical data internally
            # Without this, models like Moirai use their default (1680) which causes zero-padding
            # and results in flat/poor predictions when less data is available
            if "context_length" not in params or params["context_length"] is None:
                params["context_length"] = context_length
                self.logger.info(f"Setting model context_length to {context_length} for rolling forecast")
            elif params["context_length"] != context_length:
                self.logger.warning(
                    f"Model context_length ({params['context_length']}) differs from rolling forecast "
                    f"context_length ({context_length}). Using rolling forecast value: {context_length}"
                )
                params["context_length"] = context_length
        else:
            # For single forecast mode, if requesting more steps than max_forecast_steps,
            # set it to the test length to avoid flat line padding
            if "max_forecast_steps" in params and params["max_forecast_steps"] is not None:
                requested_steps = len(test_df)
                if params["max_forecast_steps"] < requested_steps:
                    self.logger.warning(
                        f"max_forecast_steps ({params['max_forecast_steps']}) is less than "
                        f"test length ({requested_steps}). Setting to {requested_steps} to avoid flat line predictions."
                    )
                    params["max_forecast_steps"] = requested_steps

        model_class = ModelFactory.get_model_class(algorithm)
        model = model_class(model_class.config_class(**params))

        # Handle exogenous regressors if they are supported by the model
        if model.supports_exog and len(exog_columns) > 0:
            exog_ts = TimeSeries.from_pd(pd.concat((train_df.loc[:, exog_columns], test_df.loc[:, exog_columns])))
            train_df = train_df.loc[:, [target_column] + feature_columns]
            test_df = test_df.loc[:, [target_column] + feature_columns]
        else:
            exog_ts = None

        self.logger.info(f"Training the forecasting model: {algorithm}...")
        set_progress(("2", "10"))
        train_ts = TimeSeries.from_pd(train_df)
        predictions = model.train(train_ts, exog_data=exog_ts)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        self.logger.info("Computing training performance metrics...")
        set_progress(("6", "10"))
        evaluator = ForecastEvaluator(model, config=ForecastEvaluator.config_class())
        train_metrics = ForecastModel._compute_metrics(evaluator, train_ts, predictions)
        set_progress(("7", "10"))

        test_ts = TimeSeries.from_pd(test_df)

        # Handle different forecast modes
        if forecast_mode == "rolling_update":
            # Original rolling forecast: updates context with actual values
            self.logger.info(f"Using rolling forecast (context update) mode with window size {rolling_window_size}...")
            test_pred, test_err = self._rolling_forecast_update(
                model, train_ts, test_ts, exog_ts, rolling_window_size, set_progress
            )
        elif forecast_mode == "rolling_sliding":
            # New efficient rolling forecast: sliding window without retraining
            self.logger.info(f"Using rolling forecast (sliding window) mode...")
            self.logger.info(f"  Context length: {context_length}")
            self.logger.info(f"  Prediction length: {rolling_window_size}")
            self.logger.info(f"  Prediction stride: {prediction_stride if prediction_stride else rolling_window_size}")

            # Combine train and test for sliding window
            full_ts = train_ts + test_ts

            all_predictions, test_err = model.rolling_forecast(
                time_series=full_ts,
                context_length=context_length,
                prediction_length=rolling_window_size,
                prediction_stride=prediction_stride,
                exog_data=exog_ts,
                return_iqr=False,
            )

            # Extract only the test portion - filter predictions that fall within test period
            # Convert to dataframe for easier filtering
            all_pred_df = all_predictions.to_pd()

            # Use pandas Timestamp for comparison (convert from Unix timestamp)
            import pandas as pd

            test_start = pd.Timestamp(test_ts.time_stamps[0], unit="s")
            test_end = pd.Timestamp(test_ts.time_stamps[-1], unit="s")

            # Keep only predictions within test period
            test_pred_df = all_pred_df[(all_pred_df.index >= test_start) & (all_pred_df.index <= test_end)]

            # Convert back to TimeSeries (already imported at top of file)
            test_pred = TimeSeries.from_pd(test_pred_df)

            self.logger.info(f"  Total rolling predictions: {len(all_predictions)}")
            self.logger.info(f"  Test period predictions: {len(test_pred)}")
        else:
            # Single forecast mode
            if "max_forecast_steps" in params and params["max_forecast_steps"] is not None:
                n = min(len(test_ts) - 1, int(params["max_forecast_steps"]))
                test_ts, _ = test_ts.bisect(t=test_ts.time_stamps[n])

            self.logger.info("Computing test performance metrics...")
            test_pred, test_err = model.forecast(time_stamps=test_ts.time_stamps, exog_data=exog_ts)

        test_metrics = ForecastModel._compute_metrics(evaluator, test_ts, test_pred)
        set_progress(("8", "10"))

        self.logger.info("Plotting forecasting results...")

        # For rolling forecast modes, we can't call model.plot_forecast_plotly() directly
        # because it would try to forecast the entire test period, exceeding max_forecast_steps.
        # Instead, we create a Figure object directly with the predictions we already have.
        if forecast_mode in ["rolling_update", "rolling_sliding"]:
            from merlion.plot import Figure

            # Extract univariate time series for the target variable
            test_uni = test_ts.univariates[test_ts.names[model.target_seq_index]]
            pred_uni = test_pred.univariates[test_pred.names[0]]

            # For time_series_prev, we want to show the training data
            if train_ts is not None:
                train_uni = train_ts.univariates[train_ts.names[model.target_seq_index]]
            else:
                train_uni = None

            # Create a Figure object similar to what get_figure() returns
            # We don't have uncertainty estimates for rolling forecasts, so lb/ub are None
            fig_obj = Figure(
                y=test_uni,  # Actual test values
                yhat=pred_uni,  # Rolling forecast predictions
                yhat_lb=None,  # No uncertainty bounds
                yhat_ub=None,
                y_prev=train_uni,  # Training data (previous values)
                yhat_prev=None,  # We don't show predictions on training data
                yhat_prev_lb=None,
                yhat_prev_ub=None,
            )

            # Use the Figure's plot_plotly method to create the figure
            # This ensures consistent styling with single forecast mode
            title = f"{type(model).__name__}: Rolling Forecast ({forecast_mode})"
            figure = fig_obj.plot_plotly(title=title, metric_name=model.target_name, figsize=(1000, 500))
        else:
            # Single forecast mode - use standard plotting
            figure = model.plot_forecast_plotly(
                time_series=test_ts, time_series_prev=train_ts, exog_data=exog_ts, plot_forecast_uncertainty=True
            )
            figure.update_layout(width=None, height=500)

        self.logger.info("Finished.")
        set_progress(("10", "10"))

        return model, train_metrics, test_metrics, figure

    def _rolling_forecast_update(self, model, train_ts, test_ts, exog_ts, window_size, set_progress):
        """
        Perform rolling forecast with context update: predict in small windows and update context
        with actual values after each window.

        Args:
            model: Trained forecasting model
            train_ts: Training time series (used as initial context)
            test_ts: Test time series to forecast
            exog_ts: Exogenous time series (if any)
            window_size: Number of steps to predict in each window
            set_progress: Progress callback

        Returns:
            (predictions, errors): TimeSeries of predictions and errors
        """
        import numpy as np

        all_predictions = []
        prediction_timestamps = []

        # Number of windows needed
        num_steps = len(test_ts)
        num_windows = (num_steps + window_size - 1) // window_size  # Ceiling division

        self.logger.info(f"Rolling forecast: {num_steps} steps in {num_windows} windows of size {window_size}")

        # Start with training data as context
        current_context = train_ts

        for i in range(num_windows):
            # Calculate window boundaries
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, num_steps)
            current_window_size = end_idx - start_idx

            # Get timestamps for this window
            window_timestamps = test_ts.time_stamps[start_idx:end_idx]

            # Make prediction for this window
            pred, err = model.forecast(
                time_stamps=window_timestamps, time_series_prev=current_context, exog_data=exog_ts
            )

            # Store predictions
            all_predictions.append(pred)

            # Update context: append actual values from test set to context for next iteration
            # This simulates having observed the actual values up to this point
            actual_window = test_ts.window(window_timestamps[0], window_timestamps[-1], include_tf=True)
            current_context = current_context + actual_window

            # Update progress
            progress_val = 7 + int((i + 1) / num_windows)
            set_progress((str(progress_val), "10"))

            self.logger.info(f"  Window {i+1}/{num_windows}: predicted {current_window_size} steps")

        # Concatenate all predictions
        from merlion.utils.time_series import UnivariateTimeSeries

        if len(all_predictions) > 0:
            # Combine all prediction windows
            combined_pred = all_predictions[0]
            for pred in all_predictions[1:]:
                combined_pred = combined_pred + pred

            self.logger.info(f"Rolling forecast complete: {len(combined_pred)} total predictions")
            return combined_pred, None
        else:
            return test_ts * 0, None  # Return zeros if no predictions
