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
from merlion.models.anomaly.base import DetectorConfig

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
        transform_config=None,
    ):
        if target_column not in train_df:
            target_column = int(target_column)
        assert target_column in train_df, f"The target variable {target_column} is not in the time series."
        try:
            feature_columns = [int(c) if c not in train_df else c for c in feature_columns]
        except ValueError:
            feature_columns = []
        try:
            exog_columns = [int(c) if c not in train_df else c for c in exog_columns]
        except ValueError:
            exog_columns = []
        for exog_column in exog_columns:
            assert exog_column in train_df, f"Exogenous variable {exog_column} is not in the time series."

        # Re-arrange dataframe so that the target column is first, and exogenous columns are last
        columns = [target_column] + feature_columns + exog_columns
        train_df = train_df.loc[:, columns]
        test_df = test_df.loc[:, columns]

        # Get the target_seq_index & initialize the model
        params["target_seq_index"] = columns.index(target_column)

        # Handle transform if provided - DON'T add to model config yet
        # We'll apply it manually and then add the trained transform to the model
        user_transform = None
        train_ts_original = None
        if transform_config:
            from merlion.dashboard.utils.transform_utils import create_transform_sequence

            self.logger.info(f"Creating transform: {transform_config['name']}...")
            user_transform = create_transform_sequence([transform_config])

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
        train_ts_original = train_ts

        # Apply user transform if provided
        if user_transform:
            self.logger.info(f"Adding transform to model: {transform_config['name']}...")
            # Now add the transform to the model's transform
            if model.transform is not None:
                # Model already has a transform (e.g., normalization), combine them
                from merlion.transform.sequence import TransformSequence

                model.transform = TransformSequence([user_transform, model.transform])
            else:
                model.transform = user_transform
            model.config.invert_transform = model.transform.proper_inversion and not isinstance(
                model.config, DetectorConfig
            )

        predictions = model.train(train_ts, exog_data=exog_ts)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        self.logger.info("Computing training performance metrics...")
        set_progress(("6", "10"))
        evaluator = ForecastEvaluator(model, config=ForecastEvaluator.config_class())

        is_custom_non_invertible = transform_config and transform_config["name"] in [
            "PeakMultiplier",
            "ConvexHullMethod",
            "RollingMeans",
        ]

        if model.config.invert_transform:
            ground_truth_train = train_ts
        elif is_custom_non_invertible:
            ground_truth_train = train_ts  # Evaluate against original data as requested
        else:
            ground_truth_train = model.transform(train_ts)
        train_metrics = ForecastModel._compute_metrics(evaluator, ground_truth_train, predictions)
        set_progress(("7", "10"))

        test_ts = TimeSeries.from_pd(test_df)
        if "max_forecast_steps" in params and params["max_forecast_steps"] is not None:
            n = min(len(test_ts) - 1, int(params["max_forecast_steps"]))
            test_ts, _ = test_ts.bisect(t=test_ts.time_stamps[n])

        self.logger.info("Computing test performance metrics...")
        # Get predictions and IQR bounds in one call
        test_pred, lb, ub = model.forecast(
            time_stamps=test_ts.time_stamps, exog_data=exog_ts, return_iqr=True
        )

        # Metrics are calculated by comparing predictions against original data for the custom transforms
        if model.config.invert_transform:
            ground_truth_test = test_ts
        elif is_custom_non_invertible:
            ground_truth_test = test_ts  # Evaluate against original data as requested
        else:
            ground_truth_test = model.transform(test_ts)
        test_metrics = ForecastModel._compute_metrics(evaluator, ground_truth_test, test_pred)
        set_progress(("8", "10"))

        self.logger.info("Plotting forecasting results...")
        import plotly.graph_objs as go

        figure = go.Figure()

        # 1. Original test data
        original_test_df = test_ts.to_pd()
        var_name = original_test_df.columns[0]
        figure.add_trace(
            go.Scatter(
                x=original_test_df.index,
                y=original_test_df[var_name],
                name="Original Test Data",
                mode="lines",
                line=dict(color="blue"),
            )
        )

        # 2. Transformed test data (if applicable)
        if user_transform:
            transformed_test = model.transform(test_ts)
            transformed_test_df = transformed_test.to_pd()
            figure.add_trace(
                go.Scatter(
                    x=transformed_test_df.index,
                    y=transformed_test_df[var_name],
                    name="Transformed Test Data",
                    mode="lines",
                    line=dict(color="orange", dash="dot"),
                )
            )

        # 3. Predicted data
        test_pred_df = test_pred.to_pd()
        figure.add_trace(
            go.Scatter(
                x=test_pred_df.index,
                y=test_pred_df[test_pred.names[0]],
                name="Prediction",
                mode="lines",
                line=dict(color="green"),
            )
        )

        # 4. Uncertainty bounds
        if lb is not None and ub is not None:
            lb_df = lb.to_pd()
            ub_df = ub.to_pd()
            figure.add_trace(
                go.Scatter(
                    x=lb_df.index,
                    y=lb_df[lb.names[0]],
                    name="Lower Bound",
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=ub_df.index,
                    y=ub_df[ub.names[0]],
                    name="Upper Bound",
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(0, 176, 246, 0.2)",
                    fill="tonexty",
                    showlegend=False,
                )
            )

        figure.update_layout(width=None, height=500, title_text="Forecast on Test Data")
        self.logger.info("Finished.")
        set_progress(("10", "10"))

        return model, train_metrics, test_metrics, figure
