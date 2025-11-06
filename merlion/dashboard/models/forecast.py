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

        # Apply user transform if provided
        if user_transform:
            train_ts_original = train_ts
            self.logger.info(f"Training and applying transform: {transform_config['name']}...")
            user_transform.train(train_ts)
            train_ts = user_transform(train_ts)
            # Now add the TRAINED transform to the model's transform
            if model.transform is not None:
                # Model already has a transform (e.g., normalization), combine them
                from merlion.transform.sequence import TransformSequence

                model.transform = TransformSequence([user_transform, model.transform])
            else:
                model.transform = user_transform

        predictions = model.train(train_ts, exog_data=exog_ts)
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        self.logger.info("Computing training performance metrics...")
        set_progress(("6", "10"))
        evaluator = ForecastEvaluator(model, config=ForecastEvaluator.config_class())
        train_metrics = ForecastModel._compute_metrics(evaluator, train_ts, predictions)
        set_progress(("7", "10"))

        test_ts = TimeSeries.from_pd(test_df)
        if "max_forecast_steps" in params and params["max_forecast_steps"] is not None:
            n = min(len(test_ts) - 1, int(params["max_forecast_steps"]))
            test_ts, _ = test_ts.bisect(t=test_ts.time_stamps[n])

        self.logger.info("Computing test performance metrics...")
        test_pred, test_err = model.forecast(time_stamps=test_ts.time_stamps, exog_data=exog_ts)
        test_metrics = ForecastModel._compute_metrics(evaluator, test_ts, test_pred)
        set_progress(("8", "10"))

        self.logger.info("Plotting forecasting results...")

        # Use original plotting (transform inversion happens automatically in model.forecast)
        figure = model.plot_forecast_plotly(
            time_series=test_ts, time_series_prev=train_ts, exog_data=exog_ts, plot_forecast_uncertainty=True
        )

        # Optional: Add transformed data trace to show the transformation effect
        # Disabled for now - uncomment the code below to enable
        # if transform_config and train_ts_original is not None:
        #     import plotly.graph_objs as go
        #
        #     # Get the variable name for plotting
        #     var_name = train_ts.names[0]
        #
        #     # Check if transformation actually changed the data
        #     train_orig_df = train_ts_original.to_pd()
        #     train_trans_df = train_ts.to_pd()
        #
        #     if not train_orig_df.equals(train_trans_df):
        #         # Add transformed data trace (insert after the training data trace)
        #         transformed_trace = go.Scatter(
        #             x=train_trans_df.index,
        #             y=train_trans_df[var_name] if var_name in train_trans_df.columns else train_trans_df.iloc[:, 0],
        #             name=f"Transformed ({transform_config['name']})",
        #             mode="lines",
        #             line=dict(color="orange", width=1.5, dash="dot"),
        #             opacity=0.7
        #         )
        #         # Add the trace to the figure (after the first trace which is the training data)
        #         figure.add_trace(transformed_trace, row=1, col=1)

        figure.update_layout(width=None, height=500)
        self.logger.info("Finished.")
        set_progress(("10", "10"))

        return model, train_metrics, test_metrics, figure
