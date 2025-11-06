#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from merlion.dashboard.pages.utils import create_modal, create_param_table, create_metric_table, create_empty_figure


def create_control_panel() -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Select Training Data File"),
            html.Div(
                id="forecasting-select-file-parent",
                children=[
                    dbc.RadioItems(
                        id="forecasting-file-radio",
                        options=[
                            {"label": "Single data file", "value": "single"},
                            {"label": "Separate train/test files", "value": "separate"},
                        ],
                        value="single",
                        inline=True,
                    ),
                    dcc.Dropdown(id="forecasting-select-file", options=[], style={"width": "100%"}),
                ],
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Training Data Percentage"),
                        dcc.Slider(
                            id="forecasting-training-slider",
                            min=5,
                            max=95,
                            step=1,
                            marks={t * 10: str(t * 10) for t in range(1, 10)},
                            value=80,
                        ),
                    ],
                ),
                id="forecasting-slider-collapse",
                is_open=True,
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Select Test Data File"),
                        html.Div(
                            id="forecasting-select-test-file-parent",
                            children=[
                                dcc.Dropdown(id="forecasting-select-test-file", options=[], style={"width": "100%"})
                            ],
                        ),
                    ],
                ),
                id="forecasting-test-file-collapse",
                is_open=False,
            ),
            html.Br(),
            html.P("Select Target Column"),
            html.Div(
                id="forecasting-select-target-parent",
                children=[dcc.Dropdown(id="forecasting-select-target", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Other Features (Optional)"),
            html.Div(
                id="forecasting-select-features-parent",
                children=[
                    dcc.Dropdown(id="forecasting-select-features", options=[], multi=True, style={"width": "100%"})
                ],
            ),
            html.Br(),
            html.P("Select Exogenous Variables (Optional; Known A Priori)"),
            html.Div(
                id="forecasting-select-exog-parent",
                children=[dcc.Dropdown(id="forecasting-select-exog", options=[], multi=True, style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Select Forecasting Algorithm"),
            html.Div(
                id="forecasting-select-algorithm-parent",
                children=[dcc.Dropdown(id="forecasting-select-algorithm", options=[], style={"width": "100%"})],
            ),
            html.Br(),
            html.P("Forecast Mode"),
            dbc.RadioItems(
                id="forecasting-mode-radio",
                options=[
                    {"label": "Single Forecast (fastest, predict all at once)", "value": "single"},
                    {
                        "label": "Rolling Forecast - Context Update (updates with actual values)",
                        "value": "rolling_update",
                    },
                    {
                        "label": "Rolling Forecast - Sliding Window (efficient, no retraining)",
                        "value": "rolling_sliding",
                    },
                ],
                value="single",
                inline=False,
            ),
            dbc.Collapse(
                html.Div(
                    id="control-card",
                    children=[
                        html.Br(),
                        html.P("Context Length (historical steps for prediction)"),
                        dcc.Input(
                            id="forecasting-context-length",
                            type="number",
                            value=168,
                            min=1,
                            max=2000,
                            step=1,
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.P("Prediction Length (steps to predict per window)"),
                        dcc.Input(
                            id="forecasting-rolling-window-size",
                            type="number",
                            value=24,
                            min=1,
                            max=500,
                            step=1,
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.P("Prediction Stride (steps to move window, leave empty for non-overlapping)"),
                        dcc.Input(
                            id="forecasting-prediction-stride",
                            type="number",
                            value=None,
                            min=1,
                            max=500,
                            step=1,
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.Small(
                            id="forecasting-mode-description",
                            children="Context Update: Updates context with actual values after each window. "
                            "Sliding Window: Uses fixed-size sliding window without incorporating actuals.",
                        ),
                    ],
                ),
                id="forecasting-rolling-collapse",
                is_open=False,
            ),
            html.Br(),
            html.P("Algorithm Setting"),
            html.Div(id="forecasting-param-table", children=[create_param_table()]),
            html.Progress(id="forecasting-progressbar", style={"width": "100%", "color": "#1AB9FF"}),
            html.Br(),
            html.Div(
                children=[
                    html.Button(id="forecasting-train-btn", children="Train", n_clicks=0),
                    html.Button(id="forecasting-cancel-btn", children="Cancel", style={"margin-left": "15px"}),
                ],
                style={"textAlign": "center"},
            ),
            html.Br(),
            create_modal(
                modal_id="forecasting-exception-modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="forecasting-exception-modal-content",
                button_id="forecasting-exception-modal-close",
            ),
        ],
    )


def create_right_column() -> html.Div:
    return html.Div(
        id="right-column-data",
        children=[
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Forecasting Results"),
                    html.Hr(),
                    html.Div(id="forecasting-plots", children=[create_empty_figure()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Testing Metrics"),
                    html.Hr(),
                    html.Div(id="forecasting-test-metrics", children=[create_metric_table()]),
                ],
            ),
            html.Div(
                id="result_table_card",
                children=[
                    html.B("Training Metrics"),
                    html.Hr(),
                    html.Div(id="forecasting-training-metrics", children=[create_metric_table()]),
                ],
            ),
        ],
    )


def create_forecasting_layout() -> html.Div:
    return html.Div(
        id="forecasting_views",
        children=[
            # Left column
            html.Div(id="left-column-data", className="three columns", children=[create_control_panel()]),
            # Right column
            html.Div(className="nine columns", children=create_right_column()),
        ],
    )
