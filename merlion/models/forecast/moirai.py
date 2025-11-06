#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Wrapper around Salesforce's MOIRAI (Masked Encoder-based Universal Time Series Forecasting Transformer) model.
"""
import logging
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    from einops import rearrange
    from uni2ts.model.moirai import MoiraiForecast as MoiraiModule, MoiraiModule as MoiraiModuleBase
    from uni2ts.model.moirai_moe import MoiraiMoEForecast as MoiraiMoEModule, MoiraiMoEModule as MoiraiMoEModuleBase
    from uni2ts.model.moirai2 import Moirai2Forecast as Moirai2Module, Moirai2Module as Moirai2ModuleBase
except ImportError as e:
    err = (
        "Try installing uni2ts using `pip install uni2ts` to use MOIRAI models. "
        "Also ensure deep learning dependencies are installed with "
        "`pip install salesforce-merlion[deep-learning]` or `pip install salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)

from merlion.models.forecast.base import ForecasterBase, ForecasterConfig
from merlion.transform.base import Identity
from merlion.utils import TimeSeries, UnivariateTimeSeries, to_pd_datetime, to_timestamp

logger = logging.getLogger(__name__)


class MoiraiConfig(ForecasterConfig):
    """
    Configuration class for Salesforce's MOIRAI model.

    MOIRAI is a Universal Time Series Forecasting Transformer that supports:
    - Multiple model sizes (small, base, large)
    - Multiple model variants (moirai, moirai-moe, moirai2)
    - Probabilistic forecasting with quantile predictions
    - Univariate and multivariate time series
    - Flexible context and prediction lengths
    """

    _default_transform = Identity()

    def __init__(
        self,
        max_forecast_steps: int = None,
        target_seq_index: int = None,
        model_variant: str = "moirai2",
        model_size: str = "small",
        context_length: int = 1680,
        patch_size: Union[int, str] = "auto",
        num_samples: int = 100,
        batch_size: int = 32,
        pretrained_model_path: Optional[str] = None,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
        device: str = "auto",
        **kwargs,
    ):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for (prediction_length).
        :param target_seq_index: The index of the univariate (amongst all univariates in a general
            multivariate time series) whose value we would like to forecast.
        :param model_variant: Which MOIRAI variant to use. Options: 'moirai', 'moirai-moe', 'moirai2'.
            Default is 'moirai2' which is the latest version.
        :param model_size: Size of the pretrained model. Options: 'small', 'base', 'large'.
            Default is 'small'.
        :param context_length: Number of past time steps to use as context for forecasting.
            For moirai2: recommended 1680 (default)
            For moirai/moirai-moe: can vary based on use case (e.g., 200, 500, 1000)
        :param patch_size: Size of patches for the transformer model.
            For moirai: Options are "auto", 8, 16, 32, 64, 128
            For moirai-moe: Fixed at 16
            For moirai2: Automatically determined by the model
            Default is "auto".
        :param num_samples: Number of samples to draw for probabilistic forecasting.
            Higher values give better uncertainty estimates but slower inference.
            Not applicable for moirai2 which uses quantile forecasting.
            Default is 100.
        :param batch_size: Batch size for inference. Default is 32.
        :param pretrained_model_path: Custom path/identifier for pretrained model on HuggingFace.
            If None, uses default: f"Salesforce/{model_variant}-{version}-R-{model_size}"
            where version is "1.1" for moirai, "1.0" for moirai-moe, "2.0" for moirai2.
        :param target_dim: Number of target dimensions (1 for univariate, >1 for multivariate).
        :param feat_dynamic_real_dim: Dimension of dynamic real features (exogenous variables).
        :param past_feat_dynamic_real_dim: Dimension of past-only dynamic real features.
        :param device: Device to run the model on. Options: 'auto', 'cpu', 'cuda', 'cuda:0', etc.
            'auto' will use CUDA if available, otherwise CPU.
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)

        # Validate model_variant
        valid_variants = ["moirai", "moirai-moe", "moirai2"]
        if model_variant not in valid_variants:
            raise ValueError(f"model_variant must be one of {valid_variants}, got {model_variant}")

        # Validate model_size
        valid_sizes = ["small", "base", "large"]
        if model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}, got {model_size}")

        # Set configuration
        self.model_variant = model_variant
        self.model_size = model_size
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.device = device

        # Set pretrained model path
        if pretrained_model_path is None:
            if model_variant == "moirai":
                pretrained_model_path = f"Salesforce/moirai-1.1-R-{model_size}"
            elif model_variant == "moirai-moe":
                pretrained_model_path = f"Salesforce/moirai-moe-1.0-R-{model_size}"
            else:  # moirai2
                pretrained_model_path = f"Salesforce/moirai-2.0-R-{model_size}"

        self.pretrained_model_path = pretrained_model_path

        # Validate patch_size based on variant
        if model_variant == "moirai-moe" and patch_size != 16 and patch_size != "auto":
            logger.warning(f"moirai-moe only supports patch_size=16, changing from {patch_size} to 16")
            self.patch_size = 16


class Moirai(ForecasterBase):
    """
    Wrapper around Salesforce's MOIRAI (Masked Encoder-based Universal Time Series Forecasting Transformer).

    MOIRAI is a foundation model for time series forecasting that has been pre-trained on a diverse
    collection of time series datasets. It supports:

    - Zero-shot forecasting without fine-tuning
    - Multiple model sizes (small, base, large) for different compute budgets
    - Three model variants: moirai, moirai-moe, and moirai2 (latest)
    - Probabilistic forecasting with uncertainty quantification
    - Univariate and multivariate time series

    The model uses a transformer architecture with patching to handle variable-length time series
    and can forecast multiple steps into the future.

    Example usage:
        ```python
        from merlion.models.forecast.moirai import Moirai, MoiraiConfig
        from merlion.utils import TimeSeries

        # Create config
        config = MoiraiConfig(
            max_forecast_steps=24,
            model_variant="moirai2",
            model_size="small",
            context_length=168,  # 1 week of hourly data
        )

        # Initialize model
        model = Moirai(config)

        # Train (loads pretrained model)
        model.train(train_data)

        # Forecast
        forecast, err = model.forecast(time_stamps)
        ```
    """

    config_class = MoiraiConfig

    def __init__(self, config: MoiraiConfig):
        super().__init__(config)
        self.model = None
        self._device = None

    @property
    def require_even_sampling(self) -> bool:
        """MOIRAI can handle irregularly sampled time series."""
        return False

    @property
    def require_univariate(self) -> bool:
        """MOIRAI supports both univariate and multivariate time series."""
        return False

    @property
    def model_variant(self):
        return self.config.model_variant

    @property
    def model_size(self):
        return self.config.model_size

    @property
    def context_length(self):
        return self.config.context_length

    @property
    def patch_size(self):
        return self.config.patch_size

    @property
    def num_samples(self):
        return self.config.num_samples

    @property
    def batch_size(self):
        return self.config.batch_size

    @property
    def device(self):
        """Get the device to run the model on."""
        if self._device is None:
            device_str = self.config.device
            if device_str == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(device_str)
            logger.info(f"Using device: {self._device}")
        return self._device

    def _load_pretrained_model(self):
        """Load the pretrained MOIRAI model from HuggingFace."""
        if self.model is not None:
            return

        logger.info(f"Loading pretrained MOIRAI model: {self.config.pretrained_model_path}")

        try:
            prediction_length = self.config.max_forecast_steps
            if prediction_length is None:
                prediction_length = 100  # Default prediction length

            if self.model_variant == "moirai":
                module = MoiraiModuleBase.from_pretrained(self.config.pretrained_model_path)
                self.model = MoiraiModule(
                    module=module,
                    prediction_length=prediction_length,
                    context_length=self.context_length,
                    patch_size=self.patch_size,
                    num_samples=self.num_samples,
                    target_dim=self.config.target_dim,
                    feat_dynamic_real_dim=self.config.feat_dynamic_real_dim,
                    past_feat_dynamic_real_dim=self.config.past_feat_dynamic_real_dim,
                )
            elif self.model_variant == "moirai-moe":
                module = MoiraiMoEModuleBase.from_pretrained(self.config.pretrained_model_path)
                self.model = MoiraiMoEModule(
                    module=module,
                    prediction_length=prediction_length,
                    context_length=self.context_length,
                    patch_size=16,  # Fixed for moirai-moe
                    num_samples=self.num_samples,
                    target_dim=self.config.target_dim,
                    feat_dynamic_real_dim=self.config.feat_dynamic_real_dim,
                    past_feat_dynamic_real_dim=self.config.past_feat_dynamic_real_dim,
                )
            else:  # moirai2
                module = Moirai2ModuleBase.from_pretrained(self.config.pretrained_model_path)
                self.model = Moirai2Module(
                    module=module,
                    prediction_length=prediction_length,
                    context_length=self.context_length,
                    target_dim=self.config.target_dim,
                    feat_dynamic_real_dim=self.config.feat_dynamic_real_dim,
                    past_feat_dynamic_real_dim=self.config.past_feat_dynamic_real_dim,
                )

            # Move model to appropriate device
            if hasattr(self.model, "module"):
                self.model.module.to(self.device)

            logger.info(f"Successfully loaded MOIRAI {self.model_variant} ({self.model_size}) model")

        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

    def _prepare_input_tensor(self, time_series: TimeSeries) -> torch.Tensor:
        """
        Convert Merlion TimeSeries to tensor format expected by MOIRAI.

        :param time_series: Input time series
        :return: Tensor of shape (batch=1, time, variate)
        """
        # Convert to pandas DataFrame
        df = time_series.to_pd()

        # Handle target_seq_index if specified
        if self.target_seq_index is not None and len(df.columns) > 1:
            target_col = df.columns[self.target_seq_index]
            df = df[[target_col]]

        # Convert to numpy array
        values = df.values

        # Reshape to (batch, time, variate)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # Add batch dimension
        values = values[np.newaxis, :, :]

        # Convert to tensor
        tensor = torch.as_tensor(values, dtype=torch.float32)

        return tensor

    def _train(self, train_data: pd.DataFrame, train_config=None) -> Tuple[pd.DataFrame, None]:
        """
        Load the pretrained MOIRAI model.

        Note: MOIRAI is a pretrained foundation model designed for zero-shot forecasting.
        This method loads the model but does not perform additional training/fine-tuning.

        :param train_data: Training data as pandas DataFrame
        :param train_config: Additional training configuration (not used for pretrained model)
        :return: Tuple of (predictions on training data, None for uncertainty)
        """
        # Load the pretrained model if not already loaded
        self._load_pretrained_model()

        # Store the last training time for forecasting
        self.last_train_time = to_pd_datetime(train_data.index[-1])

        # Set target name if not already set
        if self.target_seq_index is not None:
            self.target_name = train_data.columns[self.target_seq_index]
        elif len(train_data.columns) == 1:
            self.target_name = train_data.columns[0]
        else:
            self.target_name = train_data.columns[0]
            logger.warning(
                f"Multiple columns found but target_seq_index not specified. "
                f"Using first column '{self.target_name}' as target."
            )

        # For pretrained models, we don't have train predictions in the traditional sense
        # Return empty predictions
        logger.info("MOIRAI is a pretrained model - no additional training performed")
        return pd.DataFrame(index=train_data.index, columns=[self.target_name]), None

    def _forecast(
        self,
        time_stamps: Union[int, List[int]],
        time_series_prev: TimeSeries = None,
        return_prev: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate forecasts for the specified time stamps.

        :param time_stamps: Time stamps to forecast for
        :param time_series_prev: Previous time series data to use as context
        :param return_prev: Whether to return previous values as well
        :return: Tuple of (forecast DataFrame, uncertainty/error DataFrame)
        """
        # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() first.")

        # Determine the number of forecast steps
        if isinstance(time_stamps, (int, float)):
            n_steps = int(time_stamps)
        else:
            n_steps = len(time_stamps)

        # Get context data - use train_data if time_series_prev not provided
        # time_series_prev comes as pd.DataFrame from base class
        if time_series_prev is None or (isinstance(time_series_prev, pd.DataFrame) and time_series_prev.empty):
            if self.train_data is None:
                raise ValueError(
                    "time_series_prev must be provided for MOIRAI forecasting when model has no training data"
                )
            # Use train_data as context
            time_series_prev_ts = self.train_data
        else:
            # Convert DataFrame to TimeSeries
            time_series_prev_ts = TimeSeries.from_pd(time_series_prev)

        # Convert time series to tensor
        past_target = self._prepare_input_tensor(time_series_prev_ts)

        # Ensure we have enough context
        actual_context = past_target.shape[1]
        if actual_context < self.context_length:
            logger.warning(
                f"Context length ({actual_context}) is less than configured "
                f"context_length ({self.context_length}). Padding with zeros."
            )
            # Pad with zeros at the beginning
            padding = torch.zeros(
                (past_target.shape[0], self.context_length - actual_context, past_target.shape[2]),
                dtype=past_target.dtype,
                device=past_target.device,
            )
            past_target = torch.cat([padding, past_target], dim=1)
        elif actual_context > self.context_length:
            # Take the most recent context_length values
            past_target = past_target[:, -self.context_length :, :]

        # Move to device
        past_target = past_target.to(self.device)

        # Prepare masks
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros_like(past_target[:, :, 0], dtype=torch.bool)

        # Generate forecast
        with torch.no_grad():
            if self.model_variant in ["moirai", "moirai-moe"]:
                # For moirai and moirai-moe, use the forward method
                forecast = self.model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad,
                )
                # forecast shape: (batch=1, num_samples, prediction_length, variate)
                # Get median prediction
                forecast_values = torch.median(forecast[0], dim=0).values.cpu().numpy()
                # Get uncertainty as std across samples
                forecast_std = torch.std(forecast[0], dim=0).cpu().numpy()

            else:  # moirai2
                # For moirai2, use the predict method which returns quantiles
                forecast = self.model.predict(past_target)
                # forecast shape: (batch, num_quantiles, prediction_length)
                # Remove batch dimension first
                forecast_no_batch = forecast[0]  # Shape: (num_quantiles, prediction_length)

                # Extract median quantile (typically at middle index)
                median_idx = len(self.model.module.quantile_levels) // 2

                # Handle both tensor and numpy array outputs
                if hasattr(forecast_no_batch, "cpu"):
                    forecast_values = forecast_no_batch[median_idx, :].cpu().numpy()
                else:
                    forecast_values = forecast_no_batch[median_idx, :]

                # Reshape to (prediction_length, 1) for consistency
                if forecast_values.ndim == 1:
                    forecast_values = forecast_values.reshape(-1, 1)

                # Calculate uncertainty from quantile spread (IQR / 1.35 approximates std)
                q25_idx = int(len(self.model.module.quantile_levels) * 0.25)
                q75_idx = int(len(self.model.module.quantile_levels) * 0.75)
                iqr = forecast_no_batch[q75_idx, :] - forecast_no_batch[q25_idx, :]

                # Handle both tensor and numpy array outputs
                if hasattr(iqr, "cpu"):
                    forecast_std = (iqr / 1.35).cpu().numpy()
                else:
                    forecast_std = iqr / 1.35

                # Reshape std to (prediction_length, 1)
                if forecast_std.ndim == 1:
                    forecast_std = forecast_std.reshape(-1, 1)

        # Handle length mismatch - pad or truncate to match requested steps
        actual_steps = forecast_values.shape[0]
        if actual_steps > n_steps:
            # Truncate if we have too many
            forecast_values = forecast_values[:n_steps, :]
            forecast_std = forecast_std[:n_steps, :]
        elif actual_steps < n_steps:
            # Model cannot generate enough steps - this is a limitation
            # Instead of padding with flat line, raise an informative error
            raise ValueError(
                f"Model can only generate {actual_steps} steps (max_forecast_steps={self.config.max_forecast_steps}), "
                f"but {n_steps} were requested. Please either:\n"
                f"  1. Set max_forecast_steps={n_steps} or higher in MoiraiConfig, OR\n"
                f"  2. Use rolling_forecast() method for longer forecast horizons.\n"
                f"Example: model.rolling_forecast(time_series=ts, context_length=168, "
                f"prediction_length={min(actual_steps, 100)}, prediction_stride={min(actual_steps, 100)})"
            )

        # Create time index for forecast
        if isinstance(time_stamps, (int, float)):
            # Generate time stamps based on the last time in time_series_prev
            if time_series_prev is not None and not time_series_prev.is_empty():
                last_time = time_series_prev.tf
            else:
                last_time = self.last_train_time

            # Infer frequency from time_series_prev if available
            if time_series_prev is not None and len(time_series_prev) > 1:
                times = to_pd_datetime(time_series_prev.time_stamps)
                freq = pd.infer_freq(times)
                if freq is None:
                    freq = times[1] - times[0]
            else:
                freq = pd.Timedelta(hours=1)  # Default to hourly

            forecast_index = pd.date_range(start=to_pd_datetime(last_time) + freq, periods=n_steps, freq=freq)
        else:
            forecast_index = to_pd_datetime(time_stamps)

        # Create DataFrames
        columns = [self.target_name] if self.target_name else ["target"]
        forecast_df = pd.DataFrame(
            forecast_values,
            index=forecast_index,
            columns=columns
            if forecast_values.shape[1] == 1
            else [f"{columns[0]}_{i}" for i in range(forecast_values.shape[1])],
        )

        err_df = pd.DataFrame(forecast_std, index=forecast_index, columns=forecast_df.columns)

        return forecast_df, err_df

    def __getstate__(self):
        """Custom serialization to handle the MOIRAI model."""
        state = self.__dict__.copy()
        # Don't serialize the model itself, just the config
        state["model"] = None
        state["_device"] = None
        return state

    def __setstate__(self, state):
        """Custom deserialization to reload the MOIRAI model."""
        self.__dict__.update(state)
        # Model will be reloaded when needed
