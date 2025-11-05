#
# Copyright (c) 2025 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Transforms that rescale the input or otherwise normalize it.
"""
from collections import OrderedDict
import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from sklearn.preprocessing import StandardScaler

from merlion.transform.base import InvertibleTransformBase, TransformBase
from merlion.utils import UnivariateTimeSeries, TimeSeries

logger = logging.getLogger(__name__)


class AbsVal(TransformBase):
    """
    Takes the absolute value of the input time series.
    """

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the "pseudo-inverse" is just the identity (i.e. we lose sign information).
        """
        return False

    @property
    def identity_inversion(self):
        return True

    def train(self, time_series: TimeSeries):
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        return TimeSeries(
            OrderedDict(
                (name, UnivariateTimeSeries(var.index, np.abs(var.np_values))) for name, var in time_series.items()
            )
        )


class Rescale(InvertibleTransformBase):
    """
    Rescales the bias & scale of input vectors or scalars by pre-specified amounts.
    """

    def __init__(self, bias=0.0, scale=1.0, normalize_bias=True, normalize_scale=True):
        super().__init__()
        self.bias = bias
        self.scale = scale
        self.normalize_bias = normalize_bias
        self.normalize_scale = normalize_scale

    @property
    def requires_inversion_state(self):
        """
        ``False`` because rescaling operations are stateless to invert.
        """
        return False

    def train(self, time_series: TimeSeries):
        pass

    @property
    def is_trained(self):
        return self.bias is not None and self.scale is not None

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        if not self.is_trained:
            raise RuntimeError(f"Cannot use {type(self).__name__} without training it first!")

        bias = self.bias if isinstance(self.bias, Mapping) else {name: self.bias for name in time_series.names}
        scale = self.scale if isinstance(self.scale, Mapping) else {name: self.scale for name in time_series.names}
        assert set(time_series.names).issubset(bias.keys()) and set(time_series.names).issubset(scale.keys())

        new_vars = OrderedDict()
        for name, var in time_series.items():
            if self.normalize_bias:
                var = var - bias[name]
            if self.normalize_scale:
                var = var / scale[name]
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series._is_aligned
        return ret

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        if not self.is_trained:
            raise RuntimeError(f"Cannot use {type(self).__name__} without training it first!")
        bias = self.bias if isinstance(self.bias, Mapping) else {name: self.bias for name in time_series.names}
        scale = self.scale if isinstance(self.scale, Mapping) else {name: self.scale for name in time_series.names}
        assert set(time_series.names).issubset(bias.keys()) and set(time_series.names).issubset(scale.keys())

        new_vars = OrderedDict()
        for name, var in time_series.items():
            if self.normalize_scale:
                var = var * scale[name]
            if self.normalize_bias:
                var = var + bias[name]
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        ret = TimeSeries(new_vars, check_aligned=False)
        ret._is_aligned = time_series._is_aligned
        return ret


class MeanVarNormalize(Rescale):
    """
    A learnable transform that rescales the values of a time series to have
    zero mean and unit variance.
    """

    def __init__(self, bias=None, scale=None, normalize_bias=True, normalize_scale=True):
        super().__init__(bias, scale, normalize_bias, normalize_scale)

    def train(self, time_series: TimeSeries):
        bias, scale = {}, {}
        for name, var in time_series.items():
            scaler = StandardScaler().fit(var.np_values.reshape(-1, 1))
            bias[name] = float(scaler.mean_)
            scale[name] = float(scaler.scale_)
        self.bias = bias
        self.scale = scale


class MinMaxNormalize(Rescale):
    """
    A learnable transform that rescales the values of a time series to be
    between zero and one.
    """

    def __init__(self, bias=None, scale=None, normalize_bias=True, normalize_scale=True):
        super().__init__(bias, scale, normalize_bias, normalize_scale)

    def train(self, time_series: TimeSeries):
        bias, scale = {}, {}
        for name, var in time_series.items():
            minval, maxval = var.min(), var.max()
            bias[name] = minval
            scale[name] = np.maximum(1e-8, maxval - minval)
        self.bias = bias
        self.scale = scale


class BoxCoxTransform(InvertibleTransformBase):
    """
    Applies the Box-Cox power transform to the time series, with power lmbda.
    When lmbda is None, we
    When lmbda > 0, it is ((x + offset) ** lmbda - 1) / lmbda.
    When lmbda == 0, it is ln(lmbda + offset).
    """

    def __init__(self, lmbda=None, offset=0.0):
        super().__init__()
        if lmbda is not None:
            if isinstance(lmbda, dict):
                assert all(isinstance(x, (int, float)) for x in lmbda.values())
            else:
                assert isinstance(lmbda, (int, float))
        self.lmbda = lmbda
        self.offset = offset

    @property
    def requires_inversion_state(self):
        """
        ``False`` because the Box-Cox transform does is stateless to invert.
        """
        return False

    def train(self, time_series: TimeSeries):
        if self.lmbda is None:
            self.lmbda = {name: scipy.stats.boxcox(var.np_values + self.offset)[1] for name, var in time_series.items()}
            logger.info(f"Chose Box-Cox lambda = {self.lmbda}")
        elif not isinstance(self.lmbda, Mapping):
            self.lmbda = {name: self.lmbda for name in time_series.names}
        assert len(self.lmbda) == time_series.dim

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = OrderedDict()
        for name, var in time_series.items():
            y = scipy.special.boxcox(var + self.offset, self.lmbda[name])
            var = pd.Series(y, index=var.index, name=var.name)
            new_vars[name] = UnivariateTimeSeries.from_pd(var)

        return TimeSeries(new_vars)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        new_vars = []
        for name, var in time_series.items():
            lmbda = self.lmbda[name]
            if lmbda > 0:
                var = (lmbda * var + 1) ** (1 / lmbda)
                nanvals = var.isna()
                if nanvals.any():
                    var[nanvals] = 0
            else:
                var = var.apply(np.exp)
            new_vars.append(UnivariateTimeSeries.from_pd(var - self.offset))

        return TimeSeries(new_vars)


class SpikeAmplification(InvertibleTransformBase):
    """
    Amplifies spikes (extreme values) in the time series to improve sensitivity
    to high quantile losses (e.g., 0.99 quantile) which punish underprediction.

    This transform identifies values above a certain percentile threshold and
    amplifies them using a power transformation, making the model more sensitive
    to extreme upward movements while preserving the overall distribution shape.

    The transformation works as follows:
    1. Compute baseline statistics (mean, std, or quantiles) from training data
    2. For values above the spike threshold, apply amplification:
       - Linear amplification: spike_value * amplification_factor
       - Power amplification: spike_value ** amplification_power (when power > 1)
    3. Smooth transition using a sigmoid function to avoid discontinuities
    """

    def __init__(
        self,
        spike_threshold_quantile=0.75,
        amplification_factor=1.5,
        amplification_power=1.2,
        smoothing_window=0.1,
        use_power=False,
    ):
        """
        :param spike_threshold_quantile: Quantile threshold above which values are considered spikes (0-1).
            Default 0.75 means top 25% of values are amplified.
        :param amplification_factor: Linear amplification factor for spike values. Default 1.5.
        :param amplification_power: Power to raise spike values to (when use_power=True). Default 1.2.
        :param smoothing_window: Window for smooth transition (as fraction of threshold). Default 0.1.
        :param use_power: If True, use power amplification instead of linear. Default False.
        """
        super().__init__()
        assert 0 < spike_threshold_quantile < 1, "spike_threshold_quantile must be between 0 and 1"
        assert amplification_factor >= 1, "amplification_factor must be >= 1"
        assert amplification_power >= 1, "amplification_power must be >= 1"
        assert 0 < smoothing_window < 1, "smoothing_window must be between 0 and 1"

        self.spike_threshold_quantile = spike_threshold_quantile
        self.amplification_factor = amplification_factor
        self.amplification_power = amplification_power
        self.smoothing_window = smoothing_window
        self.use_power = use_power

        # These will be set during training
        self.threshold = None
        self.mean = None
        self.std = None

    @property
    def requires_inversion_state(self):
        """
        ``False`` because we store the statistics needed for inversion.
        """
        return False

    def train(self, time_series: TimeSeries):
        """
        Compute spike thresholds and statistics from training data.
        """
        self.threshold = {}
        self.mean = {}
        self.std = {}

        for name, var in time_series.items():
            values = var.np_values
            self.threshold[name] = float(np.quantile(values, self.spike_threshold_quantile))
            self.mean[name] = float(np.mean(values))
            self.std[name] = float(np.std(values))

        logger.info(
            f"SpikeAmplification trained with thresholds: {self.threshold}, "
            f"amplification_factor={self.amplification_factor}"
        )

    def _smooth_amplification(self, x, threshold, smoothing_range):
        """
        Apply smooth amplification using a sigmoid-like transition.

        :param x: Input values
        :param threshold: Spike threshold
        :param smoothing_range: Range for smooth transition
        :return: Amplification weights (1.0 for normal values, up to amplification_factor for spikes)
        """
        # Sigmoid function centered at threshold
        # For x < threshold - smoothing_range: weight = 1.0
        # For x > threshold + smoothing_range: weight = amplification_factor
        # In between: smooth transition

        normalized = (x - threshold) / (smoothing_range + 1e-8)
        sigmoid = 1 / (1 + np.exp(-6 * normalized))  # Steepness factor = 6

        # Scale sigmoid from [0, 1] to [1, amplification_factor]
        weight = 1.0 + (self.amplification_factor - 1.0) * sigmoid

        return weight

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        """
        Apply spike amplification to the time series.
        """
        if self.threshold is None:
            raise RuntimeError("SpikeAmplification must be trained before use!")

        new_vars = OrderedDict()

        for name, var in time_series.items():
            values = var.np_values.copy()
            threshold = self.threshold[name]
            mean = self.mean[name]
            std = self.std[name]

            # Compute smoothing range
            smoothing_range = self.smoothing_window * std

            if self.use_power:
                # Power-based amplification for values above threshold
                # Normalize, apply power, denormalize
                above_threshold = values > threshold
                if above_threshold.any():
                    normalized = (values[above_threshold] - mean) / (std + 1e-8)
                    # Apply power only to positive deviations
                    amplified = np.sign(normalized) * (np.abs(normalized) ** self.amplification_power)
                    values[above_threshold] = mean + amplified * std
            else:
                # Smooth linear amplification
                weights = self._smooth_amplification(values, threshold, smoothing_range)
                # Amplify deviation from mean
                values = mean + (values - mean) * weights

            new_vars[name] = UnivariateTimeSeries(var.index, values, name=var.name)

        return TimeSeries(new_vars)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        """
        Invert the spike amplification transform.
        """
        if self.threshold is None:
            raise RuntimeError("SpikeAmplification must be trained before inversion!")

        new_vars = OrderedDict()

        for name, var in time_series.items():
            values = var.np_values.copy()
            threshold = self.threshold[name]
            mean = self.mean[name]
            std = self.std[name]

            # Compute smoothing range
            smoothing_range = self.smoothing_window * std

            if self.use_power:
                # Invert power amplification
                # Estimate which values were amplified (approximate)
                # This is an approximation since we don't know exact original values
                amplified_threshold = mean + ((threshold - mean) * self.amplification_factor)
                above_amplified = values > amplified_threshold

                if above_amplified.any():
                    normalized = (values[above_amplified] - mean) / (std + 1e-8)
                    # Invert power
                    deamplified = np.sign(normalized) * (np.abs(normalized) ** (1.0 / self.amplification_power))
                    values[above_amplified] = mean + deamplified * std
            else:
                # Invert smooth linear amplification
                weights = self._smooth_amplification(values, threshold, smoothing_range)
                # De-amplify deviation from mean
                values = mean + (values - mean) / (weights + 1e-8)

            new_vars[name] = UnivariateTimeSeries(var.index, values, name=var.name)

        return TimeSeries(new_vars)
