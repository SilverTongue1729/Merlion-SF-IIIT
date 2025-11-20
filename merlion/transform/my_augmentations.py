import pandas as pd
import numpy as np

from merlion.utils import TimeSeries
from merlion.transform.base import TransformBase


class ConvexHullMethod(TransformBase):
    """
    Applies the Convex Hull Method augmentation.
    This transform is not invertible.
    """
    invertible = False

    def __init__(self, slope_limit: float = 1000000000.0, relaxation_tp: int = 10):
        """
        :param slope_limit: Maximum slope limit for the convex hull.
        :param relaxation_tp: Look-ahead window for calculating slopes.
        """
        super().__init__()
        self.slope_limit = slope_limit
        self.relaxation_tp = relaxation_tp

    def train(self, time_series: TimeSeries) -> None:
        # This transform is stateless (doesn't learn from data)
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        # Apply the augmentation to each column in the TimeSeries
        augmented_df = time_series.to_pd()
        for col in augmented_df.columns:
            augmented_df[col] = self._run_chm(
                augmented_df[col], self.slope_limit, self.relaxation_tp
            )
        return TimeSeries.from_pd(augmented_df)

    @staticmethod
    def _run_chm(series, slope_limit, relaxation_tp):
        """
        Static method containing your original convex_hull_method logic.
        """
        NO_OF_ENTRIES = len(series)
        new_value = []
        
        # Handle empty series
        if NO_OF_ENTRIES == 0:
            return pd.Series([], index=series.index, dtype=series.dtype)

        cur = series.iloc[0]
        
        for i in range(NO_OF_ENTRIES):
            vmax = cur
            imax = i
            new_value.append(cur)
            slope = -slope_limit
            for j in range(i, min(i + relaxation_tp, NO_OF_ENTRIES)):
                # (j - i + 1) is 1 when j == i, avoiding division by zero
                current_slope = (series.iloc[j] - cur) / (j - i + 1)
                if current_slope > slope:
                    slope = current_slope
                    vmax = series.iloc[j]
                    imax = j
            cur += slope
            
        return pd.Series(new_value, index=series.index)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        # This transform is not invertible
        print(f"Warning: {type(self).__name__} is not invertible. "
              "Returning time series as-is.")
        return time_series



class RollingMeans(TransformBase):
    """
    Applies a rolling mean (moving average) to the time series.
    This transform is not invertible.
    
    Note: Merlion has a built-in `merlion.transform.moving_average.MovingAverage`
    which is invertible, but this class implements your specific logic.
    """
    invertible = False

    def __init__(self, window_size: int = 5):
        """
        :param window_size: The size of the rolling window.
        """
        super().__init__()
        self.window_size = window_size

    def train(self, time_series: TimeSeries) -> None:
        # This transform is stateless
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        # .rolling().mean() works directly on the entire DataFrame
        augmented_df = time_series.to_pd().rolling(
            window=self.window_size, min_periods=1
        ).mean()
        return TimeSeries.from_pd(augmented_df)

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        print(f"Warning: {type(self).__name__} is not invertible. "
              "Returning time series as-is.")
        return time_series



class PeakMultiplier(TransformBase):
    """
    Applies the Peak Multiplier augmentation to identify and scale peaks.
    This transform is not invertible.
    """
    invertible = False

    def __init__(self, peak_multiplier: float = 1.5, 
                 peak_threshold: float = 8.0, 
                 hill_dist: int = 5):
        """
        :param peak_multiplier: Factor to multiply peaks by.
        :param peak_threshold: The change threshold to identify the start of a peak.
        :param hill_dist: Window size to smooth the multiplied peaks.
        """
        super().__init__()
        self.peak_multiplier = peak_multiplier
        self.peak_threshold = peak_threshold
        self.hill_dist = hill_dist

    def train(self, time_series: TimeSeries) -> None:
        # This transform is stateless
        pass

    def __call__(self, time_series: TimeSeries) -> TimeSeries:
        # Apply the augmentation to each column in the TimeSeries
        augmented_df = time_series.to_pd()
        for col in augmented_df.columns:
            augmented_df[col] = self._run_pm(
                augmented_df[col], 
                self.peak_multiplier, 
                self.peak_threshold, 
                self.hill_dist
            )
        return TimeSeries.from_pd(augmented_df)

    @staticmethod
    def _run_pm(series, peak_multiplier, peak_threshold, hill_dist):
        """
        Static method containing your original peak_multiplier logic.
        """
        modified = series.copy()
        is_peak = [False] * len(series)
        
        for i in range(1, len(series)):
            if (series.iloc[i] - series.iloc[i-1]) > peak_threshold:
                is_peak[i] = True
            elif is_peak[i-1] and (series.iloc[i] > series.iloc[i-1]):
                is_peak[i] = True
        
        for i in range(len(series)):
            if is_peak[i]:
                modified.iloc[i] = series.iloc[i] * peak_multiplier
        
        mod = modified.copy()
        for i in range(len(modified)):
            for j in range(1, hill_dist + 1): # Fix: include hill_dist
                if i + j < len(modified):
                    mod.iloc[i] = max(mod.iloc[i], modified.iloc[i+j] * (1 - (j / hill_dist)))
                if i - j >= 0:
                    mod.iloc[i] = max(mod.iloc[i], modified.iloc[i-j] * (1 - (j / hill_dist)))
        modified = mod.copy()
        return modified

    def _invert(self, time_series: TimeSeries) -> TimeSeries:
        print(f"Warning: {type(self).__name__} is not invertible. "
              "Returning time series as-is.")
        return time_series