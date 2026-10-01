import numpy as np
import pandas as pd
from scipy import stats

class TimeSeriesStationarityTester:
    @staticmethod
    def log_returns(prices: np.ndarray) -> np.ndarray:
        return np.diff(np.log(np.maximum(prices, 1e-8)))

    @staticmethod
    def rolling_moments(series: pd.Series, window: int = 20) -> pd.DataFrame:
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return pd.DataFrame({
            "value": series,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std
        })
