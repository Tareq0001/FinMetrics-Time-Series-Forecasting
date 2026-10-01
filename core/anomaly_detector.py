import numpy as np
import pandas as pd

class TimeSeriesAnomalyDetector:
    @staticmethod
    def detect_anomalies(series: pd.Series, window: int = 30, k_sigma: float = 2.5) -> pd.DataFrame:
        rolling_mean = series.rolling(window=window, min_periods=5).mean()
        rolling_std = series.rolling(window=window, min_periods=5).std()

        upper_bound = rolling_mean + (k_sigma * rolling_std)
        lower_bound = rolling_mean - (k_sigma * rolling_std)

        is_anomaly = (series > upper_bound) | (series < lower_bound)
        z_scores = (series - rolling_mean) / np.maximum(rolling_std, 1e-6)

        return pd.DataFrame({
            "actual": series,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "z_score": z_scores,
            "anomaly_detected": is_anomaly
        })
