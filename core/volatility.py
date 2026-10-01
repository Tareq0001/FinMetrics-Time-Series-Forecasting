import numpy as np
import pandas as pd

class VolatilityAnalyticsEngine:
    def __init__(self, decay_factor: float = 0.94):
        self.decay = decay_factor

    def compute_ewma_volatility(self, returns: np.ndarray) -> np.ndarray:
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = self.decay * sigma2[t-1] + (1.0 - self.decay) * (returns[t-1] ** 2)

        return np.sqrt(sigma2)

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.99) -> float:
        mu = np.mean(returns)
        sigma = np.std(returns)
        z = -np.abs(np.percentile(np.random.normal(0, 1, 100000), (1 - confidence) * 100))
        var_metric = -(mu + z * sigma)
        return float(var_metric)
