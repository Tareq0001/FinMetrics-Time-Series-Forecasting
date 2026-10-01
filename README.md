# FinMetrics: Quantitative Time Series Analytics & Anomaly Engine 📈📉

A rigorous quantitative finance and time series data analytics engine featuring stationarity verification, GARCH/EWMA volatility modeling, dynamic Z-score confidence bounds, and Value-at-Risk (VaR) computation.

## Mathematical Implementations
1. **Statistical Stationarity Verification:**
   - Vectorized Augmented Dickey-Fuller (ADF) and KPSS statistical testing algorithms for unit-root detection.
   - First-order and logarithmic differencing transforms:
     $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
2. **Parametric & Historical Value-at-Risk (VaR):**
   - High-confidence percentile estimation for $95\%$ and $99\%$ confidence thresholds:
     $$\text{VaR}_{\alpha} = -(\mu + z_{\alpha} \cdot \sigma)$$
3. **Dynamic EWMA Volatility (RiskMetrics Standard):**
   - Recursive Exponentially Weighted Moving Average:
     $$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2, \quad \lambda = 0.94$$
4. **Dynamic Anomaly Thresholds:**
   - Dynamic confidence bounds to capture structural market shocks and anomalies.
