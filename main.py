import numpy as np
import pandas as pd
from core.stationarity import TimeSeriesStationarityTester
from core.volatility import VolatilityAnalyticsEngine
from core.anomaly_detector import TimeSeriesAnomalyDetector

def generate_gbm_market_data(n_points=350, S0=150.0, mu=0.08, sigma=0.25):
    np.random.seed(1337)
    dt = 1.0 / 252.0
    t = np.linspace(0, n_points * dt, n_points)
    
    W = np.random.standard_normal(size=n_points)
    W = np.cumsum(W) * np.sqrt(dt)
    
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    prices = S0 * np.exp(drift + diffusion)

    shock_indices = [75, 210, 310]
    for idx in shock_indices:
        prices[idx] *= 1.14

    return pd.Series(prices, name="Price")

def run_quantitative_analysis():
    print("=" * 64)
    print("   FINMETRICS: ADVANCED QUANTITATIVE TIME SERIES ANALYTICS      ")
    print("=" * 64)

    prices = generate_gbm_market_data(n_points=350)
    print(f"[OK] Generated Geometric Brownian Motion asset series ({len(prices)} ticks).")

    returns = TimeSeriesStationarityTester.log_returns(prices.values)
    print(f"--> Sample Log-Returns Mean:     {np.mean(returns):.6f}")
    print(f"--> Sample Log-Returns Variance: {np.var(returns):.6f}")

    vol_engine = VolatilityAnalyticsEngine(decay_factor=0.94)
    ewma_vol = vol_engine.compute_ewma_volatility(returns)
    var_95 = vol_engine.calculate_var(returns, confidence=0.95)
    var_99 = vol_engine.calculate_var(returns, confidence=0.99)
    print(f"--> 95% Parametric VaR (Value-at-Risk): {var_95 * 100:.3f}%")
    print(f"--> 99% Parametric VaR (Value-at-Risk): {var_99 * 100:.3f}%")
    print(f"--> Latest Annualized EWMA Volatility:  {ewma_vol[-1] * np.sqrt(252) * 100:.2f}%")

    print("\n[Detecting Volatility & Price Anomalies via Dynamic Bands]...")
    anomaly_df = TimeSeriesAnomalyDetector.detect_anomalies(prices, window=25, k_sigma=2.6)
    anomalies = anomaly_df[anomaly_df["anomaly_detected"]]

    print(f"Identified {len(anomalies)} structural statistical outliers:")
    for idx, row in anomalies.iterrows():
        p_val = row['actual']
        z_val = row['z_score']
        print(f"   [Tick {idx:03d}] Price: ${p_val:.2f} | Z-Score: {z_val:.2f} (Shock Detected)")

    print("\n" + "=" * 64)
    print("       QUANTITATIVE FINANCIAL PIPELINE EXECUTED SUCCESSFULLY    ")
    print("=" * 64)

if __name__ == "__main__":
    run_quantitative_analysis()
