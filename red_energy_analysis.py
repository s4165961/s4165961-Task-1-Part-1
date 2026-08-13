"""
RMIT Data Science Assignment - Part 1.3 Data Analysis
Job role: Commercial Analyst (Data Science focus), Red Energy

Two models:
  1. Random Forest Regressor  -> forecasts daily electricity demand (NSW)
     (Dataset 1: NSW Electricity Price & Demand 2018-2023, Kaggle/AEMO --
      66 monthly CSVs, e.g. PRICE_AND_DEMAND_201801_NSW1.csv)
  2. K-Means Clustering       -> segments households by usage pattern
     (Dataset 2: Electricity Consumption Benchmarks, data.gov.au --
      electricityconsumptionbenchmarkssurveydataaergovhack.csv)

SETUP:
  1. Dataset 1: unzip archive.zip into a folder (default: ./data/price_demand/)
     so it contains all 66 PRICE_AND_DEMAND_*.csv files.
  2. Dataset 2: place electricityconsumptionbenchmarkssurveydataaergovhack.csv
     in ./data/
  3. pip install pandas numpy scikit-learn

Run with: python red_energy_analysis.py
"""

import glob
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# PATHS -- adjust if your files live somewhere else
# ============================================================
PRICE_DEMAND_DIR = "data/price_demand"  # folder containing the 66 monthly CSVs
BENCHMARKS_CSV = "data/electricityconsumptionbenchmarkssurveydataaergovhack.csv"


# ============================================================
# MODEL 1: Demand Forecasting (Random Forest Regressor)
# ============================================================
def load_demand_data(folder):
    """Concatenates all monthly PRICE_AND_DEMAND_*.csv files and resamples
    to daily averages. Daily resampling sidesteps the settlement-interval
    change AEMO made partway through 2021 (30-min periods before, 5-min
    after), so lag features stay consistent across the whole date range."""
    files = sorted(glob.glob(os.path.join(folder, "PRICE_AND_DEMAND_*_NSW1.csv")))
    if not files:
        raise FileNotFoundError(f"No PRICE_AND_DEMAND_*_NSW1.csv files found in {folder}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)

    daily = df.set_index("SETTLEMENTDATE").resample("D").agg(
        {"TOTALDEMAND": "mean", "RRP": "mean"}
    ).dropna().reset_index()

    daily["dayofweek"] = daily["SETTLEMENTDATE"].dt.dayofweek
    daily["month"] = daily["SETTLEMENTDATE"].dt.month
    daily["demand_lag1"] = daily["TOTALDEMAND"].shift(1)
    daily["demand_lag7"] = daily["TOTALDEMAND"].shift(7)
    daily["price_lag1"] = daily["RRP"].shift(1)

    daily = daily.dropna().reset_index(drop=True)
    return daily


def build_forecasting_model(df):
    feature_cols = ["dayofweek", "month", "demand_lag1", "demand_lag7", "price_lag1"]
    X = df[feature_cols]
    y = df["TOTALDEMAND"]

    # Time-ordered split (not random) since this is a time series
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print("=== Model 1: Demand Forecasting (Random Forest) ===")
    print(f"Rows used: {len(df)} daily observations "
          f"({df['SETTLEMENTDATE'].min().date()} to {df['SETTLEMENTDATE'].max().date()})")
    print(f"MAE:  {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R^2:  {r2:.3f}")
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    return model, {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ============================================================
# MODEL 2: Customer Segmentation (K-Means Clustering)
# ============================================================
def load_household_data(path):
    """Loads the wide-format half-hourly benchmarks survey, keeps only
    'general' (regular household) usage rows, and aggregates to one row
    per respondent with summary consumption stats for clustering."""
    df = pd.read_csv(path)
    df = df[df["TYPE"] == "general"].copy()

    usage_cols = [c for c in df.columns if c.startswith("E_") and c.endswith("_WH")]
    df["total_daily_wh"] = df[usage_cols].sum(axis=1)
    df["peak_wh"] = df[usage_cols].max(axis=1)
    df["avg_wh"] = df[usage_cols].mean(axis=1)
    # Simple day/night ratio: sum of 6pm-11:30pm columns vs 6am-11:30am columns
    evening_cols = [c for c in usage_cols if c[2:4] in
                    [f"{h:02d}" for h in range(18, 24)]]
    morning_cols = [c for c in usage_cols if c[2:4] in
                    [f"{h:02d}" for h in range(6, 12)]]
    df["evening_wh"] = df[evening_cols].sum(axis=1)
    df["morning_wh"] = df[morning_cols].sum(axis=1)
    df["evening_morning_ratio"] = df["evening_wh"] / df["morning_wh"].replace(0, np.nan)

    summary = df.groupby("respondent").agg(
        total_daily_wh=("total_daily_wh", "mean"),
        peak_wh=("peak_wh", "mean"),
        avg_wh=("avg_wh", "mean"),
        evening_morning_ratio=("evening_morning_ratio", "mean"),
    ).dropna().reset_index()

    return summary


def build_segmentation_model(df, n_clusters=4):
    feature_cols = ["total_daily_wh", "peak_wh", "avg_wh", "evening_morning_ratio"]
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = labels

    score = silhouette_score(X_scaled, labels)

    print("\n=== Model 2: Customer Segmentation (K-Means) ===")
    print(f"Households used: {len(df)}")
    print(f"Silhouette score: {score:.3f}")
    print("\nCluster sizes:")
    print(df["cluster"].value_counts())
    print("\nCluster profiles (mean values):")
    print(df.groupby("cluster")[feature_cols].mean())

    return kmeans, df, score


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading and processing demand data...")
    demand_df = load_demand_data(PRICE_DEMAND_DIR)
    forecast_model, forecast_metrics = build_forecasting_model(demand_df)

    print("\nLoading and processing household benchmarks data...")
    household_df = load_household_data(BENCHMARKS_CSV)
    cluster_model, clustered_df, silhouette = build_segmentation_model(household_df)

    print("\n\n=== SUMMARY FOR YOUR REPORT ===")
    print(f"Forecasting model: MAE={forecast_metrics['mae']:.2f} MW, "
          f"RMSE={forecast_metrics['rmse']:.2f} MW, MAPE={forecast_metrics['mape']:.2f}%, "
          f"R^2={forecast_metrics['r2']:.3f}")
    print(f"Segmentation model: Silhouette score={silhouette:.3f}, "
          f"{clustered_df['cluster'].nunique()} clusters found "
          f"across {len(clustered_df)} households")
