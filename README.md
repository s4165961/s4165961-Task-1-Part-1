# s4165961-Task-1-Part-1
Source code for the Data Analysis section (03-analysis) of my RMIT Case Studies in Data Science assignment. This repository contains the code used to produce the machine learning results discussed in my executive summary.

## Job Role
Commercial Analyst (Data Science focused) for Red Energy, an Australian energy retailer. The role centres on revenue forecasting, customer-level demand projections, and time-of-use (TOU) volume analytics.

## Datasets
1. **NSW, AUS, Electricity Price & Demand 2018-2023** (Kaggle, sourced from AEMO)
   https://www.kaggle.com/datasets/joebeachcapital/nsw-australia-electricity-demand-2018-2023

2. **Electricity Consumption Benchmarks** (data.gov.au)
   https://data.gov.au/data/dataset/electricity-consumption-benchmarks

## What the Code Does
Two models are built, one on each dataset:
- **Model 1: Demand Forecasting (Random Forest Regressor):** predicts daily electricity demand using time-based and lagged features. Evaluated with MAE, RMSE, MAPE, and R².
- **Model 2: Customer Segmentation (K-Means Clustering):** groups households by usage pattern (total, peak, and average consumption, plus an evening-to-morning ratio). Evaluated with silhouette score.

## Setup
1. Download Dataset 1 (auto-downloads via `kagglehub` when the script runs — requires a free Kaggle account and API token).
2. Download Dataset 2 manually from the link above and place the CSV in a `data/` folder.
3. Unzip Dataset 1 into `data/price_demand/` so it contains all 66 monthly `PRICE_AND_DEMAND_*.csv` files.
4. Install dependencies:
   ```
   pip install pandas numpy scikit-learn kagglehub
   ```
   
## Running

```
python red_energy_analysis.py
```

This prints the evaluation metrics and cluster profiles used in the report's Data Analysis section.

## Results Summary
- **Forecasting model:** MAPE 3.50%, RMSE 343.11 MW, R² 0.845 (1,998 daily observations, 2018–2023)
- **Segmentation model:** 4 clusters across 25 households, silhouette score 0.477
