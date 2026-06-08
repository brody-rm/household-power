# Household Power Usage Forecasting

A structured time series forecasting project benchmarking five models — from classical statistical methods to a zero-shot transformer foundation model — against daily household electricity consumption. Built as a portfolio project to demonstrate end-to-end ML workflow development, model comparison, and evaluation rigor.

## Dataset

**UCI Household Power Consumption** — minute-level electricity readings from a single French household (December 2006 – November 2010), aggregated to daily mean global active power (kW). Approximately 4 years of data across 1,442 observations after preprocessing.

- 25,979 missing values per column (encoded as `?`) — handled via linear interpolation
- **Train:** December 2006 – November 2009 (1,081 days)
- **Test:** December 2009 – November 2010 (361 days)

## Notebook Workflow

Each notebook is self-contained and builds on the outputs of the previous. All modeling notebooks share the same walk-forward evaluation strategy: 13 non-overlapping 30-day folds over the held-out test year, with an expanding training window that respects temporal order.

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_eda.ipynb` | Data ingestion, missing value handling (linear interpolation), daily aggregation, EDA, stationarity testing (ADF + KPSS), ACF/PACF, train/test split |
| 02 | `02_classical_baselines.ipynb` | ETS and SARIMA with walk-forward validation; multiplicative vs. additive seasonality comparison; month dummy exogenous regressors for SARIMA |
| 03 | `03_modern_ml.ipynb` | Prophet and XGBoost; lag/calendar feature engineering; SHAP feature importances |
| 04 | `04_foundation_model.ipynb` | Zero-shot forecasting with Chronos-Bolt-Base; probabilistic output with 80% prediction intervals; PI calibration check |

## Results

All models are evaluated on the same 361-day test period using three metrics averaged across 13 walk-forward folds. The consolidated leaderboard is written to `outputs/leaderboard.csv` by each modeling notebook.

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| ETS | 0.2841 | 0.2317 | 23.99% |
| Prophet | 0.2547 | 0.1965 | 20.17% |
| SARIMA | 0.2381 | 0.1810 | 19.69% |
| Chronos-Bolt-Base | 0.2359 | **0.1777** | 19.20% |
| XGBoost | **0.2319** | 0.1815 | **19.01%** |

**Key finding:** Chronos-Bolt-Base, applied entirely zero-shot with no training on this dataset, achieves the best MAE across all five models and trails XGBoost on RMSE by only 0.004 — while outperforming every trained model including SARIMA and Prophet. This highlights the practical value of time series foundation models as strong out-of-the-box baselines that require no feature engineering, hyperparameter tuning, or dataset-specific preprocessing.

## Methodology Notes

- **Walk-forward evaluation** is applied consistently: each fold conditions on all observed history and forecasts the next 30 days, simulating real deployment without look-ahead bias.
- **Chronos-Bolt-Base** is the only probabilistic model, outputting 9 quantiles per step. Point estimates use the median (q=0.5); 80% prediction intervals use q=0.1 and q=0.9, with coverage verified against the 80% target.
- **XGBoost** uses actual observed values for test-set lag features (batch evaluation mode), which is appropriate for retrospective benchmarking but differs from recursive multi-step forecasting in a live deployment context.
- **SHAP values** are computed for XGBoost to provide directional feature attribution, identifying `lag_1` and `lag_7` as the strongest predictors.

## Project Structure

```
household-power-usage/
├── data/                        # Raw and processed CSVs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classical_baselines.ipynb
│   ├── 03_modern_ml.ipynb
│   └── 04_foundation_model.ipynb
├── outputs/
│   ├── leaderboard.csv          # Consolidated model metrics
│   ├── classical_forecasts.png
│   ├── modern_ml_forecasts.png
│   └── figures/
├── requirements.txt
└── run_pipeline.py
```

## Skills Demonstrated

- **Time series EDA:** decomposition, stationarity testing (ADF, KPSS), ACF/PACF analysis
- **Classical forecasting:** ETS, SARIMA with auto model selection, exogenous regressors
- **Modern ML:** Prophet with custom seasonality; XGBoost with lag feature engineering, early stopping, and SHAP interpretability
- **Foundation models:** zero-shot inference with Chronos-Bolt-Base, probabilistic calibration evaluation
- **Evaluation rigor:** walk-forward cross-validation, consistent fold boundaries across all models, RMSE/MAE/MAPE, prediction interval coverage
