# Quant Stock Signal Research

This repository is a research-oriented prototype for stock return prediction and signal-based backtesting.

It compares statistical machine learning and deep learning models on OHLCV-based financial time series data. The workflow includes feature engineering, rolling-window training, forward-looking evaluation, signal generation, and basic strategy backtesting.

The repository includes a small synthetic OHLCV sample file only for demonstrating the expected data format. Real market data should be downloaded separately before running meaningful experiments.
## Project Structure

```text
quant_stock_signal_research/
├── config.yaml
├── requirements.txt
├── run_experiment.py
├── data/
│   └── raw/
│       └── sample_ohlcv.csv
├── outputs/
└── src/
    ├── data.py
    ├── features.py
    ├── models.py
    ├── backtest.py
    └── train.py
```

## Data Format

Put your OHLCV data in:

```text
data/raw/sample_ohlcv.csv
```

Expected columns:

```text
date,open,high,low,close,volume
```

## Models

The current version includes:

- Logistic Regression
- Random Forest
- LSTM
- 2D CNN style sequence classifier

## Evaluation

The project reports:

- Accuracy
- F1 score
- Cumulative return
- Sharpe ratio
- Maximum drawdown

## Run

```bash
pip install -r requirements.txt
python run_experiment.py
```

## Research Positioning

This is a reproducible research prototype for learning-based quantitative financial signal generation. It is not investment advice and should not be used for live trading without further validation.
