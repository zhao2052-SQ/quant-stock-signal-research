import numpy as np
import pandas as pd

def add_features(df, target_horizon=1):
    data = df.copy()
    data["return_1d"] = data["close"].pct_change()
    data["log_return"] = np.log(data["close"]).diff()
    data["range_pct"] = (data["high"] - data["low"]) / data["close"]
    data["volume_change"] = data["volume"].pct_change()
    data["ma_5"] = data["close"].rolling(5).mean()
    data["ma_10"] = data["close"].rolling(10).mean()
    data["ma_ratio_5_10"] = data["ma_5"] / data["ma_10"] - 1
    data["volatility_5"] = data["return_1d"].rolling(5).std()
    data["volatility_10"] = data["return_1d"].rolling(10).std()
    data["future_return"] = data["close"].shift(-target_horizon) / data["close"] - 1
    data["target"] = (data["future_return"] > 0).astype(int)
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data

def feature_columns():
    return [
        "return_1d",
        "log_return",
        "range_pct",
        "volume_change",
        "ma_ratio_5_10",
        "volatility_5",
        "volatility_10",
    ]

def make_sequence_data(df, columns, lookback):
    x_values = df[columns].values.astype("float32")
    y_values = df["target"].values.astype("int64")
    sequences = []
    labels = []
    dates = []
    returns = []
    for i in range(lookback, len(df)):
        sequences.append(x_values[i - lookback:i])
        labels.append(y_values[i])
        dates.append(df["date"].iloc[i])
        returns.append(df["future_return"].iloc[i])
    return np.array(sequences), np.array(labels), np.array(dates), np.array(returns)
