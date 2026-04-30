from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_ohlcv
from src.features import add_features, feature_columns, make_sequence_data
from src.train import train_sklearn_model, train_torch_sequence_model
from src.backtest import evaluate_strategy

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)

    raw = load_ohlcv(config["data_path"])
    df = add_features(raw, target_horizon=config["target_horizon"])
    columns = feature_columns()
    model_name = config["model_name"]

    if model_name in {"logistic", "random_forest"}:
        pred, dates, future_returns, clf_metrics = train_sklearn_model(
            df,
            columns,
            config["train_ratio"],
            model_name,
        )
    else:
        x, y, seq_dates, seq_returns = make_sequence_data(df, columns, config["lookback"])
        pred, split, clf_metrics = train_torch_sequence_model(
            x,
            y,
            config["train_ratio"],
            model_name,
        )
        dates = seq_dates[split:]
        future_returns = seq_returns[split:]

    backtest_df, strategy_metrics = evaluate_strategy(
        dates,
        future_returns,
        pred,
        config["initial_capital"],
        config["transaction_cost"],
    )

    metrics = {**clf_metrics, **strategy_metrics}
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    backtest_df.to_csv(output_dir / "backtest.csv", index=False)

    plt.figure()
    plt.plot(backtest_df["date"], backtest_df["equity"])
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png")

    print(metrics)

if __name__ == "__main__":
    main()
