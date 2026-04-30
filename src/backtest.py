import numpy as np
import pandas as pd

def evaluate_strategy(dates, future_returns, predictions, initial_capital=1.0, transaction_cost=0.0005):
    signal = np.where(predictions == 1, 1.0, -1.0)
    turnover = np.abs(np.diff(signal, prepend=0.0))
    strategy_returns = signal * future_returns - turnover * transaction_cost
    equity = initial_capital * np.cumprod(1 + strategy_returns)
    cumulative_return = equity[-1] / initial_capital - 1
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()
    result = pd.DataFrame({
        "date": dates,
        "future_return": future_returns,
        "prediction": predictions,
        "signal": signal,
        "strategy_return": strategy_returns,
        "equity": equity,
        "drawdown": drawdown,
    })
    metrics = {
        "cumulative_return": float(cumulative_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }
    return result, metrics
