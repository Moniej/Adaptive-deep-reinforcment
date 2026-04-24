import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    total_return,
    win_rate,
    average_win,
    average_loss,
    profit_factor,
    expectancy,
    max_drawdown,
)


def analyze_baseline(trades_path: str):
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])

    results = {
        "total_trades": len(trades),
        "total_return": total_return(trades),
        "win_rate": win_rate(trades),
        "average_win": average_win(trades),
        "average_loss": average_loss(trades),
        "profit_factor": profit_factor(trades),
        "expectancy": expectancy(trades),
        "max_drawdown": max_drawdown(trades),
    }

    results_df = pd.DataFrame([results])

    print("\nBaseline Performance Summary")
    print(results_df.T)

    Path("../../results/metrics").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("../../results/metrics/baseline_metrics.csv", index=False)

    return results_df


if __name__ == "__main__":
    analyze_baseline("../../results/trades/baseline_trades.csv")