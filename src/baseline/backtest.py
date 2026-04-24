import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.baseline.strategy import generate_signal


def run_backtest(df: pd.DataFrame, stop_loss=0.0010, take_profit=0.0020, initial_balance=10000):
    balance = initial_balance
    position = None
    entry_price = None
    trade_log = []

    for time, row in df.iterrows():
        signal = generate_signal(row)

        if position is None:
            if signal == 1:
                position = "buy"
                entry_price = row["close"]
                entry_time = time
            elif signal == -1:
                position = "sell"
                entry_price = row["close"]
                entry_time = time

        else:
            current_price = row["close"]

            if position == "buy":
                pnl = current_price - entry_price
                if pnl <= -stop_loss or pnl >= take_profit:
                    balance += pnl * 100000
                    trade_log.append({
                        "entry_time": entry_time,
                        "exit_time": time,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl": pnl * 100000,
                        "balance": balance
                    })
                    position = None
                    entry_price = None

            elif position == "sell":
                pnl = entry_price - current_price
                if pnl <= -stop_loss or pnl >= take_profit:
                    balance += pnl * 100000
                    trade_log.append({
                        "entry_time": entry_time,
                        "exit_time": time,
                        "position": position,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl": pnl * 100000,
                        "balance": balance
                    })
                    position = None
                    entry_price = None

    trades = pd.DataFrame(trade_log)
    return trades, balance


if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/eurusd_features.csv", index_col="time", parse_dates=True)

    trades, final_balance = run_backtest(df)

    print("Final Balance:", final_balance)
    print("Number of Trades:", len(trades))
    print(trades.head())

    Path("../../results/trades").mkdir(parents=True, exist_ok=True)
    trades.to_csv("../../results/trades/baseline_trades.csv", index=False)