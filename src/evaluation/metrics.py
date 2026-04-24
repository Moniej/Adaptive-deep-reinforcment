import pandas as pd


def total_return(trades: pd.DataFrame, initial_balance: float = 10000) -> float:
    if trades.empty:
        return 0.0
    final_balance = trades["balance"].iloc[-1]
    return (final_balance - initial_balance) / initial_balance


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    wins = (trades["pnl"] > 0).sum()
    return wins / len(trades)


def average_win(trades: pd.DataFrame) -> float:
    wins = trades[trades["pnl"] > 0]
    if wins.empty:
        return 0.0
    return wins["pnl"].mean()


def average_loss(trades: pd.DataFrame) -> float:
    losses = trades[trades["pnl"] < 0]
    if losses.empty:
        return 0.0
    return losses["pnl"].mean()


def profit_factor(trades: pd.DataFrame) -> float:
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())

    if gross_loss == 0:
        return 0.0

    return gross_profit / gross_loss


def expectancy(trades: pd.DataFrame) -> float:
    wr = win_rate(trades)
    avg_w = average_win(trades)
    avg_l = abs(average_loss(trades))

    return (wr * avg_w) - ((1 - wr) * avg_l)


def max_drawdown(trades: pd.DataFrame, initial_balance: float = 10000) -> float:
    if trades.empty:
        return 0.0

    equity = pd.concat(
        [
            pd.Series([initial_balance]),
            trades["balance"].reset_index(drop=True)
        ],
        ignore_index=True
    )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()