import pandas as pd


def bullish_condition(row: pd.Series) -> bool:
    return (
        row["close"] > row["ma_10"]
        and row["ma_10"] > row["ma_20"]
        and row["momentum_10"] > 0
    )


def bearish_condition(row: pd.Series) -> bool:
    return (
        row["close"] < row["ma_10"]
        and row["ma_10"] < row["ma_20"]
        and row["momentum_10"] < 0
    )