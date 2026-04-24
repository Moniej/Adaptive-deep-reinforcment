import pandas as pd


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1"] = df["close"].pct_change()
    df["return_5"] = df["close"].pct_change(5)

    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["momentum_10"] = df["close"] - df["close"].shift(10)

    df["rolling_high_20"] = df["high"].rolling(20).max()
    df["rolling_low_20"] = df["low"].rolling(20).min()

    return df