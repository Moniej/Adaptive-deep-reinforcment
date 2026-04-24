import pandas as pd


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["candle_range"] = df["high"] - df["low"]
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()

    return df