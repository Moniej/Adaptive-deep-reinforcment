import pandas as pd


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df.index.hour

    df["london_session"] = df["hour"].between(7, 15).astype(int)
    df["newyork_session"] = df["hour"].between(13, 21).astype(int)
    df["asia_session"] = df["hour"].between(0, 8).astype(int)

    return df