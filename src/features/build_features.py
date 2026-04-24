import sys
from pathlib import Path
import pandas as pd

# add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_forex_data
from src.features.price_features import add_price_features
from src.features.volatility_features import add_volatility_features
from src.features.session_features import add_session_features


def build_feature_set(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_forex_data(input_path)

    df = add_price_features(df)
    df = add_volatility_features(df)
    df = add_session_features(df)

    df = df.dropna()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    print("✅ Features built and saved")
    print("Final shape:", df.shape)
    print(df.head())

    return df


if __name__ == "__main__":
    build_feature_set(
        input_path="../../data/raw/eurusd.csv",
        output_path="../../data/processed/eurusd_features.csv"
    )