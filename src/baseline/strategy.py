import pandas as pd
from src.baseline.rules import bullish_condition, bearish_condition


def generate_signal(row: pd.Series) -> int:
    """
    Returns:
        1 for buy
       -1 for sell
        0 for hold
    """
    if bullish_condition(row):
        return 1
    elif bearish_condition(row):
        return -1
    return 0