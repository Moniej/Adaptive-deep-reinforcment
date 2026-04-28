import sys
import os

# ensure project root is visible to Python
sys.path.append(os.path.abspath("."))

from src.data.loader import load_forex_data
from src.features.build_features import build_feature_set
from src.environment.trading_env import TradingEnv
from src.agents.dqn_agent import DQNAgent


def main():
    print("Loading data...")
    data = load_forex_data("data/raw/eurusd.csv")

    print("Building features...")
    data = build_feature_set(data)

    # safety check (prevents silent RL crashes)
    if data.isnull().values.any():
        print("❌ NaN values detected after feature engineering")
        return

    print("Initializing environment...")
    env = TradingEnv(data)

    print("Training agent...")
    agent = DQNAgent(env)
    agent.train()

    print("Evaluating agent...")
    results = agent.evaluate()

    print("Done.")
    print("Results:", results)


if __name__ == "__main__":
    main()
