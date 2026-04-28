import sys
import os
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

    print("Initializing environment...")
    env = TradingEnv(data)

    print("Training agent...")
    agent = DQNAgent(env)
    agent.train()

    print("Done. Evaluating...")


if __name__ == "__main__":
    main()
