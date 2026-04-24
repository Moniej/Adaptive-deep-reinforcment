from src.data.loader import load_data
from src.features.build_features import build_features
from src.environment.trading_env import TradingEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent

def main():
    print("Loading data...")
    data = load_data()

    print("Building features...")
    data = build_features(data)

    print("Initializing environment...")
    env = TradingEnv(data)

    print("Training agent...")
    agent = DQNAgent(env)   # or PPOAgent(env)
    agent.train()

    print("Done. Evaluating...")
    results = agent.evaluate()

    print(results)

if __name__ == "__main__":
    main()
