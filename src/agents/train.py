import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.baseline.backtest import run_backtest
from src.environment.trading_env import TradingEnv
from src.evaluation.metrics import max_drawdown, profit_factor, total_return, win_rate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "eurusd_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_feature_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = max(2, int(len(df) * train_ratio))
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    if len(test_df) < 2:
        raise ValueError("Not enough rows in test split. Increase dataset size or lower train_ratio.")
    return train_df, test_df


def train_dqn(env: TradingEnv, episodes: int = 20) -> tuple[DQNAgent, list[dict]]:
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss:
                losses.append(loss)
            total_reward += reward
            state = next_state

        history.append(
            {
                "episode": episode,
                "reward": total_reward,
                "balance": info["balance"],
                "profit": info["total_profit"],
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "epsilon": agent.epsilon,
            }
        )
        print(
            f"[DQN] Episode {episode}/{episodes} | "
            f"balance={info['balance']:.2f} profit={info['total_profit']:.2f} "
            f"reward={total_reward:.2f} epsilon={agent.epsilon:.4f}"
        )

    return agent, history


def train_ppo(env: TradingEnv, episodes: int = 10) -> tuple[PPOAgent, list[dict]]:
    agent = PPOAgent(state_size=env.state_size, action_size=env.action_size)
    history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(reward, done)
            total_reward += reward
            state = next_state

        loss = agent.learn()
        history.append(
            {
                "episode": episode,
                "reward": total_reward,
                "balance": info["balance"],
                "profit": info["total_profit"],
                "avg_loss": loss,
            }
        )
        print(
            f"[PPO] Episode {episode}/{episodes} | "
            f"balance={info['balance']:.2f} profit={info['total_profit']:.2f} "
            f"reward={total_reward:.2f} loss={loss:.4f}"
        )

    return agent, history


def evaluate_agent(
    env: TradingEnv,
    policy_fn,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = env.reset()
    done = False
    step_records = []

    while not done:
        action = int(policy_fn(state))
        next_state, reward, done, info = env.step(action)
        step_records.append(
            {
                "step": env.current_step,
                "reward": reward,
                "balance": info["balance"],
                "equity": info["equity"],
                "drawdown": info["drawdown"],
                "position": info["position"],
                "action": action,
            }
        )
        state = next_state

    trades = pd.DataFrame(env.trade_log)
    if trades.empty:
        summary = {
            "agent": label,
            "final_balance": env.balance,
            "total_profit": env.total_profit,
            "total_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "max_drawdown": env.current_drawdown,
            "profit_factor": 0.0,
        }
    else:
        trade_metrics = trades.rename(columns={"net_pnl": "pnl"})
        trade_metrics["balance"] = env.initial_balance + trade_metrics["pnl"].cumsum()
        summary = {
            "agent": label,
            "final_balance": env.balance,
            "total_profit": env.total_profit,
            "total_trades": len(trades),
            "win_rate": win_rate(trade_metrics),
            "total_return": total_return(trade_metrics, initial_balance=env.initial_balance),
            "max_drawdown": max_drawdown(trade_metrics, initial_balance=env.initial_balance),
            "profit_factor": profit_factor(trade_metrics),
        }

    return pd.DataFrame(step_records), pd.DataFrame([summary])


def evaluate_baseline(test_df: pd.DataFrame) -> pd.DataFrame:
    trades, final_balance = run_backtest(test_df)
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "agent": "baseline",
                    "final_balance": final_balance,
                    "total_profit": final_balance - 10000,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "agent": "baseline",
                "final_balance": final_balance,
                "total_profit": final_balance - 10000,
                "total_trades": len(trades),
                "win_rate": win_rate(trades),
                "total_return": total_return(trades),
                "max_drawdown": max_drawdown(trades),
                "profit_factor": profit_factor(trades),
            }
        ]
    )


def evaluate_baseline_detailed(test_df: pd.DataFrame, initial_balance: float = 10000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades, final_balance = run_backtest(test_df, initial_balance=initial_balance)

    if trades.empty:
        step_records = pd.DataFrame(
            [{"step": 0, "balance": initial_balance, "equity": initial_balance, "drawdown": 0.0}]
        )
        summary = pd.DataFrame(
            [
                {
                    "agent": "baseline",
                    "final_balance": final_balance,
                    "total_profit": final_balance - initial_balance,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                }
            ]
        )
        return step_records, summary

    equity = pd.concat(
        [pd.Series([initial_balance]), trades["balance"].reset_index(drop=True)],
        ignore_index=True,
    )
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    step_records = pd.DataFrame(
        {
            "step": np.arange(len(equity)),
            "balance": equity,
            "equity": equity,
            "drawdown": drawdown,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "agent": "baseline",
                "final_balance": final_balance,
                "total_profit": final_balance - initial_balance,
                "total_trades": len(trades),
                "win_rate": win_rate(trades),
                "total_return": total_return(trades, initial_balance=initial_balance),
                "max_drawdown": max_drawdown(trades, initial_balance=initial_balance),
                "profit_factor": profit_factor(trades),
            }
        ]
    )
    return step_records, summary


def save_training_history(history: list[dict], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def plot_training_curves(dqn_history: list[dict], ppo_history: list[dict]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot([row["episode"] for row in dqn_history], [row["balance"] for row in dqn_history], label="DQN")
    plt.plot([row["episode"] for row in ppo_history], [row["balance"] for row in ppo_history], label="PPO")
    plt.xlabel("Episode")
    plt.ylabel("Episode-End Balance")
    plt.title("Training Balance by Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_balance.png")
    plt.close()


def plot_equity_curves(
    baseline_steps: pd.DataFrame,
    dqn_steps: pd.DataFrame,
    ppo_steps: pd.DataFrame,
) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(baseline_steps["step"], baseline_steps["equity"], label="Baseline", linewidth=2)
    plt.plot(dqn_steps["step"], dqn_steps["equity"], label="DQN", linewidth=2)
    plt.plot(ppo_steps["step"], ppo_steps["equity"], label="PPO", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.title("Equity Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "equity_curve_comparison.png")
    plt.close()


def plot_drawdown_curves(
    baseline_steps: pd.DataFrame,
    dqn_steps: pd.DataFrame,
    ppo_steps: pd.DataFrame,
) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(baseline_steps["step"], baseline_steps["drawdown"], label="Baseline", linewidth=2)
    plt.plot(dqn_steps["step"], dqn_steps["drawdown"], label="DQN", linewidth=2)
    plt.plot(ppo_steps["step"], ppo_steps["drawdown"], label="PPO", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Drawdown")
    plt.title("Drawdown Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "drawdown_comparison.png")
    plt.close()


def plot_summary_bars(comparison: pd.DataFrame) -> None:
    ordered = comparison.set_index("agent").reindex(["baseline", "dqn", "ppo"]).reset_index()

    plt.figure(figsize=(9, 5))
    plt.bar(ordered["agent"], ordered["total_profit"], color=["#6c757d", "#1f77b4", "#ff7f0e"])
    plt.xlabel("Agent")
    plt.ylabel("Total Profit")
    plt.title("Profit Comparison")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "profit_comparison.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(ordered["agent"], ordered["total_trades"], color=["#6c757d", "#1f77b4", "#ff7f0e"])
    plt.xlabel("Agent")
    plt.ylabel("Number of Trades")
    plt.title("Trade Count Comparison")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "trade_count_comparison.png")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN and PPO forex trading agents.")
    parser.add_argument("--dqn-episodes", type=int, default=20, help="Number of DQN training episodes.")
    parser.add_argument("--ppo-episodes", type=int, default=10, help="Number of PPO training episodes.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows loaded from the feature dataset for faster experiments.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio between 0 and 1.",
    )
    return parser.parse_args()


def main(
    dqn_episodes: int = 20,
    ppo_episodes: int = 10,
    max_rows: int | None = None,
    train_ratio: float = 0.8,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    df = load_feature_data()
    if max_rows is not None:
        df = df.head(max_rows).copy()
        print(f"Using a subset of {len(df)} rows for this run.")

    train_df, test_df = split_data(df, train_ratio=train_ratio)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    train_env_dqn = TradingEnv(train_df)
    train_env_ppo = TradingEnv(train_df)
    test_env_dqn = TradingEnv(test_df)
    test_env_ppo = TradingEnv(test_df)

    dqn_agent, dqn_history = train_dqn(train_env_dqn, episodes=dqn_episodes)
    ppo_agent, ppo_history = train_ppo(train_env_ppo, episodes=ppo_episodes)

    dqn_agent.save(str(MODELS_DIR / "dqn_agent.pt"))
    ppo_agent.save(str(MODELS_DIR / "ppo_agent.pt"))

    save_training_history(dqn_history, RESULTS_DIR / "metrics" / "dqn_training_history.csv")
    save_training_history(ppo_history, RESULTS_DIR / "metrics" / "ppo_training_history.csv")

    dqn_steps, dqn_summary = evaluate_agent(test_env_dqn, lambda state: dqn_agent.act(state, greedy=True), "dqn")
    ppo_steps, ppo_summary = evaluate_agent(test_env_ppo, lambda state: ppo_agent.act(state, greedy=True), "ppo")
    baseline_steps, baseline_summary = evaluate_baseline_detailed(test_df)

    baseline_steps.to_csv(RESULTS_DIR / "metrics" / "baseline_step_results.csv", index=False)
    dqn_steps.to_csv(RESULTS_DIR / "metrics" / "dqn_step_results.csv", index=False)
    ppo_steps.to_csv(RESULTS_DIR / "metrics" / "ppo_step_results.csv", index=False)
    dqn_summary.to_csv(RESULTS_DIR / "dqn_results.csv", index=False)
    ppo_summary.to_csv(RESULTS_DIR / "ppo_results.csv", index=False)
    baseline_summary.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

    comparison = pd.concat([baseline_summary, dqn_summary, ppo_summary], ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "metrics" / "agent_comparison.csv", index=False)

    plot_training_curves(dqn_history, ppo_history)
    plot_equity_curves(baseline_steps, dqn_steps, ppo_steps)
    plot_drawdown_curves(baseline_steps, dqn_steps, ppo_steps)
    plot_summary_bars(comparison)

    print("\nTraining complete.")
    print(comparison)


if __name__ == "__main__":
    args = parse_args()
    main(
        dqn_episodes=args.dqn_episodes,
        ppo_episodes=args.ppo_episodes,
        max_rows=args.max_rows,
        train_ratio=args.train_ratio,
    )
