import numpy as np
import pandas as pd


class TradingEnv:
    """Discrete-action forex environment with behavioral reward shaping."""

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        trade_size: float = 100000.0,
        trading_cost: float = 5.0,
        reward_scale: float = 1000.0,
        loss_penalty: float = 0.2,
        drawdown_penalty: float = 0.5,
        account_drawdown_penalty: float = 0.1,
        inactivity_penalty: float = 0.01,
        invalid_action_penalty: float = 0.005,
        consistency_bonus: float = 0.02,
        consistency_window: int = 10,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.initial_balance = float(initial_balance)
        self.trade_size = float(trade_size)
        self.trading_cost = float(trading_cost)
        self.reward_scale = float(reward_scale)
        self.loss_penalty = float(loss_penalty)
        self.drawdown_penalty = float(drawdown_penalty)
        self.account_drawdown_penalty = float(account_drawdown_penalty)
        self.inactivity_penalty = float(inactivity_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.consistency_bonus = float(consistency_bonus)
        self.consistency_window = int(consistency_window)

        self.feature_columns = [
            column for column in self.df.columns if column != "time"
        ]
        self.n_market_features = len(self.feature_columns)
        self.state_size = self.n_market_features + 4
        self.action_size = 3
        self.n_steps = len(self.df)

        if self.n_steps < 2:
            raise ValueError("TradingEnv requires at least two rows of market data.")

        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.entry_step = None
        self.total_profit = 0.0
        self.max_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.reward_history = []
        self.equity_curve = [self.initial_balance]
        self.trade_log = []
        return self._get_state()

    def _get_row(self) -> pd.Series:
        step = min(self.current_step, self.n_steps - 1)
        return self.df.iloc[step]

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        if self.position == 1:
            return (price - self.entry_price) * self.trade_size
        return (self.entry_price - price) * self.trade_size

    def _update_drawdown(self, equity: float) -> float:
        self.max_balance = max(self.max_balance, equity)
        if self.max_balance == 0:
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = max(0.0, (self.max_balance - equity) / self.max_balance)
        return self.current_drawdown

    def _get_state(self) -> np.ndarray:
        row = self._get_row()
        market_state = row[self.feature_columns].to_numpy(dtype=np.float32)
        price = float(row["close"])
        unrealized_pnl = self._unrealized_pnl(price)
        equity = self.balance + unrealized_pnl
        drawdown = self._update_drawdown(equity)

        agent_state = np.array(
            [
                float(self.position),
                unrealized_pnl / self.reward_scale,
                self.balance / self.initial_balance,
                drawdown,
            ],
            dtype=np.float32,
        )
        return np.concatenate([market_state, agent_state]).astype(np.float32)

    def _open_position(self, direction: int, price: float) -> float:
        self.position = direction
        self.entry_price = price
        self.entry_step = self.current_step
        self.balance -= self.trading_cost
        return -self.trading_cost / self.reward_scale

    def _close_position(self, price: float, forced: bool = False) -> tuple[float, dict]:
        raw_pnl = self._unrealized_pnl(price)
        net_pnl = raw_pnl - self.trading_cost
        self.balance += net_pnl
        self.total_profit += net_pnl

        trade = {
            "entry_step": self.entry_step,
            "exit_step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": price,
            "raw_pnl": raw_pnl,
            "net_pnl": net_pnl,
            "forced_exit": forced,
        }
        self.trade_log.append(trade)

        reward = net_pnl / self.reward_scale
        if net_pnl < 0:
            reward -= self.loss_penalty * abs(net_pnl) / self.reward_scale

        self.position = 0
        self.entry_price = 0.0
        self.entry_step = None
        return reward, trade

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if action not in (self.HOLD, self.BUY, self.SELL):
            raise ValueError(f"Invalid action {action}. Expected one of 0, 1, 2.")

        row = self._get_row()
        price = float(row["close"])
        reward = 0.0
        trade_info = None

        if self.position == 0:
            if action == self.BUY:
                reward += self._open_position(direction=1, price=price)
            elif action == self.SELL:
                reward += self._open_position(direction=-1, price=price)
            else:
                reward -= self.inactivity_penalty
        elif self.position == 1 and action == self.SELL:
            close_reward, trade_info = self._close_position(price)
            reward += close_reward
        elif self.position == -1 and action == self.BUY:
            close_reward, trade_info = self._close_position(price)
            reward += close_reward
        elif (self.position == 1 and action == self.BUY) or (self.position == -1 and action == self.SELL):
            reward -= self.invalid_action_penalty

        equity = self.balance + self._unrealized_pnl(price)
        drawdown = self._update_drawdown(equity)
        reward -= self.drawdown_penalty * drawdown
        reward -= (
            self.account_drawdown_penalty
            * max(0.0, self.initial_balance - equity)
            / self.reward_scale
        )

        if (
            len(self.reward_history) >= self.consistency_window
            and np.mean(self.reward_history[-self.consistency_window:]) > 0
        ):
            reward += self.consistency_bonus

        self.reward_history.append(reward)
        self.equity_curve.append(equity)

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        if done and self.position != 0:
            final_price = float(self.df.iloc[self.n_steps - 1]["close"])
            close_reward, trade_info = self._close_position(final_price, forced=True)
            reward += close_reward
            equity = self.balance
            self.equity_curve[-1] = equity
            self._update_drawdown(equity)

        next_state = self._get_state()
        info = {
            "balance": self.balance,
            "equity": equity,
            "position": self.position,
            "total_profit": self.total_profit,
            "drawdown": self.current_drawdown,
            "trade": trade_info,
        }
        return next_state, float(reward), done, info


if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/eurusd_features.csv")
    env = TradingEnv(df)

    state = env.reset()
    done = False

    while not done:
        action = np.random.choice([0, 1, 2])
        state, reward, done, info = env.step(action)

    print("Final Balance:", info["balance"])
    print("Total Profit:", info["total_profit"])
    print("Trades:", len(env.trade_log))
