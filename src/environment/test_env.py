#import from trading_env  # ❌ remove this

import pandas as pd

df = pd.read_csv("../../data/processed/eurusd_features.csv")
print(df.columns)

# define the class BEFORE using it
class TradingEnv:
    def __init__(self, df):
        self.df = df.reset_index()
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.entry_price = 0
        return self._get_state()

    def _get_state(self):
        return self.df.iloc[self.current_step].drop("time").values

    def step(self, action):
        # simple step logic for testing
        self.current_step += 1
        reward = 0
        done = self.current_step >= len(self.df) - 1
        return self._get_state(), reward, done

# now you can create the environment
env = TradingEnv(df)

done = False
state = env.reset()  # ✅ always reset first

while not done:
    action = 1  # test: always buy
    state, reward, done = env.step(action)

print("Final Balance:", env.balance)
