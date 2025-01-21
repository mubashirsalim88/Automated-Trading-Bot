import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import gym

def load_data(filepath):
    """
    Load and preprocess data from a CSV file.
    """
    data = pd.read_csv(filepath, parse_dates=['Timestamp'], dayfirst=True)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d-%m-%Y %H:%M')
    data.set_index('Timestamp', inplace=True)

    scaler = MinMaxScaler()
    scaled_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 
                      'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])
    return data

class CryptoTradingEnv(gym.Env):
    """
    Custom Gym environment for cryptocurrency trading.
    """
    def __init__(self, data, window_size=10):
        super(CryptoTradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.window_size, data.shape[1]), dtype=np.float32
        )

    def reset(self):
        self.current_step = self.window_size
        return self._next_observation()

    def _next_observation(self):
        return np.array(self.data.iloc[self.current_step - self.window_size:self.current_step])

    def step(self, action):
        prev_state = self.data.iloc[self.current_step - 1]
        current_state = self.data.iloc[self.current_step]
        reward = 0
        if action == 1:  # Buy
            reward = current_state['Close'] - prev_state['Close']
        elif action == 2:  # Sell
            reward = prev_state['Close'] - current_state['Close']
        self.current_step += 1
        done = self.current_step >= len(self.data)
        return self._next_observation(), reward, done, {}

    def render(self):
        print(f'Step: {self.current_step}, Close Price: {self.data["Close"].iloc[self.current_step]}')
