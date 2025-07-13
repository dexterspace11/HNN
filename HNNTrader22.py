# ---------------- Unified True Self-Learning Dream-Hybrid Neural Trading System -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt, time, pickle, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
from sklearn.decomposition import PCA
from collections import Counter

# ---------------- Reinforcement Unit -------------------
class SelfLearningNeuron:
    def __init__(self, state_size):
        self.state = np.random.rand(state_size)
        self.value = 0.0
        self.usage = 0
        self.reward = 0.0
        self.connections = {}

    def distance(self, input_state):
        return np.linalg.norm(self.state - input_state)

    def activate(self, input_state):
        self.usage += 1
        return np.exp(-self.distance(input_state))

    def reinforce(self, reward):
        self.reward += reward
        self.value += reward * 0.1

# ---------------- Agent -------------------
class SelfLearningTrader:
    def __init__(self):
        self.neurons = []
        self.memory = []
        self.epsilon = 1.0  # Exploration rate

    def act(self, state):
        if not self.neurons or np.random.rand() < self.epsilon:
            neuron = SelfLearningNeuron(len(state))
            neuron.state = state
            self.neurons.append(neuron)
            return neuron, np.random.choice(['buy', 'sell', 'hold'])

        for n in self.neurons:
            if not hasattr(n, 'activate'):
                raise TypeError(f"Neuron of type {type(n)} does not have an 'activate' method")

        scores = [(n, n.activate(state)) for n in self.neurons]
        best_neuron = max(scores, key=lambda x: x[1])[0]
        return best_neuron, max(self.memory[-1]["action_values"], key=self.memory[-1]["action_values"].get) if self.memory else 'hold'

    def learn(self, state, action, reward):
        neuron = SelfLearningNeuron(len(state))
        neuron.state = state
        neuron.reinforce(reward)
        self.neurons.append(neuron)
        self.memory.append({"state": state, "action": action, "reward": reward, "action_values": {'buy': 0, 'sell': 0, 'hold': 0}})

        if len(self.memory) >= 2:
            self.memory[-2]["action_values"][action] += reward * 0.1

        self.epsilon = max(0.05, self.epsilon * 0.995)

    def save(self):
        with open("agent_state.pkl", "wb") as f:
            pickle.dump((self.neurons, self.memory), f)

    def load(self):
        if os.path.exists("agent_state.pkl"):
            with open("agent_state.pkl", "rb") as f:
                self.neurons, self.memory = pickle.load(f)

# ---------------- Indicator Utilities -------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return pd.Series(tr).rolling(window=period).mean()

def get_kucoin_data(symbol='BTC/USDT', timeframe='1m', limit=200):
    exchange = ccxt.kucoin()
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df.dropna(inplace=True)
    return df

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠🤖 True Self-Learning Dream-Hybrid BTC/USDT Neural Trader")

    if 'capital' not in st.session_state:
        st.session_state.capital = 1000.0
    if 'position' not in st.session_state:
        st.session_state.position = None
    if 'prediction_log' not in st.session_state:
        st.session_state.prediction_log = []

    agent = SelfLearningTrader()
    agent.load()

    while True:
        try:
            df = get_kucoin_data('BTC/USDT', '1m', 200)
            features = ['close', 'RSI', 'MA20', 'ATR']

            imputer = SimpleImputer(strategy='mean')
            scaler = MinMaxScaler()
            data_imputed = imputer.fit_transform(df[features])
            data_scaled = scaler.fit_transform(data_imputed)

            smoothing_factor = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7
            input_window = data_scaled[-5:].flatten()

            neuron, action = agent.act(input_window)
            current_price = df['close'].iloc[-1]

            reward = 0
            if action == 'buy' and st.session_state.position is None:
                st.session_state.position = current_price
            elif action == 'sell' and st.session_state.position is not None:
                reward = current_price - st.session_state.position
                st.session_state.capital += reward
                st.session_state.position = None
            elif action == 'hold' and st.session_state.position is not None:
                reward = (current_price - st.session_state.position) * 0.01

            agent.learn(input_window, action, reward)
            agent.save()

            predicted_price = current_price + np.random.randn() * 10  # Placeholder

            st.session_state.prediction_log.append({
                'Time': df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                'Predicted': predicted_price,
                'Actual': current_price,
                'Action': action,
                'Reward': reward,
                'Capital': st.session_state.capital
            })

            log_df = pd.DataFrame(st.session_state.prediction_log)
            st.line_chart(log_df.set_index('Time')[['Predicted', 'Actual']].tail(100))

            st.metric("Current Close", f"{current_price:.2f}")
            st.metric("Predicted Close", f"{predicted_price:.2f}")
            st.metric("Capital", f"{st.session_state.capital:.2f}")
            st.metric("Action", action)
            st.metric("Reward", f"{reward:.4f}")

            if len(log_df) > 10:
                returns = log_df['Reward'].fillna(0)
                sharpe = returns.mean() / returns.std() * np.sqrt(60) if returns.std() else 0
                drawdown = (log_df['Capital'].cummax() - log_df['Capital']).max()
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                st.metric("Max Drawdown", f"{drawdown:.2f}")

            st.dataframe(log_df.tail(10))

            time.sleep(60)

        except Exception as e:
            st.error(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

