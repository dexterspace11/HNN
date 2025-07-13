# ---------------- Unified Dream-Hybrid Neural Trading System (Complete Single Script) -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import time
import dill as pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime

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
        self.epsilon = 1.0
        self.drawdowns = []
        self.capital_history = []

    def act(self, state):
        # Exploration or if no neurons yet
        if not self.neurons or np.random.rand() < self.epsilon:
            neuron = SelfLearningNeuron(len(state))
            neuron.state = state
            self.neurons.append(neuron)
            return neuron, np.random.choice(['buy', 'sell', 'hold'])

        # Exploitation: pick neuron with highest activation
        scores = [(n, n.activate(state)) for n in self.neurons]
        best_neuron = max(scores, key=lambda x: x[1])[0]
        # Choose best action from last memory's action values, else hold
        if self.memory:
            last_action_vals = self.memory[-1]["action_values"]
            action = max(last_action_vals, key=last_action_vals.get)
        else:
            action = 'hold'
        return best_neuron, action

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
            pickle.dump((self.neurons, self.memory, self.epsilon, self.capital_history, self.drawdowns), f)

    def load(self):
        if os.path.exists("agent_state.pkl"):
            with open("agent_state.pkl", "rb") as f:
                self.neurons, self.memory, self.epsilon, self.capital_history, self.drawdowns = pickle.load(f)
        else:
            # Initialize missing attributes if new
            self.epsilon = 1.0
            self.capital_history = []
            self.drawdowns = []

    def compute_metrics(self, capital):
        self.capital_history.append(capital)
        if len(self.capital_history) > 1:
            max_capital = max(self.capital_history)
            drawdown = max_capital - capital
            self.drawdowns.append(drawdown)
            volatility = np.std(self.capital_history)
            avg_capital = np.mean(self.capital_history)
            return drawdown, volatility, avg_capital
        return 0, 0, capital

    def learned_rules(self):
        # Top 5 most used neurons as 'rules'
        usage_sorted = sorted(self.neurons, key=lambda x: x.usage, reverse=True)[:5]
        rules = []
        for i, neuron in enumerate(usage_sorted):
            rules.append(f"Neuron {i+1}: Usage={neuron.usage}, Reward={neuron.reward:.2f}")
        if not rules:
            rules = ["No learned rules yet."]
        return rules

# ---------------- Indicators & Data -------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_kucoin_data(symbol='BTC/USDT', timeframe='1m'):
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧐️🤖 Enhanced Dream-Hybrid BTC/USDT Neural Trader")

    global exchange
    exchange = ccxt.kucoin()

    agent = SelfLearningTrader()
    agent.load()

    capital = 1000.0
    position = None
    prediction_log = []

    placeholder = st.empty()
    chart_placeholder = st.empty()
    log_placeholder = st.empty()
    rule_placeholder = st.empty()
    dream_placeholder = st.empty()

    while True:
        df = get_kucoin_data()
        latest = df.iloc[-1:]
        features = ['close', 'RSI', 'MA20']

        imputer = SimpleImputer(strategy='mean')
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(imputer.fit_transform(df[features]))
        state = data_scaled[-5:].flatten()

        neuron, action = agent.act(state)
        current_price = latest['close'].values[0]

        reward = 0
        stop_loss_triggered = False

        if action == 'buy' and position is None:
            position = current_price
        elif action == 'sell' and position is not None:
            reward = current_price - position
            capital += reward
            position = None
        elif action == 'hold' and position is not None:
            unrealized = current_price - position
            reward = unrealized * 0.01
            # Stop loss at 1% loss
            if unrealized < -0.01 * position:
                reward = unrealized
                capital += reward
                position = None
                stop_loss_triggered = True

        agent.learn(state, action, reward)
        agent.save()

        drawdown, volatility, avg_capital = agent.compute_metrics(capital)

        prediction_log.append({
            'Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Price': current_price,
            'Action': action,
            'Reward': reward,
            'Capital': capital,
            'Drawdown': drawdown,
            'StopLossTriggered': stop_loss_triggered
        })

        df_log = pd.DataFrame(prediction_log[-100:])

        with placeholder.container():
            st.metric("Price", f"{current_price:.2f}")
            st.metric("Capital", f"{capital:.2f} USDT")
            st.metric("Last Action", action)
            st.metric("Reward", f"{reward:.2f}")
            st.metric("Drawdown", f"{drawdown:.2f}")
            st.metric("Volatility", f"{volatility:.2f}")

        with chart_placeholder.container():
            st.subheader("📊 Price & Trading Actions")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_log['Time'], df_log['Price'], label='Price', marker='.')
            buy_times = df_log[df_log['Action'] == 'buy']['Time']
            sell_times = df_log[df_log['Action'] == 'sell']['Time']
            ax.scatter(buy_times, df_log[df_log['Action'] == 'buy']['Price'], color='green', label='Buy', marker='^', s=80)
            ax.scatter(sell_times, df_log[df_log['Action'] == 'sell']['Price'], color='red', label='Sell', marker='v', s=80)
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)

        with log_placeholder.container():
            st.subheader("📅 Trading Log (Most Recent 100 Actions)")
            st.dataframe(df_log[::-1].reset_index(drop=True))

        with rule_placeholder.container():
            st.subheader("🧠 Learned Rules Summary")
            for rule in agent.learned_rules():
                st.write(rule)

        with dream_placeholder.container():
            st.subheader("💭 Dream Visualization (Recent Neuron Usage)")
            fig, ax = plt.subplots(figsize=(8, 2))
            activations = [n.usage for n in agent.neurons[-20:]]
            ax.bar(range(len(activations)), activations, color='purple')
            ax.set_xlabel("Neuron Index")
            ax.set_ylabel("Usage Count")
            ax.set_title("Recent Neuron Usage (Growth Visualization)")
            st.pyplot(fig)
            plt.close(fig)

        time.sleep(60)

if __name__ == "__main__":
    main()

