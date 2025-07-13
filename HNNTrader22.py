# ---------------- Unified Dream-Hybrid Neural Trading System with Visualizations, Rules, and Risk Controls -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt, time, pickle, os
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

    def act(self, state):
        if not self.neurons or np.random.rand() < self.epsilon:
            neuron = SelfLearningNeuron(len(state))
            neuron.state = state
            self.neurons.append(neuron)
            return neuron, np.random.choice(['buy', 'sell', 'hold'])

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

# ---------------- Indicators & Data -------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_kucoin_data(symbol='BTC/USDT', timeframe='1m'):
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

# ---------------- Evaluation -------------------
def calculate_sharpe_ratio(returns):
    if len(returns) < 2:
        return 0.0
    return np.mean(returns) / (np.std(returns) + 1e-8)

def calculate_drawdown(capital_log):
    peaks = np.maximum.accumulate(capital_log)
    drawdowns = (peaks - capital_log) / (peaks + 1e-8)
    return np.max(drawdowns) if len(drawdowns) else 0.0

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠 Unified Dream-Hybrid BTC/USDT Neural Trader")

    global exchange
    exchange = ccxt.kucoin()
    agent = SelfLearningTrader()
    agent.load()

    capital = 1000.0
    initial_capital = capital
    position = None
    stop_loss_pct = 0.02
    capital_log, return_log, prediction_log = [], [], []

    placeholder = st.empty()
    chart_placeholder = st.empty()
    dream_placeholder = st.empty()
    rule_placeholder = st.empty()

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
        if action == 'buy' and position is None:
            position = current_price
        elif action == 'sell' and position is not None:
            reward = current_price - position
            capital += reward
            return_log.append(reward)
            position = None
        elif action == 'hold' and position is not None:
            reward = (current_price - position) * 0.01

        if position is not None and current_price < position * (1 - stop_loss_pct):
            reward = (current_price - position)
            capital += reward
            return_log.append(reward)
            position = None

        agent.learn(state, action, reward)
        agent.save()

        capital_log.append(capital)
        prediction_log.append({'Time': datetime.now(), 'Price': current_price, 'Action': action, 'Reward': reward, 'Capital': capital})

        # ----- Visuals and Metrics -----
        with placeholder.container():
            st.metric("📈 Price", f"{current_price:.2f}")
            st.metric("💰 Capital", f"{capital:.2f} USDT")
            st.metric("📊 Sharpe Ratio", f"{calculate_sharpe_ratio(return_log):.2f}")
            st.metric("📉 Max Drawdown", f"{calculate_drawdown(np.array(capital_log)) * 100:.2f}%")
            st.metric("🧠 Neuron Count", len(agent.neurons))
            st.metric("🤖 Last Action", action)

        with chart_placeholder.container():
            df_log = pd.DataFrame(prediction_log[-100:])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_log['Time'], df_log['Price'], label='Price')
            ax.scatter(df_log[df_log['Action'] == 'buy']['Time'], df_log[df_log['Action'] == 'buy']['Price'], color='green', label='Buy')
            ax.scatter(df_log[df_log['Action'] == 'sell']['Time'], df_log[df_log['Action'] == 'sell']['Price'], color='red', label='Sell')
            ax.set_title("Trading Log")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with dream_placeholder.container():
            dream_states = np.array([n.state for n in agent.neurons])
            if len(dream_states) > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                projected = pca.fit_transform(dream_states)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.scatter(projected[:, 0], projected[:, 1], c='purple', alpha=0.6)
                ax2.set_title("💭 Dream State Visualization (PCA of Neurons)")
                st.pyplot(fig2)
                plt.close(fig2)

        with rule_placeholder.container():
            st.markdown("### 📘 Inferred Trading Rules")
            if agent.memory:
                last_memory = agent.memory[-1]
                top_action = max(last_memory['action_values'], key=last_memory['action_values'].get)
                st.write(f"The algorithm currently favors **{top_action.upper()}** under similar state patterns.")
            st.write(f"Exploration rate: {agent.epsilon:.2f}")
            st.write("State-to-action memory size:", len(agent.memory))

        time.sleep(60)

if __name__ == "__main__":
    main()

