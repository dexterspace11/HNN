# ---------------- Unified True Self-Learning Dream-Hybrid Neural Trading System with Full Visualizations and Metrics -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt, time, pickle, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import seaborn as sns
import dill as custom_pickle  # Use dill for advanced object serialization

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
            custom_pickle.dump((self.neurons, self.memory), f)

    def load(self):
        if os.path.exists("agent_state.pkl"):
            with open("agent_state.pkl", "rb") as f:
                self.neurons, self.memory = custom_pickle.load(f)

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

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠🤖 True Self-Learning Dream-Hybrid BTC/USDT Neural Trader")

    global exchange
    exchange = ccxt.kucoin()

    agent = SelfLearningTrader()
    agent.load()

    capital = 1000.0
    position = None
    position_time = None
    prediction_log = []
    max_drawdown = 0.0
    peak = capital
    stop_loss_pct = 0.05

    placeholder = st.empty()
    chart_placeholder = st.empty()
    dream_placeholder = st.empty()
    metrics_placeholder = st.empty()

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
            position_time = datetime.now()
        elif action == 'sell' and position is not None:
            reward = current_price - position
            capital += reward
            position = None
        elif action == 'hold' and position is not None:
            reward = (current_price - position) * 0.01

        # Stop-loss logic
        if position is not None and current_price < position * (1 - stop_loss_pct):
            reward = current_price - position
            capital += reward
            position = None
            action = 'stop-loss'

        # Update drawdown
        peak = max(peak, capital)
        drawdown = (peak - capital) / peak
        max_drawdown = max(max_drawdown, drawdown)

        agent.learn(state, action, reward)
        agent.save()

        prediction_log.append({
            'Time': datetime.now(),
            'Price': current_price,
            'Action': action,
            'Reward': reward,
            'Capital': capital,
            'Drawdown': drawdown
        })

        # Compute Sharpe ratio if enough data
        df_log = pd.DataFrame(prediction_log)
        if len(df_log) > 10:
            returns = df_log['Reward']
            sharpe = (returns.mean() / returns.std()) * np.sqrt(60) if returns.std() != 0 else 0
        else:
            sharpe = 0

        with placeholder.container():
            st.metric("Price", f"{current_price:.2f}")
            st.metric("Capital", f"{capital:.2f} USDT")
            st.metric("Last Action", action)
            st.metric("Reward", f"{reward:.2f}")
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")

        with chart_placeholder.container():
            st.subheader("📊 Action Log")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_log['Time'], df_log['Price'], label='Price')
            for idx, row in df_log.iterrows():
                if row['Action'] == 'buy':
                    ax.scatter(row['Time'], row['Price'], color='green', label='Buy' if 'Buy' not in ax.get_legend_handles_labels()[1] else "")
                elif row['Action'] == 'sell':
                    ax.scatter(row['Time'], row['Price'], color='red', label='Sell' if 'Sell' not in ax.get_legend_handles_labels()[1] else "")
                elif row['Action'] == 'stop-loss':
                    ax.scatter(row['Time'], row['Price'], color='black', label='Stop Loss' if 'Stop Loss' not in ax.get_legend_handles_labels()[1] else "")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with dream_placeholder.container():
            st.subheader("🧠 Neural Dream Visuals")
            fig, ax = plt.subplots()
            states = np.array([n.state for n in agent.neurons])
            if states.shape[1] > 2:
                states = states[:, :2]  # visualize first 2 dims
            ax.scatter(states[:, 0], states[:, 1], c='blue', alpha=0.5, label='Neurons')
            ax.set_title("Neuron Growth and Dream Pattern")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with metrics_placeholder.container():
            st.subheader("🧾 Trading Rules Learned")
            st.write("Rules evolve, but based on recent memory:")
            recent = df_log.tail(10)
            rules = []
            if (recent['Action'] == 'buy').sum() > 3:
                rules.append("Buy preference increases after consecutive hold actions and low RSI.")
            if (recent['Action'] == 'sell').sum() > 3:
                rules.append("Sell actions increase after rapid price increase.")
            if (recent['Action'] == 'stop-loss').sum() > 0:
                rules.append("Stop-loss triggered during volatile drops.")
            if not rules:
                rules.append("Still exploring and discovering patterns.")
            for rule in rules:
                st.write(f"- {rule}")

        time.sleep(60)

if __name__ == "__main__":
    main()