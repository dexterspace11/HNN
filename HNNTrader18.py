# ---------------- Fully Integrated Dream-Hybrid Neural Trading System -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt, time, pickle, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime

# ---------------- Unified Memory Structures -------------------
class EpisodicMemory:
    def __init__(self):
        self.episodes = {}
        self.current_episode = None

    def create_episode(self, timestamp):
        self.current_episode = timestamp
        self.episodes[timestamp] = {'patterns': [], 'emotional_tags': [], 'context': None}

    def store_pattern(self, pattern, emotional_tag):
        if self.current_episode is None:
            self.create_episode(datetime.now())
        self.episodes[self.current_episode]['patterns'].append(pattern)
        self.episodes[self.current_episode]['emotional_tags'].append(emotional_tag)

class WorkingMemory:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.short_term_patterns = []
        self.temporal_context = []

    def store(self, pattern, temporal_marker):
        if len(self.short_term_patterns) >= self.capacity:
            self.short_term_patterns.pop(0)
            self.temporal_context.pop(0)
        self.short_term_patterns.append(pattern)
        self.temporal_context.append(temporal_marker)

class SemanticMemory:
    def __init__(self):
        self.pattern_relationships = {}

    def store_relationship(self, pattern1, pattern2, strength):
        key = tuple(sorted([str(p) for p in [pattern1, pattern2]]))
        self.pattern_relationships[key] = strength

# ---------------- Hybrid Neurons -------------------
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

class HybridNeuralUnit(SelfLearningNeuron):
    def __init__(self, position, learning_rate=0.1):
        super().__init__(len(position))
        self.position = position
        self.learning_rate = learning_rate
        self.age = 0
        self.emotional_weight = 1.0
        self.last_spike_time = None

    def quantum_inspired_distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        decay = np.exp(-self.age / 100.0)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

    def get_attention_score(self, input_pattern):
        return self.quantum_inspired_distance(input_pattern) * self.emotional_weight

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

    def hebbian_learn(self, other_unit, strength, spike_timing=None):
        stdp_factor = 1.0
        if spike_timing:
            pre_time = spike_timing.get('pre', datetime.now())
            post_time = spike_timing.get('post', datetime.now())
            timing_diff = (pre_time - post_time).total_seconds()
            stdp_factor = np.exp(-abs(timing_diff) / 20.0)
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + strength * self.learning_rate * stdp_factor

# ---------------- Self-Learning Hybrid Trader -------------------
class SelfLearningTrader:
    def __init__(self):
        self.units = []
        self.memory = []
        self.epsilon = 1.0
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory()
        self.semantic_memory = SemanticMemory()
        self.last_prediction = None

    def act(self, state):
        if not self.units or np.random.rand() < self.epsilon:
            unit = HybridNeuralUnit(state)
            self.units.append(unit)
            return unit, np.random.choice(['buy', 'sell', 'hold'])

        scores = [(unit, unit.quantum_inspired_distance(state)) for unit in self.units]
        scores.sort(key=lambda x: x[1], reverse=True)
        best_unit, _ = scores[0]
        return best_unit, max(self.memory[-1]['action_values'], key=self.memory[-1]['action_values'].get) if self.memory else 'hold'

    def learn(self, state, action, reward):
        unit = HybridNeuralUnit(state)
        unit.reinforce(reward)
        self.units.append(unit)
        self.memory.append({"state": state, "action": action, "reward": reward, "action_values": {'buy': 0, 'sell': 0, 'hold': 0}})
        if len(self.memory) >= 2:
            self.memory[-2]['action_values'][action] += reward * 0.1
        self.epsilon = max(0.05, self.epsilon * 0.995)

    def save(self):
        with open("agent_state.pkl", "wb") as f:
            pickle.dump((self.units, self.memory), f)

    def load(self):
        if os.path.exists("agent_state.pkl"):
            with open("agent_state.pkl", "rb") as f:
                self.units, self.memory = pickle.load(f)

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

# ---------------- Streamlit UI -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠🌌 Unified Dream-Hybrid BTC/USDT Neural Trader")

    global exchange
    exchange = ccxt.kucoin()

    agent = SelfLearningTrader()
    agent.load()

    capital = 1000.0
    position = None
    prediction_log = []

    placeholder = st.empty()
    chart_placeholder = st.empty()

    while True:
        df = get_kucoin_data()
        latest = df.iloc[-1:]
        features = ['close', 'RSI', 'MA20']

        imputer = SimpleImputer(strategy='mean')
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(imputer.fit_transform(df[features]))
        state = data_scaled[-5:].flatten()

        unit, action = agent.act(state)
        current_price = latest['close'].values[0]

        reward = 0
        if action == 'buy' and position is None:
            position = current_price
        elif action == 'sell' and position is not None:
            reward = current_price - position
            capital += reward
            position = None
        elif action == 'hold' and position is not None:
            reward = (current_price - position) * 0.01

        agent.learn(state, action, reward)
        agent.save()

        prediction_log.append({'Time': datetime.now(), 'Price': current_price, 'Action': action, 'Reward': reward})

        with placeholder.container():
            st.metric("Price", f"{current_price:.2f}")
            st.metric("Capital", f"{capital:.2f} USDT")
            st.metric("Last Action", action)
            st.metric("Reward", f"{reward:.2f}")

        with chart_placeholder.container():
            st.subheader("📊 Action Log")
            df_log = pd.DataFrame(prediction_log[-100:])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_log['Time'], df_log['Price'], label='Price')
            ax.scatter(df_log[df_log['Action'] == 'buy']['Time'], df_log[df_log['Action'] == 'buy']['Price'], color='green', label='Buy')
            ax.scatter(df_log[df_log['Action'] == 'sell']['Time'], df_log[df_log['Action'] == 'sell']['Price'], color='red', label='Sell')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        time.sleep(60)

if __name__ == "__main__":
    main()