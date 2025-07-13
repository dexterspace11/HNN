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

# ---------------- Hybrid Neural Network -------------------
class HybridNeuralUnit:
    def __init__(self, position, learning_rate=0.1):
        self.position = position
        self.learning_rate = learning_rate
        self.age = 0
        self.usage_count = 0
        self.reward = 0.0
        self.emotional_weight = 1.0
        self.last_spike_time = None
        self.connections = {}

    def quantum_inspired_distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        decay = np.exp(-self.age / 100.0)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

    def get_attention_score(self, input_pattern):
        similarity = self.quantum_inspired_distance(input_pattern)
        return similarity * self.emotional_weight

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

    def hebbian_learn(self, other_unit, strength, spike_timing=None):
        stdp_factor = 1.0
        if spike_timing:
            pre_time = spike_timing.get('pre', datetime.now())
            post_time = spike_timing.get('post', datetime.now())
            timing_diff = (pre_time - post_time).total_seconds()
            stdp_factor = np.exp(-abs(timing_diff) / 20.0)
        strength *= stdp_factor
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + strength * self.learning_rate * self.emotional_weight

    def activate(self, input_state):
        return self.get_attention_score(input_state)

# ---------------- Memory Structures -------------------
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

# ---------------- Pattern Recognition Metrics -------------------
class PatternRecognitionMetrics:
    def __init__(self, network):
        self.network = network
        self.pattern_history = []
        self.signal_history = []

    def verify_pattern_learning(self, test_data):
        patterns = [test_data[['close', 'RSI', 'MA20', 'ATR']].iloc[i].values for i in range(len(test_data) - 1)]
        padded_patterns = [
            np.pad(p, (0, 20 - len(p)), mode='constant') if len(p) < 20 else p
            for p in patterns
        ]
        recognized = sum(self.network.quantum_inspired_distance(p) > 0.6 for p in padded_patterns)
        return recognized / len(padded_patterns) if padded_patterns else 0

    def evaluate_signal_reliability(self, prediction_log):
        signals = [{'buy': p['Buy'], 'sell': p['Sell'], 'predicted': p['Predicted'], 'actual': p['Actual']} for p in prediction_log]
        consistent = sum((s1['buy'] - s1['predicted']) * (s2['buy'] - s2['predicted']) > 0 for s1, s2 in zip(signals[:-1], signals[1:]))
        return consistent / len(signals) if signals else 0

# ---------------- Hybrid Neural Network -------------------
class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.gen_threshold = 0.5
        self.feature_importance = None
        self.drift_threshold = 0.05
        self.last_prediction = None
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory()
        self.semantic_memory = SemanticMemory()
        self.synaptic_scaling_factor = 1.0

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position)
        self.units.append(unit)
        return unit

    def process_input(self, input_data):
        if not self.units:
            return self.generate_unit(input_data), 0.0

        similarities = [(unit, unit.quantum_inspired_distance(input_data)) for unit in self.units]
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_unit, best_similarity = similarities[0]

        emotional_tag = 1.0 + (best_similarity * 0.5)
        self.episodic_memory.store_pattern(input_data, emotional_tag)
        self.working_memory.store(input_data, datetime.now())
        best_unit.emotional_weight = emotional_tag
        best_unit.age = 0
        best_unit.usage_count += 1
        best_unit.update_spike_time()

        if best_similarity < self.gen_threshold:
            return self.generate_unit(input_data), 0.0

        spike_timing = {'pre': best_unit.last_spike_time, 'post': datetime.now()}
        for unit, similarity in similarities[:3]:
            if unit != best_unit:
                attention_score = unit.get_attention_score(input_data)
                unit.hebbian_learn(best_unit, similarity * attention_score, spike_timing)
            unit.age += 1

        return best_unit, best_similarity

    def predict_next(self, input_data, smoothing_factor):
        unit, similarity = self.process_input(input_data)
        predicted = unit.position

        if len(self.units) > 1:
            recent_units = sorted(self.units, key=lambda x: x.usage_count, reverse=True)[:2]
            trend = recent_units[0].position - recent_units[1].position
            predicted += trend * 0.2

        if self.last_prediction is None:
            smoothed = predicted
        else:
            smoothed = self.last_prediction * smoothing_factor + predicted * (1 - smoothing_factor)

        self.last_prediction = smoothed
        return smoothed, similarity

    def estimate_uncertainty(self, similarity):
        return (1.0 - similarity) * self.synaptic_scaling_factor

    def quantum_inspired_distance(self, pattern):
        if not self.units:
            return 0.0
        similarities = [unit.quantum_inspired_distance(pattern) for unit in self.units]
        return max(similarities)

# ---------------- Utility for Performance Metrics -------------------
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0
    return (mean_return - risk_free_rate) / std_return

def calculate_drawdown(equity_curve):
    peak = equity_curve[0]
    max_drawdown = 0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    return max_drawdown

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠🤖 True Self-Learning Dream-Hybrid BTC/USDT Neural Trader")

    # Initialize session state
    if 'capital' not in st.session_state:
        st.session_state.capital = 1000.0
    if 'position' not in st.session_state:
        st.session_state.position = None
    if 'prediction_log' not in st.session_state:
        st.session_state.prediction_log = []
    if 'agent' not in st.session_state:
        st.session_state.agent = SelfLearningTrader()
        # Safe load with fallback
        if os.path.exists("agent_state.pkl"):
            try:
                st.session_state.agent.load()
            except AttributeError:
                st.warning("Incompatible agent_state.pkl file found. Removing and starting fresh.")
                os.remove("agent_state.pkl")

    # Initialize network & metrics fresh each run
    network = HybridNeuralNetwork()
    metrics = PatternRecognitionMetrics(network)

    # Load live data
    with st.spinner("Fetching latest data from KuCoin..."):
        df = get_kucoin_data('BTC/USDT', '1m', 200)

    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    data_imputed = imputer.fit_transform(df[features])
    data_scaled = scaler.fit_transform(data_imputed)

    smoothing_factor = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7
    input_window = data_scaled[-5:].flatten()

    # Agent action
    neuron, action = st.session_state.agent.act(input_window)
    current_price = df['close'].iloc[-1]
    position = st.session_state.position

    # Reward logic
    reward = 0
    if action == 'buy' and position is None:
        st.session_state.position = current_price
        action_msg = f"Bought at {current_price:.2f}"
    elif action == 'sell' and position is not None:
        reward = current_price - position
        st.session_state.capital += reward
        st.session_state.position = None
        action_msg = f"Sold at {current_price:.2f}, Profit: {reward:.2f}"
    elif action == 'hold' and position is not None:
        reward = (current_price - position) * 0.01
        action_msg = f"Holding position bought at {position:.2f}"
    else:
        action_msg = f"Holding (no position)"

    st.session_state.agent.learn(input_window, action, reward)
    st.session_state.agent.save()

    predicted_scaled, similarity = network.predict_next(input_window, smoothing_factor)
    reconstructed = np.copy(data_scaled[-1])
    reconstructed[0] = predicted_scaled[0]
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    # Price bounds for buy/sell signals from units
    buy_price, sell_price = predicted_close, predicted_close
    if network.units:
        n_features = data_scaled.shape[1]
        padded_positions = np.array([
            np.pad(unit.position[:n_features], (0, max(0, n_features - len(unit.position[:n_features]))), mode='constant')
            for unit in network.units
        ])
        closes = scaler.inverse_transform(padded_positions)[:, 0]
        buy_price = closes.min()
        sell_price = closes.max()

    uncertainty = network.estimate_uncertainty(similarity)
    ci = uncertainty * current_price

    # Append prediction log
    st.session_state.prediction_log.append({
        'Time': df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
        'Predicted': predicted_close,
        'Actual': current_price,
        'Buy': buy_price,
        'Sell': sell_price,
        'Error': abs(current_price - predicted_close),
        'Action': action,
        'Reward': reward,
        'Position': st.session_state.position if st.session_state.position else "None"
    })

    # --- Visuals & Metrics ---

    # Price plot with predicted vs actual
    log_df = pd.DataFrame(st.session_state.prediction_log)
    st.subheader("Price Prediction & Actual Close Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(log_df['Time'], log_df['Actual'], label='Actual Close', marker='o')
    ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted Close', marker='x')
    ax.fill_between(log_df['Time'], log_df['Predicted'] - ci, log_df['Predicted'] + ci, color='gray', alpha=0.2, label='Confidence Interval')
    ax.legend()
    ax.set_xticklabels(log_df['Time'], rotation=45, ha='right')
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)

    # Cumulative returns and drawdown
    log_df['Return'] = log_df['Reward']
    log_df['Equity'] = st.session_state.capital + log_df['Return'].cumsum()
    sharpe = calculate_sharpe_ratio(log_df['Return'])
    drawdown = calculate_drawdown(log_df['Equity'].values)

    st.subheader("Performance Metrics")
    st.markdown(f"- **Current Capital:** ${st.session_state.capital:.2f}")
    st.markdown(f"- **Sharpe Ratio:** {sharpe:.4f}")
    st.markdown(f"- **Max Drawdown:** {drawdown*100:.2f}%")
    st.markdown(f"- **Last Action:** {action_msg}")

    # Episodic Memory Dream Visualization
    st.subheader("Episodic Memory Dream (Recent Patterns & Emotional Tags)")
    if network.episodic_memory.current_episode:
        ep = network.episodic_memory.episodes[network.episodic_memory.current_episode]
        ep_df = pd.DataFrame({
            'Pattern': ep['patterns'][-10:],  # last 10 patterns
            'Emotional Tag': ep['emotional_tags'][-10:]
        })
        st.dataframe(ep_df)
    else:
        st.write("No episodic memory yet.")

    # Learned Trading Rules (top connected units & connection strengths)
    st.subheader("Learned Trading Rules & Neural Connections")
    conns = []
    for unit in network.units:
        for other, strength in unit.connections.items():
            conns.append({
                "From Unit": str(unit.position[:5]),
                "To Unit": str(other.position[:5]),
                "Strength": strength
            })
    if conns:
        conns_df = pd.DataFrame(conns)
        st.dataframe(conns_df.sort_values(by="Strength", ascending=False).head(10))
    else:
        st.write("No learned connections yet.")

    # Trading log table
    st.subheader("Trading Log (Recent 20 Actions)")
    st.dataframe(log_df.tail(20))

    # Sleep and refresh every 60 seconds
    st.text("Refreshing in 60 seconds...")
    time.sleep(60)
    st.experimental_rerun()

if __name__ == "__main__":
    main()

