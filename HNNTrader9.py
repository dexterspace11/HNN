# ---------------- Enhanced Hierarchical Neural Network Trader -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import ccxt
import time
from datetime import datetime

# ---------------- Hybrid Memory Structures -------------------
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

# ---------------- Hybrid Neural Structures -------------------
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

class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.gen_threshold = 0.5
        self.last_prediction = None
        self.synaptic_scaling_factor = 1.0
        self.episodic_memory = EpisodicMemory()

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

    def get_buy_sell_range(self, scaler, feature_dim):
        if not self.units:
            return 0.0, 0.0, 0.0
        padded = np.array([
            np.pad(unit.position[:feature_dim], (0, max(0, feature_dim - len(unit.position))), mode='constant')
            for unit in self.units
        ])
        closes = scaler.inverse_transform(padded)[:, 0]
        return closes.min(), closes.max(), closes.max() - closes.min()

# ---------------- Indicators & Data -------------------
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

def get_kucoin_data():
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=data_limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df.dropna(inplace=True)
    return df

# ---------------- Streamlit UI -------------------
exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'
data_limit = 200

st.set_page_config(page_title="Enhanced HNNTrader", layout="wide")
st.title("Enhanced Hierarchical Neural Network Trader")

placeholder = st.empty()
network = HybridNeuralNetwork()

capital = 1000
trade_size = 100
profits = []

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    smoothing_factor = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7
    input_window = data_scaled[-5:].flatten()

    actual_time = df.index[-1]
    actual_close = df['close'].iloc[-1]

    preds_scaled = []
    current_input = input_window.copy()
    for _ in range(3):
        pred_scaled, sim = network.predict_next(current_input, smoothing_factor)
        preds_scaled.append(pred_scaled)
        current_input[:-1] = current_input[1:]
        current_input[-1] = pred_scaled[0]

    pred_closes = []
    for p in preds_scaled:
        reconstructed = np.copy(data_scaled[-1])
        reconstructed[0] = p[0]
        pred_closes.append(scaler.inverse_transform([reconstructed])[0][0])

    buy_price, sell_price, signal_score = network.get_buy_sell_range(scaler, data_scaled.shape[1])

    trade_executed = pred_closes[0] < actual_close  # simulate buy if price expected to rise
    if trade_executed:
        profit = (pred_closes[0] - actual_close) / actual_close * trade_size
        profits.append(profit)

    win_trades = sum(1 for p in profits if p > 0)
    loss_trades = sum(1 for p in profits if p <= 0)
    win_loss_ratio = win_trades / loss_trades if loss_trades else float('inf')
    sharpe_ratio = (np.mean(profits) / np.std(profits)) * np.sqrt(len(profits)) if len(profits) > 1 else 0

    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Next Close +1", f"{pred_closes[0]:.2f}")
        col2.metric("Next Close +2", f"{pred_closes[1]:.2f}")
        col3.metric("Next Close +3", f"{pred_closes[2]:.2f}")
        col4.metric("Actual Close", f"{actual_close:.2f}")

        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Buy Value", f"{buy_price:.2f}")
        bcol2.metric("Sell Value", f"{sell_price:.2f}")
        bcol3.metric("Signal Score", f"{signal_score:.4f}")

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Capital Remaining", f"{capital + sum(profits):.2f} USDT")
        pcol2.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}")
        pcol3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], pred_closes, marker='o', label='Predicted')
        ax.axhline(actual_close, color='red', linestyle='--', label='Actual')
        ax.set_title("Predicted Next 3 Close Values")
        ax.legend()
        st.pyplot(fig)

    time.sleep(60)
