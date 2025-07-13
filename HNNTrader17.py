# ---------------- Final Adaptive Dream-Hybrid Neural Trading System -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import ccxt
import time
from datetime import datetime
import pickle
import os

# ---------------- Hybrid Neural Unit -------------------
class HybridNeuralUnit:
    def __init__(self, position, learning_rate=0.1):
        self.position = position
        self.learning_rate = learning_rate
        self.age = 0
        self.usage_count = 0
        self.reward = 0.0
        self.emotional_weight = 1.0
        self.last_spike_time = datetime.now()
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
            if pre_time and post_time:
                timing_diff = (pre_time - post_time).total_seconds()
                stdp_factor = np.exp(-abs(timing_diff) / 20.0)
        strength *= stdp_factor
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + strength * self.learning_rate * self.emotional_weight

# ---------------- Hybrid Neural Network -------------------
class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.last_prediction = None
        self.synaptic_scaling_factor = 1.0

    def process_input(self, input_data):
        if not self.units:
            return self._generate_unit(input_data), 0.0

        similarities = [(u, u.quantum_inspired_distance(input_data)) for u in self.units]
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_unit, best_sim = similarities[0]
        best_unit.age = 0
        best_unit.usage_count += 1
        best_unit.update_spike_time()

        if best_sim < 0.6:
            return self._generate_unit(input_data), 0.0

        for u, sim in similarities[:3]:
            if u != best_unit:
                u.hebbian_learn(best_unit, sim, {'pre': u.last_spike_time, 'post': datetime.now()})
            u.age += 1

        return best_unit, best_sim

    def _generate_unit(self, position):
        unit = HybridNeuralUnit(position)
        self.units.append(unit)
        return unit

    def predict_next(self, input_data, smoothing):
        unit, similarity = self.process_input(input_data)
        predicted = unit.position

        if len(self.units) > 1:
            top2 = sorted(self.units, key=lambda x: x.usage_count, reverse=True)[:2]
            trend = top2[0].position - top2[1].position
            predicted += trend * 0.2

        if self.last_prediction is None:
            smoothed = predicted
        else:
            smoothed = self.last_prediction * smoothing + predicted * (1 - smoothing)

        self.last_prediction = smoothed
        return smoothed, similarity

    def reinforce_failure(self, failed_pattern):
        for unit in self.units:
            sim = unit.quantum_inspired_distance(failed_pattern)
            if sim > 0.6:
                unit.reward -= 0.5

    def estimate_uncertainty(self, similarity):
        return (1.0 - similarity) * self.synaptic_scaling_factor

    def get_stats(self):
        return {
            'unit_count': len(self.units),
            'avg_reward': np.mean([u.reward for u in self.units]) if self.units else 0,
            'avg_usage': np.mean([u.usage_count for u in self.units]) if self.units else 0
        }

# ---------------- Memory Replay & Persistence -------------------
def save_state(network, memory_log):
    with open("network_state.pkl", "wb") as f:
        pickle.dump((network, memory_log), f)

def load_state():
    if os.path.exists("network_state.pkl"):
        with open("network_state.pkl", "rb") as f:
            return pickle.load(f)
    return HybridNeuralNetwork(), []

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
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df.dropna(inplace=True)
    return df

# ---------------- Main -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠📈 Adaptive Dream-Hybrid BTC/USDT Neural Trader")
    global exchange, symbol, timeframe
    exchange = ccxt.kucoin()
    symbol = 'BTC/USDT'
    timeframe = '1m'

    network, memory_log = load_state()
    st.write("📡 Starting adaptive trading loop...")
    st.warning("Run this script outside Streamlit to start the trading loop.")

if __name__ == "__main__":
    main()