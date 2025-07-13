# ---------------- Self-Learning Dream-Hybrid Neural Trading System -------------------
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
import random

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

    def update_spike_time(self):
        self.last_spike_time = datetime.now()

    def hebbian_learn(self, other_unit, strength):
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

        for u, sim in similarities[:3]:
            if u != best_unit:
                u.hebbian_learn(best_unit, sim)
            u.age += 1

        return best_unit, best_sim

    def _generate_unit(self, position):
        unit = HybridNeuralUnit(position)
        self.units.append(unit)
        return unit

    def predict_action(self, input_data):
        unit, similarity = self.process_input(input_data)
        action = random.choice(['buy', 'sell', 'hold']) if similarity < 0.5 else 'buy' if unit.reward > 0 else 'sell'
        confidence = similarity * unit.emotional_weight
        return action, confidence, unit

    def reinforce(self, unit, reward):
        unit.reward += reward

    def get_stats(self):
        return {
            'unit_count': len(self.units),
            'avg_reward': np.mean([u.reward for u in self.units]) if self.units else 0,
            'avg_usage': np.mean([u.usage_count for u in self.units]) if self.units else 0
        }

# ---------------- Memory Replay & Persistence -------------------
def save_state(network):
    with open("network_state.pkl", "wb") as f:
        pickle.dump(network, f)

def load_state():
    if os.path.exists("network_state.pkl"):
        with open("network_state.pkl", "rb") as f:
            return pickle.load(f)
    return HybridNeuralNetwork()

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
    st.title("🧠 Reinforcement-Driven Dream-Hybrid BTC/USDT Neural Trader")

    global exchange, symbol, timeframe
    exchange = ccxt.kucoin()
    symbol = 'BTC/USDT'
    timeframe = '1m'

    network = load_state()
    capital_usdt = 1000
    inventory = []
    trade_log = []

    while True:
        df = get_kucoin_data()
        features = ['close', 'RSI', 'MA20', 'ATR']
        imputer = SimpleImputer(strategy='mean')
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(imputer.fit_transform(df[features]))

        input_pattern = data_scaled[-5:].flatten()
        actual_price = df['close'].iloc[-1]

        action, confidence, unit = network.predict_action(input_pattern)

        reward = 0
        if action == 'buy' and capital_usdt > 0:
            inventory.append({'price': actual_price, 'time': datetime.now()})
            capital_usdt -= actual_price
        elif action == 'sell' and inventory:
            entry = inventory.pop(0)
            capital_usdt += actual_price
            reward = actual_price - entry['price']
            trade_log.append({'Buy': entry['price'], 'Sell': actual_price, 'Profit': reward})
        elif action == 'hold':
            reward = -0.01

        network.reinforce(unit, reward)
        save_state(network)

        # Display
        st.metric("Capital", f"{capital_usdt:.2f} USDT")
        st.metric("Inventory", len(inventory))
        st.metric("Last Action", action)
        st.metric("Reward", f"{reward:.4f}")

        if trade_log:
            df_trades = pd.DataFrame(trade_log)
            df_trades['Time'] = [t['time'] if 'time' in t else datetime.now() for t in inventory]
            st.subheader("Executed Trades")
            st.dataframe(df_trades.tail(10))

        stats = network.get_stats()
        st.metric("Units", stats['unit_count'])
        st.metric("Avg Reward", f"{stats['avg_reward']:.4f}")
        st.metric("Avg Usage", f"{stats['avg_usage']:.2f}")

        time.sleep(60)

if __name__ == "__main__":
    main()
