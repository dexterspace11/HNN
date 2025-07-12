# ---------------- Final Unified Dream-Hybrid Neural Trading System -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import ccxt
import time
from datetime import datetime

# ---------------- Hybrid Neural Unit -------------------
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

# ---------------- Setup -------------------
st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
st.title("🧠📈 Dream-Hybrid BTC/USDT Neural Trader")

exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'

network = HybridNeuralNetwork()
prediction_log = []
capital_usdt = 1000
trade_amount = 100
executed_trades = []
open_trades = []

placeholder = st.empty()
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(imputer.fit_transform(df[features]))

    input_pattern = data_scaled[-5:].flatten()
    actual_close = df['close'].iloc[-1]
    smoothing = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7

    predicted_scaled, similarity = network.predict_next(input_pattern, smoothing)
    reconstructed = np.copy(data_scaled[-1])
    reconstructed[0] = predicted_scaled[0]
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    prediction_log.append({
        'Time': datetime.now(),
        'Actual': actual_close,
        'Predicted': predicted_close,
        'Error': abs(actual_close - predicted_close)
    })

    # Trade Decision
    signal_score = similarity
    if signal_score > 0.3 and capital_usdt >= trade_amount:
        sell_price = predicted_close * 1.01
        open_trades.append({
            'Time': datetime.now(), 'Buy': predicted_close, 'Sell': sell_price, 'Status': 'Open'
        })
        capital_usdt -= trade_amount

    for trade in open_trades:
        if trade['Status'] == 'Open':
            if actual_close >= trade['Sell']:
                trade['Status'] = 'Closed'
                capital_usdt += trade['Sell']
                trade['CloseTime'] = datetime.now()
                executed_trades.append(trade)
            elif actual_close < trade['Buy'] * 0.985:  # Stop Loss
                trade['Status'] = 'Closed'
                capital_usdt += actual_close
                trade['Sell'] = actual_close
                trade['CloseTime'] = datetime.now()
                executed_trades.append(trade)
                network.reinforce_failure(input_pattern)

    open_trades = [t for t in open_trades if t['Status'] == 'Open']

    # Evaluation
    stats = network.get_stats()
    win_trades = [t for t in executed_trades if t['Sell'] > t['Buy']]
    returns = [(t['Sell'] - t['Buy']) / t['Buy'] for t in executed_trades]
    win_ratio = len(win_trades) / len(executed_trades) if executed_trades else 0
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if returns else 0
    profit = capital_usdt - 1000

    # Display
    with placeholder.container():
        st.metric("Predicted Close", f"{predicted_close:.2f}")
        st.metric("Actual Close", f"{actual_close:.2f}")
        st.metric("Capital", f"{capital_usdt:.2f} USDT")
        st.metric("Win Ratio", f"{win_ratio:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Cumulative Profit", f"{profit:.2f} USDT")

    with chart_placeholder.container():
        st.subheader("📊 Prediction vs Actual Close")
        hist_df = pd.DataFrame(prediction_log[-100:])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist_df['Time'], hist_df['Actual'], label='Actual', marker='o')
        ax.plot(hist_df['Time'], hist_df['Predicted'], label='Predicted', marker='x')
        ax.fill_between(hist_df['Time'], hist_df['Predicted'] - hist_df['Error'],
                        hist_df['Predicted'] + hist_df['Error'], color='orange', alpha=0.2)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("🧾 Executed Trades")
        if executed_trades:
            df_exec = pd.DataFrame(executed_trades)
            df_exec['Profit'] = df_exec['Sell'] - df_exec['Buy']
            st.dataframe(df_exec.tail(10))

        st.subheader("📌 Open Trades")
        if open_trades:
            df_open = pd.DataFrame(open_trades)
            st.dataframe(df_open.tail(10))

    with metrics_placeholder.container():
        st.subheader("📈 Neural Network Stats")
        st.metric("Neural Units", stats['unit_count'])
        st.metric("Avg Reward", f"{stats['avg_reward']:.4f}")
        st.metric("Avg Usage", f"{stats['avg_usage']:.2f}")

    time.sleep(60)