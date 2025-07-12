# ---------------- Updated Dream-Based Streamlit Neural Trading System -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import time
from datetime import datetime

# ---------------- Neural Unit with Dream Capability -------------------
class DreamUnit:
    def __init__(self, pattern):
        self.pattern = pattern
        self.usage = 0
        self.reward = 0
        self.last_activation = datetime.now()

    def similarity(self, new_pattern):
        dist = np.linalg.norm(np.array(self.pattern) - np.array(new_pattern))
        return np.exp(-dist)

    def reinforce(self, reward):
        self.reward += reward
        self.usage += 1
        self.last_activation = datetime.now()

# ---------------- DreamNet -------------------
class DreamNet:
    def __init__(self):
        self.units = []

    def process(self, input_pattern):
        if not self.units:
            new_unit = DreamUnit(input_pattern)
            self.units.append(new_unit)
            return 0

        similarities = [unit.similarity(input_pattern) for unit in self.units]
        best_index = np.argmax(similarities)
        best_unit = self.units[best_index]
        best_unit.reinforce(1 - similarities[best_index])

        if similarities[best_index] < 0.6:
            self.units.append(DreamUnit(input_pattern))

        return best_unit.reward / (best_unit.usage + 1e-5)

    def dream_learn(self):
        for unit in self.units:
            dream_input = unit.pattern + np.random.normal(0, 0.05, size=len(unit.pattern))
            self.process(dream_input)

    def reinforce_failure(self, failed_pattern):
        for unit in self.units:
            sim = unit.similarity(failed_pattern)
            if sim > 0.6:
                unit.reinforce(-0.5)

    def get_stats(self):
        return {
            'unit_count': len(self.units),
            'avg_reward': np.mean([u.reward for u in self.units]) if self.units else 0,
            'avg_usage': np.mean([u.usage for u in self.units]) if self.units else 0,
        }

# ---------------- Get KuCoin Data -------------------
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

# ---------------- Streamlit Setup -------------------
st.set_page_config(page_title="Dream-Based Neural Trader", layout="wide")
st.title("🧠 Dream-Driven BTC/USDT Predictor")

exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'

net = DreamNet()
capital_usdt = 1000
trade_amount = 100
executed_trades = []
prediction_history = []
open_trades = []

placeholder = st.empty()
chart_placeholder = st.empty()
stats_placeholder = st.empty()

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    data = df[features].dropna().values
    current_input = data[-1]

    signal_score = net.process(current_input)
    predicted_close = current_input[0] + np.random.normal(0, 0.5)
    actual_close = df['close'].iloc[-1]

    prediction_history.append({
        'Time': datetime.now(),
        'Actual': actual_close,
        'Predicted': predicted_close,
        'Error': abs(actual_close - predicted_close),
    })

    buy_signal = signal_score > 0.3 and capital_usdt >= trade_amount
    if buy_signal:
        capital_usdt -= trade_amount
        sell_price = predicted_close * 1.01
        open_trades.append({
            'Time': datetime.now(),
            'Buy': predicted_close,
            'Sell': sell_price,
            'Status': 'Open'
        })

    # Simulate trade closes
    for trade in open_trades:
        if trade['Status'] == 'Open':
            if actual_close >= trade['Sell']:
                trade['Status'] = 'Closed'
                trade['CloseTime'] = datetime.now()
                capital_usdt += trade['Sell']
                executed_trades.append(trade)
            elif actual_close < trade['Buy'] * 0.985:  # 1.5% stop loss
                trade['Status'] = 'Closed'
                trade['CloseTime'] = datetime.now()
                trade['Sell'] = actual_close
                net.reinforce_failure(current_input)
                capital_usdt += actual_close
                executed_trades.append(trade)

    open_trades = [t for t in open_trades if t['Status'] == 'Open']

    net.dream_learn()

    stats = net.get_stats()
    win_trades = [t for t in executed_trades if t['Sell'] > t['Buy']]
    total_trades = len(executed_trades)
    win_ratio = len(win_trades) / total_trades if total_trades > 0 else 0
    returns = [(t['Sell'] - t['Buy']) / t['Buy'] for t in executed_trades]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) if returns else 0
    cumulative_profit = capital_usdt - 1000

    with placeholder.container():
        st.metric("🧠 Signal Score", f"{signal_score:.4f}")
        st.metric("📈 Predicted Close", f"{predicted_close:.2f}")
        st.metric("💰 Capital", f"{capital_usdt:.2f} USDT")
        st.metric("📊 Win Ratio", f"{win_ratio:.2%}")
        st.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("📈 Cumulative Profit", f"{cumulative_profit:.2f} USDT")

    with chart_placeholder.container():
        st.subheader("Prediction vs Actual Close")
        hist_df = pd.DataFrame(prediction_history[-100:])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist_df['Time'], hist_df['Actual'], label='Actual', marker='o')
        ax.plot(hist_df['Time'], hist_df['Predicted'], label='Predicted', marker='x')
        ax.fill_between(
            hist_df['Time'],
            hist_df['Predicted'] - hist_df['Error'],
            hist_df['Predicted'] + hist_df['Error'],
            color='orange',
            alpha=0.2
        )
        ax.set_title("Prediction Accuracy")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Executed Trades")
        if executed_trades:
            log_df = pd.DataFrame(executed_trades)
            log_df['Profit'] = log_df['Sell'] - log_df['Buy']
            st.dataframe(log_df[['Time', 'Buy', 'Sell', 'Profit', 'Status', 'CloseTime']].tail(10))
        else:
            st.write("No trades executed yet.")

        st.subheader("Open Trades")
        if open_trades:
            open_df = pd.DataFrame(open_trades)
            st.dataframe(open_df[['Time', 'Buy', 'Sell', 'Status']].tail(10))
        else:
            st.write("No open trades.")

    with stats_placeholder.container():
        st.subheader("Neural Network Growth")
        st.metric("Neural Units", stats['unit_count'])
        st.metric("Avg Reward", f"{stats['avg_reward']:.4f}")
        st.metric("Avg Usage", f"{stats['avg_usage']:.2f}")

    time.sleep(60)

