# ✅ Proactive Hybrid DNN-EQIC BTC/USDT Predictor with Streamlit and Smart Buy/Sell Zones

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import ccxt
import time
from datetime import datetime

# (Memory, Neural Unit, Neural Network classes are unchanged — truncated here for brevity)
# Make sure to paste your complete existing classes before this line

# ---------------- Indicator & Data -------------------
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

# ---------------- Smart Trade Zones -------------------
def find_smart_trade_zones(network, current_input, scaler):
    if not network.units:
        return None, None

    rsi_index = 1  # Assuming RSI is at index 1
    close_index = 0

    buy_candidates = []
    sell_candidates = []

    for unit in network.units:
        rsi = unit.position[rsi_index]
        if rsi < 0.4 and unit.emotional_weight > 1.0:
            buy_candidates.append(unit)
        elif rsi > 0.6 and unit.emotional_weight > 1.0:
            sell_candidates.append(unit)

    def most_similar_unit(candidates):
        return min(candidates, key=lambda u: np.linalg.norm(current_input - u.position)) if candidates else None

    best_buy = most_similar_unit(buy_candidates)
    best_sell = most_similar_unit(sell_candidates)

    buy_price = sell_price = None

    if best_buy:
        reconstructed = current_input.copy()
        reconstructed[close_index] = best_buy.position[close_index]
        buy_price = scaler.inverse_transform([reconstructed])[0][close_index]

    if best_sell:
        reconstructed = current_input.copy()
        reconstructed[close_index] = best_sell.position[close_index]
        sell_price = scaler.inverse_transform([reconstructed])[0][close_index]

    return buy_price, sell_price

# ---------------- Streamlit UI -------------------
exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'
data_limit = 200

st.set_page_config(page_title="Hybrid DNN-EQIC BTC Predictor", layout="wide")
st.title("🚀 Real-Time Hybrid DNN-EQIC BTC/USDT Predictor with Smart Zones")

placeholder = st.empty()
chart_placeholder = st.empty()
error_chart_placeholder = st.empty()

network = HybridNeuralNetwork()
prediction_log = []

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    smoothing_factor = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7

    input_data = data_scaled[-1]
    actual_time = df.index[-1]
    actual_close = df['close'].iloc[-1]

    predicted_scaled, similarity = network.predict_next(input_data, smoothing_factor)
    reconstructed = np.copy(input_data)
    reconstructed[0] = predicted_scaled[0]
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    buy_zone, sell_zone = find_smart_trade_zones(network, input_data, scaler)

    prediction_log.append({
        'Time': actual_time.strftime("%H:%M:%S"),
        'Predicted': predicted_close,
        'Actual': actual_close,
        'Error': abs(actual_close - predicted_close)
    })

    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Close", f"{predicted_close:.2f}")
        col2.metric("Uncertainty", f"{network.estimate_uncertainty(similarity):.4f}")
        col3.metric("Smart Buy Zone", f"{buy_zone:.2f}" if buy_zone else "N/A")
        col4.metric("Smart Sell Zone", f"{sell_zone:.2f}" if sell_zone else "N/A")

        log_df = pd.DataFrame(prediction_log[-50:])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log_df['Time'], log_df['Actual'], label='Actual', color='blue', marker='o')
        ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted', color='orange', marker='x')
        if buy_zone:
            ax.axhline(y=buy_zone, color='green', linestyle='--', label='Buy Zone')
        if sell_zone:
            ax.axhline(y=sell_zone, color='red', linestyle='--', label='Sell Zone')
        ax.legend()
        ax.set_title("Actual vs Predicted Close with Zones")
        plt.xticks(rotation=45)
        chart_placeholder.pyplot(fig)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        ax2.plot(log_df['Time'], log_df['Error'], color='red', marker='.')
        ax2.set_title("Prediction Error Over Time")
        plt.xticks(rotation=45)
        error_chart_placeholder.pyplot(fig2)
        plt.close(fig2)

    time.sleep(60)