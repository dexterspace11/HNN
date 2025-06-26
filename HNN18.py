
# Optimized Hybrid DNN-EQIC BTC/USDT Predictor with Streamlit
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import ccxt
import time
from datetime import datetime

exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '5m'
data_limit = 200

smoothing_factor = 0.8
prediction_log, prediction_queue, replay_memory = [], [], []

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

    def quantum_distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff ** 2))
        return np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)

    def attention_score(self, input_pattern):
        return self.quantum_distance(input_pattern) * self.emotional_weight

    def spike(self):
        self.last_spike_time = datetime.now()

    def hebbian_learn(self, other, strength):
        self.connections[other] = self.connections.get(other, 0) + strength * self.learning_rate * self.emotional_weight

class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.threshold = 0.5
        self.last_prediction = None
        self.feature_importance = None

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position)
        self.units.append(unit)
        return unit

    def process(self, input_data):
        if not self.units:
            return self.generate_unit(input_data), 0.0

        similarities = [(u, u.quantum_distance(input_data)) for u in self.units]
        best_unit, best_sim = max(similarities, key=lambda x: x[1])

        best_unit.usage_count += 1
        best_unit.spike()

        if best_sim < self.threshold:
            return self.generate_unit(input_data), 0.0

        for unit, sim in similarities:
            if unit != best_unit:
                unit.hebbian_learn(best_unit, sim * unit.attention_score(input_data))
            unit.age += 1

        return best_unit, best_sim

    def predict(self, input_data):
        unit, sim = self.process(input_data)
        prediction = unit.position

        if len(self.units) > 1:
            recent = sorted(self.units, key=lambda u: u.usage_count, reverse=True)[:2]
            trend = recent[0].position - recent[1].position
            prediction += trend * 0.1

        smoothed = prediction if self.last_prediction is None else self.last_prediction * smoothing_factor + prediction * (1 - smoothing_factor)
        self.last_prediction = smoothed
        return smoothed, sim

    def update_importance(self, input_data, prediction):
        error = np.abs(input_data - prediction)
        self.feature_importance = error if self.feature_importance is None else self.feature_importance + error

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
    return tr.rolling(window=period).mean()

def get_data():
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=data_limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    return df.dropna()

# Streamlit App
st.set_page_config(page_title="Hybrid DNN-EQIC BTC Predictor", layout="wide")
st.title("Hybrid DNN-EQIC BTC/USDT Predictor")
placeholder = st.empty()
chart_placeholder = st.empty()
error_chart_placeholder = st.empty()
pca_placeholder = st.empty()

network = HybridNeuralNetwork()

while True:
    df = get_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data = MinMaxScaler().fit_transform(imputer.fit_transform(df[features]))
    mse_list, mae_list = [], []

    for row in data:
        pred, sim = network.predict(row)
        mse_list.append(np.mean((row - pred) ** 2))
        mae_list.append(np.mean(np.abs(row - pred)))
        network.update_importance(row, pred)
        replay_memory.append(row)
        if len(replay_memory) > 100:
            replay_memory.pop(0)

    for row in replay_memory[-10:]:
        pred, _ = network.predict(row)
        network.update_importance(row, pred)

    last_input = data[-1]
    pred_scaled, sim = network.predict(last_input)
    reconstructed = np.copy(last_input)
    reconstructed[0] = pred_scaled[0]
    predicted_close = MinMaxScaler().fit_transform(imputer.fit_transform(df[features]))[-1][0] * df['close'].max()

    timestamp = df.index[-1]
    prediction_queue.append((timestamp, predicted_close))
    if len(prediction_queue) > 1:
        prev_time, prev_pred = prediction_queue.pop(0)
        if prev_time in df.index:
            actual = df.loc[prev_time, 'close']
            prediction_log.append({
                'Time': prev_time.strftime("%H:%M:%S"),
                'Predicted': prev_pred,
                'Actual': actual,
                'Error': abs(actual - prev_pred)
            })

    pca = PCA(n_components=3).fit_transform(data)
    fig_pca = plt.figure(figsize=(6, 4))
    ax = fig_pca.add_subplot(111, projection='3d')
    ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c='blue')
    ax.set_title("PCA 3D Visualization")
    pca_placeholder.pyplot(fig_pca)
    plt.close(fig_pca)

    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{np.mean(mse_list):.5f}")
        col2.metric("MAE", f"{np.mean(mae_list):.5f}")
        col3.metric("Units", len(network.units))
        st.metric("Predicted Close", f"{predicted_close:.2f}")
        st.metric("Estimated Uncertainty", f"{1 - sim:.4f}")

        if prediction_log:
            log_df = pd.DataFrame(prediction_log)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(log_df['Time'], log_df['Actual'], label='Actual', color='blue')
            ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted', color='orange')
            ax.legend()
            chart_placeholder.pyplot(fig)
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(10, 2.5))
            ax2.plot(log_df['Time'], log_df['Error'], color='red')
            error_chart_placeholder.pyplot(fig2)
            plt.close(fig2)

        if network.feature_importance is not None:
            norm_importance = network.feature_importance / np.sum(network.feature_importance)
            st.bar_chart(pd.Series(norm_importance, index=features))

    time.sleep(60)