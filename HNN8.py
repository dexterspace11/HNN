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

# ===== KuCoin Configuration =====
exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'
data_limit = 200

# Logging history
prediction_log = []
error_trend = []

# ===== Indicators =====
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

# ===== Hybrid Neural Unit (DNN-EQIC) =====
class HybridNeuralUnit:
    def __init__(self, position, learning_rate=0.1):
        self.position = position
        self.connections = {}
        self.learning_rate = learning_rate

    def hebbian_learn(self, other_unit, strength):
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + strength * self.learning_rate

    def quantum_inspired_distance(self, input_pattern):
        diff = np.abs(input_pattern - self.position)
        dist = np.sqrt(np.sum(diff**2))
        return np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)

class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.gen_threshold = 0.5
        self.error_history = []

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position)
        self.units.append(unit)
        return unit

    def process_input(self, input_data):
        if not self.units:
            return self.generate_unit(input_data), 0.0
        similarities = [(unit, unit.quantum_inspired_distance(input_data)) for unit in self.units]
        best_unit, best_similarity = max(similarities, key=lambda x: x[1])
        if best_similarity < self.gen_threshold:
            return self.generate_unit(input_data), 0.0
        for unit, similarity in similarities:
            if unit != best_unit:
                best_unit.hebbian_learn(unit, similarity)
        return best_unit, best_similarity

    def predict_next(self, input_data):
        unit, _ = self.process_input(input_data)
        return unit.position

    def adapt_threshold(self):
        if len(self.error_history) >= 10:
            mean_err = np.mean(self.error_history)
            std_err = np.std(self.error_history)
            if mean_err > 0.05 + std_err:
                self.gen_threshold *= 1.1
            self.error_history = []

# ===== Streamlit App =====
st.set_page_config(page_title="Hybrid DNN-EQIC BTC Predictor", layout="wide")
st.title("Real-Time Hybrid DNN-EQIC BTC/USDT Predictor")
placeholder = st.empty()

mse_list, mae_list = [], []
network = HybridNeuralNetwork()

# Forward validation placeholders
prev_prediction = None
prev_timestamp = None

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    mse_list.clear()
    mae_list.clear()

    for sample in data_scaled:
        unit, similarity = network.process_input(sample)
        prediction = network.predict_next(sample)
        error = np.mean((sample - prediction)**2)
        mae = np.mean(np.abs(sample - prediction))
        mse_list.append(error)
        mae_list.append(mae)
        network.error_history.append(error)

    network.adapt_threshold()

    last_input = data_scaled[-1]
    predicted_next_scaled = network.predict_next(last_input)
    predicted_close_scaled = predicted_next_scaled[0]

    reconstructed = np.copy(last_input)
    reconstructed[0] = predicted_close_scaled
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]
    actual_close = df['close'].iloc[-1]

    now = datetime.now().strftime("%H:%M:%S")
    if prev_prediction is not None and prev_timestamp != df.index[-1]:
        forward_error = abs(prev_prediction - actual_close)
        prediction_log.append({
            'Time': prev_timestamp.strftime("%H:%M:%S"),
            'Predicted': prev_prediction,
            'Actual': actual_close,
            'Error': forward_error
        })
        error_trend.append(forward_error)

    prev_prediction = predicted_close
    prev_timestamp = df.index[-1]

    # === PCA Visualization ===
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)
    plt.close('all')
    fig_pca = plt.figure(figsize=(6, 4))
    ax = fig_pca.add_subplot(111, projection='3d')
    ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c='blue', label='Live Data')
    ax.set_title("PCA 3D Visualization")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # === Streamlit Output ===
    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction MSE", f"{np.mean(mse_list):.5f}")
        col2.metric("Prediction MAE", f"{np.mean(mae_list):.5f}")
        col3.metric("Neural Units", len(network.units))
        st.metric("Actual Last Close", f"{actual_close:.2f}")
        st.metric("Predicted Next Close", f"{predicted_close:.2f}")

    if prediction_log:
    	log_df = pd.DataFrame(prediction_log)

    	# Create a unified matplotlib chart
    	plt.figure(figsize=(10, 4))
    	plt.plot(log_df['Time'], log_df['Actual'], label='Actual Close', marker='o', color='blue')
    	plt.plot(log_df['Time'], log_df['Predicted'], label='Predicted Close', marker='x', color='orange')
    	plt.xticks(rotation=45)
    	plt.xlabel("Time")
    	plt.ylabel("Price (USDT)")
    	plt.title("Actual vs Predicted Close Price")
    	plt.legend()
    	plt.grid(True)
    	st.pyplot(plt.gcf())  # Show matplotlib figure

    	# Plot prediction error
    	plt.figure(figsize=(10, 2.5))
    	plt.plot(log_df['Time'], log_df['Error'], label='Absolute Prediction Error', color='red', marker='.')
    	plt.xticks(rotation=45)
    	plt.xlabel("Time")
    	plt.ylabel("Error")
    	plt.title("Prediction Error Over Time")
    	plt.grid(True)
    	st.pyplot(plt.gcf())

    time.sleep(60)
