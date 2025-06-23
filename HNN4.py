import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import ccxt
import time

# ===== KuCoin Configuration =====
exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '5m'
data_limit = 200

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
st.title("Real-Time Hybrid DNN-EQIC BTC/USDT Predictor with Dynamic PCA")
placeholder = st.empty()
mse_list, mae_list = [], []
network = HybridNeuralNetwork()

for _ in range(1000):  # Simulation loop; adjust range as needed
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

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
    predictions = [network.predict_next(last_input) for _ in range(7)]
    pred_close_scaled = [p[0] for p in predictions]
    full_features = np.tile(last_input, (7, 1))
    full_features[:, 0] = pred_close_scaled
    pred_close = scaler.inverse_transform(full_features)[:, 0]

    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c='blue', label='Data')
    ax.set_title("PCA 3D Visualization - Live")

    with placeholder.container():
        st.write(f"Prediction MSE: {np.mean(mse_list):.5f}")
        st.write(f"Prediction MAE: {np.mean(mae_list):.5f}")
        st.write(f"Total Neural Units (Centroids): {len(network.units)}")
        st.write(f"Next 7 predicted Close prices: {pred_close}")
        st.pyplot(fig)

    time.sleep(60)