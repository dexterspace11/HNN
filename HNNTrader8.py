import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import ccxt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN

# ----------- Utility functions -----------

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

# ----------- Memory Structures -----------

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

# ----------- Neural Units -----------

class NeuralUnit:
    def __init__(self, position, activity_dim):
        self.position = np.asarray(position).flatten()
        self.activity_dim = activity_dim
        self.age = 0
        self.usage = 0
        self.connections = {}  # unit -> weight
        self.activity = 0
        self.last_update = datetime.now()

    def update_activity(self, input_pattern):
        dist = np.linalg.norm(input_pattern - self.position)
        self.activity = np.exp(-dist**2 / (2 * 0.1**2))
        return self.activity

    def hebbian_update(self, other_unit, delta_w):
        if other_unit not in self.connections:
            self.connections[other_unit] = 0.0
        self.connections[other_unit] += delta_w
        self.connections[other_unit] = min(max(self.connections[other_unit], 0), 1)

# ----------- Ecosystem -----------

class NeuralEcosystem:
    def __init__(self, activity_dim, growth_threshold=0.6, prune_threshold=0.2):
        self.units = []
        self.activity_dim = activity_dim
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold

    def create_unit(self, position):
        position = np.asarray(position).flatten()
        u = NeuralUnit(position, self.activity_dim)
        self.units.append(u)
        return u

    def process_input(self, input_pattern):
        input_pattern = np.asarray(input_pattern).flatten()
        if not self.units:
            return self.create_unit(input_pattern)

        activities = [(u, u.update_activity(input_pattern)) for u in self.units]
        best_unit, best_act = max(activities, key=lambda x: x[1])
        learning_rate = 0.05
        best_unit.position += learning_rate * (input_pattern - best_unit.position) * best_act
        best_unit.usage += 1
        best_unit.age += 1
        best_unit.last_update = datetime.now()

        for u, act in activities:
            if u != best_unit:
                delta_w = 0.1 * best_act * act
                best_unit.hebbian_update(u, delta_w)

        if best_act > self.growth_threshold:
            self.create_unit(input_pattern)

        self.prune_units()
        return best_unit

    def prune_units(self):
        self.units = [u for u in self.units if u.usage > 2 or u.age < 100]

    def cluster_units(self, eps=0.4, min_samples=3):
        if len(self.units) < 2:
            return {}

        valid_units = []
        positions = []
        for u in self.units:
            pos = np.asarray(u.position).flatten()
            if pos.ndim == 1 and pos.shape[0] == self.activity_dim:
                valid_units.append(u)
                positions.append(pos)

        if len(positions) < 2:
            return {}

        positions = np.vstack(positions)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        clusters = {}
        for u, lbl in zip(valid_units, labels):
            clusters.setdefault(lbl, []).append(u)
        return clusters

    def form_hierarchy(self, eps=0.4, min_samples=3):
        clusters = self.cluster_units(eps, min_samples)
        hierarchy_units = []
        for lbl, units in clusters.items():
            if lbl == -1:
                continue
            centroid = np.mean([u.position for u in units], axis=0)
            hl_unit = NeuralUnit(centroid, self.activity_dim)
            for u in units:
                hl_unit.hebbian_update(u, 0.1)
            hierarchy_units.append(hl_unit)
        return hierarchy_units

# ----------- Predictor -----------

class SelfOrgPredictor:
    def __init__(self, input_dim):
        self.ecosystem = NeuralEcosystem(input_dim)
        self.history = []

    def predict(self, input_pattern):
        input_pattern = np.asarray(input_pattern).flatten()
        unit = self.ecosystem.process_input(input_pattern)
        pred = unit.position
        if len(self.ecosystem.units) > 5:
            level1 = self.ecosystem.form_hierarchy()
            if level1:
                centroid = np.mean([u.position for u in level1], axis=0)
                pred = 0.5 * pred + 0.5 * centroid
        return pred

    def learn(self, input_pattern, target, error_threshold=0.05):
        input_pattern = np.asarray(input_pattern).flatten()
        target = np.asarray(target).flatten()
        unit = self.ecosystem.process_input(input_pattern)
        pred = unit.position
        error = np.linalg.norm(pred - target)
        for u in self.ecosystem.units:
            u.position += 0.01 * (target - u.position) * u.activity
        if error > 0.1:
            self.ecosystem.create_unit(target)
        if len(self.ecosystem.units) > 5:
            level1 = self.ecosystem.form_hierarchy()
            for u in level1:
                u.position += 0.01 * (target - u.position)

# ----------- Streamlit App -----------

exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'
data_limit = 200

st.set_page_config(page_title="Hierarchical Self-Organizing Neural Predictor", layout="wide")
st.title("🧠 Brain-like Hierarchical Neural Predictor")

placeholder = st.empty()

predictor = SelfOrgPredictor(input_dim=4)
episodic_memory = EpisodicMemory()

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    input_pattern = data_scaled[-1]
    actual_time = df.index[-1]
    actual_close = df['close'].iloc[-1]

    preds = []
    current_input = input_pattern.copy()
    for _ in range(3):
        pred = predictor.predict(current_input)
        preds.append(pred)
        current_input[:-1] = current_input[1:]
        current_input[-1] = pred[0]

    pred_array = np.copy(data_scaled[-1])
    pred_array[0] = preds[0][0]
    pred_close = scaler.inverse_transform([pred_array])[0][0]

    episodic_memory.store_pattern(preds, emotional_tag=1.0)
    error = abs(actual_close - pred_close)

    predictor.learn(input_pattern, np.array([actual_close]))

    if len(predictor.ecosystem.units) > 5:
        predictor.ecosystem.form_hierarchy()

    hierarchy_units = predictor.ecosystem.form_hierarchy()
    if hierarchy_units:
        centroid = np.mean([u.position for u in hierarchy_units], axis=0)
        for u in predictor.ecosystem.units:
            u.position += 0.005 * (centroid - u.position)

    with placeholder.container():
        st.markdown(f"### ⏱ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.metric("📈 Predicted Next Close", f"{preds[0][0]:.2f}")
        st.metric("📉 Actual Close", f"{actual_close:.2f}")
        st.metric("❌ Prediction Error", f"{error:.2f}")
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [preds[0][0], preds[1][0], preds[2][0]], label='Predicted')
        ax.scatter(0, actual_close, color='red', label='Actual')
        ax.legend()
        st.pyplot(fig)

    time.sleep(60)
