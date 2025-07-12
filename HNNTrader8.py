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

class WorkingMemory:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.short_term_patterns = []
        self.temporal_context = []
    def store(self, pattern, temporal_marker):
        if len(self.short_term_patterns) >= self.capacity:
            self.short_term_patterns.pop(0)
            self.temporal_context.pop(0)
        self.short_term_patterns.append(pattern)
        self.temporal_context.append(temporal_marker)

class SemanticMemory:
    def __init__(self):
        self.pattern_relationships = {}
    def store_relationship(self, pattern1, pattern2, strength):
        key = tuple(sorted([str(p) for p in [pattern1, pattern2]]))
        self.pattern_relationships[key] = strength

# ----------- Neural Units -----------

class NeuralUnit:
    def __init__(self, position, activity_dim):
        self.position = position
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
            self.connections[other_unit] = 0
        self.connections[other_unit] += delta_w
        self.connections[other_unit] = min(max(self.connections[other_unit], 0), 1)

class NeuralEcosystem:
    def __init__(self, activity_dim, growth_threshold=0.6, prune_threshold=0.2):
        self.units = []
        self.activity_dim = activity_dim
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold

    def process_input(self, input_pattern):
        if not self.units:
            u = self.create_unit(input_pattern)
            return u
        # Compute activity for all units
        activities = [(u, u.update_activity(input_pattern)) for u in self.units]
        # Find best match
        best_unit, best_act = max(activities, key=lambda x: x[1])
        # Hebbian position update
        learning_rate = 0.05
        best_unit.position += learning_rate * (input_pattern - best_unit.position) * best_act
        best_unit.usage += 1
        best_unit.age += 1
        best_unit.last_update = datetime.now()

        # Connect units via Hebbian
        for u, act in activities:
            if u != best_unit:
                delta_w = 0.1 * best_act * act
                best_unit.hebian_update(u, delta_w)

        # Growth rule
        if best_act > self.growth_threshold:
            self.create_unit(input_pattern)

        # Prune inactive or old units
        self.prune_units()
        return best_unit

    def create_unit(self, position):
        u = NeuralUnit(position, self.activity_dim)
        self.units.append(u)
        return u

    def prune_units(self):
        # Remove units with low usage or old age
        self.units = [u for u in self.units if u.usage > 2 or u.age < 100]

    def cluster_units(self):
        # Cluster based on positions to identify groups for hierarchy
        if len(self.units) < 2:
            return []
        positions = np.array([u.position for u in self.units])
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(positions)
        labels = clustering.labels_
        clusters = {}
        for u, lbl in zip(self.units, labels):
            clusters.setdefault(lbl, []).append(u)
        return clusters

    def form_hierarchy(self):
        # For each cluster, create a higher-level unit
        clusters = self.cluster_units()
        hierarchy_units = []
        for lbl, units in clusters.items():
            if lbl == -1:
                continue  # noise
            # Average position
            centroid = np.mean([u.position for u in units], axis=0)
            # Create higher-level unit
            hl_unit = NeuralUnit(centroid, self.activity_dim)
            # Connect to cluster units
            for u in units:
                delta_w = 0.1
                hl_unit.hebian_update(u, delta_w)
            hierarchy_units.append(hl_unit)
        return hierarchy_units

# ----------- Predictor with hierarchical self-organization -----------

class SelfOrgPredictor:
    def __init__(self, input_dim):
        self.ecosystem = NeuralEcosystem(input_dim)
        self.hierarchy = []  # higher level units
        self.history = []

    def predict(self, input_pattern):
        # Activate base units
        unit = self.ecosystem.process_input(input_pattern)
        pred = unit.position
        # Form hierarchy periodically
        if len(self.ecosystem.units) > 5:
            self.hierarchy = self.ecosystem.form_hierarchy()
        # Optionally, integrate higher level predictions
        if self.hierarchy:
            # average higher level units' positions
            hl_pred = np.mean([u.position for u in self.hierarchy], axis=0)
            pred = 0.5 * pred + 0.5 * hl_pred
        return pred

    def learn(self, input_pattern, target):
        # Activate base units and update
        unit = self.ecosystem.process_input(input_pattern)
        pred = unit.position
        error = np.linalg.norm(pred - target)
        # Adjust units towards target
        for u in self.ecosystem.units:
            u.position += 0.01 * (target - u.position) * u.activity
        # Possibly add units if error high
        if error > 0.1:
            self.ecosystem.create_unit(target)
        # Update hierarchy
        if len(self.ecosystem.units) > 5:
            self.hierarchy = self.ecosystem.form_hierarchy()
            for hl_unit in self.hierarchy:
                hl_unit.position += 0.01 * (target - hl_unit.position)

# ----------- Streamlit App -----------

exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'
data_limit = 200

st.set_page_config(page_title="Hierarchical Self-Organizing Neural Predictor", layout="wide")
st.title("Brain-like Growing Hierarchical Price Predictor")

placeholder = st.empty()

# Initialize predictor
input_dim = 4
predictor = SelfOrgPredictor(input_dim)

# Initialize memory
episodic_memory = EpisodicMemory()
working_memory = None
semantic_memory = None

# Main loop
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

    # Predict next 3 steps
    preds = []
    current_input = input_pattern.copy()
    for _ in range(3):
        pred = predictor.predict(current_input)
        preds.append(pred)
        # Simulate next input
        current_input[:-1] = current_input[1:]
        current_input[-1] = pred[0]

    # Inverse transform for visualization
    pred_array = np.copy(data_scaled[-1])
    pred_array[0] = preds[0][0]
    pred_close = scaler.inverse_transform([pred_array])[0][0]

    # Store in episodic memory
    episodic_memory.store_pattern(preds, emotional_tag=1.0)

    # Compute error
    error = abs(actual_close - pred_close)

    # Learn from prediction error
    predictor.learn(input_pattern, np.array([actual_close]))

    # Hierarchical self-organization
    if len(predictor.ecosystem.units) > 5:
        predictor.ecosystem.form_hierarchy()

    # Visualization
    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.metric("Predicted Next Close", f"{preds[0][0]:.2f}")
        st.metric("Actual Close", f"{actual_close:.2f}")
        st.metric("Error", f"{error:.2f}")

        fig, ax = plt.subplots()
        ax.plot([0,1,2], [preds[0][0], preds[1][0], preds[2][0]], label='Predicted')
        ax.scatter(0, actual_close, color='red', label='Actual')
        ax.legend()
        st.pyplot(fig)

    time.sleep(60)