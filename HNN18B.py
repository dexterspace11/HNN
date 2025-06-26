# ✅ Full Hybrid DNN-EQIC BTC/USDT Predictor with Streamlit Visualization and Forward Validation Charts

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

# ---------------- Memory Structures -------------------
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

# ---------------- Neural Structures -------------------
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
        return np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)

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
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + \
                                       strength * self.learning_rate * self.emotional_weight

class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.gen_threshold = 0.5
        self.feature_importance = None
        self.drift_window = []
        self.drift_threshold = 0.05
        self.last_prediction = None
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory()
        self.semantic_memory = SemanticMemory()
        self.homeostatic_target = 0.1
        self.synaptic_scaling_factor = 1.0

    def generate_unit(self, position):
        unit = HybridNeuralUnit(position)
        self.units.append(unit)
        return unit

    def process_input(self, input_data):
        if not self.units:
            return self.generate_unit(input_data), 0.0

        similarities = [(unit, unit.quantum_inspired_distance(input_data)) for unit in self.units]
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_unit, best_similarity = similarities[0]

        emotional_tag = 1.0 + (best_similarity * 0.5)
        self.episodic_memory.store_pattern(input_data, emotional_tag)
        self.working_memory.store(input_data, datetime.now())
        best_unit.emotional_weight = emotional_tag
        best_unit.age = 0
        best_unit.usage_count += 1
        best_unit.update_spike_time()

        if best_similarity < self.gen_threshold:
            return self.generate_unit(input_data), 0.0

        spike_timing = {'pre': best_unit.last_spike_time, 'post': datetime.now()}
        # Limit Hebbian learning to top 3 units only
        for unit, similarity in similarities[:3]:
            if unit != best_unit:
                attention_score = unit.get_attention_score(input_data)
                unit.hebbian_learn(best_unit, similarity * attention_score, spike_timing)
            unit.age += 1

        return best_unit, best_similarity

    def predict_next(self, input_data, smoothing_factor):
        unit, similarity = self.process_input(input_data)
        predicted = unit.position

        if len(self.units) > 1:
            recent_units = sorted(self.units, key=lambda x: x.usage_count, reverse=True)[:2]
            trend = recent_units[0].position - recent_units[1].position
            predicted += trend * 0.1

        if self.last_prediction is None:
            smoothed = predicted
        else:
            smoothed = self.last_prediction * smoothing_factor + predicted * (1 - smoothing_factor)

        self.last_prediction = smoothed
        return smoothed, similarity

    def estimate_uncertainty(self, similarity):
        return (1.0 - similarity) * self.synaptic_scaling_factor

    def update_feature_importance(self, input_data, prediction):
        error_vector = np.abs(input_data - prediction)
        if self.feature_importance is None:
            self.feature_importance = error_vector
        else:
            self.feature_importance += error_vector

    def prune_units(self, threshold=0.01):
        self._apply_homeostatic_regulation()
        self._apply_synaptic_scaling()
        if len(self.units) < 2:
            return
        mean_vector = np.mean([u.position for u in self.units], axis=0)
        self.units = [u for u in self.units if np.linalg.norm(u.position - mean_vector) > threshold]

    def _apply_homeostatic_regulation(self):
        activity_level = np.mean([u.usage_count for u in self.units]) or 1.0
        for unit in self.units:
            unit.learning_rate *= self.homeostatic_target / activity_level

    def _apply_synaptic_scaling(self):
        avg_connections = np.mean([len(u.connections) for u in self.units]) or 1.0
        for unit in self.units:
            unit.connections = {k: v * self.synaptic_scaling_factor / avg_connections for k, v in unit.connections.items()}

    def monitor_drift(self, current_error):
        self.drift_window.append(current_error)
        if len(self.drift_window) > 20:
            self.drift_window.pop(0)
        if len(self.drift_window) == 20 and np.std(self.drift_window) > self.drift_threshold:
            self.gen_threshold *= 0.95
            for unit in self.units:
                unit.learning_rate *= 1.05


# ---------------- Indicators + Data Fetch -------------------
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


# ---------------- Real-time Streamlit App -------------------

exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '5m'
data_limit = 200

st.set_page_config(page_title="Hybrid DNN-EQIC BTC Predictor", layout="wide")
st.title("Real-Time Hybrid DNN-EQIC BTC/USDT Predictor")

placeholder = st.empty()
chart_placeholder = st.empty()
error_chart_placeholder = st.empty()
pca_placeholder = st.empty()

network = HybridNeuralNetwork()
prediction_log = []
prediction_queue = []
replay_memory = []
replay_errors = []

loop_count = 0

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    mse_list, mae_list = [], []

    # Adaptive smoothing factor based on volatility
    current_volatility = df['ATR'].iloc[-1]
    smoothing_factor = 0.3 if current_volatility > df['close'].std() * 0.01 else 0.7

    # Process all samples to train network and accumulate errors
    for sample in data_scaled:
        prediction, similarity = network.predict_next(sample, smoothing_factor)
        error = np.mean((sample - prediction) ** 2)
        mae = np.mean(np.abs(sample - prediction))
        mse_list.append(error)
        mae_list.append(mae)

        network.update_feature_importance(sample, prediction)
        network.monitor_drift(error)

        replay_memory.append(sample)
        replay_errors.append(error)
        if len(replay_memory) > 100:
            replay_memory.pop(0)
            replay_errors.pop(0)

    # Prioritized replay: top 10 samples by error
    if len(replay_memory) > 10:
        top_indices = np.argsort(replay_errors)[-10:]
        for idx in top_indices:
            prediction, _ = network.predict_next(replay_memory[idx], smoothing_factor)
            network.update_feature_importance(replay_memory[idx], prediction)

    network.prune_units()

    # Predict next step based on last sample
    last_input = data_scaled[-1]
    predicted_next_scaled, similarity = network.predict_next(last_input, smoothing_factor)
    reconstructed = np.copy(last_input)
    reconstructed[0] = predicted_next_scaled[0]  # Replace only close price feature with predicted
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    current_timestamp = df.index[-1]
    prediction_queue.append((current_timestamp, predicted_close))

    # Forward validation: Compare previous prediction with actual close at that time (one-step ahead)
    if len(prediction_queue) > 1:
        prev_time, prev_pred = prediction_queue.pop(0)
        # Find actual close corresponding to prev_time timestamp in current df
        if prev_time in df.index:
            actual_close = df.loc[prev_time, 'close']
            prediction_log.append({
                'Time': prev_time.strftime("%H:%M:%S"),
                'Predicted': prev_pred,
                'Actual': actual_close,
                'Error': abs(actual_close - prev_pred)
            })

    loop_count += 1
    if loop_count % 5 == 0:
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(data_scaled)
        fig_pca = plt.figure(figsize=(6, 4))
        ax = fig_pca.add_subplot(111, projection='3d')
        ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c='blue')
        ax.set_title("PCA 3D Visualization")
        pca_placeholder.pyplot(fig_pca)
        plt.close(fig_pca)

    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction MSE", f"{np.mean(mse_list):.5f}")
        col2.metric("Prediction MAE", f"{np.mean(mae_list):.5f}")
        col3.metric("Neural Units", len(network.units))
        st.metric("Predicted Close", f"{predicted_close:.2f}")
        st.metric("Estimated Uncertainty", f"{network.estimate_uncertainty(similarity):.4f}")

        if prediction_log:
            log_df = pd.DataFrame(prediction_log)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(log_df['Time'], log_df['Actual'], label='Actual Close', color='blue', marker='o')
            ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted Close', color='orange', marker='x')
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.set_title('Forward Validation: Actual vs Predicted Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_placeholder.pyplot(fig)
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(10, 2.5))
            ax2.plot(log_df['Time'], log_df['Error'], color='red', marker='.')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Absolute Error')
            ax2.set_title('Forward Validation: Prediction Error Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            error_chart_placeholder.pyplot(fig2)
            plt.close(fig2)

        st.markdown("### Feature Importance")
        if network.feature_importance is not None:
            norm_importance = network.feature_importance / np.sum(network.feature_importance)
            st.bar_chart(pd.Series(norm_importance, index=features))

    time.sleep(60)
