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
timeframe = '1m'
data_limit = 200

prediction_log = []
prediction_queue = []
replay_memory = []
smoothing_factor = 0.8

class EpisodicMemory:
    def __init__(self):
        self.episodes = {}
        self.current_episode = None
        
    def create_episode(self, timestamp):
        self.current_episode = timestamp
        self.episodes[timestamp] = {
            'patterns': [],
            'emotional_tags': [],
            'context': None
        }
        
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
        
    def get_related_patterns(self, pattern, threshold=0.5):
        return [(k[0], v) for k, v in self.pattern_relationships.items() 
                if str(pattern) in k and v >= threshold]

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
        
    def update_reward(self, error):
        self.reward += -error * 0.1
        
    def hebbian_learn(self, other_unit, strength, spike_timing=None):
        if spike_timing is not None:
            timing_diff = spike_timing['pre'] - spike_timing['post']
            stdp_factor = np.exp(-abs(timing_diff) / 20.0)
            strength *= stdp_factor
            
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + \
                                     strength * self.learning_rate * self.emotional_weight
        
    def update_spike_time(self):
        self.last_spike_time = datetime.now()

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
            
        # Multi-scale temporal processing
        temporal_scales = [1, 5, 15]
        similarities = []
        
        for scale in temporal_scales:
            scaled_input = self._temporal_scale(input_data, scale)
            unit_similarities = [(unit, unit.quantum_inspired_distance(scaled_input)) 
                               for unit in self.units]
            similarities.extend(unit_similarities)
            
        best_unit, best_similarity = max(similarities, key=lambda x: x[1])
        
        # Emotional tagging and memory storage
        emotional_tag = self._calculate_emotional_tag(best_similarity)
        self.episodic_memory.store_pattern(input_data, emotional_tag)
        self.working_memory.store(input_data, datetime.now())
        
        # Hierarchical pattern recognition
        related_patterns = self.semantic_memory.get_related_patterns(input_data)
        hierarchical_strength = sum(strength for _, strength in related_patterns)
        
        best_unit.emotional_weight = emotional_tag
        best_unit.age = 0
        best_unit.usage_count += 1
        best_unit.update_spike_time()
        
        if best_similarity < self.gen_threshold:
            return self.generate_unit(input_data), 0.0
            
        # Enhanced learning with STDP and attention
        spike_timing = {'pre': datetime.now(), 'post': None}
        for unit, similarity in similarities:
            if unit != best_unit:
                attention_score = unit.get_attention_score(input_data)
                best_unit.hebbian_learn(unit, similarity * attention_score, spike_timing)
            unit.age += 1
            
        return best_unit, best_similarity
        
    def _temporal_scale(self, input_data, scale):
        return input_data * (1.0 / scale)
        
    def _calculate_emotional_tag(self, similarity):
        return 1.0 + (similarity * 0.5)
        
    def predict_next(self, input_data):
        unit, similarity = self.process_input(input_data)
        predicted = unit.position
        
        # Pattern extrapolation
        if len(self.units) > 1:
            predicted = self._extrapolate_pattern(predicted, input_data)
            
        if self.last_prediction is None:
            smoothed = predicted
        else:
            smoothed = self.last_prediction * smoothing_factor + predicted * (1 - smoothing_factor)
        self.last_prediction = smoothed
        return smoothed, similarity
        
    def _extrapolate_pattern(self, predicted, input_data):
        # Simple extrapolation based on recent trends
        if len(self.units) < 2:
            return predicted
            
        recent_units = sorted(self.units, key=lambda x: x.usage_count, reverse=True)[:2]
        trend = recent_units[0].position - recent_units[1].position
        return predicted + trend * 0.1
        
    def estimate_uncertainty(self, similarity):
        uncertainty = 1.0 - similarity
        return uncertainty * self.synaptic_scaling_factor
        
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
        diversity_scores = [np.linalg.norm(u.position - mean_vector) for u in self.units]
        self.units = [u for i, u in enumerate(self.units) if diversity_scores[i] > threshold]
        
    def _apply_homeostatic_regulation(self):
        activity_level = np.mean([u.usage_count for u in self.units])
        regulation_factor = self.homeostatic_target / activity_level
        for unit in self.units:
            unit.learning_rate *= regulation_factor
            
    def _apply_synaptic_scaling(self):
        if len(self.units) > 0:
            avg_connections = np.mean([len(u.connections) for u in self.units])
            for unit in self.units:
                scale_factor = self.synaptic_scaling_factor / avg_connections
                unit.connections = {k: v*scale_factor for k,v in unit.connections.items()}
                
    def monitor_drift(self, current_error):
        self.drift_window.append(current_error)
        if len(self.drift_window) > 20:
            self.drift_window.pop(0)
        if len(self.drift_window) == 20:
            drift_std = np.std(self.drift_window)
            if drift_std > self.drift_threshold:
                self.gen_threshold *= 0.95
                for unit in self.units:
                    unit.learning_rate *= 1.05

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

st.set_page_config(page_title="Enhanced Hybrid DNN-EQIC BTC Predictor", layout="wide")
st.title("Real-Time Enhanced Hybrid DNN-EQIC BTC/USDT Predictor")
placeholder = st.empty()
chart_placeholder = st.empty()
error_chart_placeholder = st.empty()
pca_placeholder = st.empty()

network = HybridNeuralNetwork()

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    mse_list, mae_list = [], []

    for sample in data_scaled:
        prediction, similarity = network.predict_next(sample)
        error = np.mean((sample - prediction) ** 2)
        mae = np.mean(np.abs(sample - prediction))
        mse_list.append(error)
        mae_list.append(mae)
        network.update_feature_importance(sample, prediction)
        network.monitor_drift(error)
        replay_memory.append(sample)
        if len(replay_memory) > 100:
            replay_memory.pop(0)

    for sample in replay_memory[-10:]:  # replay last 10
        prediction, _ = network.predict_next(sample)
        network.update_feature_importance(sample, prediction)

    network.prune_units()

    last_input = data_scaled[-1]
    predicted_next_scaled, similarity = network.predict_next(last_input)
    reconstructed = np.copy(last_input)
    reconstructed[0] = predicted_next_scaled[0]
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    current_timestamp = df.index[-1]
    prediction_queue.append((current_timestamp, predicted_close))

    if len(prediction_queue) > 1:
        prev_time, prev_pred = prediction_queue.pop(0)
        actual_close = df.loc[prev_time, 'close'] if prev_time in df.index else None
        if actual_close is not None:
            prediction_log.append({
                'Time': prev_time.strftime("%H:%M:%S"),
                'Predicted': prev_pred,
                'Actual': actual_close,
                'Error': abs(actual_close - prev_pred)
            })

    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_scaled)
    fig_pca = plt.figure(figsize=(6, 4))
    ax = fig_pca.add_subplot(111, projection='3d')
    ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c='blue')
    ax.set_title("PCA 3D Visualization")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    pca_placeholder.pyplot(fig_pca)
    plt.close(fig_pca)

    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction MSE", f"{np.mean(mse_list):.5f}")
        col2.metric("Prediction MAE", f"{np.mean(mae_list):.5f}")
        col3.metric("Neural Units", len(network.units))

        uncertainty = network.estimate_uncertainty(similarity)
        st.metric("Latest Predicted Close", f"{predicted_close:.2f}")
        st.metric("Estimated Uncertainty", f"{uncertainty:.4f}")

        if prediction_log:
            log_df = pd.DataFrame(prediction_log)
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(log_df['Time'], log_df['Actual'], label='Actual Close', color='blue', marker='o')
            ax1.plot(log_df['Time'], log_df['Predicted'], label='Predicted Close', color='orange', marker='x')
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price (USDT)")
            ax1.set_title("Actual vs Predicted Close Price (Lagged)")
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            ax1.grid(True)
            chart_placeholder.pyplot(fig)
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(10, 2.5))
            ax2.plot(log_df['Time'], log_df['Error'], label='Prediction Error', color='red', marker='.')
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Error")
            ax2.set_title("Prediction Error Over Time")
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True)
            error_chart_placeholder.pyplot(fig2)
            plt.close(fig2)

        st.markdown("### Feature Importance (Relative Influence)")
        if network.feature_importance is not None:
            norm_importance = network.feature_importance / np.sum(network.feature_importance)
            st.bar_chart(pd.Series(norm_importance, index=features))

    time.sleep(60)