import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
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
            predicted += trend * 0.2

        if self.last_prediction is None:
            smoothed = predicted
        else:
            smoothed = self.last_prediction * smoothing_factor + predicted * (1 - smoothing_factor)

        self.last_prediction = smoothed
        return smoothed, similarity

    def estimate_uncertainty(self, similarity):
        return (1.0 - similarity) * self.synaptic_scaling_factor

    # Wrapper for pattern recognition metrics
    def quantum_inspired_distance(self, pattern):
        if not self.units:
            return 0.0
        similarities = [unit.quantum_inspired_distance(pattern) for unit in self.units]
        return max(similarities)

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

# ---------------- Pattern Recognition Metrics -------------------
class PatternRecognitionMetrics:
    def __init__(self, network):
        self.network = network
        self.pattern_history = []
        self.signal_history = []
        self.error_history = []

    def verify_pattern_learning(self, test_data):
        """Test pattern recognition accuracy"""
        test_patterns = self._extract_patterns(test_data)
        recognition_rate = self._calculate_recognition_rate(test_patterns)
        self.pattern_history.append({
            'timestamp': datetime.now(),
            'rate': recognition_rate,
            'patterns': len(test_patterns)
        })
        return recognition_rate

    def evaluate_signal_reliability(self, prediction_log):
        """Analyze signal consistency and reliability"""
        signal_patterns = self._extract_signal_patterns(prediction_log)
        reliability_score = self._calculate_signal_consistency(signal_patterns)
        self.signal_history.append({
            'timestamp': datetime.now(),
            'score': reliability_score,
            'signals': len(signal_patterns)
        })
        return reliability_score

    def _extract_patterns(self, data):
        """Extract patterns from input data"""
        features = ['close', 'RSI', 'MA20', 'ATR']
        patterns = []
        for i in range(len(data) - 1):
            pattern = data[features].iloc[i].values
            patterns.append(pattern)
        return patterns

    def _calculate_recognition_rate(self, patterns):
        """Calculate pattern recognition accuracy"""
        recognized = 0
        for pattern in patterns:
            similarity = self.network.quantum_inspired_distance(pattern)
            if similarity > 0.8:  # Threshold for pattern recognition
                recognized += 1
        return recognized / len(patterns) if patterns else 0

    def _extract_signal_patterns(self, prediction_log):
        """Extract signal patterns from prediction log"""
        signals = []
        for entry in prediction_log:
            signal_pattern = {
                'buy': entry['Buy'],
                'sell': entry['Sell'],
                'predicted': entry['Predicted'],
                'actual': entry['Actual']
            }
            signals.append(signal_pattern)
        return signals

    def _calculate_signal_consistency(self, signals):
        """Calculate signal consistency score"""
        consistent_signals = 0
        for i in range(len(signals) - 1):
            current = signals[i]
            next_signal = signals[i + 1]
            if self._is_consistent(current, next_signal):
                consistent_signals += 1
        return consistent_signals / len(signals) if signals else 0

    def _is_consistent(self, signal1, signal2):
        """Check if two consecutive signals are consistent"""
        buy_trend1 = signal1['buy'] - signal1['predicted']
        buy_trend2 = signal2['buy'] - signal2['predicted']
        return (buy_trend1 > 0 and buy_trend2 > 0) or (buy_trend1 < 0 and buy_trend2 < 0)

# ---------------- Streamlit Dashboard -------------------
exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'
data_limit = 200

st.set_page_config(page_title="Hybrid DNN-EQIC Predictor", layout="wide")
st.title("Hybrid DNN-EQIC BTC/USDT Predictor with Buy/Sell + Error + Metrics")

placeholder = st.empty()
chart_placeholder = st.empty()
error_chart_placeholder = st.empty()
metrics_placeholder = st.empty()

network = HybridNeuralNetwork()
prediction_log = []
metrics = PatternRecognitionMetrics(network)

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

    all_positions = np.array([unit.position for unit in network.units])
    buy_price, sell_price = predicted_close, predicted_close
    if len(all_positions) > 0:
        closes = scaler.inverse_transform(all_positions)[:, 0]
        buy_price = closes.min()
        sell_price = closes.max()

    prediction_log.append({
        'Time': actual_time.strftime("%H:%M:%S"),
        'Predicted': predicted_close,
        'Actual': actual_close,
        'Buy': buy_price,
        'Sell': sell_price,
        'Error': abs(actual_close - predicted_close)
    })

    # Calculate and display pattern recognition and signal reliability metrics
    pattern_rate = metrics.verify_pattern_learning(df)
    signal_score = metrics.evaluate_signal_reliability(prediction_log)

    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Close", f"{predicted_close:.2f}")
        col2.metric("Buy at Dip", f"{buy_price:.2f}")
        col3.metric("Sell at Peak", f"{sell_price:.2f}")

        log_df = pd.DataFrame(prediction_log[-50:])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log_df['Time'], log_df['Actual'], label='Actual', color='blue', marker='o')
        ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted', color='orange', marker='x')
        ax.legend()
        ax.set_title("Actual vs Predicted Close")
        plt.xticks(rotation=45)
        chart_placeholder.pyplot(fig)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        ax2.plot(log_df['Time'], log_df['Error'], color='red', marker='.')
        ax2.set_title("Prediction Error Over Time")
        plt.xticks(rotation=45)
        error_chart_placeholder.pyplot(fig2)
        plt.close(fig2)

    with metrics_placeholder.container():
        st.markdown("### System Performance Metrics")
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Pattern Recognition Rate", f"{pattern_rate:.2%}")
        mcol2.metric("Signal Consistency Score", f"{signal_score:.2%}")

    time.sleep(60)
