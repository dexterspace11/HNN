# ---------------- Unified True Self-Learning Dream-Hybrid Neural Trading System -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt, time, pickle, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
from sklearn.decomposition import PCA
from collections import Counter

# ---------------- Reinforcement Unit -------------------
class SelfLearningNeuron:
    def __init__(self, state_size):
        self.state = np.random.rand(state_size)
        self.value = 0.0
        self.usage = 0
        self.reward = 0.0
        self.connections = {}

    def distance(self, input_state):
        return np.linalg.norm(self.state - input_state)

    def activate(self, input_state):
        self.usage += 1
        return np.exp(-self.distance(input_state))

    def reinforce(self, reward):
        self.reward += reward
        self.value += reward * 0.1

# ---------------- Agent -------------------
class SelfLearningTrader:
    def __init__(self):
        self.neurons = []
        self.memory = []
        self.epsilon = 1.0  # Exploration rate

    def act(self, state):
        if not self.neurons or np.random.rand() < self.epsilon:
            neuron = SelfLearningNeuron(len(state))
            neuron.state = state
            self.neurons.append(neuron)
            return neuron, np.random.choice(['buy', 'sell', 'hold'])

        scores = [(n, n.activate(state)) for n in self.neurons]
        best_neuron = max(scores, key=lambda x: x[1])[0]
        return best_neuron, max(self.memory[-1]["action_values"], key=self.memory[-1]["action_values"].get) if self.memory else 'hold'

    def learn(self, state, action, reward):
        neuron = SelfLearningNeuron(len(state))
        neuron.state = state
        neuron.reinforce(reward)
        self.neurons.append(neuron)
        self.memory.append({"state": state, "action": action, "reward": reward, "action_values": {'buy': 0, 'sell': 0, 'hold': 0}})

        if len(self.memory) >= 2:
            self.memory[-2]["action_values"][action] += reward * 0.1

        self.epsilon = max(0.05, self.epsilon * 0.995)

    def save(self):
        with open("agent_state.pkl", "wb") as f:
            pickle.dump((self.neurons, self.memory), f)

    def load(self):
        if os.path.exists("agent_state.pkl"):
            with open("agent_state.pkl", "rb") as f:
                self.neurons, self.memory = pickle.load(f)

# ---------------- Hybrid Neural Network -------------------
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
        decay = np.exp(-self.age / 100.0)
        return (np.exp(-2.0 * dist) + 0.5 / (1 + 0.9 * dist)) * decay

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
        self.connections[other_unit] = self.connections.get(other_unit, 0.0) + strength * self.learning_rate * self.emotional_weight

class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.gen_threshold = 0.5
        self.feature_importance = None
        self.drift_threshold = 0.05
        self.last_prediction = None
        self.episodic_memory = EpisodicMemory()
        self.working_memory = WorkingMemory()
        self.semantic_memory = SemanticMemory()
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

    def quantum_inspired_distance(self, pattern):
        if not self.units:
            return 0.0
        similarities = [unit.quantum_inspired_distance(pattern) for unit in self.units]
        return max(similarities)

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

# ---------------- Indicators & Data -------------------
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

def get_kucoin_data(symbol='BTC/USDT', timeframe='1m', limit=200):
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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

    def verify_pattern_learning(self, test_data):
        patterns = [test_data[['close', 'RSI', 'MA20', 'ATR']].iloc[i].values for i in range(len(test_data) - 1)]
        padded_patterns = [
            np.pad(p, (0, 20 - len(p)), mode='constant') if len(p) < 20 else p
            for p in patterns
        ]
        recognized = sum(self.network.quantum_inspired_distance(p) > 0.6 for p in padded_patterns)
        return recognized / len(padded_patterns) if padded_patterns else 0

    def evaluate_signal_reliability(self, prediction_log):
        signals = [{'buy': p['Buy'], 'sell': p['Sell'], 'predicted': p['Predicted'], 'actual': p['Actual']} for p in prediction_log]
        consistent = sum((s1['buy'] - s1['predicted']) * (s2['buy'] - s2['predicted']) > 0 for s1, s2 in zip(signals[:-1], signals[1:]))
        return consistent / len(signals) if signals else 0

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠🤖 True Self-Learning Dream-Hybrid BTC/USDT Neural Trader")

    global exchange
    exchange = ccxt.kucoin()

    symbol = 'BTC/USDT'
    timeframe = '1m'
    data_limit = 200

    agent = SelfLearningTrader()
    agent.load()

    network = HybridNeuralNetwork()
    metrics = PatternRecognitionMetrics(network)

    capital = 1000.0
    position = None
    prediction_log = []

    placeholder = st.empty()
    chart_placeholder = st.empty()
    error_chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    neural_growth_placeholder = st.empty()
    rules_placeholder = st.empty()
    trade_log_placeholder = st.empty()
    dream_placeholder = st.empty()

    while True:
        df = get_kucoin_data(symbol, timeframe, data_limit)
        features = ['close', 'RSI', 'MA20', 'ATR']

        imputer = SimpleImputer(strategy='mean')
        scaler = MinMaxScaler()
        data_imputed = imputer.fit_transform(df[features])
        data_scaled = scaler.fit_transform(data_imputed)

        smoothing_factor = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7
        input_window = data_scaled[-5:].flatten()

        # Agent action
        neuron, action = agent.act(input_window)
        current_price = df['close'].iloc[-1]

        # Reward & position update
        reward = 0
        if action == 'buy' and position is None:
            position = current_price
        elif action == 'sell' and position is not None:
            reward = current_price - position
            capital += reward
            position = None
        elif action == 'hold' and position is not None:
            reward = (current_price - position) * 0.01

        agent.learn(input_window, action, reward)
        agent.save()

        # Hybrid Network prediction
        predicted_scaled, similarity = network.predict_next(input_window, smoothing_factor)
        reconstructed = np.copy(data_scaled[-1])
        reconstructed[0] = predicted_scaled[0]
        predicted_close = scaler.inverse_transform([reconstructed])[0][0]

        buy_price, sell_price = predicted_close, predicted_close
        if network.units:
            n_features = data_scaled.shape[1]
            padded_positions = np.array([
                np.pad(unit.position[:n_features], (0, max(0, n_features - len(unit.position[:n_features]))), mode='constant')
                for unit in network.units
            ])
            closes = scaler.inverse_transform(padded_positions)[:, 0]
            buy_price = closes.min()
            sell_price = closes.max()

        uncertainty = network.estimate_uncertainty(similarity)
        ci = uncertainty * current_price

        # Add to prediction/trading log
        prediction_log.append({
            'Time': df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            'Predicted': predicted_close,
            'Actual': current_price,
            'Buy': buy_price,
            'Sell': sell_price,
            'Error': abs(current_price - predicted_close),
            'Action': action,
            'Reward': reward,
            'Position': position if position else "None"
        })

        # Sharpe ratio calculation
        returns = [p['Reward'] for p in prediction_log if p['Reward'] != 0]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(len(returns))
        else:
            sharpe_ratio = 0

        # Max drawdown calculation
        equity_curve = np.cumsum(returns) + capital
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0

        pattern_rate = metrics.verify_pattern_learning(df)
        signal_score = metrics.evaluate_signal_reliability(prediction_log)

        # Neural growth metrics
        total_neurons = len(agent.neurons)
        total_units = len(network.units)
        avg_neuron_usage = np.mean([n.usage for n in agent.neurons]) if agent.neurons else 0
        avg_unit_usage = np.mean([u.usage_count for u in network.units]) if network.units else 0

        # Learned trading rules summary (dynamic)
        action_counts = Counter([p['Action'] for p in prediction_log])
        avg_rewards = {}
        for act in ['buy', 'sell', 'hold']:
            rewards_for_act = [p['Reward'] for p in prediction_log if p['Action'] == act]
            avg_rewards[act] = np.mean(rewards_for_act) if rewards_for_act else 0

        learned_rules_summary = f"""
        ### Learned Trading Rules Summary (Dynamic)
        - Most frequent actions: {action_counts.most_common(3)}
        - Average rewards by action:
          - Buy: {avg_rewards['buy']:.4f}
          - Sell: {avg_rewards['sell']:.4f}
          - Hold: {avg_rewards['hold']:.4f}

        - Exploration rate (epsilon): {agent.epsilon:.4f}
        - Total neurons in SelfLearningTrader: {total_neurons}
        - Total units in HybridNeuralNetwork: {total_units}
        - Avg neuron usage count: {avg_neuron_usage:.2f}
        - Avg unit usage count: {avg_unit_usage:.2f}
        """

        # Dream visualization - episodic memory patterns with emotional tags
        episode_patterns = []
        episode_weights = []
        for ep in network.episodic_memory.episodes.values():
            episode_patterns.extend(ep['patterns'])
            episode_weights.extend(ep['emotional_tags'])
        if len(episode_patterns) > 5:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(episode_patterns)
        else:
            reduced = np.array(episode_patterns)

        dream_fig, dream_ax = plt.subplots(figsize=(8, 4))
        if len(reduced) > 0:
            weights = np.array(episode_weights) if episode_weights else np.ones(len(reduced))
            sizes = 50 + (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-9) * 300
            scatter = dream_ax.scatter(reduced[:, 0], reduced[:, 1], s=sizes, c=weights, cmap='coolwarm', alpha=0.7)
            dream_ax.set_title("Dream Visualization: Episodic Memory Patterns")
            plt.colorbar(scatter, ax=dream_ax, label="Emotional Weight")
        else:
            dream_ax.text(0.5, 0.5, "No episodic memory patterns yet", ha='center', va='center')
        dream_ax.grid(True)

        # Streamlit UI update
        with placeholder.container():
            st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Current Close", f"{current_price:.2f}")
            col2.metric("Predicted Close", f"{predicted_close:.2f}")
            col3.metric("Buy at Dip", f"{buy_price:.2f}")
            col4.metric("Sell at Peak", f"{sell_price:.2f}")
            col5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            col6.metric("Max Drawdown", f"{max_drawdown:.2%}")

        log_df = pd.DataFrame(prediction_log[-50:])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log_df['Time'], log_df['Actual'], label='Actual', color='blue', marker='o')
        ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted', color='orange', marker='x')
        ax.fill_between(log_df['Time'], log_df['Predicted'] - ci, log_df['Predicted'] + ci, alpha=0.2, color='orange')

        y_min = min(log_df['Actual'].min(), log_df['Predicted'].min())
        y_max = max(log_df['Actual'].max(), log_df['Predicted'].max())
        y_min_adjusted = 1000 * (y_min // 1000)
        y_max_adjusted = 1000 * ((y_max // 1000) + 1)
        ax.set_ylim(y_min_adjusted, y_max_adjusted)

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
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Pattern Recognition Rate", f"{pattern_rate:.2%}")
            mcol2.metric("Signal Consistency Score", f"{signal_score:.2%}")
            mcol3.metric("Open Positions", str(position if position else "None"))

        with neural_growth_placeholder.container():
            st.markdown("### Neural Growth Metrics")
            st.markdown(f"- Total Neurons: {total_neurons}")
            st.markdown(f"- Total Units: {total_units}")
            st.markdown(f"- Average Neuron Usage: {avg_neuron_usage:.2f}")
            st.markdown(f"- Average Unit Usage: {avg_unit_usage:.2f}")

        with rules_placeholder.container():
            st.markdown(learned_rules_summary)

        with trade_log_placeholder.container():
            st.markdown("### Recent Trading Log (Last 10)")
            recent_trades_df = pd.DataFrame(prediction_log[-10:])
            st.dataframe(recent_trades_df[['Time', 'Action', 'Reward', 'Position', 'Actual']])

        with dream_placeholder.container():
            st.pyplot(dream_fig)
            plt.close(dream_fig)

        time.sleep(60)

if __name__ == "__main__":
    main()
