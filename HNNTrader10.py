# ---------------- Enhanced HNNTrader8 Streamlit App with Trade Simulation and Analytics -------------------
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

# ---------------- Neural Network -------------------
class HybridNeuralNetwork:
    def __init__(self):
        self.units = []
        self.gen_threshold = 0.5
        self.last_prediction = None
        self.synaptic_scaling_factor = 1.0
        self.episodic_memory = EpisodicMemory()

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

# ---------------- Data Fetch & Indicators -------------------
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

# ---------------- Streamlit UI & Main Logic -------------------
exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '1m'
data_limit = 200

st.set_page_config(page_title="HNNTrader8 Realtime", layout="wide")
st.title("HNNTrader8 - Real-time Predictor with Trade Simulation")

placeholder = st.empty()
chart_placeholder = st.empty()
log_placeholder = st.empty()

network = HybridNeuralNetwork()
prediction_log = []
capital = 1000.0
trade_size = 100.0
position_open = False
entry_price = 0
profits = []

while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(df[features])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    smoothing_factor = 0.4
    input_window = data_scaled[-5:].flatten()

    actual_close = df['close'].iloc[-1]
    actual_time = df.index[-1]

    preds = []
    current_input = input_window.copy()
    for _ in range(3):
        pred, sim = network.predict_next(current_input, smoothing_factor)
        preds.append(pred)
        current_input[:-1] = current_input[1:]
        current_input[-1] = pred[0]

    pred_array = np.copy(data_scaled[-1])
    pred_array[0] = preds[0][0]
    predicted_close = scaler.inverse_transform([pred_array])[0][0]

    buy_price = predicted_close - 0.005 * predicted_close
    sell_price = predicted_close + 0.005 * predicted_close
    signal_score = sim

    # Trade logic
    pnl = 0
    if not position_open and predicted_close > actual_close:
        entry_price = actual_close
        position_open = True
        trade_type = 'BUY'
    elif position_open and actual_close >= entry_price * 1.01:
        pnl = actual_close - entry_price
        profits.append(pnl)
        capital += pnl
        position_open = False
        trade_type = 'SELL'
    else:
        trade_type = '-'

    sharpe_ratio = 0
    if len(profits) > 1:
        returns = np.array(profits)
        sharpe_ratio = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(len(returns))

    win_trades = len([p for p in profits if p > 0])
    loss_trades = len([p for p in profits if p <= 0])
    win_loss_ratio = win_trades / loss_trades if loss_trades > 0 else float('inf')

    prediction_log.append({
        'Time': actual_time.strftime("%H:%M:%S"),
        'Predicted': predicted_close,
        'Actual': actual_close,
        'Buy': buy_price,
        'Sell': sell_price,
        'SignalScore': signal_score,
        'Trade': trade_type,
        'P&L': round(pnl, 4),
        'Capital': round(capital, 2)
    })

    with placeholder.container():
        st.markdown(f"### Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Predicted Close", f"{predicted_close:.2f}")
        col2.metric("Buy Price", f"{buy_price:.2f}")
        col3.metric("Sell Price", f"{sell_price:.2f}")
        col4.metric("Signal Score", f"{signal_score:.3f}")
        col5.metric("Capital", f"{capital:.2f}")

    with chart_placeholder.container():
        log_df = pd.DataFrame(prediction_log[-50:])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log_df['Time'], log_df['Actual'], label='Actual', color='blue')
        ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted', color='orange')
        ax.legend()
        ax.set_title("Actual vs Predicted Close")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with log_placeholder.container():
        st.markdown("### Trade Log and Performance")
        st.dataframe(pd.DataFrame(prediction_log[-20:]))
        st.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    time.sleep(60)