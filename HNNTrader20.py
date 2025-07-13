# ---------------- Unified True Self-Learning Dream-Hybrid Neural Trading System with Metrics -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt, time, pickle, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
from collections import deque

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

# ---------------- Indicators & Data -------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_kucoin_data(symbol='BTC/USDT', timeframe='1m'):
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

# --------------- Metrics functions ---------------
def sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

def max_drawdown(prices):
    cummax = np.maximum.accumulate(prices)
    drawdowns = (prices - cummax) / cummax
    return drawdowns.min()

def extract_trading_rules(neurons):
    if not neurons:
        return "No learned trading rules yet."
    # Simplified example: average neuron state vectors, interpret feature importance
    avg_state = np.mean([n.state for n in neurons], axis=0)
    rule_summary = "Current learned trading pattern feature importance (normalized):\n"
    normalized = (avg_state - np.min(avg_state)) / (np.ptp(avg_state) + 1e-9)
    for i, val in enumerate(normalized):
        rule_summary += f" - Feature {i}: {val:.2f}\n"
    return rule_summary

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader with Metrics", layout="wide")
    st.title("🧠🤖 True Self-Learning Dream-Hybrid BTC/USDT Neural Trader with Metrics")

    global exchange
    exchange = ccxt.kucoin()

    agent = SelfLearningTrader()
    agent.load()

    capital = 1000.0
    position = None
    prediction_log = []
    prices_history = deque(maxlen=252*7)  # store ~1 week of 1-min returns for Sharpe calc
    capital_history = deque(maxlen=1000)
    neuron_counts = deque(maxlen=100)

    placeholder = st.empty()
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    log_placeholder = st.empty()
    dream_placeholder = st.empty()
    rules_placeholder = st.empty()
    growth_placeholder = st.empty()

    while True:
        df = get_kucoin_data()
        latest = df.iloc[-1:]
        features = ['close', 'RSI', 'MA20']

        imputer = SimpleImputer(strategy='mean')
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(imputer.fit_transform(df[features]))
        state = data_scaled[-5:].flatten()

        neuron, action = agent.act(state)
        current_price = latest['close'].values[0]
        prices_history.append(current_price)

        reward = 0
        if action == 'buy' and position is None:
            position = current_price
            trade_action = 'BUY'
            trade_price = current_price
        elif action == 'sell' and position is not None:
            reward = current_price - position
            capital += reward
            position = None
            trade_action = 'SELL'
            trade_price = current_price
        else:
            trade_action = 'HOLD'
            trade_price = current_price
            if position is not None:
                reward = (current_price - position) * 0.01

        agent.learn(state, action, reward)
        agent.save()

        capital_history.append(capital)
        neuron_counts.append(len(agent.neurons))

        # Log trades for display
        if trade_action in ['BUY', 'SELL']:
            prediction_log.append({
                'Timestamp': datetime.now(),
                'Action': trade_action,
                'Price': trade_price,
                'Capital': capital,
                'Reward': reward
            })

        # Calculate returns for Sharpe ratio (log returns)
        returns = np.diff(np.log(np.array(prices_history))) if len(prices_history) > 1 else np.array([0])
        sharpe = sharpe_ratio(returns)
        drawdown = max_drawdown(np.array(prices_history))

        # UI Output
        with placeholder.container():
            st.metric("Current Close Price", f"{current_price:.2f} USDT")
            st.metric("Capital", f"{capital:.2f} USDT")
            st.metric("Last Action", trade_action)
            st.metric("Last Reward", f"{reward:.4f}")
            st.metric("Sharpe Ratio (est.)", f"{sharpe:.2f}")
            st.metric("Max Drawdown", f"{drawdown:.2%}")
            st.metric("Total Learned Neurons", len(agent.neurons))

        with log_placeholder.container():
            st.subheader("📜 Trading Log")
            if prediction_log:
                log_df = pd.DataFrame(prediction_log)
                log_df.index = log_df['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(log_df[['Timestamp', 'Action', 'Price', 'Capital', 'Reward']].sort_index(ascending=False))
            else:
                st.write("No trades executed yet.")

        with chart_placeholder.container():
            st.subheader("📈 Price & Trades Chart")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(list(prices_history), label='Close Price')
            buys = [log['Price'] for log in prediction_log if log['Action'] == 'BUY']
            buys_idx = [i for i, log in enumerate(prediction_log) if log['Action'] == 'BUY']
            sells = [log['Price'] for log in prediction_log if log['Action'] == 'SELL']
            sells_idx = [i for i, log in enumerate(prediction_log) if log['Action'] == 'SELL']
            ax.scatter(buys_idx, buys, marker='^', color='green', label='Buy', s=100)
            ax.scatter(sells_idx, sells, marker='v', color='red', label='Sell', s=100)
            ax.set_title("Price and Trade Actions Over Time")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with dream_placeholder.container():
            st.subheader("🌙 Dreams Visualization (Episodic Memory Snapshot)")
            # Show a simplified visualization of random neurons states as "dreams"
            dream_samples = np.array([n.state for n in agent.neurons[-10:]])
            if len(dream_samples) > 0:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.imshow(dream_samples, aspect='auto', cmap='plasma')
                ax.set_title("Neuron States Over Recent Dreams")
                ax.set_xlabel("State Feature Index")
                ax.set_ylabel("Neuron Index")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.write("No dreams recorded yet.")

        with rules_placeholder.container():
            st.subheader("📚 Current Learned Trading Rules Summary")
            rules_summary = extract_trading_rules(agent.neurons)
            st.text(rules_summary)

        with growth_placeholder.container():
            st.subheader("📊 Neural Growth Over Time")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(list(neuron_counts), label='Neuron Count')
            ax.set_title("Growth of Learned Neurons")
            ax.set_xlabel("Time (iterations)")
            ax.set_ylabel("Number of Neurons")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        time.sleep(60)

if __name__ == "__main__":
    main()
