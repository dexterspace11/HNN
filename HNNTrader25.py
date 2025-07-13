# ---------------- Unified True Self-Learning Dream-Hybrid Neural Trading System -------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt, time, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
from sklearn.decomposition import PCA
from collections import Counter

# 🧐 Import dream_core.py components
from dream_core import (
    SelfLearningTrader, HybridNeuralNetwork, PatternRecognitionMetrics,
    calculate_rsi, calculate_atr
)

# ---------------- KuCoin Data Fetcher -------------------
def get_kucoin_data(symbol='BTC/USDT', timeframe='1m', limit=200):
    exchange = ccxt.kucoin()
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['RSI'] = calculate_rsi(df['close'])
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df.dropna(inplace=True)
    return df

# ---------------- Utility for Performance Metrics -------------------
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - risk_free_rate) / std_return if std_return != 0 else 0

def calculate_drawdown(equity_curve):
    peak = equity_curve[0]
    max_drawdown = 0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        max_drawdown = max(max_drawdown, dd)
    return max_drawdown

# ---------------- Main Streamlit App -------------------
def main():
    st.set_page_config(page_title="Dream-Hybrid Trader", layout="wide")
    st.title("🧠🤖 True Self-Learning Dream-Hybrid BTC/USDT Neural Trader")

    if 'capital' not in st.session_state:
        st.session_state.capital = 1000.0
    if 'position' not in st.session_state:
        st.session_state.position = None
    if 'prediction_log' not in st.session_state:
        st.session_state.prediction_log = []

    # Properly load agent using static load method
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = SelfLearningTrader.load()
        except Exception:
            st.warning("Incompatible agent_state.pkl file or load error. Starting fresh.")
            st.session_state.agent = SelfLearningTrader()

    # Persistent Hybrid Neural Network
    if 'network' not in st.session_state:
        st.session_state.network = HybridNeuralNetwork()

    network = st.session_state.network
    metrics = PatternRecognitionMetrics(network)

    with st.spinner("Fetching latest data from KuCoin..."):
        df = get_kucoin_data('BTC/USDT', '1m', 200)

    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(imputer.fit_transform(df[features]))

    smoothing_factor = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7
    input_window = data_scaled[-5:].flatten()

    neuron, action = st.session_state.agent.act(input_window)
    current_price = df['close'].iloc[-1]
    position = st.session_state.position
    reward = 0

    if action == 'buy' and position is None:
        st.session_state.position = current_price
        action_msg = f"Bought at {current_price:.2f}"
    elif action == 'sell' and position is not None:
        reward = current_price - position
        st.session_state.capital += reward
        st.session_state.position = None
        action_msg = f"Sold at {current_price:.2f}, Profit: {reward:.2f}"
    elif action == 'hold' and position is not None:
        reward = (current_price - position) * 0.01
        action_msg = f"Holding position bought at {position:.2f}"
    else:
        action_msg = f"Holding (no position)"

    st.session_state.agent.learn(input_window, action, reward)
    st.session_state.agent.save()

    predicted_scaled, similarity = network.predict_next(input_window, smoothing_factor)
    reconstructed = np.copy(data_scaled[-1])
    reconstructed[0] = predicted_scaled[0]
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    if network.units:
        closes = scaler.inverse_transform(
            [np.pad(u.position[:data_scaled.shape[1]], (0, max(0, data_scaled.shape[1] - len(u.position))), mode='constant')
             for u in network.units]
        )[:, 0]
        buy_price, sell_price = closes.min(), closes.max()
    else:
        buy_price = sell_price = predicted_close

    uncertainty = network.estimate_uncertainty(similarity)
    ci = uncertainty * current_price

    st.session_state.prediction_log.append({
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

    log_df = pd.DataFrame(st.session_state.prediction_log)

    # 🔧 Improved Price Chart
    st.subheader("Price Prediction & Actual Close Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(log_df['Time'], log_df['Actual'], label='Actual Close', marker='o', color='blue')
    ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted Close', marker='x', color='orange')
    ax.fill_between(log_df['Time'], log_df['Predicted'] - ci, log_df['Predicted'] + ci, color='gray', alpha=0.2)
    price_min = min(log_df['Actual'].min(), log_df['Predicted'].min())
    price_max = max(log_df['Actual'].max(), log_df['Predicted'].max())
    ax.set_ylim(price_min - 300, price_max + 300)
    ax.legend()
    ax.set_xticks(np.arange(0, len(log_df['Time']), max(1, len(log_df['Time']) // 10)))
    ax.set_xticklabels(log_df['Time'][::max(1, len(log_df['Time']) // 10)], rotation=45, ha='right')
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)

    log_df['Return'] = log_df['Reward']
    log_df['Equity'] = st.session_state.capital + log_df['Return'].cumsum()
    sharpe = calculate_sharpe_ratio(log_df['Return'])
    drawdown = calculate_drawdown(log_df['Equity'].values)

    st.subheader("Performance Metrics")
    st.markdown(f"- **Current Capital:** ${st.session_state.capital:.2f}")
    st.markdown(f"- **Sharpe Ratio:** {sharpe:.4f}")
    st.markdown(f"- **Max Drawdown:** {drawdown*100:.2f}%")
    st.markdown(f"- **Last Action:** {action_msg}")

    # 📈 Neural Growth Metrics
    stats = network.neural_growth_stats()
    st.subheader("Neural Growth & Dream Clustering")
    st.markdown(f"- **Total Units:** {stats['total_units']}")
    st.markdown(f"- **Dream Clusters (Episodes):** {stats['total_episodes']}")
    st.markdown(f"- **Memory Patterns:** {stats['total_patterns']}")

    st.subheader("Episodic Memory Dream")
    if network.episodic_memory.current_episode:
        ep = network.episodic_memory.episodes[network.episodic_memory.current_episode]
        st.dataframe(pd.DataFrame({
            'Pattern': ep['patterns'][-10:],
            'Emotional Tag': ep['emotional_tags'][-10:]
        }))
    else:
        st.write("No episodic memory yet.")

    st.subheader("🌈 Dream Visualization (PCA Projection of Episodic Memory)")
    dream_patterns = [p for ep in network.episodic_memory.episodes.values() for p in ep['patterns']]
    dream_emotions = [e for ep in network.episodic_memory.episodes.values() for e in ep['emotional_tags']]

    if dream_patterns:
        try:
            pca = PCA(n_components=2)
            patterns_2d = pca.fit_transform(dream_patterns)
            df_dream = pd.DataFrame({
                'X': patterns_2d[:, 0],
                'Y': patterns_2d[:, 1],
                'Emotion': dream_emotions
            })
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            scatter = ax2.scatter(df_dream['X'], df_dream['Y'], c=df_dream['Emotion'], cmap='viridis', alpha=0.7)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label("Emotional Tag Intensity")
            ax2.set_title("Projected Memory Patterns")
            ax2.set_xlabel("PCA X")
            ax2.set_ylabel("PCA Y")
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Unable to render dream visualization: {e}")
    else:
        st.write("Not enough memory patterns to generate dream visualization.")

    st.subheader("Learned Trading Rules")
    conns = []
    for unit in network.units:
        for other, strength in unit.connections.items():
            conns.append({
                "From Unit": str(unit.position[:5]),
                "To Unit": str(other.position[:5]),
                "Strength": strength
            })
    if conns:
        st.dataframe(pd.DataFrame(conns).sort_values(by="Strength", ascending=False).head(10))
    else:
        st.write("No learned connections yet.")

    st.subheader("Trading Log (Recent 20 Actions)")
    st.dataframe(log_df.tail(20))

    st.text("Refreshing in 60 seconds...")
    time.sleep(60)
    st.rerun()

if __name__ == "__main__":
    main()
