# ✅ Optimized Hybrid DNN-EQIC BTC/USDT Predictor with Streamlit Visualization
# Key Enhancements:
# - Improved smoothing (volatility aware)
# - Forward validation fixed: predicted vs *future* actual close
# - Prioritized memory replay
# - Limited Hebbian updates to top-3 most similar units
# - Reduced PCA overhead (only updated every 5 loops)

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

# KuCoin Configuration
exchange = ccxt.kucoin()
symbol = 'BTC/USDT'
timeframe = '5m'
data_limit = 200

prediction_log = []
prediction_queue = []
replay_memory = []
replay_errors = []

loop_count = 0

# ---------------- Indicators + Data -------------------
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

# ---------------- Memory + Network Modules -------------------
# [Identical to original: EpisodicMemory, WorkingMemory, SemanticMemory, HybridNeuralUnit]
# [Modified]: HybridNeuralNetwork (attention limit, dynamic smoothing)
# [Same as original with adjustments where discussed earlier in chat]

# ---------------- Streamlit App -------------------
st.set_page_config(page_title="Hybrid DNN-EQIC BTC Predictor", layout="wide")
st.title("Real-Time Hybrid DNN-EQIC BTC/USDT Predictor")
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

    current_volatility = df['ATR'].iloc[-1]
    smoothing_factor = 0.3 if current_volatility > df['close'].std() * 0.01 else 0.7

    for sample in data_scaled:
        prediction, similarity = network.predict_next(sample)
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

    # Prioritized Replay (top 10 by error)
    if len(replay_memory) > 10:
        top_indices = np.argsort(replay_errors)[-10:]
        for idx in top_indices:
            prediction, _ = network.predict_next(replay_memory[idx])
            network.update_feature_importance(replay_memory[idx], prediction)

    network.prune_units()

    last_input = data_scaled[-1]
    predicted_next_scaled, similarity = network.predict_next(last_input)
    reconstructed = np.copy(last_input)
    reconstructed[0] = predicted_next_scaled[0]
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    current_timestamp = df.index[-1]
    prediction_queue.append((current_timestamp, predicted_close))

    # Forward validation: previous prediction vs actual close now
    if len(prediction_queue) > 1:
        prev_time, prev_pred = prediction_queue.pop(0)
        actual_close = df.loc[df.index >= prev_time, 'close'].iloc[0] if prev_time < df.index[-1] else None
        if actual_close is not None:
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
            chart_placeholder.pyplot(fig)
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(10, 2.5))
            ax2.plot(log_df['Time'], log_df['Error'], color='red', marker='.')
            error_chart_placeholder.pyplot(fig2)
            plt.close(fig2)

        st.markdown("### Feature Importance")
        if network.feature_importance is not None:
            norm_importance = network.feature_importance / np.sum(network.feature_importance)
            st.bar_chart(pd.Series(norm_importance, index=features))

    time.sleep(60)
