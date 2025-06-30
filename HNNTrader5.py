# ---------------- Fixed Streamlit App with Actual Close Line and Value Display ----------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import ccxt
import time
from datetime import datetime

# Existing class definitions (EpisodicMemory, WorkingMemory, SemanticMemory, HybridNeuralUnit, HybridNeuralNetwork, etc.)
# ... [unchanged from your current code, truncated here for brevity]

# Add in KuCoin config
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
    input_window = data_scaled[-5:].flatten()

    actual_time = df.index[-1]
    actual_close = df['close'].iloc[-1]

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
    ci = uncertainty * actual_close

    prediction_log.append({
        'Time': actual_time.strftime("%H:%M:%S"),
        'Predicted': predicted_close,
        'Actual': actual_close,
        'Buy': buy_price,
        'Sell': sell_price,
        'Error': abs(actual_close - predicted_close)
    })

    pattern_rate = metrics.verify_pattern_learning(df)
    signal_score = metrics.evaluate_signal_reliability(prediction_log)

    with placeholder.container():
        st.markdown(f"### Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Close", f"{predicted_close:.2f}")
        col2.metric("Buy at Dip", f"{buy_price:.2f}")
        col3.metric("Sell at Peak", f"{sell_price:.2f}")
        st.markdown(f"**Current Actual Close:** `{actual_close:.2f}`")

        log_df = pd.DataFrame(prediction_log[-50:])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log_df['Time'], log_df['Actual'], label='Actual', color='blue', marker='o')
        ax.plot(log_df['Time'], log_df['Predicted'], label='Predicted', color='orange', marker='x')
        ax.fill_between(log_df['Time'], log_df['Predicted'] - ci, log_df['Predicted'] + ci, alpha=0.2, color='orange')
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