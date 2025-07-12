import os
import pickle

# ---------------- Persistent State Functions -------------------
def save_state(filename, network, memory_log):
    with open(filename, 'wb') as f:
        pickle.dump({'network': network, 'memory_log': memory_log}, f)

def load_state(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            return state['network'], state['memory_log']
    return HybridNeuralNetwork(), []

def replay_memory(network, memory_log):
    for pattern in memory_log:
        network.process_input(pattern)

# ---------------- Load Previous State -------------------
STATE_FILE = 'dream_network_state.pkl'
network, memory_log = load_state(STATE_FILE)
prediction_log = []
capital_usdt = 1000
trade_amount = 100
executed_trades = []
open_trades = []

placeholder = st.empty()
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

# ---------------- Main Loop -------------------
while True:
    df = get_kucoin_data()
    features = ['close', 'RSI', 'MA20', 'ATR']
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(imputer.fit_transform(df[features]))

    input_pattern = data_scaled[-5:].flatten()
    actual_close = df['close'].iloc[-1]
    smoothing = 0.3 if df['ATR'].iloc[-1] > df['close'].std() * 0.01 else 0.7

    predicted_scaled, similarity = network.predict_next(input_pattern, smoothing)
    reconstructed = np.copy(data_scaled[-1])
    reconstructed[0] = predicted_scaled[0]
    predicted_close = scaler.inverse_transform([reconstructed])[0][0]

    memory_log.append(input_pattern)
    if len(memory_log) > 1000:
        memory_log.pop(0)

    prediction_log.append({
        'Time': datetime.now(),
        'Actual': actual_close,
        'Predicted': predicted_close,
        'Error': abs(actual_close - predicted_close)
    })

    # Adaptive position sizing
    adaptive_trade_amount = trade_amount * min(1.5, max(0.5, similarity * 2))

    # Trade decision
    if similarity > 0.3 and capital_usdt >= adaptive_trade_amount:
        sell_price = predicted_close * 1.01
        open_trades.append({
            'Time': datetime.now(), 'Buy': predicted_close, 'Sell': sell_price,
            'Status': 'Open', 'Amount': adaptive_trade_amount
        })
        capital_usdt -= adaptive_trade_amount

    # Trade evaluation
    for trade in open_trades:
        if trade['Status'] == 'Open':
            if actual_close >= trade['Sell']:
                trade['Status'] = 'Closed'
                capital_usdt += trade['Sell']
                trade['CloseTime'] = datetime.now()
                executed_trades.append(trade)
            elif actual_close < trade['Buy'] * 0.985:
                trade['Status'] = 'Closed'
                capital_usdt += actual_close
                trade['Sell'] = actual_close
                trade['CloseTime'] = datetime.now()
                executed_trades.append(trade)
                network.reinforce_failure(input_pattern)

    open_trades = [t for t in open_trades if t['Status'] == 'Open']

    # Evaluation
    stats = network.get_stats()
    win_trades = [t for t in executed_trades if t['Sell'] > t['Buy']]
    returns = [(t['Sell'] - t['Buy']) / t['Buy'] for t in executed_trades]
    win_ratio = len(win_trades) / len(executed_trades) if executed_trades else 0
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if returns else 0
    profit = capital_usdt - 1000

    # Display
    with placeholder.container():
        st.metric("Predicted Close", f"{predicted_close:.2f}")
        st.metric("Actual Close", f"{actual_close:.2f}")
        st.metric("Capital", f"{capital_usdt:.2f} USDT")
        st.metric("Win Ratio", f"{win_ratio:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Cumulative Profit", f"{profit:.2f} USDT")

    with chart_placeholder.container():
        st.subheader("📊 Prediction vs Actual Close")
        hist_df = pd.DataFrame(prediction_log[-100:])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist_df['Time'], hist_df['Actual'], label='Actual', marker='o')
        ax.plot(hist_df['Time'], hist_df['Predicted'], label='Predicted', marker='x')
        ax.fill_between(hist_df['Time'], hist_df['Predicted'] - hist_df['Error'],
                        hist_df['Predicted'] + hist_df['Error'], color='orange', alpha=0.2)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📟 Executed Trades")
        if executed_trades:
            df_exec = pd.DataFrame(executed_trades)
            df_exec['Profit'] = df_exec['Sell'] - df_exec['Buy']
            st.dataframe(df_exec.tail(10))

        st.subheader("📌 Open Trades")
        if open_trades:
            df_open = pd.DataFrame(open_trades)
            st.dataframe(df_open.tail(10))

    with metrics_placeholder.container():
        st.subheader("📈 Neural Network Stats")
        st.metric("Neural Units", stats['unit_count'])
        st.metric("Avg Reward", f"{stats['avg_reward']:.4f}")
        st.metric("Avg Usage", f"{stats['avg_usage']:.2f}")

    # Save state
    save_state(STATE_FILE, network, memory_log)

    time.sleep(60)
