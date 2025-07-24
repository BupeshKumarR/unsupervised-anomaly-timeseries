import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Streamlit setup ---
st.set_page_config(page_title="⚡ PowerGrid Live Monitor", layout="wide")
st.title("⚡ Real-Time Smart Meter Monitoring")
st.markdown("Simulating live anomaly detection with streaming data.")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/sample_power_data.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

df = load_data()

# --- Initialize state ---
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "window_data" not in st.session_state:
    st.session_state.window_data = df.iloc[0:0].copy()
if "paused" not in st.session_state:
    st.session_state.paused = False

# --- Control buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.paused = True
with col2:
    if st.button("▶️ Resume", use_container_width=True):
        st.session_state.paused = False

# --- Parameters ---
WINDOW_SIZE = 120
REFRESH_RATE = 0.5
FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]

# --- Main containers ---
plot_placeholder = st.empty()
alert_placeholder = st.empty()
shap_placeholder = st.empty()

# --- Main simulation loop ---
if not st.session_state.paused and st.session_state.current_idx < len(df):

    next_row = df.iloc[st.session_state.current_idx]
    st.session_state.window_data = pd.concat(
        [st.session_state.window_data, pd.DataFrame([next_row])],
        ignore_index=True
    ).tail(WINDOW_SIZE)

    # Prepare features
    if len(st.session_state.window_data) >= 60:
        X = st.session_state.window_data[FEATURES]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.02, random_state=42)
        preds = model.fit_predict(X_scaled)
        st.session_state.window_data["anomaly_live"] = (preds == -1).astype(int)

        # Store SHAP explanation for last anomaly
        anomalies = st.session_state.window_data[st.session_state.window_data["anomaly_live"] == 1]
        if not anomalies.empty:
            latest_idx = anomalies.index[-1]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            st.session_state.latest_shap = {
                "shap_values": shap_values[st.session_state.window_data.index.get_loc(latest_idx)],
                "expected_value": explainer.expected_value,
                "data_row": X.iloc[st.session_state.window_data.index.get_loc(latest_idx)]
            }
        else:
            st.session_state.latest_shap = None

    else:
        st.session_state.window_data["anomaly_live"] = 0
        st.session_state.latest_shap = None

    st.session_state.current_idx += 1
    time.sleep(REFRESH_RATE)

# --- Plot live chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=st.session_state.window_data["datetime"],
    y=st.session_state.window_data["Global_active_power"],
    mode="lines",
    name="Power (kW)",
    line=dict(color="blue")
))
anomalies = st.session_state.window_data[st.session_state.window_data["anomaly_live"] == 1]
fig.add_trace(go.Scatter(
    x=anomalies["datetime"],
    y=anomalies["Global_active_power"],
    mode="markers",
    name="Anomaly",
    marker=dict(color="red", size=7, symbol="x")
))
fig.update_layout(
    height=400,
    title="Live Power Usage with Anomaly Detection",
    xaxis_title="Time",
    yaxis_title="Power (kW)",
    showlegend=True,
    margin=dict(t=50, b=30)
)
plot_placeholder.plotly_chart(fig, use_container_width=True)

# --- Anomaly alert ---
if anomalies.shape[0] > 0:
    alert_placeholder.warning(
        f"🚨 {len(anomalies)} anomaly{'ies' if len(anomalies) > 1 else ''} detected in the last {WINDOW_SIZE} readings!",
        icon="⚠️"
    )
else:
    alert_placeholder.info("✅ No anomalies detected in current window.", icon="✅")

# --- SHAP Explanation (only when paused) ---
if st.session_state.paused and st.session_state.latest_shap:
    st.subheader("📊 SHAP Explanation for Latest Anomaly")
    shap_data = st.session_state.latest_shap

    explanation = shap.Explanation(
        values=shap_data["shap_values"],
        base_values=shap_data["expected_value"],
        data=shap_data["data_row"],
        feature_names=FEATURES
    )

    # Render SHAP bar plot
    fig_shap, ax = plt.subplots()
    shap.plots.bar(explanation, show=False)
    st.pyplot(fig_shap, use_container_width=True)
