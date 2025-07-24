import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
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

# --- Init session state ---
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "window_data" not in st.session_state:
    st.session_state.window_data = df.iloc[0:0].copy()
if "paused" not in st.session_state:
    st.session_state.paused = False
if "latest_shap" not in st.session_state:
    st.session_state.latest_shap = None

# --- Controls ---
col1, col2 = st.columns(2)
with col1:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.paused = True
with col2:
    if st.button("▶️ Resume", use_container_width=True):
        st.session_state.paused = False

# --- Main logic ---
WINDOW_SIZE = 120
FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]

if not st.session_state.paused and st.session_state.current_idx < len(df):

    next_row = df.iloc[st.session_state.current_idx]
    st.session_state.window_data = pd.concat(
        [st.session_state.window_data, pd.DataFrame([next_row])],
        ignore_index=True
    ).tail(WINDOW_SIZE)

    if len(st.session_state.window_data) >= 60:
        X = st.session_state.window_data[FEATURES]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.02, random_state=42)
        preds = model.fit_predict(X_scaled)
        st.session_state.window_data["anomaly_live"] = (preds == -1).astype(int)

        # SHAP only for last anomaly
        anomalies = st.session_state.window_data[st.session_state.window_data["anomaly_live"] == 1]
        if not anomalies.empty:
            latest_idx = anomalies.index[-1]
            explainer = shap.Explainer(model, X)
            shap_vals = explainer(X)
            st.session_state.latest_shap = shap_vals[latest_idx]
    else:
        st.session_state.window_data["anomaly_live"] = 0

    st.session_state.current_idx += 1

# --- Plot ---
plot_placeholder = st.empty()
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

# --- Alert ---
if anomalies.shape[0] > 0:
    st.warning(
        f"🚨 {len(anomalies)} anomaly{'ies' if len(anomalies) > 1 else ''} detected in last {WINDOW_SIZE} readings!",
        icon="⚠️"
    )
else:
    st.info("✅ No anomalies detected in current window.", icon="✅")

# --- SHAP when paused ---
if st.session_state.paused and st.session_state.latest_shap is not None:
    st.subheader("📊 SHAP Explanation for Latest Anomaly")
    fig_shap, ax = plt.subplots()
    shap.plots.bar(st.session_state.latest_shap, show=False)
    st.pyplot(fig_shap, use_container_width=True)

# --- Trigger refresh ---
if not st.session_state.paused:
    st.rerun()
