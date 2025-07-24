import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="⚡ PowerGrid Live Monitor", layout="wide")

st.title("⚡ Real-Time Smart Meter Monitoring")
st.markdown("Simulating live anomaly detection with streaming data.")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/sample_power_data.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

df = load_data()

WINDOW_SIZE = 120
FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]

if "idx" not in st.session_state:
    st.session_state.idx = WINDOW_SIZE

# This will trigger rerun every 500ms without blocking
count = st_autorefresh(interval=500, limit=None, key="autorefresh")

if st.session_state.idx >= len(df):
    st.write("⚡ Live simulation completed.")
else:
    window_df = df.iloc[st.session_state.idx - WINDOW_SIZE : st.session_state.idx]

    X = window_df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.02, random_state=42)
    preds = model.fit_predict(X_scaled)
    window_df = window_df.copy()
    window_df["anomaly"] = (preds == -1).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window_df["datetime"], y=window_df["Global_active_power"],
                             mode="lines", name="Power (kW)", line=dict(color="blue")))
    anomalies = window_df[window_df["anomaly"] == 1]
    fig.add_trace(go.Scatter(x=anomalies["datetime"], y=anomalies["Global_active_power"],
                             mode="markers", name="Anomalies", marker=dict(color="red", size=7, symbol="x")))

    st.plotly_chart(fig, use_container_width=True)

    if anomalies.shape[0] > 0:
        st.warning(f"🚨 {len(anomalies)} anomaly{'ies' if len(anomalies)>1 else ''} detected!")
    else:
        st.info("✅ No anomalies detected.")

    # Increment index for next refresh
    st.session_state.idx += 1
