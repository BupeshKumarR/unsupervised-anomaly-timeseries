import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap

# ------------------ App Config ------------------
st.set_page_config(page_title="⚡ PowerGrid Live Monitor", layout="wide")
st.title("⚡ Real-Time Smart Meter Monitoring")
st.markdown("Simulating live anomaly detection with streaming data.")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/sample_power_data.csv", parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

df = load_data()

# ------------------ Session State Init ------------------
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "window_data" not in st.session_state:
    st.session_state.window_data = pd.DataFrame(columns=df.columns)
if "paused" not in st.session_state:
    st.session_state.paused = False

# ------------------ Parameters ------------------
WINDOW_SIZE = 120
REFRESH_RATE = 0.5  # seconds
FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]

# ------------------ Controls ------------------
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("⏸ Pause" if not st.session_state.paused else "▶️ Resume"):
        st.session_state.paused = not st.session_state.paused

# ------------------ Placeholders ------------------
placeholder = st.empty()
alert_placeholder = st.empty()

# ------------------ Main Loop ------------------
while st.session_state.current_idx < len(df):

    if st.session_state.paused:
        time.sleep(0.1)
        continue

    next_row = df.iloc[st.session_state.current_idx]
    st.session_state.window_data = pd.concat(
        [st.session_state.window_data, pd.DataFrame([next_row])]
    ).tail(WINDOW_SIZE)

    if len(st.session_state.window_data) >= 60:
        X = st.session_state.window_data[FEATURES]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.02, random_state=42)
        preds = model.fit_predict(X_scaled)
        st.session_state.window_data["anomaly_live"] = (preds == -1).astype(int)
    else:
        st.session_state.window_data["anomaly_live"] = 0

    anomalies = st.session_state.window_data[st.session_state.window_data["anomaly_live"] == 1]

    with placeholder.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.window_data["datetime"],
            y=st.session_state.window_data["Global_active_power"],
            mode='lines',
            name='Power (kW)',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=anomalies["datetime"],
            y=anomalies["Global_active_power"],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=7, symbol='x')
        ))

        fig.update_layout(
            height=400,
            title="Live Power Usage with Anomaly Detection",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            showlegend=True,
            margin=dict(t=50, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

        if anomalies.shape[0] > 0:
            alert_placeholder.warning(f"\U0001F6A8 {len(anomalies)} anomaly{'ies' if len(anomalies)>1 else ''} detected in the last {WINDOW_SIZE} readings!", icon="⚠️")

            latest_anomaly = anomalies.iloc[-1]
            latest_idx = st.session_state.window_data.index.get_loc(latest_anomaly.name)

            try:
                explainer = shap.Explainer(model, X_scaled)
                shap_vals = explainer(X_scaled)

                if latest_idx < len(shap_vals):
                    base_val = shap_vals.base_values[latest_idx]
                    st.markdown("### 📊 SHAP Explanation for Latest Anomaly")
                    st.text(f"Base value: {base_val:.4f}")
                    st.text("Feature contributions:")
                    for feat, val in zip(FEATURES, shap_vals[latest_idx].values):
                        st.text(f"{feat}: {val:.4f}")
                else:
                    st.warning("⚠️ SHAP explanation not available for this anomaly (index out of range).")
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")
        else:
            alert_placeholder.info("✅ No anomalies detected in current window.", icon="✅")

    st.session_state.current_idx += 1
    time.sleep(REFRESH_RATE)
