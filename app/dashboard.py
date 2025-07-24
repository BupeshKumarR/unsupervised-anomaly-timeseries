import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Load data ---
st.set_page_config(layout="wide")

st.title("⚡ PowerGrid Guardian: Smart Energy Anomaly Detection")
st.markdown("Detecting abnormal power consumption using unsupervised ML & statistical methods.")

# Sidebar method selection
method = st.sidebar.radio("Choose Detection Method", ["STL + Z-Score", "Isolation Forest", "LSTM Autoencoder"])


# Load data
if method == "STL + Z-Score":
    df = pd.read_csv("data/processed/stl_anomaly_output.csv", parse_dates=["datetime"])
    anomaly_col = "anomaly"
elif method == "Isolation Forest":
    df = pd.read_csv("data/processed/iforest_anomaly_output.csv", parse_dates=["datetime"])
    anomaly_col = "anomaly_iforest"
else:  # LSTM Autoencoder
    df = pd.read_csv("data/processed/lstm_anomaly_output.csv", parse_dates=["datetime"])
    anomaly_col = "anomaly_lstm"


# --- Plot ---
fig = go.Figure()

# Add main line
fig.add_trace(go.Scatter(
    x=df["datetime"],
    y=df["Global_active_power"],
    mode='lines',
    name='Global Active Power',
    line=dict(color='blue')
))

# Add anomalies
anomalies = df[df[anomaly_col] == 1]
fig.add_trace(go.Scatter(
    x=anomalies["datetime"],
    y=anomalies["Global_active_power"],
    mode='markers',
    name='Anomalies',
    marker=dict(color='red', size=6, symbol='x')
))

fig.update_layout(
    title=f"Anomaly Detection: {method}",
    xaxis_title="Time",
    yaxis_title="Power (kW)",
    legend=dict(x=0, y=1.1, orientation="h"),
    margin=dict(l=20, r=20, t=60, b=20),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Show raw data (optional)
with st.expander("🔎 View Raw Data"):
    st.dataframe(df.head(100))
