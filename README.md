# Real-Time Unsupervised Anomaly Detection in Power Grid Timeseries

Live demo: [Streamlit Cloud App →](https://unsupervised-anomaly-timeseries-nhcznrbvhultujigyo7vvd.streamlit.app/)

---

## 📊 Overview

This project showcases a **real-time anomaly detection system** applied to household power consumption data. It leverages both classical unsupervised ML algorithms and deep learning architectures, wrapped inside an interactive Streamlit dashboard.

Key components:

* **Unsupervised learning (Isolation Forest)** for lightweight real-time detection
* **LSTM Autoencoder** for temporal pattern learning
* **SHAP explainability** to interpret anomaly origins
* **Streamlit app with real-time data streaming and pause/resume controls**

---

## ⚖️ Motivation

Power consumption data is inherently **time-dependent** and **non-linear**. Real-world grids demand:

* Immediate detection of anomalies (faults, surges, misuse)
* Interpretability for audit, root-cause analysis, and response

Hence, we combine **lightweight anomaly detection** for speed and **deep learning-based LSTM Autoencoders** to learn latent temporal dependencies. The model can be deployed in real-time systems or edge devices.

---

## ⚛️ Dataset

* Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
* 2M+ minute-wise readings between 2006 and 2010
* Features:

  * `Global_active_power` (kW)
  * `Global_reactive_power` (kW)
  * `Voltage` (V)
  * `Global_intensity` (A)
  * Plus others for future scope

We process and resample this into clean chunks for real-time simulation.

---

## 🔢 Methodology

### 1. **Isolation Forest**

* Fast, unsupervised algorithm using random trees to isolate anomalies.
* Works well in high dimensions and non-parametric.
* Each reading is scored, with an anomaly label assigned if the score falls below a threshold.

Used in this project for **streaming simulation due to speed**.

---

### 2. **LSTM Autoencoder** (Planned in Phase 2)

* LSTM networks are well-suited for timeseries due to their memory.
* Autoencoder architecture learns to compress (encode) and reconstruct (decode) time windows.
* **Anomalies** are detected based on high reconstruction error from the decoder.

This will be integrated to replace Isolation Forest for better long-term pattern learning.

#### Why LSTM is Important:

* Captures **long-term dependencies** unlike tree-based methods
* Learns **temporal correlations** (e.g., nighttime vs daytime usage)
* Works on **multivariate sequences** and handles missing values better

---

## 🔍 Explainability with SHAP

SHAP (SHapley Additive exPlanations) assigns contribution scores to each feature per prediction.

We use it to:

* Identify which features (e.g., `Voltage`, `Power`) caused the anomaly
* Build trust in the model
* Support human-in-the-loop analysis

In our dashboard:

* SHAP text summary is shown for the latest detected anomaly
* (Optional) SHAP force plots or summary plots can be added

---

## 🚀 Streamlit Dashboard

### Features

* Real-time simulation with `time.sleep()` and session state
* Line chart with power usage + anomalies (Plotly)
* SHAP explanation summary for each anomaly
* **Pause/Resume** toggle to inspect anomalies
* Designed for extensibility with LSTM or other detectors

---

## 🛠️ Setup & Run

```bash
# Clone repo
https://github.com/BupeshKumarR/unsupervised-anomaly-timeseries.git
cd unsupervised-anomaly-timeseries

# (Recommended) Setup virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app/realtime_dashboard.py
```

---

## 📊 Future Work

* [ ] Switch from Isolation Forest to **LSTM Autoencoder** inference
* [ ] Integrate **real-time data sources** (IoT, MQTT)
* [ ] Add alert system (Slack/email/SMS)
* [ ] Extend SHAP to handle sequence-level explanation
* [ ] Include auto model retraining pipeline

---

## 🚀 Authors

**Bupesh Kumar**
MS in Data Science, Northeastern University
[GitHub](https://github.com/BupeshKumarR) • [LinkedIn](https://www.linkedin.com/in/rbupeshkumar/)

---

## 🔗 License

This project is licensed under the MIT License.

---

> “What gets measured gets managed. What gets explained, gets trusted.”
