import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import random
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI-IDS Pro | Real-Time SIM",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN UI DESIGN (GLASSMORPHISM & GRADIENTS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    /* Smooth Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0ea5e9 100%);
        background-attachment: fixed;
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    
    .main-title {
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    /* HUD Bar */
    .status-hud {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(8px);
        padding: 12px 24px;
        border-radius: 12px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .glow-dot {
        width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px;
    }
    .dot-active {
        background-color: #14b8a6; box-shadow: 0 0 12px #14b8a6; animation: pulse-teal 2s infinite;
    }
    .dot-idle { background-color: #64748b; }
    
    @keyframes pulse-teal {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Alerts */
    .alert-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid;
    }

    .stDataFrame { background-color: transparent !important; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Prediction', 'Actual', 'Confidence', 'Drift'])
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'last_accuracy' not in st.session_state:
    st.session_state.last_accuracy = 0.0

# --- SIMULATION ENGINE ---
def generate_fake_data():
    atk_types = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    # Probability: 70% Normal, 30% Attacks
    if random.random() < 0.7:
        y_true = "Normal"
        y_pred = "Normal" if random.random() < 0.95 else random.choice(atk_types[1:])
    else:
        y_true = random.choice(atk_types[1:])
        y_pred = y_true if random.random() < 0.8 else random.choice(atk_types)
    
    conf = round(random.uniform(0.5, 0.99), 2)
    drift = random.random() < 0.05 # 5% chance of simulated drift
    ts = datetime.now().strftime("%H:%M:%S")
    
    return {
        'Timestamp': ts,
        'Prediction': y_pred,
        'Actual': y_true,
        'Confidence': conf,
        'Drift': drift
    }

def update_simulation():
    if not st.session_state.is_streaming:
        return
    
    new_row = generate_fake_data()
    # Add to history
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)
    
    # Prune rolling window (max 500)
    if len(st.session_state.history) > 500:
        st.session_state.history = st.session_state.history.tail(500)
    
    # Save Alert if attack
    if new_row['Prediction'] != "Normal":
        sev = "High" if new_row['Confidence'] > 0.8 else "Medium" if new_row['Confidence'] > 0.6 else "Low"
        alert = {
            'timestamp': new_row['Timestamp'],
            'type': new_row['Prediction'],
            'confidence': new_row['Confidence'],
            'severity': sev
        }
        st.session_state.alerts.append(alert)
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts.pop(0)

# --- SIDEBAR: SAAS CONTROLS ---
st.sidebar.markdown("<h2 style='letter-spacing: -1px;'>IDS Sim Console</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if st.session_state.is_streaming:
    if st.sidebar.button("Stop Live Stream", use_container_width=True):
        st.session_state.is_streaming = False
        st.rerun()
    st.markdown("<style>div.stButton > button { background: linear-gradient(to right, #f59e0b, #eab308) !important; color: white !important; }</style>", unsafe_allow_html=True)
else:
    if st.sidebar.button("Start Live Stream", use_container_width=True):
        st.session_state.is_streaming = True
        st.rerun()
    st.markdown("<style>div.stButton > button { background: linear-gradient(to right, #3b82f6, #14b8a6) !important; color: white !important; }</style>", unsafe_allow_html=True)

st.sidebar.markdown("### 🧩 Filter Logic")
f_atks = st.sidebar.multiselect("Attack Category", ["Normal", "DoS", "Probe", "R2L", "U2R"], default=["Normal", "DoS", "Probe", "R2L", "U2R"])
f_query = st.sidebar.text_input("Forensic Search (Query)")
f_limit = st.sidebar.slider("Historical Lookback", 10, 500, 150)

st.sidebar.markdown("### ⚙️ Engine Tunables")
use_drift = st.sidebar.checkbox("Simulate Drift Detection", value=True)
smooth_factor = st.sidebar.slider("Accuracy Smoothing Factor", 1, 50, 20)
latency = st.sidebar.select_slider("Polling Interval (s)", options=[0.1, 0.2, 0.5, 1.0, 2.0], value=1.0)

# --- HUD ---
hud_title = "🚀 Real-Time Detection Active" if st.session_state.is_streaming else "System Standby"
st.markdown(f"<p class='main-title'>{hud_title}</p>", unsafe_allow_html=True)

log_size = len(st.session_state.history)
dot_class = "dot-active" if st.session_state.is_streaming else "dot-idle"
status_lbl = "Live Processing" if st.session_state.is_streaming else "Idle"

st.markdown(f"""
<div class="status-hud">
    <div><span class="glow-dot {dot_class}"></span>{status_lbl}</div>
    <div style="color: #94a3b8; font-size: 0.9rem;">Packet Ingress: <b style="color: #3b82f6;">{log_size} samples</b></div>
    <div style="color: #94a3b8; font-size: 0.9rem;">Refresh Rate: <b style="color: #14b8a6;">{latency}s</b></div>
</div>
""", unsafe_allow_html=True)

# --- ANALYTICS HUB ---
df = st.session_state.history.copy()
if not df.empty:
    df['Confidence'] = df['Confidence'].astype(float)
    if f_atks:
        df = df[df['Prediction'].isin(f_atks)]
    if f_query:
        df = df[df.astype(str).apply(lambda x: x.str.contains(f_query, case=False)).any(axis=1)]
    df_view = df.tail(f_limit)

    # Metrics
    acc = (df['Prediction'] == df['Actual']).mean()
    threats = len(df[df['Prediction'] != "Normal"])
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="glass-card"><small style="color: #94a3b8;">Detection Accuracy</small><h3>{acc*100:.2f}%</h3></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="glass-card"><small style="color: #94a3b8;">Total Threats Blocked</small><h3>{threats}</h3></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="glass-card"><small style="color: #94a3b8;">Throughput</small><h3>{1/latency:.1f} P/s</h3></div>""", unsafe_allow_html=True)

    # Charts
    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Accuracy Momentum")
        df['Smoothing'] = (df['Prediction'] == df['Actual']).rolling(window=smooth_factor, min_periods=1).mean()
        fig_acc = px.line(df, x="Timestamp", y="Smoothing", color_discrete_sequence=['#14b8a6'])
        fig_acc.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with g2:
        st.subheader("Incident Distribution")
        dist = df['Prediction'].value_counts().reset_index()
        fig_pie = px.pie(dist, values='count', names='Prediction', hole=0.7, 
                         color_discrete_map={"Normal": "#3b82f6", "DoS": "#f97316", "Probe": "#eab308", "R2L": "#f43f5e", "U2R": "#8b5cf6"})
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Feed
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Network Forensic Log")
        def badge_color(val):
            color = "#f87171" if val < 0.5 else "#fbbf24" if val < 0.8 else "#2dd4bf"
            return f'color: {color}; font-weight: 700;'
        
        st.dataframe(
            df_view.tail(15).iloc[::-1].style.map(badge_color, subset=['Confidence']),
            use_container_width=True
        )
        st.download_button("Download Snapshot (CSV)", df.to_csv(index=False).encode('utf-8'), "ids_sim.csv", "text/csv", use_container_width=True)

    with c2:
        st.subheader("Incident Alerts")
        filtered_alerts = [a for a in st.session_state.alerts if a['type'] in f_atks]
        if not filtered_alerts:
            st.info("System monitoring... No active threats.")
        for a in reversed(filtered_alerts[-10:]):
            color = "#ef4444" if a['severity'] == "High" else "#f97316" if a['severity'] == "Medium" else "#eab308"
            st.markdown(f"""
            <div class="alert-item" style="border-left-color: {color};">
                <small style="color: #94a3b8;">{a['timestamp']}</small><br>
                <b style="color: #f1f5f9;">{a['type']} DETECTED</b><br>
                <span style="font-size: 0.85rem; color: {color};">Status: {a['severity']} | Confidence: {a['confidence']}</span>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Simulator Standby. Enable 'Start Live Stream' to begin telemetry.")

# --- AUTO UPDATE LOOP ---
if st.session_state.is_streaming:
    update_simulation()
    time.sleep(latency)
    st.rerun()
