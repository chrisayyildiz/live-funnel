# streamlit_funnel_dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Config
RAW_DATA_PATH = "simulated_funnel_events.csv"
FUNNEL_METRICS_PATH = "funnel_metrics.csv"

# Load data
@st.cache_data
def load_data():
    df_events = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    df_funnel = pd.read_csv(FUNNEL_METRICS_PATH, index_col=0)
    return df_events, df_funnel

# Filter data
def filter_data(df, device_filter, referrer_filter, date_range):
    df_filtered = df.copy()
    
    if device_filter != "All":
        df_filtered = df_filtered[df_filtered["device_type"] == device_filter]
    
    if referrer_filter != "All":
        df_filtered = df_filtered[df_filtered["referrer"] == referrer_filter]
    
    df_filtered = df_filtered[
        (df_filtered["timestamp"].dt.date >= date_range[0]) &
        (df_filtered["timestamp"].dt.date <= date_range[1])
    ]
    
    return df_filtered

# Plot funnel using Plotly
def plot_funnel(funnel_df):
    fig = go.Figure()

    fig.add_trace(go.Funnel(
        y=funnel_df.index.tolist(),
        x=funnel_df["users"].tolist(),
        textinfo="value+percent previous+percent initial",
        marker={"color": [
            "#FDE8D0", "#FDD1A0", "#FDB870", "#FD9F50", "#FD8040", "#FD6030"
        ]}
    ))

    fig.update_layout(
        title="Funnel Metrics",
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit app ---
st.set_page_config(page_title="Live Funnel Dashboard", layout="centered")

st.title("📊 Live Funnel Analysis Dashboard")
st.caption("Conceptual real-time funnel visualisation for e-commerce conversion optimisation")

# Load data
df_events, df_funnel = load_data()

# Sidebar filters
st.sidebar.header("Filters")

device_filter = st.sidebar.selectbox(
    "Device Type", ["All"] + sorted(df_events["device_type"].unique().tolist())
)

referrer_filter = st.sidebar.selectbox(
    "Referrer", ["All"] + sorted(df_events["referrer"].unique().tolist())
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=[df_events["timestamp"].dt.date.min(), df_events["timestamp"].dt.date.max()],
    min_value=df_events["timestamp"].dt.date.min(),
    max_value=df_events["timestamp"].dt.date.max()
)

# Apply filters
df_filtered = filter_data(df_events, device_filter, referrer_filter, date_range)

# Recompute funnel metrics from filtered data
users_per_step = df_filtered.groupby("event_type")["user_id"].nunique().reindex([
    "view_homepage", "view_category", "view_product",
    "add_to_cart", "checkout", "purchase"
]).fillna(0).astype(int)

conversion_rates = users_per_step / users_per_step.shift(1)
conversion_rates.iloc[0] = None
drop_off_rates = 1 - conversion_rates
anomaly_flags = drop_off_rates > 0.3

funnel_df_filtered = pd.DataFrame({
    "users": users_per_step,
    "conversion_rate": conversion_rates,
    "drop_off_rate": drop_off_rates,
    "anomaly": anomaly_flags
})

# Show anomaly warnings
if funnel_df_filtered["anomaly"].any():
    st.error("⚠️ Anomaly detected! High drop-off rate at one or more stages.")
else:
    st.success("✅ No anomalies detected.")

# Show funnel chart (Plotly)
plot_funnel(funnel_df_filtered)

# Optional — show raw data
with st.expander("Show Raw Events Data"):
    st.dataframe(df_filtered)

# Optional — rerun ETL button (conceptual)
if st.button("🔄 Rerun ETL & Refresh Dashboard"):
    st.warning("Manual rerun concept — please execute the ETL script and refresh Streamlit app.")
