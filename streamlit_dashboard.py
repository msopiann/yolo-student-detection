import streamlit as st
import pandas as pd
import json
import plotly.graph_objs as go
import os

# === Load Data ===
DATA_FOLDER = "dashboard_data"
ACTIVITY_STATS_FILE = os.path.join(DATA_FOLDER, "activity_stats.json")
TIME_SERIES_FILE = os.path.join(DATA_FOLDER, "time_series.csv")

st.set_page_config(page_title="Classroom Activity Dashboard", layout="wide")
st.title("📊 Classroom Activity Dashboard")

# === Load JSON data ===
try:
    with open(ACTIVITY_STATS_FILE) as f:
        stats = json.load(f)
except FileNotFoundError:
    st.error("❌ File 'activity_stats.json' not found.")
    st.stop()

activity_count = stats["counts"]
average_durations = stats["durations"]

activities = list(activity_count.keys())
counts = [activity_count[act] for act in activities]
durations = [average_durations[act] for act in activities]

colors_hex = {
    "hand-raising": "#2EB88A",
    "read": "#2662D9",
    "write": "#E88C30",
    "inactive": "#E13670",
}
colors_plot = [colors_hex.get(act, "#888888") for act in activities]

# === Charts ===
# Combo Chart
st.subheader("📈 Activity Counts & Average Durations")
combo_fig = go.Figure()
combo_fig.add_trace(
    go.Bar(x=activities, y=counts, name="Occurrences", marker_color=colors_plot)
)
combo_fig.add_trace(
    go.Scatter(
        x=activities,
        y=durations,
        name="Avg Duration (s)",
        mode="lines+markers",
        marker=dict(color="black"),
    )
)
combo_fig.update_layout(
    yaxis=dict(title="Occurrences"),
    yaxis2=dict(title="Duration (s)", overlaying="y", side="right"),
    title="Activity Counts & Avg Durations",
    barmode="group",
)
st.plotly_chart(combo_fig, use_container_width=True)

# Donut Chart
st.subheader("🍩 Activity Distribution")
donut_fig = go.Figure(
    data=[
        go.Pie(
            labels=activities,
            values=counts,
            hole=0.4,
            marker=dict(colors=colors_plot),
        )
    ]
)
donut_fig.update_layout(title="Activity Distribution (by Count)")
st.plotly_chart(donut_fig, use_container_width=True)

# Gauge Indicators
st.subheader("🎯 Average Duration by Activity")
cols = st.columns(len(activities))
for i, act in enumerate(activities):
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=average_durations[act],
            title={"text": act.capitalize()},
            gauge={
                "axis": {"range": [0, max(durations) + 5]},
                "bar": {"color": colors_hex.get(act, "#888888")},
            },
        )
    )
    gauge_fig.update_layout(height=250, margin=dict(t=0, b=0, l=10, r=10))
    cols[i].plotly_chart(gauge_fig, use_container_width=True)

# Time-series Timeline (optional)
if os.path.exists(TIME_SERIES_FILE):
    st.subheader("⏱️ Activity Timeline")
    df_time = pd.read_csv(TIME_SERIES_FILE)

    timeline_fig = go.Figure()
    for act in activities:
        if act != "inactive":
            timeline_fig.add_trace(
                go.Scatter(
                    x=df_time[df_time["activity"] == act]["frame"],
                    y=df_time[df_time["activity"] == act]["track_id"],
                    mode="markers",
                    name=act,
                    marker=dict(color=colors_hex.get(act, "#888888")),
                )
            )
    timeline_fig.update_layout(
        title="Activity Over Time (Frame Index)",
        xaxis_title="Frame",
        yaxis_title="Track ID",
        height=800,
    )
    st.plotly_chart(timeline_fig, use_container_width=True)
else:
    st.info("ℹ️ No time-series data available.")
