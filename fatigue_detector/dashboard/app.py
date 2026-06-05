"""Streamlit dashboard for fatigue detection session analysis.

Connects to the same SQLite database used by the fatigue detector
and provides interactive visualizations of EAR, MAR, PERCLOS,
head pose angles, state transitions, and session summaries.

Run with: streamlit run dashboard/app.py
"""

import os
import sys

import streamlit as st
import sqlite3
import pandas as pd

# Plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Default DB path (sibling of dashboard/ directory)
DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'fatigue_log.db'
)


def load_data(db_path: str) -> pd.DataFrame:
    """Load event data from SQLite database into a DataFrame."""
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM events ORDER BY id ASC', conn)
    conn.close()

    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Driver Fatigue Monitor",
        layout="wide",
        page_icon="🚗"
    )

    st.title("🚗 Driver Fatigue Monitor — Session Dashboard")

    # ── Sidebar Controls ──
    db_path = st.sidebar.text_input("Database Path", DEFAULT_DB)

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Refresh (5s)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Usage:** Run the fatigue detector first to generate data, "
        "then open this dashboard to visualize the session."
    )

    # ── Load Data ──
    df = load_data(db_path)

    if df.empty:
        st.warning(
            "⚠️ No data found. Run the fatigue detector first to "
            "generate session data, then refresh this dashboard."
        )
        return

    # ── Summary Cards ──
    st.subheader("📊 Session Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    state_changes = df[df['event_type'] == 'STATE_CHANGE']
    col1.metric("Total Events", len(df))
    col2.metric("Drowsy Alerts",
                len(state_changes[state_changes['state'] == 'DROWSY']))
    col3.metric("Very Drowsy",
                len(state_changes[state_changes['state'] == 'VERY_DROWSY']))
    col4.metric("Asleep Alerts",
                len(state_changes[state_changes['state'] == 'ASLEEP']))
    col5.metric("Distractions", int(df['distracted'].sum()))
    col6.metric("Yawns", int(df['yawning'].sum()))

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    st.markdown("---")

    # ── EAR & MAR Timeline ──
    st.subheader("📈 EAR & MAR Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['ear'],
        name='EAR', line=dict(color='cyan', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['mar'],
        name='MAR', line=dict(color='orange', width=1.5)
    ))
    fig.add_hline(y=0.25, line_dash='dash', line_color='red',
                  annotation_text='EAR Threshold (0.25)')
    fig.add_hline(y=0.60, line_dash='dash', line_color='yellow',
                  annotation_text='MAR Threshold (0.60)')
    fig.update_layout(
        template='plotly_dark', height=400,
        xaxis_title='Time', yaxis_title='Ratio',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, width="stretch")

    # ── PERCLOS Trend ──
    st.subheader("📉 PERCLOS Trend")
    fig2 = px.line(df, x='timestamp', y='perclos', template='plotly_dark')
    fig2.add_hline(y=0.15, line_dash='dash', line_color='red',
                   annotation_text='Fatigue Threshold (0.15)')
    fig2.update_layout(height=300, xaxis_title='Time', yaxis_title='PERCLOS')
    st.plotly_chart(fig2, width="stretch")

    # ── Head Pose ──
    st.subheader("🔄 Head Pose Angles")
    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.line(df, x='timestamp', y='pitch',
                       title='Pitch (Nodding)', template='plotly_dark')
        fig3.add_hline(y=20, line_dash='dash', line_color='orange',
                       annotation_text='+20°')
        fig3.add_hline(y=-20, line_dash='dash', line_color='orange',
                       annotation_text='-20°')
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, width="stretch")

    with col2:
        fig4 = px.line(df, x='timestamp', y='yaw',
                       title='Yaw (Looking Away)', template='plotly_dark')
        fig4.add_hline(y=30, line_dash='dash', line_color='orange',
                       annotation_text='+30°')
        fig4.add_hline(y=-30, line_dash='dash', line_color='orange',
                       annotation_text='-30°')
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, width="stretch")

    st.markdown("---")

    # ── State Changes Table ──
    st.subheader("📋 State Change Log (Latest First)")
    if not state_changes.empty:
        display_cols = ['timestamp', 'state', 'ear', 'mar',
                        'perclos', 'pitch', 'yaw']
        st.dataframe(
            state_changes[display_cols].iloc[::-1].head(50),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No state changes recorded yet.")

    # ── Raw Data ──
    with st.expander("🗄️ View Raw Data (Latest First)"):
        st.dataframe(df.iloc[::-1].head(200), width="stretch", hide_index=True)

    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


if __name__ == '__main__':
    main()
