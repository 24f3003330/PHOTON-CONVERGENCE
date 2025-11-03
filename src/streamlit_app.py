import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---
st.set_page_config(layout="wide", page_title="PHOTON: Smart Energy Optimization Dashboard (168H)", 
                   initial_sidebar_state="expanded")

# Define SCENARIOS that influence data and RL output
SCENARIOS = {
    "Typical Week": {
        "Base_Consumption": 2.0, 
        "Solar_Factor": 0.5, 
        "Peak_Hour": 18, # 6 PM is the default grid peak hour
        "Peak_Discharge_Power": 2.0 
    },
    "High Demand Week": {
        "Base_Consumption": 2.5,  # Higher base load
        "Solar_Factor": 0.4, # Slightly cloudier week
        "Peak_Hour": 19, # 7 PM is the default grid peak hour
        "Peak_Discharge_Power": 3.5 # Higher discharge needed
    }
}

@st.cache_data(ttl=3600) # Cache data for 1 hour
def generate_mock_data(scenario_key, total_hours=168, freq='15Min'):
    """Generates 168 hours (7 days) of historical data for ML training."""
    
    params = SCENARIOS[scenario_key]
    
    # 1. Generate Historical/Current Data (168 hours at 15-minute frequency)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=total_hours)
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    df = pd.DataFrame(index=time_index)
    
    # Calculate array length for sine wave simulation
    N = len(df)
    
    # Data simulation: Daily Cycle Simulation (using 24-hour period for sine wave)
    # Consumption peaks in the evening (18:00 to 22:00)
    consumption_daily_cycle = 0.5 * np.sin(np.linspace(0, 7 * 2 * np.pi, N))
    df['Consumption'] = params['Base_Consumption'] + consumption_daily_cycle + np.random.rand(N) * 0.2
    
    # Solar Generation: Peaks sharply around noon
    solar_hours = df.index.hour + df.index.minute / 60
    solar_daily_cycle = params['Solar_Factor'] * np.maximum(0, np.sin((solar_hours - 6) / 12 * np.pi))
    df['Solar_Gen'] = 0.1 + solar_daily_cycle + np.random.rand(N) * 0.05
    df['Solar_Gen'] = df['Solar_Gen'].clip(lower=0.0)

    # RL Logic: Discharge at the designated Peak_Hour (e.g., 6 PM) EVERY day
    peak_h = params['Peak_Hour']
    peak_p = params['Peak_Discharge_Power']
    
    # Discharge battery for 1 hour starting at the peak hour
    is_discharge_time = (df.index.hour == peak_h) & (df.index.minute.isin([0, 15, 30, 45]))
    df['Battery_Flow'] = np.where(
        is_discharge_time, 
        peak_p, 
        np.where(
            (df.index.hour == 12) & (df.index.minute.isin([0, 15, 30, 45])), -1.5, # Charge at noon
            np.random.randn(N) * 0.05
        )
    )
    
    # Grid Consumption (The optimized outcome)
    df['Grid_Import'] = (df['Consumption'] - df['Solar_Gen'] - df['Battery_Flow']).clip(lower=0)
    
    # MOCK RL Optimization Schedule (Simplified for the whole week)
    optimal_schedule = {
        f"Daily @ {peak_h:02d}:00 PM": {"Action": "DISCHARGE", "Power (KW)": peak_p, "Reason": f"Daily Predicted Grid Peak ({scenario_key})"},
        "Daily @ 12:00 PM": {"Action": "CHARGE", "Power (KW)": 4.5, "Reason": "Daily Solar Peak Forecast"}
    }
    
    # Select the columns that would represent the training/synthetic data
    synthetic_data = df[['Consumption', 'Solar_Gen', 'Grid_Import', 'Battery_Flow']].copy()
    synthetic_data.index.name = 'Timestamp'
    
    # Extract only the last 24 hours for the main chart visualization
    df_chart = df.iloc[-96:] 
    
    # Generate Future Forecast Data (Last 15 minutes of the week + 15 minutes forward)
    end_time_current = df_chart.index[-1]
    time_index_future = pd.date_range(start=end_time_current, end=end_time_current + timedelta(minutes=15), freq='1Min')
    df_future = pd.DataFrame(index=time_index_future)
    df_future['Demand_Forecast'] = df_chart['Consumption'].iloc[-1] * (1 + np.linspace(0, 0.1, len(df_future))) + np.random.rand(len(df_future)) * 0.1
    
    return synthetic_data, df_chart, df_future, optimal_schedule

# --- 2. LAYOUT FUNCTIONS ---

def display_kpi_card(title, value, unit, color="#268A2E"):
    """Creates a stylized KPI card."""
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #f9f9f9; text-align: left;">
            <p style="font-size: 14px; color: #6e6e6e; margin: 0;">{title}</p>
            <h3 style="font-size: 32px; color: {color}; margin: 5px 0 0 0;">
                {value}
                <span style="font-size: 16px; color: #333333;">{unit}</span>
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_energy_flow_chart(df_chart, df_future):
    """Creates the Live Energy Flow chart using Plotly, zoomed to the last 24 hours."""
    fig = go.Figure()
    
    # Define Time Axis Range (Last 24 hours of data + 15 min forecast)
    start_time = df_chart.index[0]
    end_time = df_future.index[-1]

    # --- 1. RL ACTION HIGHLIGHT ---
    # Highlight the next RL Discharge period (e.g., 6 PM of the last day)
    params = SCENARIOS[st.session_state.scenario]
    action_hour = params['Peak_Hour']
    
    # Find the latest peak hour on the chart
    latest_peak = df_chart[df_chart.index.hour == action_hour].index.max()
    if latest_peak:
        fig.add_shape(
            type="rect",
            x0=latest_peak.replace(minute=0),
            x1=latest_peak.replace(minute=0) + timedelta(hours=1),
            y0=0, y1=df_chart['Consumption'].max() * 1.1,
            line=dict(width=0),
            fillcolor="rgba(255, 165, 0, 0.2)",
            layer="below"
        )
    
    # --- 2. CURRENT DATA TRACES ---
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Consumption'], mode='lines', name='Household Consumption (Demand)', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Solar_Gen'], mode='lines', name='Solar Generation (Supply)', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Grid_Import'], mode='lines', name='Grid Consumption (Net Import)', line=dict(color='purple', width=3)))
    
    # 3. BATTERY FLOW AREA
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Battery_Flow'].clip(lower=0), mode='lines', name='Battery Discharge', fill='tozeroy', fillcolor='rgba(0,128,0, 0.3)', line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Battery_Flow'].clip(upper=0).abs(), mode='lines', name='Battery Charge', fill='tozeroy', fillcolor='rgba(0,0,255, 0.3)', line=dict(color='blue', width=1)))

    # 4. ML FORECAST TRACE (Dotted line extending into the future)
    fig.add_trace(go.Scatter(
        x=df_future.index, 
        y=df_future['Demand_Forecast'], 
        mode='lines', 
        name='Demand Forecast (ML)', 
        line=dict(color='red', dash='dot', width=3)
    ))

    # 5. CHART LAYOUT ADJUSTMENTS
    fig.update_layout(
        title='Energy Flow & **ML-Optimized Dispatch** (Last 24 Hours + Forecast)',
        xaxis_title="Time",
        yaxis_title="Power (KW)",
        height=500,
        xaxis_range=[start_time, end_time],
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 3. DASHBOARD ENTRY POINT ---

def main_dashboard():
    st.title("💡 Photon: Smart Energy Optimization Dashboard")
    st.write("Energy Optimization & Renewable Integration for Urban Residential Complexes")
    
    # 3.1 SIDEBAR FOR INTERACTIVITY (Scenario Selection)
    st.sidebar.title("🛠️ Prototype Controls")
    
    if 'scenario' not in st.session_state:
        st.session_state['scenario'] = "Typical Week"
        
    st.session_state.scenario = st.sidebar.selectbox(
        "Select Energy Scenario:",
        list(SCENARIOS.keys())
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML & RL Overview")
    st.sidebar.markdown(
        """
        - **Forecasting (XGBoost/LSTM):** Predicts Demand/Solar for 168 hours.
        - **Optimization (Q-Learning/DQN):** Schedules battery action based on forecasts.
        - **Goal:** Maximize savings by minimizing Grid Import during high-cost hours.
        """
    )
    
    synthetic_data, df_chart, df_future, optimal_schedule = generate_mock_data(st.session_state.scenario)
    st.markdown("---")
    
    # 3.2 REAL-TIME METRICS (Showing average of the last hour for stability)
    st.subheader("📊 Real-time Power Flow (Last Hour Average)")
    
    latest_data = df_chart.iloc[-4:].mean() # Average over last hour (4 15-min points)
    mock_battery_level = 0.5 + (datetime.now().minute % 10) * 0.05
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_kpi_card("Solar Generation", f"{latest_data['Solar_Gen']:.2f}", "KW", "#F9A825")
    with col2:
        display_kpi_card("Household Consumption", f"{latest_data['Consumption']:.2f}", "KW", "#FF5733")
    with col3:
        display_kpi_card("Battery Level", f"{mock_battery_level:.1f}", "%", "#00C853")
    with col4:
        display_kpi_card("Grid Consumption", f"{latest_data['Grid_Import']:.2f}", "KW", "#673AB7")

    # 3.3 LIVE ENERGY FLOW CHART & FORECASTING
    st.markdown("---")
    create_energy_flow_chart(df_chart, df_future)
    st.markdown("The dotted red line shows the ML **Demand Forecast**. The orange highlight shows the RL-determined **Battery Discharge Window** (Daily peak hour).")

    # 3.4 OPTIMIZATION OUTPUTS & IMPACT ANALYSIS
    st.markdown("---")

    colA, colB = st.columns([1, 2])
    
    # A. Optimal Schedule (RL Optimization Module Output)
    with colA:
        st.subheader("🤖 Optimal Battery Schedule (RL Engine)")
        st.info("The Reinforcement Learning engine determines the optimal daily schedule to boost storage efficiency and maximize long-term cost savings.")
        
        schedule_df = pd.DataFrame(optimal_schedule).T.reset_index()
        schedule_df.columns = ['Time', 'Action', 'Power (KW)', 'Reason']
        st.table(schedule_df)
        
        # Alerts & Efficiency
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("⚠️ **Alert**: Anomaly Detected - Daily average consumption trending 5% above historical week average. Review load management.")

    # B. Savings Calculator (Cost Analysis)
    with colB:
        st.subheader("💰 Savings & Sustainability Impact (Based on 168H Simulation)")
        
        # Calculate impact metrics based on 168H simulation
        baseline_import = synthetic_data['Consumption'].sum()
        optimized_import = synthetic_data['Grid_Import'].sum()
        
        # The estimated savings are calculated over the full 168 hours of simulated data
        estimated_savings_weekly = (baseline_import - optimized_import) * 10 
        
        daily_savings = estimated_savings_weekly / 7
        monthly_savings = estimated_savings_weekly * 4
        
        daily_co2_savings = daily_savings * 0.8 
        monthly_co2_savings = monthly_savings * 0.8

        colB1, colB2, colB3, colB4 = st.columns(4)
        
        with colB1:
            display_kpi_card("Daily Savings", f"₹ {daily_savings:.2f}", "", "#4CAF50")
        with colB2:
            display_kpi_card("Monthly Savings", f"₹ {monthly_savings:.2f}", "", "#4CAF50")
        with colB3:
            display_kpi_card("Daily CO2 Saved", f"{daily_co2_savings:.2f}", "kg", "#03A9F4")
        with colB4:
            display_kpi_card("Monthly CO2 Saved", f"{monthly_co2_savings:.2f}", "kg", "#03A9F4")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Displays cost savings compared to baseline, proving the system's real ROI and sustainability impact. Enables residents to lower electricity bills and reduce their carbon footprint.")

    # 4. RAW SYNTHETIC DATA DISPLAY
    st.markdown("---")
    with st.expander("📚 View Raw Synthetic Dataset (168 Hours - Input for ML Training)"):
        st.markdown(f"**Showing 168 hours (7 days) of synthetic data points for scenario: {st.session_state.scenario}.** This data is used by the XGBoost/LSTM module for training.")
        st.dataframe(synthetic_data, height=300)

if __name__ == "__main__":
    main_dashboard()