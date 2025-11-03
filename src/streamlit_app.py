import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---
st.set_page_config(layout="wide", page_title="PHOTON: Smart Energy Optimization Dashboard", 
                   initial_sidebar_state="expanded")

# Define SCENARIOS that influence data and RL output
SCENARIOS = {
    "Standard Day": {
        "Base_Consumption": 2.0, 
        "Solar_Factor": 0.4, 
        "RL_Discharge_Hour": 18, # 6 PM discharge
        "RL_Discharge_Power": 1.0 
    },
    "High Peak Day": {
        "Base_Consumption": 2.5,  # Higher base load
        "Solar_Factor": 0.3, # Less sun
        "RL_Discharge_Hour": 19, # 7 PM discharge to counter higher peak
        "RL_Discharge_Power": 3.0 # Higher discharge power
    }
}

@st.cache_data(ttl=5) # Cache data for 5 seconds for re-run on scenario change
def generate_mock_data(scenario_key):
    """Generates mock time-series data based on the selected scenario."""
    
    # Use selected scenario parameters
    params = SCENARIOS[scenario_key]
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=60)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1Min')
    df = pd.DataFrame(index=time_index)
    
    # Household Consumption (Demand Forecasting Input)
    fluctuation = 0.5 * np.sin(np.linspace(0, 4*np.pi, len(df)))
    df['Consumption'] = params['Base_Consumption'] + fluctuation + np.random.rand(len(df)) * 0.2
    
    # Solar Generation (Solar Forecasting Input)
    solar_curve = params['Solar_Factor'] * np.sin(np.linspace(0, np.pi, len(df))) * (df.index.hour % 24).isin(range(10, 17))
    df['Solar_Gen'] = 0.2 + solar_curve + np.random.rand(len(df)) * 0.1
    df['Solar_Gen'] = df['Solar_Gen'].clip(lower=0.0)

    # Battery and Grid Simulation (MOCK RL Strategy Implementation)
    discharge_h = params['RL_Discharge_Hour']
    discharge_p = params['RL_Discharge_Power']
    
    # Simulate RL action: Discharge the battery during the peak hour
    df['Battery_Flow'] = np.where(
        (df.index.hour == discharge_h) & (df.index.minute < 30), discharge_p, # Discharge (Positive = Export from Battery)
        np.where(
            (df.index.hour == 12) & (df.index.minute < 30), -1.5, # Charge (Negative = Import to Battery)
            np.random.randn(len(df)) * 0.05
        )
    )
    
    # Grid Consumption (The optimized outcome)
    df['Grid_Import'] = df['Consumption'] - df['Solar_Gen'] - df['Battery_Flow']
    df['Grid_Import'] = df['Grid_Import'].clip(lower=0)
    
    # MOCK RL Optimization Schedule (Updated based on scenario)
    optimal_schedule = {
        f"{discharge_h:02d}:00 PM": {"Action": "DISCHARGE", "Power (KW)": discharge_p, "Reason": f"Predicted Grid Peak ({scenario_key})"},
        "12:00 PM": {"Action": "CHARGE", "Power (KW)": 4.5, "Reason": "Solar Peak Forecast"},
        "09:00 PM": {"Action": "IDLE", "Power (KW)": 0.0, "Reason": "Low Demand Period"}
    }
    
    return df, optimal_schedule

# --- 2. LAYOUT FUNCTIONS (Minor improvements for clarity) ---

def display_kpi_card(title, value, unit, color="#268A2E"):
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

def create_energy_flow_chart(df, optimal_schedule):
    """Creates the Live Energy Flow chart and highlights the RL action."""
    fig = go.Figure()

    # Highlight the RL Discharge Period for visual clarity
    discharge_hour = SCENARIOS[st.session_state.scenario]['RL_Discharge_Hour']
    
    # Add a subtle shape to highlight the optimization action time window
    fig.add_shape(
        type="rect",
        x0=datetime.now() - timedelta(minutes=60) + timedelta(hours=discharge_hour),
        x1=datetime.now() - timedelta(minutes=60) + timedelta(hours=discharge_hour, minutes=30),
        y0=0, y1=df['Consumption'].max() * 1.1,
        line=dict(width=0),
        fillcolor="rgba(255, 165, 0, 0.1)",
        layer="below"
    )
    
    # Add traces for the different energy flows
    fig.add_trace(go.Scatter(x=df.index, y=df['Consumption'], mode='lines', name='Household Consumption (Demand)', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Solar_Gen'], mode='lines', name='Solar Generation (Supply)', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Grid_Import'], mode='lines', name='Grid Consumption (Net Import)', line=dict(color='purple', width=3)))
    
    # Battery Flow is shown via the discharge/charge fill
    fig.add_trace(go.Scatter(x=df.index, y=df['Battery_Flow'].clip(lower=0), mode='lines', name='Battery Discharge', fill='tozeroy', fillcolor='rgba(0,128,0, 0.3)', line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Battery_Flow'].clip(upper=0).abs(), mode='lines', name='Battery Charge', fill='tozeroy', fillcolor='rgba(0,0,255, 0.3)', line=dict(color='blue', width=1)))

    # MOCK FORECAST (Dotted line for predictive insights)
    forecast_points = int(len(df) * 0.1)
    mock_forecast = df['Consumption'].iloc[-forecast_points:].values * 1.05 
    fig.add_trace(go.Scatter(x=df.index[-forecast_points:], y=mock_forecast, mode='lines', name='Demand Forecast (ML)', line=dict(color='red', dash='dot')))

    fig.update_layout(
        title='Live Energy Flow & **ML-Optimized Dispatch** (Last Hour)',
        xaxis_title="Time",
        yaxis_title="Power (KW)",
        height=500,
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
    
    # Initialize session state for scenario
    if 'scenario' not in st.session_state:
        st.session_state['scenario'] = "Standard Day"
        
    st.session_state.scenario = st.sidebar.selectbox(
        "Select Energy Scenario:",
        list(SCENARIOS.keys()),
        key='scenario'
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML & RL Overview")
    st.sidebar.markdown(
        """
        - **Forecasting (XGBoost/LSTM):** Predicts Demand/Solar.
        - **Optimization (Q-Learning/DQN):** Schedules battery action based on forecasts.
        - **Goal:** Maximize savings by minimizing Grid Import during high-cost hours.
        """
    )
    
    df, optimal_schedule = generate_mock_data(st.session_state.scenario)

    # 3.2 REAL-TIME METRICS
    st.markdown("---")
    [cite_start]st.subheader("📊 Real-time Power Flow [cite: 19]")
    
    latest_data = df.iloc[-1]
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

    # [cite_start]3.3 LIVE ENERGY FLOW CHART & FORECASTING [cite: 20]
    st.markdown("---")
    create_energy_flow_chart(df, optimal_schedule)
    [cite_start]st.markdown("The dotted red line shows the ML **Demand Forecast**[cite: 20]. The orange highlight shows the RL-determined **Battery Discharge Window**.")

    # [cite_start]3.4 OPTIMIZATION OUTPUTS & IMPACT ANALYSIS [cite: 21]
    st.markdown("---")

    colA, colB = st.columns([1, 2])
    
    # A. Optimal Schedule (RL Optimization Module Output)
    with colA:
        st.subheader("🤖 Optimal Battery Schedule (RL Engine)")
        [cite_start]st.info("The Reinforcement Learning engine schedules optimal charging/discharging to boost storage efficiency and maximize savings[cite: 39].")
        
        schedule_df = pd.DataFrame(optimal_schedule).T.reset_index()
        schedule_df.columns = ['Time', 'Action', 'Power (KW)', 'Reason']
        st.table(schedule_df)
        
        # [cite_start]Alerts & Efficiency [cite: 48]
        st.markdown("<br>", unsafe_allow_html=True)
        [cite_start]st.warning("⚠️ **Alert**: Anomaly Detected - Consumption is 10% above 24-hr historical average. Check for unoptimized appliances. [cite: 48]")

    # [cite_start]B. Savings Calculator (Cost Analysis) [cite: 47]
    with colB:
        st.subheader("💰 Savings & Sustainability Impact")
        
        # Calculate impact metrics
        # Mock savings data: Baseline Grid Import (if no solar/battery used)
        baseline_import = df['Consumption'].sum()
        optimized_import = df['Grid_Import'].sum()
        
        # Assuming a cost of 10 Rs/KWh
        estimated_savings = (baseline_import - optimized_import) * 10 
        
        daily_savings = estimated_savings * 24 # Extrapolate to daily
        daily_co2_savings = daily_savings * 0.8 # Mock calculation (0.8 kg CO2/Rs)
        monthly_savings = daily_savings * 30
        monthly_co2_savings = daily_co2_savings * 30

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
        [cite_start]st.write("Displays cost savings compared to baseline, proving the system's real ROI and sustainability impact[cite: 47]. [cite_start]Enables residents to lower electricity bills and reduce their carbon footprint[cite: 147].")

if __name__ == "__main__":
    main_dashboard()