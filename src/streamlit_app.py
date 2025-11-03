import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---

# Use a wide layout for the dashboard look
st.set_page_config(layout="wide", page_title="PHOTON: Smart Energy Optimization Dashboard")

# Define the ML-based optimal battery schedule (MOCK RL OUTPUT)
OPTIMAL_SCHEDULE = {
    "12:00 PM": {"Action": "CHARGE", "Power (KW)": 4.5, "Reason": "Solar Peak Forecast"},
    "06:00 PM": {"Action": "DISCHARGE", "Power (KW)": 3.2, "Reason": "Predicted Grid Peak"},
    "09:00 PM": {"Action": "IDLE", "Power (KW)": 0.0, "Reason": "Low Demand Period"},
}

@st.cache_data
def generate_mock_data(minutes=60, freq='1Min'):
    """Generates mock time-series data for the energy flow chart."""
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=minutes)
    
    # Create the time index
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    df = pd.DataFrame(index=time_index)

    # Household Consumption (Baseline 2.20 KW)
    df['Consumption'] = 2.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, len(df))) + np.random.rand(len(df)) * 0.2
    
    # Solar Generation (Baseline 0.31 KW)
    solar_base = 0.2
    solar_curve = 0.4 * np.sin(np.linspace(0, np.pi, len(df))) * (df.index.hour % 24).isin(range(10, 17))
    df['Solar_Gen'] = solar_base + solar_curve + np.random.rand(len(df)) * 0.1
    df['Solar_Gen'] = df['Solar_Gen'].clip(lower=0.0)

    # Battery and Grid Simulation (MOCK RL Strategy Implementation)
    df['Battery_Flow'] = np.where(
        (df.index.hour == 12) & (df.index.minute < 30), -1.5, # Mock CHARGE
        np.where(
            (df.index.hour == 18) & (df.index.minute < 30), 1.0, # Mock DISCHARGE
            np.random.randn(len(df)) * 0.05
        )
    )
    
    # Grid Consumption (The crucial calculation)
    df['Grid_Import'] = df['Consumption'] - df['Solar_Gen'] + df['Battery_Flow']
    df['Grid_Import'] = df['Grid_Import'].clip(lower=0)
    
    return df

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

def create_energy_flow_chart(df):
    """Creates the Live Energy Flow chart using Plotly."""
    fig = go.Figure()

    # Add traces for the different energy flows
    fig.add_trace(go.Scatter(x=df.index, y=df['Solar_Gen'], mode='lines', name='Solar Generation', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Consumption'], mode='lines', name='Household Consumption', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Grid_Import'], mode='lines', name='Grid Consumption', line=dict(color='purple', width=2)))
    
    # Battery Flow: Discharge (Positive) and Charge (Negative)
    fig.add_trace(go.Scatter(x=df.index, y=df['Battery_Flow'].clip(lower=0), mode='lines', name='Battery Discharge', fill='tozeroy', fillcolor='rgba(0,128,0, 0.2)', line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Battery_Flow'].clip(upper=0).abs(), mode='lines', name='Battery Charge', fill='tozeroy', fillcolor='rgba(0,0,255, 0.2)', line=dict(color='blue', width=1)))

    # MOCK FORECAST (Dotted line for predictive insights - XGBoost/LSTM output)
    forecast_points = int(len(df) * 0.1)
    mock_forecast = df['Consumption'].iloc[-forecast_points:].values * 1.1 
    fig.add_trace(go.Scatter(x=df.index[-forecast_points:], y=mock_forecast, mode='lines', name='Demand Forecast (ML)', line=dict(color='red', dash='dot')))

    fig.update_layout(
        title='Live Energy Flow (Last Hour) & Forecast',
        xaxis_title="Time",
        yaxis_title="Power (KW)",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 3. DASHBOARD ENTRY POINT ---

def main_dashboard():
    st.title("💡 Photon: Smart Energy Optimization Dashboard")
    st.write("Energy Optimization & Renewable Integration for Urban Residential Complexes")
    st.markdown("---")

    df = generate_mock_data(minutes=60, freq='1Min')

    # Calculate real-time metrics
    latest_data = df.iloc[-1]
    mock_battery_level = (datetime.now().minute % 10) * 0.3
    
    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_kpi_card("Solar Generation", f"{latest_data['Solar_Gen']:.2f}", "KW", "#F9A825")
    with col2:
        display_kpi_card("Household Consumption", f"{latest_data['Consumption']:.2f}", "KW", "#FF5733")
    with col3:
        display_kpi_card("Battery Level", f"{mock_battery_level:.1f}", "%", "#00C853")
    with col4:
        display_kpi_card("Grid Consumption", f"{latest_data['Grid_Import']:.2f}", "KW", "#673AB7")

    st.markdown("---")
    
    # LIVE ENERGY FLOW CHART
    create_energy_flow_chart(df)

    st.markdown("---")

    # OPTIMIZATION OUTPUTS & COST ANALYSIS
    colA, colB = st.columns([1, 2])
    
    # A. Optimal Schedule (RL Optimization Module Output)
    with colA:
        st.subheader("🤖 Optimal Battery Schedule (RL Engine)")
        st.info("The RL engine determines the optimal charge/discharge schedule to maximize cost savings by reducing grid imports during peak hours.")
        
        schedule_df = pd.DataFrame(OPTIMAL_SCHEDULE).T.reset_index()
        schedule_df.columns = ['Time', 'Action', 'Power (KW)', 'Reason']
        st.table(schedule_df)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("⚠️ **Alert**: Anomaly Detected - Unusually High Morning Grid Usage. Review Household Load.")


    # B. Savings Calculator (Cost Analysis)
    with colB:
        st.subheader("💰 Savings & Sustainability Impact")
        
        # Mock savings calculation
        baseline_cost = df['Consumption'].sum() * 10 
        optimized_cost = baseline_cost - 5.0 
        
        daily_savings = (baseline_cost - optimized_cost) * 24 
        daily_co2_savings = daily_savings * 0.8 
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
        st.write("This section proves the system's **Real ROI** and **Sustainability Impact** by minimizing energy wastage and optimizing renewable use.")

if __name__ == "__main__":
    main_dashboard()