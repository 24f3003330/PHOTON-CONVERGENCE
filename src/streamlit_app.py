import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---
# Set the page configuration just once
st.set_page_config(layout="wide", page_title="PHOTON: Smart Energy Optimization Dashboard",
                    initial_sidebar_state="expanded")

# Define SCENARIOS
SCENARIOS = {
    "Typical Week": {
        "Base_Consumption": 2.0,
        "Solar_Factor": 0.5,
        "Peak_Hour": 18,
        "Peak_Discharge_Power": 2.0
    },
    "High Demand Week": {
        "Base_Consumption": 2.5,
        "Solar_Factor": 0.4,
        "Peak_Hour": 19,
        "Peak_Discharge_Power": 3.5
    }}

@st.cache_data(ttl=3600) # Cache data for 1 hour
def generate_mock_data(scenario_key, total_hours=168, freq='15Min'):
    """Generates 168 hours (7 days) of synthetic data and 30-min future forecast."""

    params = SCENARIOS[scenario_key]

    # 1. Generate FULL DATA (168 hours at 15-minute frequency)
    end_time_fixed = datetime(2025, 11, 7, 0, 0, 0)
    start_time_fixed = end_time_fixed - timedelta(hours=total_hours)

    time_index = pd.date_range(start=start_time_fixed, end=end_time_fixed, freq=freq, inclusive='left')
    df = pd.DataFrame(index=time_index)
    N = len(df)

    # Data simulation: Daily Cycle Simulation
    consumption_daily_cycle = 0.5 * np.sin(np.linspace(0, 7 * 2 * np.pi, N))
    df['Consumption'] = params['Base_Consumption'] + consumption_daily_cycle + np.random.rand(N) * 0.2

    solar_hours = df.index.hour + df.index.minute / 60
    solar_daily_cycle = params['Solar_Factor'] * np.maximum(0, np.sin((solar_hours - 6) / 12 * np.pi))
    df['Solar_Gen'] = 0.1 + solar_daily_cycle + np.random.rand(N) * 0.05
    df['Solar_Gen'] = df['Solar_Gen'].clip(lower=0.0)

    # RL Logic and Net Flow Calculations (unchanged)
    peak_h = params['Peak_Hour']
    peak_p = params['Peak_Discharge_Power']
    is_discharge_time = (df.index.hour == peak_h) & (df.index.minute.isin([0, 15, 30, 45]))
    df['Battery_Flow'] = np.where(
        is_discharge_time, peak_p,
        np.where((df.index.hour == 12) & (df.index.minute.isin([0, 15, 30, 45])), -1.5,
            np.random.randn(N) * 0.05)
    )
    df['Net_Grid_Flow'] = df['Consumption'] - df['Solar_Gen'] - df['Battery_Flow']
    df['Grid_Import'] = df['Net_Grid_Flow'].clip(lower=0)
    df['Net_Energy'] = df['Solar_Gen'] + df['Battery_Flow'] - df['Consumption']
    
    # MOCK RL Optimization Schedule
    optimal_schedule = {
        f"Daily @ {peak_h:02d}:00 PM": {"Action": "DISCHARGE", "Power (KW)": peak_p, "Reason": f"Daily Predicted Grid Peak ({scenario_key})"},
        "Daily @ 12:00 PM": {"Action": "CHARGE", "Power (KW)": 4.5, "Reason": "Daily Solar Peak Forecast"}
    }

    # 2. Generate Future Forecast Data (30 minutes)
    end_time_fixed = df.index[-1] + timedelta(minutes=15)
    time_index_future = pd.date_range(start=end_time_fixed, end=end_time_fixed + timedelta(minutes=30), freq='1Min')
    df_future = pd.DataFrame(index=time_index_future, columns=['Demand_Forecast', 'Solar_Forecast'])

    last_consumption = df['Consumption'].iloc[-1]
    last_solar = df['Solar_Gen'].iloc[-1]
    forecast_factor_demand = (1 + np.linspace(0, 0.1, len(df_future)))
    df_future['Demand_Forecast'] = last_consumption * forecast_factor_demand + np.random.rand(len(df_future)) * 0.1
    df_future['Solar_Forecast'] = last_solar * (1 + np.linspace(0, 0.05, len(df_future))) + np.random.rand(len(df_future)) * 0.05

    synthetic_data = df[['Consumption', 'Solar_Gen', 'Grid_Import', 'Battery_Flow', 'Net_Energy']].copy()
    synthetic_data.index.name = 'Timestamp'

    return synthetic_data, df_future, optimal_schedule

# --- 2. LAYOUT UTILITIES ---
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

def create_energy_flow_chart(df_full, df_future):
    """Creates the energy flow chart with a small scroll panel at the bottom."""
    fig = go.Figure()
    end_time_full = df_future.index[-1]
    max_power = df_full['Consumption'].max() * 1.1
    
    # RL Highlight (unchanged logic)
    params = SCENARIOS[st.session_state.scenario]
    action_hour = params['Peak_Hour']
    latest_peak = df_full[df_full.index.date == df_full.index.date.max()]
    latest_peak = latest_peak[latest_peak.index.hour == action_hour].index.max()
    if latest_peak:
        fig.add_shape(
            type="rect", x0=latest_peak.replace(minute=0), x1=latest_peak.replace(minute=0) + timedelta(hours=1),
            y0=0, y1=max_power, line=dict(width=0), fillcolor="rgba(255, 165, 0, 0.2)", layer="below"
        )

    # DATA TRACES
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Battery_Flow'].clip(lower=0), mode='lines', name='Battery Discharge', fill='tozeroy', fillcolor='rgba(0,128,0, 0.3)', line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Battery_Flow'].clip(upper=0).abs(), mode='lines', name='Battery Charge', fill='tozeroy', fillcolor='rgba(0,0,255, 0.3)', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Solar_Gen'], mode='lines', name='Solar Generation (Supply)', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Consumption'], mode='lines', name='Household Consumption (Demand)', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Grid_Import'], mode='lines', name='Grid Consumption (Net Import)', line=dict(color='purple', width=3)))

    # ML FORECAST TRACE (30 min)
    fig.add_trace(go.Scatter(
        x=df_future.index, y=df_future['Demand_Forecast'], mode='lines', name='Demand Forecast (ML) - 30 Min',
        line=dict(color='red', dash='dot', width=3), showlegend=True,
    ))

    # CHART LAYOUT ADJUSTMENTS
    initial_view_start = df_full.index[-96] if len(df_full) >= 96 else df_full.index[0]

    fig.update_layout(
        title='Energy Flow & **ML-Optimized Dispatch** (24H Default View)',
        xaxis_title="Time", yaxis_title="Power (KW)", height=550,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False)
    )

    # Scroll Panel Implementation (Fixed Plotly error)
    fig.update_xaxes(
        range=[initial_view_start, end_time_full],
        rangeslider_visible=True,
        rangeslider_thickness=0.08,
        rangeslider=dict(
            bgcolor="#444444", bordercolor="gray",
            yaxis=dict(rangemode="fixed", range=[0, max_power])
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=24, label="24H", step="hour", stepmode="backward"),
                dict(count=3, label="3D", step="day", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_yaxes(range=[0, max_power])
    st.plotly_chart(fig, use_container_width=True)

# --- 3. PAGE FUNCTIONS ---

def page_dashboard(synthetic_data, df_future):
    """Displays the main chart, real-time metrics, and general info."""
    st.title("💡 Dashboard: Real-Time Monitoring & Dispatch")
    st.write("Live tracking of your energy generation, consumption, and storage, plus optimized flow.")
    st.markdown("---")
    
    ## 📊 Real-Time Energy Monitoring (Based on image_05e321.jpg)
    st.header("Real-Time Energy Monitoring")
    st.write("Live tracking of your energy generation, consumption, and storage")
    if st.button("Refresh 🔄", key="refresh_monitor"):
        st.experimental_rerun() # Simple way to simulate refresh

    st.markdown(
        """
        <div style="background-color: #f0f0f5; padding: 15px; border-radius: 10px; margin-top: 10px;">
        <h4 style="margin-top: 0;">About Real-Time Monitoring</h4>
        <p>PHOTON's real-time monitoring provides instant visibility into your energy system. Track solar generation patterns, monitor consumption demands, and observe battery storage levels as they fluctuate throughout the day.</p>
        <ul>
            <li>**Solar Generation:** Real-time output from your solar panels in kilowatt-hours</li>
            <li>**Energy Demand:** Current consumption across your building</li>
            <li>**Net Energy:** Surplus (positive) or deficit (negative) balance</li>
            <li>**Battery Level:** Current charge state and percentage</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    latest_data = synthetic_data.iloc[-1]
    mock_battery_level = 75 + (datetime.now().minute % 10) * 0.5
    mock_net_energy_flow = (latest_data['Solar_Gen'] - latest_data['Consumption'] + latest_data['Battery_Flow']).round(2)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_kpi_card("Solar Generation (Now)", f"{latest_data['Solar_Gen']:.2f}", "kW", "#F9A825")
    with col2:
        display_kpi_card("Energy Demand (Now)", f"{latest_data['Consumption']:.2f}", "kW", "#FF5733")
    with col3:
        display_kpi_card("Net Energy Flow (Now)", f"{mock_net_energy_flow:.2f}", "kW", "#00C853" if mock_net_energy_flow > 0 else "#673AB7")
    with col4:
        display_kpi_card("Battery Level (Now)", f"{mock_battery_level:.1f}", "%", "#00C853")
    
    st.markdown("---")
    
    ## 📈 Live Energy Flow Chart & Forecast
    create_energy_flow_chart(synthetic_data, df_future)
    st.markdown("Use the **small scroll panel at the bottom** to easily slide and view the next 24 hours of data across the full 168-hour timeline.")


def page_forecast(df_future):
    """Displays the forecast model explanation and data."""
    st.title("☀️ Forecast")
    st.write("AI-powered 24 hour prediction of solar generation and demand")
    st.markdown("---")
    
    ## 🤖 Model Explained
    st.markdown(
        """
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #ffffff;">
        <h4 style="margin-top: 0;">How it works</h4>
        <p style="margin: 0; font-size: 14px;">A gradient boosting model uses the last 24 hours and time features to predict the next 24 hours. Use this to plan battery usage and grid interactions.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>")

    ## 📈 Next 30 Minutes Prediction
    st.subheader("Next 30 Minutes Prediction")
    forecast_display = df_future.head(5).reset_index()
    forecast_display.columns = ['Time', 'Demand Forecast (kW)', 'Solar Forecast (kW)']
    forecast_display['Time'] = forecast_display['Time'].dt.strftime('%H:%M:%S')
    st.table(forecast_display)
    st.markdown("The dotted red line on the **Dashboard** shows this demand forecast.")


def page_optimization(synthetic_data, optimal_schedule):
    """Displays the battery optimization schedule and impact analysis."""
    st.title("🔋 Battery Optimization")
    st.write("Reinforcement learning schedule and savings impact for efficient energy management.")
    st.markdown("---")

    ## 🤖 Optimal Battery Schedule
    st.header("Reinforcement Learning Schedule")
    st.info("The Reinforcement Learning engine determines the optimal daily schedule to boost storage efficiency and maximize long-term cost savings.")

    # Schedule Table
    schedule_df = pd.DataFrame(optimal_schedule).T.reset_index()
    schedule_df.columns = ['Time', 'Action', 'Power (KW)', 'Reason']
    st.table(schedule_df)

    # Action Legend Card
    st.markdown("<br>")
    st.markdown(
        """
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #ffffff;">
        <h4 style="margin-top: 0;">Action legend</h4>
        <ul>
            <li><span style="color: green; font-weight: bold;">Green</span>: charge when solar is surplus</li>
            <li><span style="color: red; font-weight: bold;">Red</span>: discharge during expensive peak hours</li>
            <li><span style="color: gray; font-weight: bold;">Gray</span>: hold when conditions are neutral</li>
        </ul>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    ## 💰 Savings & Sustainability Impact
    st.header("Savings & Sustainability Impact (168H Simulation)")
    
    baseline_import = synthetic_data['Consumption'].sum()
    optimized_import = synthetic_data['Grid_Import'].sum()
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
    st.write("These metrics display the cost savings compared to a non-optimized baseline, proving the system's real ROI and sustainability impact.")


def page_reports(synthetic_data):
    """Displays the daily and weekly energy reports."""
    st.title("📑 Reports")
    st.write("Summary of energy totals for Day and Week periods.")
    st.markdown("---")
    
    # Calculation (Same as before, moved here)
    day_data = synthetic_data.iloc[-96:] # Last 24 hours
    day_solar_total = (day_data['Solar_Gen'].sum() * 0.25).round(2)
    day_demand_total = (day_data['Consumption'].sum() * 0.25).round(2)
    day_net_energy = (day_data['Net_Energy'].sum() * 0.25).round(2)

    week_data = synthetic_data # Last 7 Days
    week_solar_total = (week_data['Solar_Gen'].sum() * 0.25).round(2)
    week_demand_total = (week_data['Consumption'].sum() * 0.25).round(2)
    week_net_energy = (week_data['Net_Energy'].sum() * 0.25).round(2)

    col1, col2 = st.columns(2)
    
    # Day Report Card
    with col1:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #ffffff;">
            <h3 style="margin-top: 0; margin-bottom: 20px;">Day Report</h3>
            <p>☀️ Total Solar <span style="float: right; color: #F9A825; font-weight: bold;">{day_solar_total} kWh</span></p>
            <p>ᑎ Total Demand <span style="float: right; color: #FF5733; font-weight: bold;">{day_demand_total} kWh</span></p>
            <p>➕ Net Energy <span style="float: right; color: {'#00C853' if day_net_energy >= 0 else '#FF5733'}; font-weight: bold;">{day_net_energy} kWh</span></p>
        </div>
        """, unsafe_allow_html=True)

    # Week Report Card
    with col2:
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #ffffff;">
            <h3 style="margin-top: 0; margin-bottom: 20px;">Week Report</h3>
            <p>☀️ Total Solar <span style="float: right; color: #F9A825; font-weight: bold;">{week_solar_total} kWh</span></p>
            <p>ᑎ Total Demand <span style="float: right; color: #FF5733; font-weight: bold;">{week_demand_total} kWh</span></p>
            <p>➕ Net Energy <span style="float: right; color: {'#00C853' if week_net_energy >= 0 else '#FF5733'}; font-weight: bold;">{week_net_energy} kWh</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Understanding Optimization Card
    st.markdown(
        """
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #ffffff; display: flex; align-items: center;">
            <div style="font-size: 40px; margin-right: 15px;">💡</div>
            <div>
                <h4 style="margin: 0 0 5px 0;">Understanding Optimization</h4>
                <p style="margin: 0; font-size: 14px;">The system charges batteries during solar-rich hours and discharges during high-tariff periods, reducing grid imports and maximizing savings.</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )


# --- 4. DASHBOARD ENTRY POINT ---
def main_dashboard():
    
    # --- Sidebar Controls ---
    st.sidebar.title("PHO⚡TON")
    st.sidebar.subheader("Navigation")
    
    page = st.sidebar.radio(
        "Go To:",
        ("Dashboard", "Forecast", "Battery Optimization", "Reports"),
        index=0 # Default to Dashboard
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Prototype Controls")
    
    # Scenario Selection
    if 'scenario' not in st.session_state:
        st.session_state['scenario'] = "Typical Week"

    selected_scenario = st.sidebar.selectbox(
        "Select Energy Scenario:",
        list(SCENARIOS.keys())
    )
    st.session_state.scenario = selected_scenario
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML & RL Overview")
    st.sidebar.markdown(
        """
        - **Forecasting (XGBoost/LSTM):** Predicts Demand/Solar for 168 hours.
        - **Optimization (Q-Learning/DQN):** Schedules battery action based on forecasts.
        - **Goal:** Maximize savings by minimizing Grid Import during high-cost hours.
        """
    )

    # --- Data Generation (Run once per scenario change) ---
    synthetic_data, df_future, optimal_schedule = generate_mock_data(st.session_state.scenario)

    # --- Page Routing ---
    if page == "Dashboard":
        page_dashboard(synthetic_data, df_future)
    elif page == "Forecast":
        page_forecast(df_future)
    elif page == "Battery Optimization":
        page_optimization(synthetic_data, optimal_schedule)
    elif page == "Reports":
        page_reports(synthetic_data)
        
    # Optional: Raw data view at the bottom of the sidebar or its own page
    # with st.expander("📚 View Raw Data"):
    #     st.dataframe(synthetic_data, height=300)

if __name__ == "__main__":
    main_dashboard()