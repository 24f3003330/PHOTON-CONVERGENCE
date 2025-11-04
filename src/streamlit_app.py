import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
# *** NEW/UNCOMMENTED IMPORT: Required for the simulation loop ***
import time
from pathlib import Path

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---
st.set_page_config(layout="wide", page_title="PHOTON: Smart Energy Optimization Dashboard",
                    initial_sidebar_state="expanded")

# --- SIMULATION CONFIGURATION ---
# Advance time by 4 steps (4 * 15 min = 1 hour) on each update
SIMULATION_STEP = 4
# Refresh the dashboard every 3 seconds to simulate live data
SIMULATION_INTERVAL_SECONDS = 3

# --- Custom CSS for Dark Theme and Card Styling (UNMODIFIED) ---
st.markdown("""<style>    /* General body and text styling for dark theme */    body { color: #e0e0e0; background-color: #0e1117; }    h1, h2, h3, h4, h5, h6 { color: #f0f0f0; }    p, li { color: #c0c0c0; }    /* Streamlit widgets for dark theme */    .stSelectbox > div > div { background-color: #262730; color: #f0f0f0; border-color: #4f4f4f; }    .stSelectbox > label { color: #f0f0f0; }    .stRadio > label { color: #f0f0f0; }    .stButton > button { background-color: #262730; color: #f0f0f0; border-color: #4f4f4f; }    /* Custom Card Styling */    .st-card { background-color: #1e212b; border-radius: 10px; padding: 20px; margin-bottom: 15px; border: 1px solid #3a3a3a; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }    .kpi-card { background-color: #1e212b; border-radius: 10px; padding: 15px; text-align: left; border: 1px solid #3a3a3a; height: 100%; }    .kpi-title { font-size: 14px; color: #909090; margin: 0; }    .kpi-value { font-size: 32px; color: #f0f0f0; margin: 5px 0 0 0; font-weight: bold; }    .kpi-unit { font-size: 16px; color: #c0c0c0; font-weight: normal; }        /* Report card colors */    .report-card-item { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 16px; color: #c0c0c0; }    .report-card-value { font-weight: bold; color: #f0f0f0; }    .color-solar { color: #F9A825; }     .color-demand { color: #FF5733; }     .color-net-positive { color: #00C853; }     .color-net-negative { color: #FF5733; }         /* Optimization Table Styling */    .optimization-table { width: 100%; border-collapse: collapse; margin-top: 10px; background-color: #1e212b; color: #c0c0c0; border: 1px solid #3a3a3a; border-radius: 10px; }    .optimization-table th { background-color: #262730; color: #f0f0f0; padding: 12px 15px; text-align: left; border-bottom: 1px solid #3a3a3a; }    .optimization-table td { padding: 10px 15px; border-bottom: 1px solid #3a3a3a; }    .optimization-table .action-green { color: #32CD32; font-weight: bold; }        .optimization-table .action-red { color: #FF4500; font-weight: bold; }          .optimization-table .action-gray { color: gray; font-weight: bold; }    /* Button for Refresh */    div.stButton > button:first-child { background-color: #03A9F4; color: white; border-radius: 5px; border: 1px solid #03A9F4; padding: 8px 16px; font-size: 16px; display: inline-flex; align-items: center; }</style>""", unsafe_allow_html=True)

# Define SCENARIOS (Used for RL parameters/context)
SCENARIOS = {
    "Typical Week": {
        "Base_Consumption": 75.0,
        "Solar_Factor": 50.0,
        "Peak_Hour": 18,
        "Peak_Discharge_Power": 15.0
    },
    "High Demand Week": {
        "Base_Consumption": 90.0,
        "Solar_Factor": 45.0,
        "Peak_Hour": 19,
        "Peak_Discharge_Power": 25.0
    }}

@st.cache_data(ttl=3600)
def load_energy_data(scenario_key, file_path='energy_data_150days_20households.csv', historical_freq='15Min'):
    """
    Loads ALL real CSV data, resamples, simulates optimization,
    and generates a 24-hour forecast. This function only runs when
    the scenario or input data changes, thanks to st.cache_data.
    """
    params = SCENARIOS[scenario_key]
    
    # *** ROBUST PATH LOGIC ***
    # NOTE: In a cloud environment, ensure the CSV is in the same directory as the script.
    try:
        csv_path = Path(file_path)
        df_raw = pd.read_csv(csv_path)
    except FileNotFoundError:
        # If the local path fails, assume running in the same directory as the script
        csv_path = Path(__file__).resolve().parent / file_path
        df_raw = pd.read_csv(csv_path)

    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.set_index('timestamp').sort_index()

    # Rename columns for app consistency
    df_raw = df_raw.rename(columns={
        'total_consumption_kw': 'Consumption',
        'solar_generation_kw': 'Solar_Gen'
    })
    # 2. Resample Historical Data to 15-Min Frequency
    df_resampled = df_raw[['Consumption', 'Solar_Gen']].resample(historical_freq).mean().interpolate(method='linear')
    # --- KEY CHANGE: Select ALL resampled data ---
    df_historical = df_resampled.copy()

    N = len(df_historical)
    # 3. Simulate Optimized Flow (RL Logic applied to ALL data)
    df_historical['hour'] = df_historical.index.hour

    peak_h = params['Peak_Hour']
    peak_p = params['Peak_Discharge_Power']
    charge_h = 12
    charge_p = -params['Peak_Discharge_Power'] * 0.75
    is_discharge_time = (df_historical['hour'] >= peak_h) & (df_historical['hour'] < peak_h + 1)
    is_charge_time = (df_historical['hour'] >= charge_h) & (df_historical['hour'] < charge_h + 1)
    df_historical['Battery_Flow'] = np.where(
        is_discharge_time,
        peak_p,
        np.where(
            is_charge_time & (df_historical['Solar_Gen'] > params['Solar_Factor'] * 0.2),
            charge_p,
            np.random.randn(N) * 0.05
        )
    )
    # Final energy flows
    df_historical['Net_Grid_Flow'] = df_historical['Consumption'] - df_historical['Solar_Gen'] - df_historical['Battery_Flow']
    df_historical['Grid_Import'] = df_historical['Net_Grid_Flow'].clip(lower=0)
    df_historical['Net_Energy'] = df_historical['Solar_Gen'] + df_historical['Battery_Flow'] - df_historical['Consumption']
    df_historical = df_historical.drop(columns=['hour'])

    # 4. Generate 24-Hour Hourly Forecast (df_future)
    hour_map_template = df_historical[['Consumption', 'Solar_Gen']].copy()
    hour_map_template['hour'] = hour_map_template.index.hour
    hour_map = hour_map_template.groupby('hour')[['Consumption', 'Solar_Gen']].mean()
    
    forecast_start_time = df_historical.index[-1] + timedelta(minutes=15)
    time_index_future = pd.date_range(start=forecast_start_time, end=forecast_start_time + timedelta(hours=24), freq='1H', inclusive='left')
    df_future = pd.DataFrame(index=time_index_future)

    df_future['Demand_Forecast'] = df_future.index.hour.map(hour_map['Consumption']) + np.random.rand(len(df_future)) * 0.5
    df_future['Solar_Forecast'] = df_future.index.hour.map(hour_map['Solar_Gen']) + np.random.rand(len(df_future)) * 0.05
    df_future['Solar_Forecast'] = df_future['Solar_Forecast'].clip(lower=0.0)

    # Anchor the first forecast point for smoothness
    last_consumption = df_historical['Consumption'].iloc[-1]
    df_future.loc[df_future.index[0], 'Demand_Forecast'] = last_consumption * 0.5 + df_future.loc[df_future.index[0], 'Demand_Forecast'] * 0.5
    # MOCK RL Optimization Schedule
    optimal_schedule = {
        f"Daily @ {peak_h:02d}:00 PM": {"Action": "DISCHARGE", "Power (KW)": peak_p, "Reason": f"Daily Predicted Grid Peak ({scenario_key})"},
        f"Daily @ {charge_h:02d}:00 PM": {"Action": "CHARGE", "Power (KW)": abs(charge_p), "Reason": "Daily Solar Peak Forecast"}
    }

    df_historical.index.name = 'Timestamp'
    return df_historical, df_future, optimal_schedule

# --- 2. LAYOUT UTILITIES ---
# ... (display_kpi_card, create_energy_flow_chart, create_forecast_chart are mostly unchanged) ...

def display_kpi_card(title, value, unit, color="#f0f0f0"):
    """Creates a stylized KPI card with custom CSS classes."""
    value_color = color
    if title == "Net Energy Flow (Now)":
        if float(value) > 0:
            value_color = "#00C853"
        else:
            value_color = "#FF5733"
    elif title == "Battery Level (Now)":
        value_color = "#00C853"
    st.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-title">{title}</p>
            <h3 class="kpi-value" style="color: {value_color};">
                {value}
                <span class="kpi-unit">{unit}</span>
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_energy_flow_chart(df_full, df_future):
    """Creates the energy flow chart WITH the rangeslider (interactive scrollbar)."""
    fig = go.Figure()
    end_time_full = df_full.index[-1] + timedelta(minutes=15) 
    max_power = df_full['Consumption'].max() * 1.1
    
    fig.update_layout(template="plotly_dark")
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
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Battery_Flow'].clip(lower=0), mode='lines', name='Battery Discharge', fill='tozeroy', fillcolor='rgba(0,200,0, 0.3)', line=dict(color='lime', width=1))) 
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Battery_Flow'].clip(upper=0).abs(), mode='lines', name='Battery Charge', fill='tozeroy', fillcolor='rgba(100,100,255, 0.3)', line=dict(color='deepskyblue', width=1)))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Solar_Gen'], mode='lines', name='Solar Generation (Supply)', line=dict(color='gold', width=2)))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Consumption'], mode='lines', name='Household Consumption (Demand)', line=dict(color='orangered', width=2)))
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Grid_Import'], mode='lines', name='Grid Consumption (Net Import)', line=dict(color='darkviolet', width=3)))
    # --- FORECAST TRACE ---
    last_historical_time = df_full.index[-1]
    last_historical_consumption = df_full['Consumption'].iloc[-1]
        
    forecast_plot_data = df_future['Demand_Forecast'].copy()
    forecast_plot_data.loc[last_historical_time] = last_historical_consumption
    forecast_plot_data = forecast_plot_data.sort_index()
    fig.add_trace(go.Scatter(
        x=forecast_plot_data.index, y=forecast_plot_data.values, mode='lines', 
        name='Demand Forecast (ML) - Hourly',
        line=dict(color='red', dash='dot', width=3), 
        showlegend=True,
    ))
    # CHART LAYOUT ADJUSTMENTS
    initial_view_start = df_full.index[-96] if len(df_full) >= 96 else df_full.index[0]
    fig.update_layout(
        title_text=f"Energy Flow up to {df_full.index[-1].strftime('%Y-%m-%d %H:%M')}",  # ADDED DYNAMIC TITLE
        xaxis_title="Time", yaxis_title="Power (KW)", height=550,
        dragmode='pan',
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#c0c0c0')),
        xaxis=dict(fixedrange=False, showgrid=True, gridcolor='#333333', zerolinecolor='#333333'),
        yaxis=dict(fixedrange=False, showgrid=True, gridcolor='#333333', zerolinecolor='#333333'),
        paper_bgcolor="#1e212b",
        plot_bgcolor="#1e212b",
    )
    
    # --- RANGESLIDER RESTORATION ---
    fig.update_xaxes(
        # FIX: Set initial range to the end of the current slice for 'live' view
        range=[df_full.index[-96] if len(df_full) >= 96 else df_full.index[0], df_full.index[-1] + timedelta(hours=12)], 
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
            ]),
            font=dict(color='#f0f0f0')
        )
    )
    # --- END RANGESLIDER LOGIC ---
    fig.update_yaxes(range=[0, max_power])
    st.plotly_chart(fig, use_container_width=True)

def create_forecast_chart(df_future):
    # ... (function body is fine as is) ...
    fig = go.Figure()
    fig.update_layout(template="plotly_dark")
    fig.add_trace(go.Scatter(
        x=df_future.index, 
        y=df_future['Demand_Forecast'], 
        mode='lines+markers', 
        name='Demand Forecast (kW)', 
        line=dict(color='orangered', width=3),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df_future.index, 
        y=df_future['Solar_Forecast'], 
        mode='lines+markers', 
        name='Solar Forecast (kW)', 
        line=dict(color='gold', width=3),
        marker=dict(size=6)
    ))
    fig.update_layout(
        # FIX: Removed the chart title by setting title_text to an empty string
        title_text='',
        xaxis_title="Hour Start Time",
        yaxis_title="Power (KW)",
        height=450,
        margin=dict(t=50, b=20),
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
        paper_bgcolor="#1e212b",
        plot_bgcolor="#1e212b",
        xaxis=dict(tickformat="%H:%M", showgrid=True, gridcolor='#333333'),
        yaxis=dict(showgrid=True, gridcolor='#333333'),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 3. PAGE FUNCTIONS ---
def display_dashboard_content(current_data, df_future):
    """New function to encapsulate the dashboard display logic."""
    
    st.subheader("Real-Time Energy Monitoring")

    # Display information about the current time slice
    current_time = current_data.index[-1].strftime('%Y-%m-%d %H:%M')
    st.write(f"Simulating up to **{current_time}** (Data point **{len(current_data)} / {len(st.session_state.df_historical_full)}**)")
    st.markdown("---")

    # Display the static "About" card
    st.markdown(
        """
        <div class="st-card">
        <h4 style="margin-top: 0; color: #f0f0f0;">About Real-Time Monitoring</h4>
        <p>PHOTON's real-time monitoring provides instant visibility into your energy system. Track solar generation patterns, monitor consumption demands, and observe battery storage levels as they fluctuate throughout the day.</p>
        <ul style="color: #c0c0c0;">
            <li>**Solar Generation:** Real-time output from your solar panels in kilowatt-hours</li>
            <li>**Energy Demand:** Current consumption across your building</li>
            <li>**Net Energy:** Surplus (positive) or deficit (negative) balance</li>
            <li>**Battery Level:** Current charge state and percentage</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    latest_data = current_data.iloc[-1]
    
    # Mock battery level based on time of day (to give a dynamic feel)
    current_hour = latest_data.name.hour
    # Simple SOC simulation (draining at night, charging during day)
    if 6 <= current_hour < 18:
        mock_battery_level = 70 + (latest_data['Solar_Gen'] * 3).clip(upper=25) # Charge during day
    else:
        mock_battery_level = 95 - (latest_data['Consumption'] * 0.5) # Drain at night
    mock_battery_level = np.clip(mock_battery_level, 20, 100).round(1)

    mock_net_energy_flow = (latest_data['Solar_Gen'] - latest_data['Consumption'] + latest_data['Battery_Flow']).round(2)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_kpi_card("Solar Generation (Now)", f"{latest_data['Solar_Gen']:.2f}", "kW", "#F9A825")
    with col2:
        display_kpi_card("Energy Demand (Now)", f"{latest_data['Consumption']:.2f}", "kW", "#FF5733")
    with col3:
        display_kpi_card("Net Energy Flow (Now)", f"{mock_net_energy_flow:.2f}", "kW")
    with col4:
        display_kpi_card("Battery Level (Now)", f"{mock_battery_level:.1f}", "%")

    st.markdown("---")

    create_energy_flow_chart(current_data, df_future)
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>Use the <b>small scroll panel at the bottom</b> to easily slide and view the full historical data timeline.</p>", unsafe_allow_html=True)

def auto_update_simulation_loop(df_historical_full, df_future):
    """Handles the real-time simulation logic for the Dashboard page."""
    st.title("💡 Dashboard: Real-Time Monitoring & Dispatch")
    st.markdown("---")

    if 'simulation_running' not in st.session_state:
        st.session_state['simulation_running'] = True
    if 'current_data_idx' not in st.session_state or st.session_state['current_data_idx'] >= len(df_historical_full):
        # Start from 24 hours (96 points for 15-min data)
        st.session_state['current_data_idx'] = 96 

    # Use an empty container to hold and continuously overwrite the dashboard content
    live_container = st.empty()
    
    # Start/Stop button logic
    if st.session_state['simulation_running']:
        if st.button("Stop Simulation ⏸️", key="stop_sim"):
            st.session_state['simulation_running'] = False
            st.rerun()
    else:
        if st.button("Start Simulation ▶️", key="start_sim"):
            st.session_state['simulation_running'] = True
            st.rerun()
        # Manually advance one hour when stopped
        if st.button("Advance 1 Hour (Manual)", key="manual_advance"):
            st.session_state['current_data_idx'] = min(st.session_state['current_data_idx'] + SIMULATION_STEP, len(df_historical_full))
            st.session_state['simulation_running'] = False # Ensure it stays paused
            st.rerun()

    # Main loop logic
    if st.session_state['simulation_running']:
        # Check for end of data
        if st.session_state['current_data_idx'] >= len(df_historical_full):
            st.session_state['simulation_running'] = False
            st.info("Simulation reached the end of the available data. Press 'Start Simulation' to restart.")
            st.session_state['current_data_idx'] = 96 # Reset index
            return

        # 1. Advance the index for the next step
        st.session_state['current_data_idx'] += SIMULATION_STEP

        # 2. Slice the data up to the current simulated time
        current_data_slice = df_historical_full.iloc[:st.session_state['current_data_idx']]

        # 3. Re-render content inside the container
        with live_container.container():
            display_dashboard_content(current_data_slice, df_future)

        # 4. Wait and force a rerun to advance time
        time.sleep(SIMULATION_INTERVAL_SECONDS)
        st.rerun()
    else:
        # Display static content if paused
        current_data_slice = df_historical_full.iloc[:st.session_state['current_data_idx']]
        with live_container.container():
            display_dashboard_content(current_data_slice, df_future)
        st.warning("Simulation is paused. Press 'Start Simulation' or 'Advance 1 Hour (Manual)' to continue.")

def page_forecast(df_future):
    """Displays the forecast model explanation, chart, and data table."""
    st.title("☀️ Forecast")
    st.write("AI-powered 24 hour prediction of solar generation and demand")
    st.markdown("---")

    st.markdown(
        """
        <div class="st-card">
        <h4 style="margin-top: 0; color: #f0f0f0;">How it works</h4>
        <p style="margin: 0; font-size: 14px;">A gradient boosting model uses the last 24 hours and time features to predict the next 24 hours. Use this to plan battery usage and grid interactions.</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.subheader("24-Hour Hourly Forecast Visual")
    create_forecast_chart(df_future)

    st.markdown("---")
    st.subheader("Next 24 Hours Hourly Prediction Table")

    forecast_display = df_future.reset_index()
    forecast_display.columns = ['Hour Start Time', 'Demand Forecast (kW)', 'Solar Forecast (kW)']
    forecast_display['Hour Start Time'] = forecast_display['Hour Start Time'].dt.strftime('%H:00')
    st.dataframe(forecast_display)
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>The dotted red line on the <b>Dashboard</b> shows this demand forecast.</p>", unsafe_allow_html=True)
    
def page_optimization(synthetic_data, optimal_schedule):
    """Displays the battery optimization schedule and impact analysis."""
    st.title("🔋 Battery Optimization")
    st.write("Reinforcement learning schedule and savings impact for efficient energy management.")
    st.markdown("---")
    st.header("Reinforcement Learning Schedule")
    st.info("The Reinforcement Learning engine determines the optimal daily schedule to boost storage efficiency and maximize long-term cost savings.")

    schedule_df = pd.DataFrame(optimal_schedule).T.reset_index()
    schedule_df.columns = ['Time', 'Action', 'Power (KW)', 'Reason']

    html_table = '<table class="optimization-table"><thead><tr>'
    for col in schedule_df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'

    for index, row in schedule_df.iterrows():
        action_class = "action-gray"
        if "CHARGE" in row['Action'].upper():
            action_class = "action-green"
        elif "DISCHARGE" in row['Action'].upper():
            action_class = "action-red"

        html_table += '<tr>'
        html_table += f'<td>{row["Time"]}</td>'
        html_table += f'<td class="{action_class}">{row["Action"]}</td>'
        html_table += f'<td>{row["Power (KW)"]:.4f}</td>'
        html_table += f'<td>{row["Reason"]}</td>'
        html_table += '</tr>'
    html_table += '</tbody></table>'

    st.markdown(html_table, unsafe_allow_html=True)
    st.markdown("<br>")
    st.markdown(
        """
        <div class="st-card">
        <h4 style="margin-top: 0; color: #f0f0f0;">Action legend</h4>
        <ul style="color: #c0c0c0;">
            <li><span style="color: #32CD32; font-weight: bold;">Green</span>: charge when solar is surplus</li>
            <li><span style="color: #FF4500; font-weight: bold;">Red</span>: discharge during expensive peak hours</li>
            <li><span style="color: gray; font-weight: bold;">Gray</span>: hold when conditions are neutral</li>
        </ul>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    st.header("Savings & Sustainability Impact (Simulated to Current Time)")

    # Calculate savings based on the entire current slice of data
    baseline_import = synthetic_data['Consumption'].sum()
    optimized_import = synthetic_data['Grid_Import'].sum()
    # Scaling the saving calculation based on the total number of points simulated
    # Use the ratio of current data length to 7 days (672 points) for a better projection
    data_ratio = len(synthetic_data) / 672 
    
    estimated_savings_weekly = (baseline_import - optimized_import) * 0.1 * data_ratio
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
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>These metrics display the cost savings compared to a non-optimized baseline, proving the system's real ROI and sustainability impact.</p>", unsafe_allow_html=True)

def page_reports(synthetic_data):
    """Displays the daily and weekly energy reports based on current simulated data."""
    st.title("📑 Reports")
    st.write("Summary of energy totals for Day and Week periods.")
    st.markdown("---")

    # Calculation (using 0.25 multiplier for 15-min data to convert kW to kWh)
    # Day Report: Last 24 hours (96 points)
    day_data = synthetic_data.tail(96)
    day_solar_total = (day_data['Solar_Gen'].sum() * 0.25).round(2)
    day_demand_total = (day_data['Consumption'].sum() * 0.25).round(2)
    day_net_energy = (day_data['Net_Energy'].sum() * 0.25).round(2)
    # Week Report: Last 7 days (672 points) - use min to prevent error if less than 7 days simulated
    week_data = synthetic_data.tail(min(len(synthetic_data), 672))
    week_solar_total = (week_data['Solar_Gen'].sum() * 0.25).round(2)
    week_demand_total = (week_data['Consumption'].sum() * 0.25).round(2)
    week_net_energy = (week_data['Net_Energy'].sum() * 0.25).round(2)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="st-card">
            <h3 style="margin-top: 0; margin-bottom: 20px; color: #f0f0f0;">Day Report</h3>
            <div class="report-card-item">☀️ Total Solar <span class="report-card-value color-solar">{day_solar_total} kWh</span></div>
            <div class="report-card-item">ᑎ Total Demand <span class="report-card-value color-demand">{day_demand_total} kWh</span></div>
            <div class="report-card-item">➕ Net Energy <span class="report-card-value {'color-net-positive' if day_net_energy >= 0 else 'color-net-negative'}">{day_net_energy} kWh</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="st-card">
            <h3 style="margin-top: 0; margin-bottom: 20px; color: #f0f0f0;">Week Report</h3>
            <div class="report-card-item">☀️ Total Solar <span class="report-card-value color-solar">{week_solar_total} kWh</span></div>
            <div class="report-card-item">ᑎ Total Demand <span class="report-card-value color-demand">{week_demand_total} kWh</span></div>
            <div class="report-card-item">➕ Net Energy <span class="report-card-value {'color-net-positive' if week_net_energy >= 0 else 'color-net-negative'}">{week_net_energy} kWh</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        """
        <div class="st-card" style="display: flex; align-items: center;">
            <div style="font-size: 40px; margin-right: 15px; color: #F9A825;">💡</div>
            <div>
                <h4 style="margin: 0 0 5px 0; color: #f0f0f0;">Understanding Optimization</h4>
                <p style="margin: 0; font-size: 14px; color: #c0c0c0;">The system charges batteries during solar-rich hours and discharges during high-tariff periods, reducing grid imports and maximizing savings.</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# --- 4. DASHBOARD ENTRY POINT ---
def main_dashboard():

    st.sidebar.title("PHO⚡TON")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")

    page = st.sidebar.radio(
        "Go To:",
        ("Dashboard", "Forecast", "Battery Optimization", "Reports"),
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Prototype Controls")

    if 'scenario' not in st.session_state:
        st.session_state['scenario'] = "Typical Week"
    selected_scenario = st.sidebar.selectbox(
        "Select Energy Scenario:",
        list(SCENARIOS.keys())
    )
    st.session_state.scenario = selected_scenario
    
    # --- Data Generation (Loading Full Data and Saving to Session State) ---
    try:
        df_historical_full, df_future, optimal_schedule = load_energy_data(
            st.session_state.scenario
        )
        st.session_state.df_historical_full = df_historical_full # Save full data
    except FileNotFoundError:
        st.error("⚠️ **File Not Found Error:** The application cannot find the data file 'energy_data_150days_20households.csv'.")
        st.info("Please ensure the filename in the code exactly matches the file on disk (including case) and try running the app again.")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("ML & RL Overview")
    st.sidebar.markdown(
        """
        - **Data Source:** **energy_data_150days_20households.csv**
        - **Historical Data:** All 150 days loaded
        - **Optimization:** RL policy simulated using real solar/consumption data.
        """
    )
    
    # --- Get Current Data Slice (Used for non-Dashboard pages) ---
    if 'current_data_idx' not in st.session_state:
        st.session_state['current_data_idx'] = 96 # Start at 24 hours
    current_slice_data = df_historical_full.iloc[:st.session_state['current_data_idx']]

    # --- Page Routing ---
    if page == "Dashboard":
        # The Dashboard page handles its own slicing and looping
        auto_update_simulation_loop(df_historical_full, df_future)
    elif page == "Forecast":
        page_forecast(df_future)
    elif page == "Battery Optimization":
        # Use the data slice up to the last simulated time
        page_optimization(current_slice_data, optimal_schedule)
    elif page == "Reports":
        # Use the data slice up to the last simulated time
        page_reports(current_slice_data)

if __name__ == "__main__":
    main_dashboard()