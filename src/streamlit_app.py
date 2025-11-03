import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---
st.set_page_config(layout="wide", page_title="PHOTON: Smart Energy Optimization Dashboard",
                    initial_sidebar_state="expanded")

# --- Custom CSS for Dark Theme and Card Styling ---
st.markdown("""
<style>
    /* General body and text styling for dark theme */
    body {
        color: #e0e0e0; 
        background-color: #0e1117; 
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0; 
    }
    p, li {
        color: #c0c0c0; 
    }

    /* Streamlit widgets for dark theme */
    .stSelectbox > div > div {
        background-color: #262730;
        color: #f0f0f0;
        border-color: #4f4f4f;
    }
    .stSelectbox > label {
        color: #f0f0f0;
    }
    .stRadio > label {
        color: #f0f0f0;
    }
    .stButton > button {
        background-color: #262730;
        color: #f0f0f0;
        border-color: #4f4f4f;
    }

    /* Custom Card Styling (consistent with the image look) */
    .st-card {
        background-color: #1e212b; /* Darker background for cards */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #3a3a3a; 
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Specific styling for KPI cards */
    .kpi-card {
        background-color: #1e212b;
        border-radius: 10px;
        padding: 15px;
        text-align: left;
        border: 1px solid #3a3a3a;
        height: 100%; 
    }
    .kpi-title {
        font-size: 14px;
        color: #909090; 
        margin: 0;
    }
    .kpi-value {
        font-size: 32px;
        color: #f0f0f0; 
        margin: 5px 0 0 0;
        font-weight: bold;
    }
    .kpi-unit {
        font-size: 16px;
        color: #c0c0c0;
        font-weight: normal;
    }
    
    /* For the specific report cards */
    .report-card-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        font-size: 16px;
        color: #c0c0c0;
    }
    .report-card-value {
        font-weight: bold;
        color: #f0f0f0;
    }
    /* Specific colors for report items */
    .color-solar { color: #F9A825; } 
    .color-demand { color: #FF5733; } 
    .color-net-positive { color: #00C853; } 
    .color-net-negative { color: #FF5733; } 
    .color-info { color: #03A9F4; } 

    /* Custom info/warning boxes */
    .st.info {
        background-color: #1e212b; 
        color: #03A9F4; 
        border-left: 5px solid #03A9F4;
        padding: 10px;
        border-radius: 5px;
    }
    .st.warning {
        background-color: #1e212b; 
        color: #FFC107; 
        border-left: 5px solid #FFC107;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Adjust Streamlit info box style */
    .stAlert {
        background-color: #1e212b !important;
        color: #c0c0c0 !important;
    }
    .stAlert > div[data-testid="stMarkdownContainer"] p {
        color: #c0c0c0 !important;
    }
    
    /* Button for Refresh */
    div.stButton > button:first-child {
        background-color: #03A9F4; 
        color: white;
        border-radius: 5px;
        border: 1px solid #03A9F4;
        padding: 8px 16px;
        font-size: 16px;
        display: inline-flex;
        align-items: center;
    }
    div.stButton > button:first-child:hover {
        background-color: #0288d1; 
        border-color: #0288d1;
    }
    
    /* Optimization Table Styling */
    .optimization-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        background-color: #1e212b; 
        color: #c0c0c0; 
        border: 1px solid #3a3a3a;
        border-radius: 10px;
    }
    .optimization-table th {
        background-color: #262730; 
        color: #f0f0f0;
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #3a3a3a;
    }
    .optimization-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #3a3a3a;
    }
    .optimization-table tr:last-child td {
        border-bottom: none;
    }
    .optimization-table tbody tr:hover {
        background-color: #262730; 
    }
    .optimization-table .action-green { color: lime; font-weight: bold; }       
    .optimization-table .action-red { color: orangered; font-weight: bold; }    
    .optimization-table .action-gray { color: gray; font-weight: bold; }
    
    /* Dataframe styling */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        background-color: #1e212b; 
        color: #c0c0c0; 
        border: 1px solid #3a3a3a;
        border-radius: 10px;
    }
    .dataframe th {
        background-color: #262730; 
        color: #f0f0f0;
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #3a3a3a;
    }
    .dataframe td {
        padding: 10px 15px;
        border-bottom: 1px solid #3a3a3a;
    }
    .dataframe tr:last-child td {
        border-bottom: none;
    }
    .dataframe tbody tr:hover {
        background-color: #262730; 
    }

</style>
""", unsafe_allow_html=True)


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
    """Generates 168 hours (7 days) of synthetic data and 24-hour future forecast at 1-hour freq."""

    params = SCENARIOS[scenario_key]

    # 1. Generate FULL DATA (168 hours at 15-minute frequency)
    end_time_fixed = datetime(2025, 11, 7, 0, 0, 0)
    start_time_fixed = end_time_fixed - timedelta(hours=total_hours)

    time_index = pd.date_range(start=start_time_fixed, end=end_time_fixed, freq=freq, inclusive='left')
    df = pd.DataFrame(index=time_index)
    N = len(df)

    # Data simulation: Daily Cycle Simulation (Historical Data)
    consumption_daily_cycle = 0.5 * np.sin(np.linspace(0, 7 * 2 * np.pi, N))
    df['Consumption'] = params['Base_Consumption'] + consumption_daily_cycle + np.random.rand(N) * 0.2

    solar_hours = df.index.hour + df.index.minute / 60
    solar_daily_cycle = params['Solar_Factor'] * np.maximum(0, np.sin((solar_hours - 6) / 12 * np.pi))
    df['Solar_Gen'] = 0.1 + solar_daily_cycle + np.random.rand(N) * 0.05
    df['Solar_Gen'] = df['Solar_Gen'].clip(lower=0.0)

    # RL Logic and Net Flow Calculations
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

    # 2. Generate Future Forecast Data (24 hours at 1-hour frequency - Realism Improved)
    forecast_start_time = df.index[-1] + timedelta(minutes=15)
    time_index_future = pd.date_range(start=forecast_start_time, end=forecast_start_time + timedelta(hours=24), freq='1H', inclusive='left')
    df_future = pd.DataFrame(index=time_index_future, columns=['Demand_Forecast', 'Solar_Forecast'])

    future_N = len(df_future)
    
    # Create the cyclic pattern for the next 24 hours
    future_hours = time_index_future.hour + time_index_future.minute / 60
    
    # Consumption Cycle (Peaking around 6 PM/18 hours)
    phase_shift_h = 6 
    future_consumption_cycle = 0.5 * np.sin((future_hours - phase_shift_h) / 12 * np.pi) 

    # Demand Forecast
    df_future['Demand_Forecast'] = params['Base_Consumption'] + future_consumption_cycle * 0.8 + np.random.rand(future_N) * 0.1
    
    # Solar Forecast (Peak around 12 PM/12 hours)
    future_solar_cycle = params['Solar_Factor'] * np.maximum(0, np.sin((future_hours - 6) / 12 * np.pi))
    df_future['Solar_Forecast'] = 0.1 + future_solar_cycle * 0.9 + np.random.rand(future_N) * 0.03
    df_future['Solar_Forecast'] = df_future['Solar_Forecast'].clip(lower=0.0)
    
    # Anchor the first forecast point to the last historical point for smoothness
    last_consumption = df['Consumption'].iloc[-1]
    last_solar = df['Solar_Gen'].iloc[-1]
    
    # Simple smoothing correction (using linear decay towards the forecast value)
    df_future.loc[df_future.index[0], 'Demand_Forecast'] = last_consumption * 0.5 + df_future.loc[df_future.index[0], 'Demand_Forecast'] * 0.5
    df_future.loc[df_future.index[0], 'Solar_Forecast'] = last_solar * 0.5 + df_future.loc[df_future.index[0], 'Solar_Forecast'] * 0.5


    synthetic_data = df[['Consumption', 'Solar_Gen', 'Grid_Import', 'Battery_Flow', 'Net_Energy']].copy()
    synthetic_data.index.name = 'Timestamp'

    return synthetic_data, df_future, optimal_schedule

# --- 2. LAYOUT UTILITIES ---

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
    """Creates the energy flow chart with a small scroll panel at the bottom."""
    fig = go.Figure()
    end_time_full = df_full.index[-1] + timedelta(minutes=15) 
    
    max_power = df_full['Consumption'].max() * 1.1
    
    fig.update_layout(template="plotly_dark")

    # RL Highlight
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
        title_text='Energy Flow & **ML-Optimized Dispatch** (24H Default View)',
        xaxis_title="Time", yaxis_title="Power (KW)", height=550,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#c0c0c0')),
        xaxis=dict(fixedrange=False, showgrid=True, gridcolor='#333333', zerolinecolor='#333333'),
        yaxis=dict(fixedrange=False, showgrid=True, gridcolor='#333333', zerolinecolor='#333333'),
        paper_bgcolor="#1e212b",
        plot_bgcolor="#1e212b",
    )

    # Scroll Panel Implementation
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
            ]),
            font=dict(color='#f0f0f0')
        )
    )
    fig.update_yaxes(range=[0, max_power])
    st.plotly_chart(fig, use_container_width=True)

def create_forecast_chart(df_future):
    """Creates a chart showing 24-hour Demand and Solar Forecasts."""
    fig = go.Figure()
    fig.update_layout(template="plotly_dark")

    # Demand Forecast
    fig.add_trace(go.Scatter(
        x=df_future.index, 
        y=df_future['Demand_Forecast'], 
        mode='lines+markers', 
        name='Demand Forecast (kW)', 
        line=dict(color='orangered', width=3),
        marker=dict(size=6)
    ))

    # Solar Forecast
    fig.add_trace(go.Scatter(
        x=df_future.index, 
        y=df_future['Solar_Forecast'], 
        mode='lines+markers', 
        name='Solar Forecast (kW)', 
        line=dict(color='gold', width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title_text='**24-Hour Hourly Demand & Solar Forecast**',
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

def page_dashboard(synthetic_data, df_future):
    """Displays the main chart, real-time metrics, and general info."""
    st.title("💡 Dashboard: Real-Time Monitoring & Dispatch")
    st.write("Live tracking of your energy generation, consumption, and storage, plus optimized flow.")
    st.markdown("---")
    
    st.subheader("Real-Time Energy Monitoring")
    # Refresh button (FIXED: st.rerun())
    if st.button("Refresh 🔄", key="refresh_monitor"):
        st.rerun() 

    # About Real-Time Monitoring card
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

    latest_data = synthetic_data.iloc[-1]
    mock_battery_level = 75 + (datetime.now().minute % 10) * 0.5
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
    
    create_energy_flow_chart(synthetic_data, df_future)
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>Use the <b>small scroll panel at the bottom</b> to easily slide and view the next 24 hours of data across the full 168-hour timeline.</p>", unsafe_allow_html=True)


def page_forecast(df_future):
    """Displays the forecast model explanation, chart, and data table."""
    st.title("☀️ Forecast")
    st.write("AI-powered 24 hour prediction of solar generation and demand")
    st.markdown("---")
    
    # How it works card
    st.markdown(
        """
        <div class="st-card">
        <h4 style="margin-top: 0; color: #f0f0f0;">How it works</h4>
        <p style="margin: 0; font-size: 14px;">A gradient boosting model uses the last 24 hours and time features to predict the next 24 hours. Use this to plan battery usage and grid interactions.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>")
    
    # Forecast Chart
    st.subheader("24-Hour Hourly Forecast Visual")
    create_forecast_chart(df_future)
    
    st.markdown("---")

    # Table Display
    st.subheader("Next 24 Hours Hourly Prediction Table")
    
    forecast_display = df_future.reset_index()
    forecast_display.columns = ['Hour Start Time', 'Demand Forecast (kW)', 'Solar Forecast (kW)']
    forecast_display['Hour Start Time'] = forecast_display['Hour Start Time'].dt.strftime('%H:00')
    st.dataframe(forecast_display)
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>The dotted red line on the <b>Dashboard</b> shows this demand forecast.</p>", unsafe_allow_html=True)


# --- 3. PAGE FUNCTIONS ---

def page_optimization(synthetic_data, optimal_schedule):
    """Displays the battery optimization schedule and impact analysis."""
    st.title("🔋 Battery Optimization")
    st.write("Reinforcement learning schedule and savings impact for efficient energy management.")
    st.markdown("---")

    st.header("Reinforcement Learning Schedule")
    st.info("The Reinforcement Learning engine determines the optimal daily schedule to boost storage efficiency and maximize long-term cost savings.")

    # Schedule Table (Styled for dark theme)
    st.markdown("""
        <style>
            .optimization-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                background-color: #1e212b; 
                color: #c0c0c0; 
                border: 1px solid #3a3a3a;
                border-radius: 10px;
            }
            .optimization-table th {
                background-color: #262730; 
                color: #f0f0f0;
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #3a3a3a;
            }
            .optimization-table td {
                padding: 10px 15px;
                border-bottom: 1px solid #3a3a3a;
            }
            .optimization-table tr:last-child td {
                border-bottom: none;
            }
            .optimization-table tbody tr:hover {
                background-color: #262730; 
            }
            /* EXPLICIT COLOR DEFINITIONS FOR CONTRAST */
            .optimization-table .action-green { color: #32CD32; font-weight: bold; }    /* Lime Green for CHARGE */
            .optimization-table .action-red { color: #FF4500; font-weight: bold; }      /* Stronger Red/Tomato for DISCHARGE */
            .optimization-table .action-gray { color: gray; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    # Convert schedule_df to HTML with custom classes for action colors
    schedule_df = pd.DataFrame(optimal_schedule).T.reset_index()
    schedule_df.columns = ['Time', 'Action', 'Power (KW)', 'Reason']
    
    # Custom HTML generation for table to apply colors based on 'Action'
    html_table = '<table class="optimization-table"><thead><tr>'
    for col in schedule_df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'
    
    for index, row in schedule_df.iterrows():
        action_class = "action-gray" # Default to gray/hold if neither charge nor discharge
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


    # Action Legend Card (Colors updated to match the stronger HTML colors)
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
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>These metrics display the cost savings compared to a non-optimized baseline, proving the system's real ROI and sustainability impact.</p>", unsafe_allow_html=True)

def page_reports(synthetic_data):
    """Displays the daily and weekly energy reports."""
    st.title("📑 Reports")
    st.write("Summary of energy totals for Day and Week periods.")
    st.markdown("---")
    
    # Calculation 
    day_data = synthetic_data.iloc[-96:] 
    day_solar_total = (day_data['Solar_Gen'].sum() * 0.25).round(2)
    day_demand_total = (day_data['Consumption'].sum() * 0.25).round(2)
    day_net_energy = (day_data['Net_Energy'].sum() * 0.25).round(2)

    week_data = synthetic_data 
    week_solar_total = (week_data['Solar_Gen'].sum() * 0.25).round(2)
    week_demand_total = (week_data['Consumption'].sum() * 0.25).round(2)
    week_net_energy = (week_data['Net_Energy'].sum() * 0.25).round(2)

    col1, col2 = st.columns(2)
    
    # Day Report Card
    with col1:
        st.markdown(f"""
        <div class="st-card">
            <h3 style="margin-top: 0; margin-bottom: 20px; color: #f0f0f0;">Day Report</h3>
            <div class="report-card-item">☀️ Total Solar <span class="report-card-value color-solar">{day_solar_total} kWh</span></div>
            <div class="report-card-item">ᑎ Total Demand <span class="report-card-value color-demand">{day_demand_total} kWh</span></div>
            <div class="report-card-item">➕ Net Energy <span class="report-card-value {'color-net-positive' if day_net_energy >= 0 else 'color-net-negative'}">{day_net_energy} kWh</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Week Report Card
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
    
    # Understanding Optimization Card
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
    
    # --- Sidebar Controls ---
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

if __name__ == "__main__":
    main_dashboard()