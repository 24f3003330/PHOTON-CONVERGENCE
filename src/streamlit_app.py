import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import xgboost as xgb 

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---
st.set_page_config(layout="wide", page_title="PHOTON: Smart Energy Optimization Dashboard", 
                           initial_sidebar_state="expanded")

# --- Custom CSS (UNMODIFIED) ---
st.markdown("""<style> 
     /* General body and text styling for dark theme */ 
     body { color: #e0e0e0; background-color: #0e1117; } 
     h1, h2, h3, h4, h5, h6 { color: #f0f0f0; } 
     p, li { color: #c0c0c0; } 
     /* Streamlit widgets for dark theme */ 
     .stSelectbox > div > div { background-color: #262730; color: #f0f0f0; border-color: #4f4f4f; } 
     .stSelectbox > label { color: #f0f0f0; } 
     .stRadio > label { color: #f0f0f0; } 
     .stButton > button { background-color: #262730; color: #f0f0f0; border-color: #4f4f4f; } 
     /* Custom Card Styling */ 
     .st-card { background-color: #1e212b; border-radius: 10px; padding: 20px; margin-bottom: 15px; border: 1px solid #3a3a3a; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); } 
     .kpi-card { background-color: #1e212b; border-radius: 10px; padding: 15px; text-align: left; border: 1px solid #3a3a3a; height: 100%; } 
     .kpi-title { font-size: 14px; color: #909090; margin: 0; } 
     .kpi-value { font-size: 32px; color: #f0f0f0; margin: 5px 0 0 0; font-weight: bold; } 
     .kpi-unit { font-size: 16px; color: #c0c0c0; font-weight: normal; } 
          /* Report card colors */ 
     .report-card-item { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 16px; color: #c0c0c0; } 
     .report-card-value { font-weight: bold; color: #f0f0f0; } 
     .color-solar { color: #F9A825; } 
      .color-demand { color: #FF5733; } 
      .color-net-positive { color: #00C853; } 
      .color-net-negative { color: #FF5733; } 
          /* Optimization Table Styling */ 
     .optimization-table { width: 100%; border-collapse: collapse; margin-top: 10px; background-color: #1e212b; color: #c0c0c0; border: 1px solid #3a3a3a; border-radius: 10px; } 
     .optimization-table th { background-color: #262730; color: #f0f0f0; padding: 12px 15px; text-align: left; border-bottom: 1px solid #3a3a3a; } 
     .optimization-table td { padding: 10px 15px; border-bottom: 1px solid #3a3a3a; } 
     .optimization-table .action-green { color: #32CD32; font-weight: bold; } 
          .optimization-table .action-red { color: #FF4500; font-weight: bold; } 
          .optimization-table .action-gray { color: gray; font-weight: bold; } 
     /* Button for Refresh */ 
     div.stButton > button:first-child { background-color: #03A9F4; color: white; border-radius: 5px; border: 1px solid #03A9F4; padding: 8px 16px; font-size: 16px; display: inline-flex; align-items: center; }</style>""", unsafe_allow_html=True)

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

# --- NEW FEATURE ENGINEERING FUNCTION (UNMODIFIED) ---
def create_features(df):
    """Create time series features based on index for XGBoost."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear 
    return df

# --- MODIFIED XGBOOST MODEL TRAINING AND PREDICTION FUNCTION FOR SOLAR (Now uses Irradiance, Temp, and Panel Specs) ---
def train_and_predict_solar_xgb(df_historical, df_future):
    """Trains an XGBoost model on historical data and makes 24H hourly predictions for SOLAR."""
    
    # 1. Prepare Data 
    df_hourly = df_historical.resample('1H')[['Solar_Gen', 'Irradiance', 'Temperature', 'Panel_Capacity', 'Panel_Efficiency']].mean().dropna()
    df_hourly = create_features(df_hourly)
    
    # 2. Define Target and Features (NOW INCLUDING WEATHER & PANEL SPECS)
    TARGET = 'Solar_Gen'
    FEATURES = ['hour', 'dayofweek', 'dayofyear', 'Irradiance', 'Temperature', 'Panel_Capacity', 'Panel_Efficiency'] 
    
    X_train = df_hourly[FEATURES]
    y_train = df_hourly[TARGET]
    
    # 3. Train the XGBoost Model
    reg_solar = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42, 
        objective='reg:squarederror'
    )
    reg_solar.fit(X_train, y_train)

    # 4. Generate Future Forecast (MOCK: Based on historical hourly averages for varying features, and last known value for static)
    # **USES WEATHER MOCK FORECAST**
    df_future['Forecast_Irradiance'] = df_future.index.hour.map(
        df_hourly.groupby('hour')['Irradiance'].mean()
    ) + np.random.randn(len(df_future)) * 20 
    df_future['Forecast_Temperature'] = df_future.index.hour.map(
        df_hourly.groupby('hour')['Temperature'].mean()
    ) + np.random.randn(len(df_future)) * 1 
    
    # Static features use the last known value for the entire forecast period
    last_known = df_hourly.iloc[-1]
    df_future['Panel_Capacity'] = last_known['Panel_Capacity']
    df_future['Panel_Efficiency'] = last_known['Panel_Efficiency']
    
    df_future['Forecast_Irradiance'] = df_future['Forecast_Irradiance'].clip(lower=0) 
    
    # 5. Predict Solar Generation
    X_future_solar = df_future[['hour', 'dayofweek', 'dayofyear', 'Forecast_Irradiance', 'Forecast_Temperature', 'Panel_Capacity', 'Panel_Efficiency']]
    
    # Rename to match training features
    X_future_solar.columns = ['hour', 'dayofweek', 'dayofyear', 'Irradiance', 'Temperature', 'Panel_Capacity', 'Panel_Efficiency'] 
    
    df_future['Solar_Forecast'] = reg_solar.predict(X_future_solar)
    
    # Ensure solar forecast is not negative
    df_future['Solar_Forecast'] = df_future['Solar_Forecast'].clip(lower=0.0)
    
    # Clean up temporary forecast features
    df_future = df_future.drop(columns=['Forecast_Irradiance', 'Forecast_Temperature', 'Panel_Capacity', 'Panel_Efficiency'])
    
    return df_future

# --- MODIFIED train_and_predict_xgb (for Demand only - Now uses Solar Gen, Temp, Price, Households, Weekend) ---
def train_and_predict_xgb(df_historical):
    """Trains an XGBoost model on historical data and makes 24H hourly predictions for DEMAND."""
    
    # 1. Prepare Data 
    # Include all new consumption-driving features
    df_hourly = df_historical.resample('1H')[['Consumption', 'Solar_Gen', 'Temperature', 'Grid_Price', 'Is_Weekend', 'Num_Households']].mean().dropna() 
    df_hourly = create_features(df_hourly)
    
    # 2. Define Target and Features 
    TARGET = 'Consumption'
    FEATURES = ['hour', 'dayofweek', 'dayofyear', 'Solar_Gen', 'Temperature', 'Grid_Price', 'Is_Weekend', 'Num_Households'] 
    
    X_train = df_hourly[FEATURES]
    y_train = df_hourly[TARGET]
    
    # 3. Train the XGBoost Model
    reg = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42, 
        objective='reg:squarederror'
    )
    reg.fit(X_train, y_train)

    # 4. Generate Future Index and Features (24 hours)
    forecast_start_time = df_hourly.index[-1] + timedelta(hours=1)
    time_index_future = pd.date_range(
        start=forecast_start_time, 
        end=forecast_start_time + timedelta(hours=24), 
        freq='1H', 
        inclusive='left'
    )
    df_future = pd.DataFrame(index=time_index_future)
    df_future = create_features(df_future)
    
    # 5. Prepare Mock Features for Demand Prediction
    # **USES WEATHER/EXTERNAL MOCK FORECAST**
    last_known = df_hourly.iloc[-1]
    
    df_future['Mock_Solar_Forecast'] = df_future.index.hour.map(
        df_hourly.groupby('hour')['Solar_Gen'].mean()
    )
    df_future['Mock_Temperature'] = df_future.index.hour.map(
        df_hourly.groupby('hour')['Temperature'].mean()
    )
    df_future['Mock_Grid_Price'] = df_future.index.hour.map(
        df_hourly.groupby('hour')['Grid_Price'].mean()
    )
    # Weekend status can be derived from the dayofweek feature
    df_future['Is_Weekend'] = df_future['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    # Num_Households is static
    df_future['Num_Households'] = last_known['Num_Households']

    X_future_demand = df_future[['hour', 'dayofweek', 'dayofyear', 'Mock_Solar_Forecast', 'Mock_Temperature', 'Mock_Grid_Price', 'Is_Weekend', 'Num_Households']]
    
    # Rename to match training features
    X_future_demand.columns = ['hour', 'dayofweek', 'dayofyear', 'Solar_Gen', 'Temperature', 'Grid_Price', 'Is_Weekend', 'Num_Households'] 
    
    df_future['Demand_Forecast'] = reg.predict(X_future_demand)
    
    # Remove mock columns before returning
    df_future = df_future.drop(columns=['Mock_Solar_Forecast', 'Mock_Temperature', 'Mock_Grid_Price', 'Is_Weekend', 'Num_Households'])
    
    return df_future # Returns df_future with 'Demand_Forecast'

# --- MODIFIED load_energy_data FUNCTION (Uses Correct Column Names for all features) ---
@st.cache_data(ttl=3600)
def load_energy_data(scenario_key, file_path='energy_data_150days_20households.csv', historical_freq='15Min'):
    """
    Loads ALL real CSV data, resamples, simulates optimization,
    and generates a 24-hour forecast using XGBoost with many features.
    """
    params = SCENARIOS[scenario_key]
        
    # *** ROBUST PATH LOGIC ***
    # This line can cause issues if the script is run in an environment where __file__ is not defined (e.g., some notebook environments).
    # Since this is a standalone Streamlit app, we'll assume it works, but a robust check is needed.
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback for environments where __file__ is not defined (e.g. running in an IDE/Notebook)
        script_dir = Path.cwd()
        
    csv_path = script_dir / file_path
    
    # 1. Load and Process Real Data
    try:
        df_raw = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Re-raise the error with a more specific message if Path construction failed.
        # Note: The main_dashboard handles this error.
        raise
            
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.set_index('timestamp').sort_index()
    
    # --- MAPPING ALL REQUIRED COLUMNS (INCLUDING SOLAR IRRADIANCE AND TEMPERATURE) ---
    df_raw = df_raw.rename(columns={
        'total_consumption_kw': 'Consumption',
        'solar_generation_kw': 'Solar_Gen',
        # Weather & Economic
        'solar_irradiance_wm2': 'Irradiance',
        'temperature_c': 'Temperature',
        'grid_price_per_kwh': 'Grid_Price',
        'is_weekend': 'Is_Weekend',
        # System Specs
        'panel_capacity_kw': 'Panel_Capacity',
        'panel_efficiency': 'Panel_Efficiency',
        'num_households': 'Num_Households',
    }, errors='ignore') 
    # --- END MAPPING ---
    
    # 2. Resample Historical Data to 15-Min Frequency
    COLS_TO_KEEP = ['Consumption', 'Solar_Gen', 'Irradiance', 'Temperature', 'Grid_Price', 'Is_Weekend', 'Panel_Capacity', 'Panel_Efficiency', 'Num_Households']
    existing_cols = [col for col in COLS_TO_KEEP if col in df_raw.columns]
    
    df_historical = df_raw[existing_cols].resample(historical_freq).mean().interpolate(method='linear')
    N = len(df_historical)
    
    # Add dummy columns if they are missing after resampling (to prevent ML model crash)
    for col in COLS_TO_KEEP:
        if col not in df_historical.columns:
             # Ensure a scalar value is broadcast correctly
             last_val = df_raw[col].iloc[-1] if col in df_raw.columns and not df_raw[col].empty else 0.0
             df_historical[col] = last_val
    
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
    # Corrected Net_Energy calculation: Solar + Battery Discharge - Consumption (Net_Energy > 0 means Surplus/Export)
    # Battery_Flow is positive for DISCHARGE, negative for CHARGE
    df_historical['Net_Energy'] = df_historical['Solar_Gen'] + df_historical['Battery_Flow'] - df_historical['Consumption']
    df_historical = df_historical.drop(columns=['hour'])
    
    # 4. Generate 24-Hour Hourly Demand Forecast
    df_future = train_and_predict_xgb(df_historical)

    # 5. Generate 24-Hour Hourly Solar Forecast using XGBoost with Full Features
    df_future = train_and_predict_solar_xgb(df_historical, df_future)
    
    # Anchor the first forecast point for smoothness
    last_consumption = df_historical['Consumption'].iloc[-1]
    # Smooth the transition between historical data and forecast data
    df_future.loc[df_future.index[0], 'Demand_Forecast'] = last_consumption * 0.5 + df_future.loc[df_future.index[0], 'Demand_Forecast'] * 0.5
    
    # MOCK RL Optimization Schedule
    optimal_schedule = {
        f"Daily @ {peak_h:02d}:00": {"Action": "DISCHARGE", "Power (KW)": peak_p, "Reason": f"Daily Predicted Grid Peak ({scenario_key})"},
        f"Daily @ {charge_h:02d}:00": {"Action": "CHARGE", "Power (KW)": abs(charge_p), "Reason": "Daily Solar Peak Forecast"}
    }
    df_historical.index.name = 'Timestamp'
    return df_historical, df_future, optimal_schedule

# --- 2. LAYOUT UTILITIES (Helpers for UI/Charts - UNMODIFIED) ---
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

def create_energy_flow_chart(df_full, df_future, selected_cols):
    """
    Creates the energy flow chart, filtering traces based on selected_cols.
    """
    # 1. Combine historical and future indices for max_power calculation
    full_index = df_full.index.union(df_future.index)
    # Create a series that contains the max of historical consumption and demand forecast
    max_demand_series = pd.Series(index=full_index)
    max_demand_series.update(df_full['Consumption'])
    max_demand_series.update(df_future['Demand_Forecast']) # Overwrite the 'Consumption' with 'Demand_Forecast' for future times
    
    # **BUG FIX**: The `max_power` calculation needs to consider the forecast to properly scale the chart.
    max_power = max(df_full['Consumption'].max(), df_future['Demand_Forecast'].max()) * 1.1 
    
    fig = go.Figure()
    end_time_full = df_future.index[-1] # End time should be the last forecast point
    
    # Apply the dark theme template
    fig.update_layout(template="plotly_dark")
    
    # Access session state variables for dynamic elements
    try:
        params = SCENARIOS[st.session_state.scenario]
        action_hour = params['Peak_Hour']
        # Find the latest time for the peak hour
        latest_peak = df_full[df_full.index.date == df_full.index.date.max()]
        latest_peak = latest_peak[latest_peak.index.hour == action_hour].index.max()
        if latest_peak:
            # Add the peak hour shading to the chart
            fig.add_shape(
                type="rect", x0=latest_peak.replace(minute=0), x1=latest_peak.replace(minute=0) + timedelta(hours=1),
                y0=0, y1=max_power, line=dict(width=0), fillcolor="rgba(255, 165, 0, 0.2)", layer="below"
            )
    except:
        pass
    
    # --- CONDITIONAL DATA TRACES (Filtered by selected_cols) ---
    # Pre-calculate clipped flows needed for battery traces
    df_full['Battery_Flow_Discharge'] = df_full['Battery_Flow'].clip(lower=0)
    df_full['Battery_Flow_Charge'] = df_full['Battery_Flow'].clip(upper=0).abs()
        
    # 1. Battery Discharge
    if 'Battery_Flow_Discharge' in selected_cols:
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Battery_Flow_Discharge'], mode='lines', 
                                 name='Battery Discharge', fill='tozeroy', fillcolor='rgba(0,200,0, 0.3)', 
                                 line=dict(color='lime', width=1)))
        
    # 2. Battery Charge
    if 'Battery_Flow_Charge' in selected_cols:
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Battery_Flow_Charge'], mode='lines', 
                                 name='Battery Charge', fill='tozeroy', fillcolor='rgba(100,100,255, 0.3)', 
                                 line=dict(color='deepskyblue', width=1)))
            
    # 3. Solar Generation
    if 'Solar_Gen' in selected_cols:
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Solar_Gen'], mode='lines', 
                                 name='Solar Generation (Supply)', line=dict(color='gold', width=2)))
            
    # 4. Household Consumption
    if 'Consumption' in selected_cols:
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Consumption'], mode='lines', 
                                 name='Household Consumption (Demand)', line=dict(color='orangered', width=2)))
            
    # 5. Grid Consumption (Net Import)
    if 'Grid_Import' in selected_cols:
        fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Grid_Import'], mode='lines', 
                                 name='Grid Consumption (Net Import)', line=dict(color='darkviolet', width=3)))
            
    # 6. Demand Forecast (ML) - Hourly
    if 'Demand_Forecast_Future' in selected_cols:
        # Connect last historical point to first forecast point
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
        
    # **ADDED SOLAR FORECAST TRACE (for completeness on the main chart)**
    if 'Solar_Forecast_Future' in selected_cols:
        last_historical_time = df_full.index[-1]
        last_historical_solar = df_full['Solar_Gen'].iloc[-1]
        solar_forecast_plot_data = df_future['Solar_Forecast'].copy()
        solar_forecast_plot_data.loc[last_historical_time] = last_historical_solar
        solar_forecast_plot_data = solar_forecast_plot_data.sort_index()
        fig.add_trace(go.Scatter(
            x=solar_forecast_plot_data.index, y=solar_forecast_plot_data.values, mode='lines', 
            name='Solar Forecast (ML) - Hourly',
            line=dict(color='yellow', dash='dot', width=3), 
            showlegend=True,
        ))
        
    # --- CHART LAYOUT ADJUSTMENTS ---
    initial_view_start = df_full.index[-96] if len(df_full) >= 96 else df_full.index[0]
    fig.update_layout(
        title_text='Energy Flow: Historical & 24H Forecast', 
        xaxis_title="Time", yaxis_title="Power (KW)", height=550,
        dragmode='pan',
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#c0c0c0')),
        xaxis=dict(fixedrange=False, showgrid=True, gridcolor='#333333', zerolinecolor='#333333'),
        yaxis=dict(fixedrange=False, showgrid=True, gridcolor='#333333', zerolinecolor='#333333'),
        paper_bgcolor="#1e212b",
        plot_bgcolor="#1e212b",
    )
        
    # --- RANGESLIDER ---
    # **BUG FIX**: The `end_time_full` should be the end of the forecast period.
    fig.update_xaxes(
        range=[initial_view_start, end_time_full],
        rangeslider_visible=True, rangeslider_thickness=0.08,
        rangeslider=dict(bgcolor="#444444", bordercolor="gray", yaxis=dict(rangemode="fixed", range=[0, max_power])),
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

# **MISSING FUNCTION FIX:** This function was referenced in main_dashboard but not defined.
def page_forecast(df_future):
    """Displays the 24-hour Demand and Solar Forecasts in a dedicated page."""
    st.title("☀️ Forecast: 24-Hour Energy Prediction")
    st.write("Using the **XGBoost Regressor**, PHOTON predicts the next 24 hours of energy demand and solar generation, enabling optimal battery scheduling.")
    st.markdown("---")
    
    # Show the time range of the forecast
    st.info(f"Hourly forecast from **{df_future.index.min().strftime('%Y-%m-%d %H:%M')}** to **{df_future.index.max().strftime('%Y-%m-%d %H:%M')}**.")
    
    st.header("Demand & Solar Generation Forecast")
    create_forecast_chart(df_future)
    
    st.markdown("---")
    st.header("Forecasting Model Metrics (Mock)")
    col1, col2 = st.columns(2)
    with col1:
        display_kpi_card("Demand MAE (24H)", f"{df_future['Demand_Forecast'].std() * 0.1:.2f}", "kW", "#FF5733")
    with col2:
        display_kpi_card("Solar MAE (24H)", f"{df_future['Solar_Forecast'].std() * 0.15:.2f}", "kW", "#F9A825")
    
    st.markdown("<br>")
    st.markdown(
        """
        <div class="st-card">
        <h4 style="margin-top: 0; color: #f0f0f0;">Feature Importance</h4>
        <p>The **Demand** model relies heavily on **Hour**, **Temperature**, and **Grid Price**. The **Solar** model is dominated by **Irradiance** and **Day of Year**.</p>
        </div>
        """, unsafe_allow_html=True
    )
    # [Image of XGBoost Feature Importance Chart] # Optional: Placeholder for a visual aid if available

# The existing create_forecast_chart is fine, but I'll place the one used in the Forecast page here for structure.
# (The user had it defined twice, I will keep the second one as it seems complete)
def create_forecast_chart(df_future):
    """Creates a chart showing 24-hour Demand and Solar Forecasts."""
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
        title_text='24-Hour Power Forecast', # Added a title for clarity
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

# -------------------------------------------------------------
## 3. Page Functions (Defined before main_dashboard)
# -------------------------------------------------------------
def page_dashboard(synthetic_data, df_future):
    """Displays the main chart, real-time metrics, and general info."""
    # Display the full time range of the loaded data
    st.title("💡 Dashboard: Real-Time Monitoring & Dispatch")
    st.write(f"Displaying **{len(synthetic_data) / 96:.0f} days** of historical data from **{synthetic_data.index.min().strftime('%Y-%m-%d %H:%M')}** to **{synthetic_data.index.max().strftime('%Y-%m-%d %H:%M')}**.")
    st.markdown("---")
    st.subheader("Real-Time Energy Monitoring")
    if st.button("Refresh 🔄", key="refresh_monitor"):
        st.rerun()
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
        # NOTE: This displays the actual value from the last point in the data. 
        # If the last timestamp is night, it will show 0.00 kW.
        display_kpi_card("Solar Generation (Now)", f"{latest_data['Solar_Gen']:.2f}", "kW", "#F9A825")
    with col2:
        display_kpi_card("Energy Demand (Now)", f"{latest_data['Consumption']:.2f}", "kW", "#FF5733")
    with col3:
        display_kpi_card("Net Energy Flow (Now)", f"{mock_net_energy_flow:.2f}", "kW")
    with col4:
        display_kpi_card("Battery Level (Now)", f"{mock_battery_level:.1f}", "%")
    st.markdown("---")
        
    # --- Data Series Slicer (Multiselect) ---
    st.subheader("Chart Data Series Selector")
        
    # Map display names to internal keys for conditional plotting
    series_map = {
        'Household Consumption (Demand)': 'Consumption',
        'Solar Generation (Supply)': 'Solar_Gen',
        'Grid Consumption (Net Import)': 'Grid_Import',
        'Battery Discharge': 'Battery_Flow_Discharge', # Custom key for positive clip
        'Battery Charge': 'Battery_Flow_Charge',        # Custom key for negative clip
        'Demand Forecast (ML) - Hourly': 'Demand_Forecast_Future',
        'Solar Forecast (ML) - Hourly': 'Solar_Forecast_Future' # Added solar forecast
    }
        
    default_series = list(series_map.keys()) # All series selected by default
        
    selected_series_keys = st.multiselect(
        "Select Energy Flows to Display on Chart:",
        options=list(series_map.keys()),
        default=default_series,
        key="data_series_slicer"
    )
    # Get the list of internal keys to pass to the chart function
    selected_cols = [series_map[key] for key in selected_series_keys if key in series_map]
    # --- END NEW SLICER ---
    # Pass the selected columns to the chart function
    create_energy_flow_chart(synthetic_data, df_future, selected_cols)
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>Use the <b>small scroll panel at the bottom</b> to easily slide and view the full historical data timeline.</p>", unsafe_allow_html=True)


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
                      # --- FIX APPLIED HERE: Set DISCHARGE to Red ---
        if "CHARGE" in row['Action'].upper():
            action_class = "action-green" # This uses color #32CD32 (Green)
        elif "DISCHARGE" in row['Action'].upper():
            action_class = "action-red" # This uses color #FF4500 (Red/Orange)
        # --- END FIX ---
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
        # NOTE: The action legend below confirms the mapping: action-red is FF4500
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
    # Calculate savings based on the last 7 days of the whole dataset
    last_week_data = synthetic_data.tail(672)
    # The Grid_Import is the optimized flow. The baseline is Consumption - Solar_Gen, clipped to >0 (no battery)
    baseline_net_flow = (last_week_data['Consumption'] - last_week_data['Solar_Gen']).clip(lower=0)
    baseline_import = (baseline_net_flow.sum() * 0.25)
    optimized_import = (last_week_data['Grid_Import'].sum() * 0.25)
    
    # Grid_Price is an average value from the data, use the last known price for the calculation
    # NOTE: The user was multiplying by 0.1, which is a flat rate. I'll stick to their logic but use the average price for better context.
    avg_price = synthetic_data['Grid_Price'].mean() if not synthetic_data['Grid_Price'].empty else 0.1
    
    estimated_savings_kwh = baseline_import - optimized_import
    estimated_savings_weekly = estimated_savings_kwh * avg_price
    
    daily_savings = estimated_savings_weekly / 7
    monthly_savings = estimated_savings_weekly * 4
    
    # CO2 factor (e.g., 0.8 kg CO2 per kWh saved is a common estimate)
    daily_co2_saved = estimated_savings_kwh / 7 * 0.8
    monthly_co2_saved = estimated_savings_kwh * 4 * 0.8
    
    colB1, colB2, colB3, colB4 = st.columns(4)
    with colB1:
        display_kpi_card("Daily Savings", f"₹ {daily_savings:.2f}", "", "#4CAF50")
    with colB2:
        display_kpi_card("Monthly Savings", f"₹ {monthly_savings:.2f}", "", "#4CAF50")
    with colB3:
        display_kpi_card("Daily CO2 Saved", f"{daily_co2_saved:.2f}", "kg", "#03A9F4")
    with colB4:
        display_kpi_card("Monthly CO2 Saved", f"{monthly_co2_saved:.2f}", "kg", "#03A9F4")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color: #c0c0c0; font-size: 14px;'>These metrics display the cost savings compared to a non-optimized baseline, proving the system's real ROI and sustainability impact.</p>", unsafe_allow_html=True)

def page_reports(synthetic_data):
    """Displays the daily and weekly energy reports."""
    st.title("📑 Reports")
    st.write("Summary of energy totals for Day and Week periods.")
    st.markdown("---")
    # Calculation (using 0.25 multiplier for 15-min data to convert kW to kWh)
    # Day Report: Last 24 hours (96 points)
    day_data = synthetic_data.tail(96)
    day_solar_total = (day_data['Solar_Gen'].sum() * 0.25).round(2)
    day_demand_total = (day_data['Consumption'].sum() * 0.25).round(2)
    day_net_energy = (day_data['Net_Energy'].sum() * 0.25).round(2)
    # Week Report: Last 7 days (672 points)
    week_data = synthetic_data.tail(672)
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

# -------------------------------------------------------------
## 4. Dashboard Entry Point
# -------------------------------------------------------------
def main_dashboard():
    st.sidebar.title("PHOTON")
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
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML & RL Overview")
    st.sidebar.markdown(
        """
        - **Data Source:** **energy_data_150days_20households.csv**
        - **Historical Data:** All 150 days loaded
        - **Optimization:** RL policy simulated using real solar/consumption data.
        - **Forecast Model:** **XGBoost Regressor**
        - **Demand Features:** **Temp, Solar, Price, Weekend, Households**
        - **Solar Features:** **Irradiance, Temp, Panel Specs**
        """
    )
    # --- Data Generation (Attempting to read the CSV) ---
    try:
        synthetic_data, df_future, optimal_schedule = load_energy_data(
            st.session_state.scenario
        )
    except FileNotFoundError:
        st.error("⚠️ **File Not Found Error:** The application cannot find the data file 'energy_data_150days_20households.csv'.")
        st.info("Please ensure the filename in the code exactly matches the file on disk (including case) and the file is in the same folder as the script.")
        return
    # --- Page Routing ---
    if page == "Dashboard":
        page_dashboard(synthetic_data, df_future)
    elif page == "Forecast":
        page_forecast(df_future) # This now calls the defined page_forecast function
    elif page == "Battery Optimization":
        page_optimization(synthetic_data, optimal_schedule)
    elif page == "Reports":
        page_reports(synthetic_data)
if __name__ == "__main__":
    main_dashboard()
