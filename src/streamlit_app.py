def create_energy_flow_chart(df_chart, df_future):
    """
    Creates the Live Energy Flow chart using Plotly, zoomed to the last 24 hours 
    and adds a Range Slider for scrolling through the 168-hour synthetic data.
    """
    fig = go.Figure()
    
    # Define Time Axis Range (Last 24 hours of data + 15 min forecast)
    # The initial view is zoomed to the last 24 hours of the full 168H data.
    start_time = df_chart.index[0]
    end_time = df_future.index[-1]

    # Retrieve the discharge hour from the currently active scenario for highlighting
    params = SCENARIOS[st.session_state.scenario]
    action_hour = params['Peak_Hour']
    
    # --- 1. RL ACTION HIGHLIGHT ---
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
    # (Note: df_chart actually contains the last 168 hours of data now, based on the previous code update)
    # We combine the historical data with the future forecast for the visual.
    
    # Combine data frames for plotting all data points on the rangeslider
    df_full_data = df_chart.copy()
    
    # We use df_full_data for the plot to enable scrolling through the 168H data
    fig.add_trace(go.Scatter(x=df_full_data.index, y=df_full_data['Consumption'], mode='lines', name='Household Consumption (Demand)', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=df_full_data.index, y=df_full_data['Solar_Gen'], mode='lines', name='Solar Generation (Supply)', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df_full_data.index, y=df_full_data['Grid_Import'], mode='lines', name='Grid Consumption (Net Import)', line=dict(color='purple', width=3)))
    
    # 3. BATTERY FLOW AREA
    fig.add_trace(go.Scatter(x=df_full_data.index, y=df_full_data['Battery_Flow'].clip(lower=0), mode='lines', name='Battery Discharge', fill='tozeroy', fillcolor='rgba(0,128,0, 0.3)', line=dict(color='green', width=1)))
    fig.add_trace(go.Scatter(x=df_full_data.index, y=df_full_data['Battery_Flow'].clip(upper=0).abs(), mode='lines', name='Battery Charge', fill='tozeroy', fillcolor='rgba(0,0,255, 0.3)', line=dict(color='blue', width=1)))

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
        title='Energy Flow & **ML-Optimized Dispatch** (Last 24 Hours Default View)',
        xaxis_title="Time",
        yaxis_title="Power (KW)",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- KEY CHANGE: Add Range Slider for Scrolling ---
    fig.update_xaxes(
        rangeslider_visible=True,
        # Set the initial view to the last 24 hours of data
        range=[df_full_data.index[-96], end_time] 
    )

    st.plotly_chart(fig, use_container_width=True)