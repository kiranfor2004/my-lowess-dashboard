#!/usr/bin/env python3
"""
Streamlit Lightweight Charts Dashboard
A web-based financial dashboard using streamlit and lightweight-charts-python
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from lightweight_charts.widgets import StreamlitChart

# Page configuration
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_csv_data(csv_file='Latest file 1.csv'):
    """Load and prepare data from CSV file"""
    try:
        if not os.path.exists(csv_file):
            st.error(f"CSV file '{csv_file}' not found!")
            return None
            
        # Read CSV file
        data = pd.read_csv(csv_file)
        
        # Convert date and time to datetime
        data['datetime'] = pd.to_datetime(
            data['date'] + ' ' + data['time'], 
            format='%d-%m-%Y %H:%M:%S'
        )
        
        # Convert to Unix timestamp for lightweight charts
        data['time'] = data['datetime'].astype('int64') // 10**9
        
        # Rename columns to match lightweight charts format
        column_mapping = {
            'Volume': 'volume',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close'
        }
        data.rename(columns=column_mapping, inplace=True)
        
        # Sort by datetime
        data.sort_values('datetime', inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    result = data.copy()
    result['value'] = result['close'].rolling(window=period).mean()
    return result[['time', 'value']].dropna()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    result = data.copy()
    result['value'] = result['close'].ewm(span=period).mean()
    return result[['time', 'value']].dropna()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    result = data.copy()
    result['value'] = rsi
    return result[['time', 'value']].dropna()

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        'upper': pd.DataFrame({'time': data['time'], 'value': upper_band}).dropna(),
        'middle': pd.DataFrame({'time': data['time'], 'value': sma}).dropna(),
        'lower': pd.DataFrame({'time': data['time'], 'value': lower_band}).dropna()
    }

def get_timeframe_data(data, timeframe_key):
    """Get data for specified timeframe"""
    timeframes = {
        '1 Day': 288,      # 288 * 5min = 1 day
        '3 Days': 864,     # 3 days
        '1 Week': 2016,    # 7 days
        '2 Weeks': 4032,   # 14 days
        '1 Month': 8640,   # 30 days
        '3 Months': 25920, # 90 days
        '6 Months': 51840, # 180 days
        '1 Year': 103680,  # 360 days
        'All Data': -1     # All available data
    }
    
    periods = timeframes[timeframe_key]
    
    if periods == -1:  # All data
        result = data.copy()
    else:
        # Get the last N periods
        result = data.tail(periods).copy()
        
    # Select required columns for lightweight charts
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    return result[required_cols].copy()

def create_sample_data():
    """Create sample data if CSV is not available"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='5T')
    np.random.seed(42)
    
    data = []
    price = 15000  # Starting price similar to your data
    
    for i, date in enumerate(dates):
        if i >= 1000:  # Limit to 1000 points for demo
            break
            
        change = np.random.normal(0, 10)
        price += change
        
        open_price = price + np.random.normal(0, 5)
        high_price = max(open_price, price) + abs(np.random.normal(0, 10))
        low_price = min(open_price, price) - abs(np.random.normal(0, 10))
        close_price = price + np.random.normal(0, 5)
        volume = np.random.randint(0, 1000)
        
        data.append({
            'time': int(date.timestamp()),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📈 Financial Dashboard")
    st.markdown("### Lightweight Charts Python - CSV Data Analysis")
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Load data
    data = load_csv_data()
    
    if data is None:
        st.warning("Could not load CSV data. Using sample data instead.")
        data = create_sample_data()
        st.info("📊 Using sample data with 1,000 data points")
    else:
        # Data information
        start_date = data['datetime'].min()
        end_date = data['datetime'].max()
        total_points = len(data)
        
        st.sidebar.success("✅ Data loaded successfully!")
        st.sidebar.info(f"""
        **Data Information:**
        - **Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
        - **Total Points:** {total_points:,}
        - **Frequency:** 5-minute intervals
        """)
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ['1 Day', '3 Days', '1 Week', '2 Weeks', '1 Month', '3 Months', '6 Months', '1 Year', 'All Data'],
        index=4  # Default to 1 Month
    )
    
    # Technical indicators
    st.sidebar.subheader("Technical Indicators")
    show_sma20 = st.sidebar.checkbox("SMA 20", value=True)
    show_sma50 = st.sidebar.checkbox("SMA 50", value=True)
    show_ema12 = st.sidebar.checkbox("EMA 12", value=False)
    show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
    
    # Chart options
    st.sidebar.subheader("Chart Options")
    chart_height = st.sidebar.slider("Chart Height", 400, 800, 600)
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    
    # Get data for selected timeframe
    chart_data = get_timeframe_data(data, timeframe)
    
    if chart_data is None or len(chart_data) == 0:
        st.error("No data available for selected timeframe.")
        return
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("📊 Data Statistics")
        
        # Current data stats
        latest = chart_data.iloc[-1]
        previous = chart_data.iloc[-2] if len(chart_data) > 1 else latest
        
        price_change = latest['close'] - previous['close']
        price_change_pct = (price_change / previous['close']) * 100 if previous['close'] != 0 else 0
        
        # Display metrics
        st.metric(
            label="Current Price",
            value=f"{latest['close']:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
        
        st.metric(
            label="High",
            value=f"{chart_data['high'].max():.2f}"
        )
        
        st.metric(
            label="Low", 
            value=f"{chart_data['low'].min():.2f}"
        )
        
        st.metric(
            label="Data Points",
            value=f"{len(chart_data):,}"
        )
        
        # Price distribution
        st.subheader("📈 Price Analysis")
        
        # Recent performance
        if len(chart_data) >= 20:
            recent_high = chart_data.tail(100)['high'].max()
            recent_low = chart_data.tail(100)['low'].min()
            current_price = latest['close']
            
            position_in_range = ((current_price - recent_low) / (recent_high - recent_low)) * 100
            
            st.write(f"**Position in Recent Range:** {position_in_range:.1f}%")
            st.progress(position_in_range / 100)
            
            # Support and resistance
            st.write(f"**Recent High:** {recent_high:.2f}")
            st.write(f"**Recent Low:** {recent_low:.2f}")
    
    with col1:
        st.subheader(f"💹 Price Chart - {timeframe}")
        
        # Create chart
        chart = StreamlitChart(width=900, height=chart_height)
        
        # Set main data
        chart.set(chart_data)
        
        # Add technical indicators
        indicators_added = []
        
        if show_sma20 and len(chart_data) >= 20:
            sma20_data = calculate_sma(chart_data, 20)
            sma20 = chart.create_line('SMA 20', color='#FF6B6B', width=2)
            sma20.set(sma20_data)
            indicators_added.append('SMA 20')
        
        if show_sma50 and len(chart_data) >= 50:
            sma50_data = calculate_sma(chart_data, 50)
            sma50 = chart.create_line('SMA 50', color='#4ECDC4', width=2)
            sma50.set(sma50_data)
            indicators_added.append('SMA 50')
        
        if show_ema12 and len(chart_data) >= 12:
            ema12_data = calculate_ema(chart_data, 12)
            ema12 = chart.create_line('EMA 12', color='#45B7D1', width=2)
            ema12.set(ema12_data)
            indicators_added.append('EMA 12')
        
        if show_bb and len(chart_data) >= 20:
            bb_data = calculate_bollinger_bands(chart_data, 20)
            
            bb_upper = chart.create_line('BB Upper', color='#9C27B0', width=1)
            bb_upper.set(bb_data['upper'])
            
            bb_middle = chart.create_line('BB Middle', color='#9C27B0', width=1)
            bb_middle.set(bb_data['middle'])
            
            bb_lower = chart.create_line('BB Lower', color='#9C27B0', width=1)
            bb_lower.set(bb_data['lower'])
            
            indicators_added.append('Bollinger Bands')
        
        # Style the chart
        chart.layout(
            background_color='#131722',
            text_color='#d1d4dc',
            font_size=12
        )
        
        chart.candle_style(
            up_color='#26a69a',
            down_color='#ef5350',
            border_up_color='#26a69a',
            border_down_color='#ef5350',
            wick_up_color='#26a69a',
            wick_down_color='#ef5350'
        )
        
        if show_volume:
            chart.volume_config(
                up_color='rgba(38, 166, 154, 0.7)',
                down_color='rgba(239, 83, 80, 0.7)'
            )
        
        chart.watermark(f'Data Analysis - {timeframe}', color='rgba(180, 180, 240, 0.3)')
        chart.legend(visible=True)
        
        # Load the chart
        chart.load()
        
        # Show indicators info
        if indicators_added:
            st.info(f"📈 **Active Indicators:** {', '.join(indicators_added)}")
    
    # Additional analysis
    st.subheader("📋 Technical Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Moving Averages**")
        if len(chart_data) >= 20:
            sma20_current = chart_data['close'].rolling(20).mean().iloc[-1]
            current_price = chart_data['close'].iloc[-1]
            
            if current_price > sma20_current:
                st.success("🟢 Price above SMA 20 (Bullish)")
            else:
                st.error("🔴 Price below SMA 20 (Bearish)")
        
        if len(chart_data) >= 50:
            sma50_current = chart_data['close'].rolling(50).mean().iloc[-1]
            
            if current_price > sma50_current:
                st.success("🟢 Price above SMA 50 (Bullish)")
            else:
                st.error("🔴 Price below SMA 50 (Bearish)")
    
    with col2:
        st.write("**Volatility**")
        if len(chart_data) >= 20:
            volatility = chart_data['close'].pct_change().rolling(20).std() * 100
            current_vol = volatility.iloc[-1]
            avg_vol = volatility.mean()
            
            if current_vol > avg_vol:
                st.warning(f"⚡ High volatility: {current_vol:.2f}%")
            else:
                st.info(f"📊 Normal volatility: {current_vol:.2f}%")
    
    with col3:
        st.write("**Price Action**")
        if len(chart_data) >= 5:
            recent_changes = chart_data['close'].pct_change().tail(5) * 100
            positive_days = (recent_changes > 0).sum()
            
            if positive_days >= 3:
                st.success(f"🚀 {positive_days}/5 recent periods positive")
            else:
                st.error(f"📉 {positive_days}/5 recent periods positive")

if __name__ == "__main__":
    main()
