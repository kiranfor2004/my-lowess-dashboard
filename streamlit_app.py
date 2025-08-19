import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="TradingView Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Authentic TradingView Dark Theme CSS
# -------------------------------
st.markdown("""
<style>
    /* Full dark theme matching TradingView */
    .stApp {
        background-color: #131722;
        color: #D1D4DC;
    }
    
    .main .block-container {
        padding: 0rem;
        background-color: #131722;
        max-width: 100%;
        margin: 0;
    }
    
    /* TradingView header bar */
    .tv-header {
        background: #1E222D;
        border-bottom: 1px solid #2A2E39;
        padding: 8px 16px;
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Trebuchet MS", Arial, sans-serif;
        display: flex;
        align-items: center;
        gap: 16px;
        font-size: 13px;
        color: #D1D4DC;
        height: 48px;
    }
    
    .tv-symbol {
        background: #2962FF;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .tv-price {
        font-weight: 700;
        font-size: 16px;
        color: #FFFFFF;
    }
    
    .tv-change.up {
        color: #4CAF50;
    }
    
    .tv-change.down {
        color: #FF5252;
    }
    
    .tv-stats {
        color: #9598A1;
        font-size: 12px;
    }
    
    /* Chart container - exact TradingView styling */
    .stPlotlyChart {
        background-color: #131722;
        margin: 0;
        padding: 0;
        border: none;
    }
    
    /* Hide all Streamlit branding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar dark theme */
    .css-1d391kg {
        background-color: #1E222D;
        border-right: 1px solid #2A2E39;
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: #D1D4DC;
    }
    
    /* Remove all padding and margins */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        gap: 0rem;
    }
    
    /* Dark theme for inputs */
    .stSelectbox > div > div {
        background-color: #2A2E39;
        color: #D1D4DC;
        border: 1px solid #434651;
    }
    
    .stFileUploader > div {
        background-color: #2A2E39;
        border: 1px solid #434651;
    }
    
    /* Remove white backgrounds */
    .stDateInput > div > div {
        background-color: #2A2E39;
        color: #D1D4DC;
        border: 1px solid #434651;
    }
    
    /* Metrics dark theme */
    [data-testid="metric-container"] {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        color: #D1D4DC;
    }
    
    [data-testid="metric-container"] > div {
        color: #D1D4DC;
    }
    
    /* Text elements */
    .stMarkdown, .stText {
        color: #D1D4DC;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #D1D4DC !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Technical Indicators
# -------------------------------
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period).mean()

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_support_resistance(prices, window=20):
    """Calculate support and resistance levels"""
    highs = prices.rolling(window=window).max()
    lows = prices.rolling(window=window).min()
    
    resistance_levels = []
    support_levels = []
    
    for i in range(window, len(prices) - window):
        if prices.iloc[i] == highs.iloc[i]:
            resistance_levels.append(prices.iloc[i])
        if prices.iloc[i] == lows.iloc[i]:
            support_levels.append(prices.iloc[i])
    
    resistance_levels = sorted(set(resistance_levels), reverse=True)[:2]
    support_levels = sorted(set(support_levels))[:2]
    
    return resistance_levels, support_levels

# -------------------------------
# Data Loading
# -------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    """Load and process trading data"""
    try:
        if uploaded_file is not None:
            content = str(uploaded_file.read(), 'latin1')
            df = pd.read_csv(StringIO(content), header=None, encoding='latin1', on_bad_lines='skip')
        else:
            try:
                df = pd.read_csv('Latest file.csv', header=None, encoding='latin1', on_bad_lines='skip')
            except FileNotFoundError:
                return generate_sample_data()
        
        if df.shape[1] >= 11:
            df = df.iloc[:, :11]
            df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'Volume',
                         'vix_open', 'vix_high', 'vix_low', 'vix_close']
        elif df.shape[1] >= 7:
            df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'Volume'] + \
                        [f'col_{i}' for i in range(7, df.shape[1])]
        else:
            st.error("❌ Expected at least 7 columns")
            return None
            
        numeric_columns = ['open', 'high', 'low', 'close', 'Volume']
        if 'vix_open' in df.columns:
            numeric_columns.extend(['vix_open', 'vix_high', 'vix_low', 'vix_close'])
            
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['close'])
        df = df[df['close'] > 0]
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['close']]
        
        if df.empty:
            st.error("❌ No valid data")
            return None
        
        try:
            df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S')
        except:
            try:
                df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
            except:
                try:
                    df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                except:
                    st.error("❌ Could not parse datetime")
                    return None
        
        df = df.sort_values('Datetime').reset_index(drop=True)
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load data: {str(e)}")
        return None

def generate_sample_data():
    """Generate realistic sample trading data"""
    np.random.seed(42)
    n_points = 1000
    base_price = 380
    
    start_date = datetime.now() - timedelta(days=180)
    dates = pd.date_range(start=start_date, periods=n_points, freq='1H')
    
    # Create realistic price movements with trends
    price_changes = np.random.normal(0, 1.5, n_points)
    trend = np.sin(np.arange(n_points) / 100) * 10
    noise = np.random.normal(0, 0.5, n_points)
    
    prices = base_price + np.cumsum(price_changes + trend * 0.2 + noise)
    
    data = []
    for i in range(n_points):
        open_price = prices[i] + np.random.normal(0, 0.5)
        close_price = prices[i] + np.random.normal(0, 0.5)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 1))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 1))
        volume = np.random.randint(50000, 500000)
        
        data.append({
            'Datetime': dates[i],
            'date': dates[i].strftime('%d-%m-%Y'),
            'time': dates[i].strftime('%H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'Volume': volume
        })
    
    return pd.DataFrame(data)

# -------------------------------
# Sidebar (minimal)
# -------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Upload Data")
    uploaded_file = st.file_uploader("📁 CSV File", type=['csv'])

# Default settings
rsi_period = 14
show_volume = True

# -------------------------------
# Load Data
# -------------------------------
df = load_data(uploaded_file=uploaded_file)
if df is None:
    st.stop()

# -------------------------------
# TradingView Header
# -------------------------------
current_price = df['close'].iloc[-1]
prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
change_class = "up" if price_change >= 0 else "down"
change_sign = "+" if price_change >= 0 else ""

st.markdown(f"""
<div class="tv-header">
    <span class="tv-symbol">MARKET</span>
    <span class="tv-price">{current_price:.2f}</span>
    <span class="tv-change {change_class}">{change_sign}{price_change:.2f} ({change_sign}{price_change_pct:.2f}%)</span>
    <span class="tv-stats">O {df['open'].iloc[-1]:.2f}</span>
    <span class="tv-stats">H {df['high'].iloc[-1]:.2f}</span>
    <span class="tv-stats">L {df['low'].iloc[-1]:.2f}</span>
    <span class="tv-stats">C {df['close'].iloc[-1]:.2f}</span>
    <span class="tv-stats">Vol {df['Volume'].iloc[-1]:,.0f}</span>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Date Filter (minimal)
# -------------------------------
col1, col2, col3 = st.columns([1, 1, 4])
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()

with col1:
    from_date = st.date_input("From", max_date - timedelta(days=30), min_value=min_date, max_value=max_date, label_visibility="collapsed")
with col2:
    to_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")

# Filter data
mask = (df['Datetime'] >= pd.Timestamp(from_date)) & (df['Datetime'] <= pd.Timestamp(to_date) + pd.Timedelta(days=1))
df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning("No data in range")
    st.stop()

df_filtered.sort_values('Datetime', inplace=True)
df_filtered['TimeLabel'] = df_filtered['Datetime'].dt.strftime('%d %b')

# Calculate indicators
df_filtered['RSI'] = calculate_rsi(df_filtered['close'], period=rsi_period)
df_filtered['EMA20'] = calculate_ema(df_filtered['close'], 20)
df_filtered['SMA50'] = calculate_sma(df_filtered['close'], 50)

# Support/Resistance
resistance_levels, support_levels = calculate_support_resistance(df_filtered['close'])

# -------------------------------
# Create TradingView-Style Chart
# -------------------------------
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.01,
    row_heights=[0.65, 0.25, 0.1],
    subplot_titles=("", "", "")
)

# Authentic TradingView colors
colors = {
    'bg': '#131722',
    'grid': '#2A2E39',
    'text': '#D1D4DC',
    'bull': '#26a69a',
    'bear': '#ef5350',
    'volume_bull': '#26a69a40',
    'volume_bear': '#ef535040',
    'ema': '#FF9800',
    'sma': '#2196F3',
    'rsi': '#9C27B0',
    'support': '#4CAF50',
    'resistance': '#FF5252'
}

# --- Main Price Chart ---
fig.add_trace(go.Candlestick(
    x=df_filtered['TimeLabel'],
    open=df_filtered['open'],
    high=df_filtered['high'],
    low=df_filtered['low'],
    close=df_filtered['close'],
    name='Price',
    increasing_line_color=colors['bull'],
    decreasing_line_color=colors['bear'],
    increasing_fillcolor=colors['bull'],
    decreasing_fillcolor=colors['bear'],
    line=dict(width=0.8),
    showlegend=False
), row=1, col=1)

# Moving Averages
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], y=df_filtered['EMA20'],
    mode='lines', name='EMA20',
    line=dict(color=colors['ema'], width=1.5),
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], y=df_filtered['SMA50'],
    mode='lines', name='SMA50',
    line=dict(color=colors['sma'], width=1.5),
    showlegend=False
), row=1, col=1)

# Support/Resistance levels
for level in resistance_levels:
    fig.add_hline(y=level, line_dash="dot", line_color=colors['resistance'], 
                 line_width=1, opacity=0.8, row=1, col=1)
for level in support_levels:
    fig.add_hline(y=level, line_dash="dot", line_color=colors['support'], 
                 line_width=1, opacity=0.8, row=1, col=1)

# --- Volume Chart ---
volume_colors = [colors['volume_bull'] if close >= open else colors['volume_bear'] 
                for close, open in zip(df_filtered['close'], df_filtered['open'])]

fig.add_trace(go.Bar(
    x=df_filtered['TimeLabel'],
    y=df_filtered['Volume'],
    name='Volume',
    marker_color=volume_colors,
    showlegend=False
), row=2, col=1)

# --- RSI Chart ---
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], y=df_filtered['RSI'],
    mode='lines', name='RSI',
    line=dict(color=colors['rsi'], width=1.5),
    showlegend=False
), row=3, col=1)

# RSI levels
fig.add_hline(y=70, line_dash="dash", line_color=colors['resistance'], line_width=0.8, opacity=0.6, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color=colors['support'], line_width=0.8, opacity=0.6, row=3, col=1)
fig.add_hline(y=50, line_dash="solid", line_color=colors['grid'], line_width=0.5, opacity=0.4, row=3, col=1)

# --- Authentic TradingView Layout ---
fig.update_layout(
    plot_bgcolor=colors['bg'],
    paper_bgcolor=colors['bg'],
    font=dict(family="-apple-system, BlinkMacSystemFont, Trebuchet MS", size=11, color=colors['text']),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=colors['bg'], 
        font_size=11, 
        font_color=colors['text'],
        bordercolor=colors['grid']
    ),
    height=750,
    margin=dict(l=0, r=60, t=0, b=0),
    showlegend=False,
    dragmode='pan',
    xaxis=dict(rangeslider=dict(visible=False))
)

# TradingView-style axes
for i in range(1, 4):
    fig.update_xaxes(
        row=i, col=1,
        type='category',
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showline=False,
        tickfont=dict(size=10, color=colors['text']),
        showticklabels=(i == 3),
        fixedrange=False
    )
    
    fig.update_yaxes(
        row=i, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showline=False,
        tickfont=dict(size=10, color=colors['text']),
        side='right',
        fixedrange=False
    )

# Format axes
fig.update_yaxes(title_text="", tickformat=".2f", row=1, col=1)
fig.update_yaxes(title_text="", tickformat=".0s", row=2, col=1)
fig.update_yaxes(title_text="", range=[0, 100], tickformat=".0f", row=3, col=1)

# Display chart
st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset+autosize"
})

# -------------------------------
# Statistics
# -------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Price", f"{current_price:.2f}", f"{change_sign}{price_change:.2f}")
with col2:
    high_24h = df_filtered['high'].max()
    st.metric("24h High", f"{high_24h:.2f}")
with col3:
    low_24h = df_filtered['low'].min()
    st.metric("24h Low", f"{low_24h:.2f}")
with col4:
    volume_24h = df_filtered['Volume'].sum()
    st.metric("24h Volume", f"{volume_24h:,.0f}")
with col5:
    rsi_current = df_filtered['RSI'].iloc[-1]
    st.metric("RSI", f"{rsi_current:.1f}")
