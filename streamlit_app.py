import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="TradingView",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Exact TradingView CSS Styling
# -------------------------------
st.markdown("""
<style>
    /* Complete TradingView dark theme */
    .stApp {
        background-color: #0D1421;
    }
    
    .main .block-container {
        padding: 0;
        background-color: #0D1421;
        max-width: 100%;
    }
    
    /* TradingView top toolbar */
    .tv-toolbar {
        background: #1E2329;
        height: 48px;
        display: flex;
        align-items: center;
        padding: 0 16px;
        border-bottom: 1px solid #2B3139;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        gap: 16px;
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .tv-symbol-info {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .tv-symbol {
        background: #2962FF;
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    
    .tv-price {
        color: #FFFFFF;
        font-size: 18px;
        font-weight: 700;
        margin-right: 8px;
    }
    
    .tv-change {
        font-size: 14px;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .tv-change.positive {
        color: #02C076;
        background: rgba(2, 192, 118, 0.1);
    }
    
    .tv-change.negative {
        color: #FF4976;
        background: rgba(255, 73, 118, 0.1);
    }
    
    .tv-timeframes {
        display: flex;
        gap: 4px;
        margin-left: auto;
        margin-right: 16px;
    }
    
    .tv-timeframe {
        background: transparent;
        color: #B2B5BE;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        border: 1px solid transparent;
    }
    
    .tv-timeframe.active {
        background: #2962FF;
        color: white;
    }
    
    .tv-timeframe:hover {
        background: rgba(41, 98, 255, 0.1);
        color: #2962FF;
    }
    
    .tv-time {
        color: #B2B5BE;
        font-size: 13px;
        font-weight: 500;
    }
    
    /* Chart container */
    .tv-chart-container {
        background: #0D1421;
        height: calc(100vh - 120px);
        position: relative;
    }
    
    .stPlotlyChart {
        background: #0D1421 !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1E2329;
        border-right: 1px solid #2B3139;
    }
    
    /* Remove all default spacing */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        gap: 0;
    }
    
    /* Volume and indicators text colors */
    .tv-volume-label {
        position: absolute;
        top: 10px;
        left: 10px;
        color: #B2B5BE;
        font-size: 12px;
        font-weight: 600;
        z-index: 1000;
    }
    
    /* Price labels on right */
    .tv-price-labels {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    
    .tv-price-label {
        background: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Data Loading Functions
# -------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            content = str(uploaded_file.read(), 'latin1')
            df = pd.read_csv(StringIO(content), header=None, encoding='latin1', on_bad_lines='skip')
        else:
            try:
                df = pd.read_csv('Latest file.csv', header=None, encoding='latin1', on_bad_lines='skip')
            except FileNotFoundError:
                return generate_realistic_data()
        
        if df.shape[1] >= 7:
            df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'Volume'] + \
                        [f'col_{i}' for i in range(7, df.shape[1])]
        else:
            return generate_realistic_data()
            
        for col in ['open', 'high', 'low', 'close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['close'])
        df = df[df['close'] > 0]
        
        try:
            df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        except:
            return generate_realistic_data()
        
        return df.sort_values('Datetime').reset_index(drop=True)
        
    except:
        return generate_realistic_data()

def generate_realistic_data():
    """Generate realistic AAPL-style data"""
    np.random.seed(42)
    n_points = 500
    base_price = 185
    
    # Generate dates (6 months of hourly data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    dates = dates[:n_points]
    
    # Create realistic price movement with volatility
    returns = np.random.normal(0.0002, 0.02, n_points)  # Realistic daily returns
    trend = np.linspace(0, 0.1, n_points)  # Slight upward trend
    volatility_clusters = np.abs(np.random.normal(0, 0.01, n_points))
    
    # Generate price series
    log_prices = np.log(base_price) + np.cumsum(returns + trend/n_points)
    prices = np.exp(log_prices)
    
    data = []
    for i in range(n_points):
        base = prices[i]
        
        # Generate realistic OHLC
        volatility = volatility_clusters[i] * base
        
        open_price = base + np.random.normal(0, volatility * 0.3)
        close_price = base + np.random.normal(0, volatility * 0.3)
        
        high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility * 0.5))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility * 0.5))
        
        # Realistic volume (higher on big moves)
        price_change = abs(close_price - open_price) / open_price
        base_volume = np.random.lognormal(15, 0.5)  # Log-normal distribution
        volume_multiplier = 1 + price_change * 10
        volume = int(base_volume * volume_multiplier)
        
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
# Sidebar
# -------------------------------
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

# Load data
df = load_data(uploaded_file)
if df is None:
    st.stop()

# -------------------------------
# TradingView Toolbar
# -------------------------------
current_price = df['close'].iloc[-1]
prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

change_class = "positive" if price_change >= 0 else "negative"
change_sign = "+" if price_change >= 0 else ""

current_time = datetime.now().strftime("%H:%M:%S")

st.markdown(f"""
<div class="tv-toolbar">
    <div class="tv-symbol-info">
        <span class="tv-symbol">AAPL</span>
        <span class="tv-price">{current_price:.2f}</span>
        <span class="tv-change {change_class}">{change_sign}{price_change:.2f} {change_sign}{price_change_pct:.2f}%</span>
    </div>
    
    <div class="tv-timeframes">
        <span class="tv-timeframe">1m</span>
        <span class="tv-timeframe">5m</span>
        <span class="tv-timeframe active">1h</span>
        <span class="tv-timeframe">1D</span>
        <span class="tv-timeframe">1W</span>
    </div>
    
    <div class="tv-time">{current_time}</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Create Multi-Panel Chart
# -------------------------------
# Filter to recent data for better visualization
df_recent = df.tail(200).copy()
df_recent['Date'] = df_recent['Datetime'].dt.strftime('%d %b')

# Calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df_recent['RSI'] = calculate_rsi(df_recent['close'])

# Create subplot layout matching TradingView
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.7, 0.2, 0.1],
    subplot_titles=('', '', '')
)

# TradingView color scheme
tv_colors = {
    'bg': '#0D1421',
    'grid': '#1E2329',
    'text': '#B2B5BE',
    'green': '#02C076',
    'red': '#FF4976',
    'blue': '#2962FF',
    'orange': '#FF8A00',
    'purple': '#9C40FF'
}

# Main candlestick chart
fig.add_trace(go.Candlestick(
    x=df_recent['Date'],
    open=df_recent['open'],
    high=df_recent['high'],
    low=df_recent['low'],
    close=df_recent['close'],
    name='',
    increasing_line_color=tv_colors['green'],
    decreasing_line_color=tv_colors['red'],
    increasing_fillcolor=tv_colors['green'],
    decreasing_fillcolor=tv_colors['red'],
    line=dict(width=1),
    showlegend=False
), row=1, col=1)

# Support and resistance lines
current_high = df_recent['high'].max()
current_low = df_recent['low'].min()
support_level = current_low + (current_high - current_low) * 0.2
resistance_level = current_high - (current_high - current_low) * 0.15

fig.add_hline(y=support_level, line_dash="dot", line_color=tv_colors['green'], 
             line_width=1, opacity=0.8, row=1, col=1)
fig.add_hline(y=resistance_level, line_dash="dot", line_color=tv_colors['red'], 
             line_width=1, opacity=0.8, row=1, col=1)

# Volume bars
volume_colors = []
for i in range(len(df_recent)):
    if df_recent['close'].iloc[i] >= df_recent['open'].iloc[i]:
        volume_colors.append('rgba(2, 192, 118, 0.8)')
    else:
        volume_colors.append('rgba(255, 73, 118, 0.8)')

fig.add_trace(go.Bar(
    x=df_recent['Date'],
    y=df_recent['Volume'],
    name='Volume',
    marker_color=volume_colors,
    showlegend=False
), row=2, col=1)

# RSI indicator
fig.add_trace(go.Scatter(
    x=df_recent['Date'],
    y=df_recent['RSI'],
    mode='lines',
    name='RSI',
    line=dict(color=tv_colors['purple'], width=2),
    showlegend=False
), row=3, col=1)

# RSI reference lines
fig.add_hline(y=70, line_dash="dash", line_color=tv_colors['text'], 
             line_width=1, opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color=tv_colors['text'], 
             line_width=1, opacity=0.5, row=3, col=1)

# Layout styling to match TradingView exactly
fig.update_layout(
    plot_bgcolor=tv_colors['bg'],
    paper_bgcolor=tv_colors['bg'],
    font=dict(family="SF Pro Display, -apple-system, BlinkMacSystemFont", 
              size=11, color=tv_colors['text']),
    showlegend=False,
    height=700,
    margin=dict(l=0, r=60, t=0, b=0),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor='rgba(30, 35, 41, 0.95)',
        bordercolor=tv_colors['grid'],
        font_size=11,
        font_color=tv_colors['text']
    ),
    xaxis=dict(rangeslider=dict(visible=False))
)

# Update all axes
for i in range(1, 4):
    fig.update_xaxes(
        showgrid=True,
        gridcolor=tv_colors['grid'],
        gridwidth=1,
        showline=False,
        tickfont=dict(size=10, color=tv_colors['text']),
        showticklabels=(i == 3),
        row=i, col=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor=tv_colors['grid'],
        gridwidth=1,
        showline=False,
        tickfont=dict(size=10, color=tv_colors['text']),
        side='right',
        row=i, col=1
    )

# Format specific axes
fig.update_yaxes(tickformat='.2f', row=1, col=1)
fig.update_yaxes(tickformat='.0s', row=2, col=1)
fig.update_yaxes(range=[0, 100], tickformat='.0f', row=3, col=1)

# Display the chart
st.markdown('<div class="tv-chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={
    'displayModeBar': False,
    'displaylogo': False,
    'scrollZoom': True,
    'doubleClick': 'reset+autosize'
})
st.markdown('</div>', unsafe_allow_html=True)

# Add price labels and volume indicators as overlays
st.markdown(f"""
<div class="tv-volume-label">Vol {df_recent['Volume'].iloc[-1]:,.0f}</div>
<div class="tv-price-labels">
    <div class="tv-price-label">{resistance_level:.2f}</div>
    <div class="tv-price-label" style="margin-top: 100px;">{current_price:.2f}</div>
    <div class="tv-price-label" style="margin-top: 100px;">{support_level:.2f}</div>
</div>
""", unsafe_allow_html=True)
