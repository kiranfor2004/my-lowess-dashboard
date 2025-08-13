import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="LOWESS + RSI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS for TradingView Style
# -------------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2962FF;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .sub-header {
        text-align: center;
        color: #787B86;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .stPlotlyChart {
        background-color: #FFFFFF;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-row {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E4E8;
        padding: 1rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    .stSelectbox, .stSlider {
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# App Title
# -------------------------------
st.markdown('<h1 class="main-header">📊 LOWESS + RSI Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">TradingView-style interactive charts</p>', unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    st.subheader("📈 LOWESS Settings")
    lowess_frac = st.slider("Smoothing Factor", 0.05, 0.3, 0.1, 0.01)
    lowess_iter = st.selectbox("Iterations", [1, 2, 3], index=2)
    
    st.subheader("📊 Technical Settings")
    rsi_period = st.selectbox("RSI Period", [14, 21], index=0)
    band1_mult = st.slider("Band 1 Multiplier", 0.5, 2.0, 1.0, 0.1)
    band2_mult = st.slider("Band 2 Multiplier", 1.0, 3.0, 2.0, 0.1)

# -------------------------------
# Load and Clean Data Function
# -------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, header=None, encoding='latin1', on_bad_lines='skip')
        else:
            df = pd.read_csv('Latest file.csv', header=None, encoding='latin1', on_bad_lines='skip')
        
        # Handle different column structures
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
            
    except Exception as e:
        # Fallback method
        try:
            if uploaded_file is not None:
                content = str(uploaded_file.read(), 'latin1')
            else:
                with open('Latest file.csv', 'r', encoding='latin1', errors='ignore') as f:
                    content = f.read()
            
            lines = content.split('\n')
            valid_lines = [line for line in lines if line.count(',') >= 6]
            
            if not valid_lines:
                st.error("❌ No valid data found")
                return None
                
            df = pd.read_csv(StringIO('\n'.join(valid_lines)), header=None)
            
            if df.shape[1] >= 11:
                df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'Volume',
                             'vix_open', 'vix_high', 'vix_low', 'vix_close']
            elif df.shape[1] >= 7:
                df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'Volume'] + \
                            [f'col_{i}' for i in range(7, df.shape[1])]
                
        except Exception as e2:
            st.error(f"❌ Failed to load data: {str(e2)}")
            return None

    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'Volume']
    if 'vix_open' in df.columns:
        numeric_columns.extend(['vix_open', 'vix_high', 'vix_low', 'vix_close'])
        
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['close'], inplace=True)
    
    # Parse datetime
    try:
        df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S')
    except:
        try:
            df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
        except:
            df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    return df

# -------------------------------
# Technical Indicators
# -------------------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_lowess_bands(prices, frac=0.1, it=3, band1_mult=1.0, band2_mult=2.0):
    x = np.arange(len(prices))
    y = prices.values
    
    lowess_result = lowess(y, x, frac=frac, it=it)
    trend = lowess_result[:, 1]
    
    residuals = y - trend
    window = min(50, len(prices))
    rolling_std = pd.Series(residuals).rolling(window=window, center=True).std()
    
    return {
        'trend': trend,
        'upper_1': trend + (band1_mult * rolling_std),
        'upper_2': trend + (band2_mult * rolling_std),
        'lower_1': trend - (band1_mult * rolling_std),
        'lower_2': trend - (band2_mult * rolling_std)
    }

# -------------------------------
# Load and Process Data
# -------------------------------
df = load_data(uploaded_file=uploaded_file)

if df is None:
    st.error("Please upload a CSV file or ensure 'Latest file.csv' exists")
    st.stop()

# Date filtering
col1, col2 = st.columns(2)
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()

with col1:
    from_date = st.date_input("📅 From Date", min_date, min_value=min_date, max_value=max_date)
with col2:
    to_date = st.date_input("📅 To Date", max_date, min_value=min_date, max_value=max_date)

# Filter data
mask = (df['Datetime'] >= pd.Timestamp(from_date)) & (df['Datetime'] <= pd.Timestamp(to_date) + pd.Timedelta(days=1))
df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning(f"⚠️ No data between {from_date} and {to_date}")
    st.stop()

df_filtered.sort_values('Datetime', inplace=True)
df_filtered['TimeLabel'] = df_filtered['Datetime'].dt.strftime('%Y-%m-%d %H:%M')

# Calculate indicators
df_filtered['RSI'] = calculate_rsi(df_filtered['close'], period=rsi_period)
bands = calculate_lowess_bands(df_filtered['close'], frac=lowess_frac, it=lowess_iter, 
                              band1_mult=band1_mult, band2_mult=band2_mult)

for key, value in bands.items():
    df_filtered[key.title().replace('_', '_Band_') if '_' in key else key.title()] = value

# -------------------------------
# Create TradingView-Style Chart
# -------------------------------
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.01,
    row_heights=[0.7, 0.15, 0.15],
    subplot_titles=[
        f"Price & LOWESS Channel ({from_date} to {to_date})",
        "Volume", 
        f"RSI ({rsi_period})"
    ]
)

# Define TradingView colors
colors = {
    'bullish': '#26A69A',
    'bearish': '#EF5350', 
    'trend': '#2962FF',
    'upper_band': '#F23645',
    'lower_band': '#089981',
    'rsi': '#9C27B0',
    'volume_bull': '#26A69A',
    'volume_bear': '#EF5350',
    'grid': '#F0F0F0',
    'text': '#363A45'
}

# --- Price Chart ---
fig.add_trace(go.Candlestick(
    x=df_filtered['TimeLabel'],
    open=df_filtered['open'],
    high=df_filtered['high'],
    low=df_filtered['low'],
    close=df_filtered['close'],
    name='Price',
    increasing_line_color=colors['bullish'],
    decreasing_line_color=colors['bearish'],
    increasing_fillcolor=colors['bullish'],
    decreasing_fillcolor=colors['bearish'],
    line=dict(width=1),
    opacity=0.9
), row=1, col=1)

# LOWESS Trend - thicker main line
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], 
    y=df_filtered['Trend'],
    mode='lines', 
    name='Trend', 
    line=dict(color=colors['trend'], width=2.5),
    opacity=0.9
), row=1, col=1)

# Upper Bands
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], 
    y=df_filtered['Upper_Band_1'], 
    mode='lines', 
    name='U1', 
    line=dict(color=colors['upper_band'], width=1, dash='dot'),
    opacity=0.7
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], 
    y=df_filtered['Upper_Band_2'], 
    mode='lines', 
    name='U2', 
    line=dict(color=colors['upper_band'], width=1, dash='dash'),
    opacity=0.5
), row=1, col=1)

# Lower Bands
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], 
    y=df_filtered['Lower_Band_1'], 
    mode='lines', 
    name='L1', 
    line=dict(color=colors['lower_band'], width=1, dash='dot'),
    opacity=0.7
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], 
    y=df_filtered['Lower_Band_2'], 
    mode='lines', 
    name='L2', 
    line=dict(color=colors['lower_band'], width=1, dash='dash'),
    opacity=0.5
), row=1, col=1)

# --- Volume Chart ---
volume_colors = [colors['volume_bull'] if close >= open else colors['volume_bear'] 
                for close, open in zip(df_filtered['close'], df_filtered['open'])]

fig.add_trace(go.Bar(
    x=df_filtered['TimeLabel'],
    y=df_filtered['Volume'],
    name='Volume',
    marker_color=volume_colors,
    opacity=0.6,
    showlegend=False
), row=2, col=1)

# --- RSI Chart ---
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], 
    y=df_filtered['RSI'],
    mode='lines', 
    name='RSI', 
    line=dict(color=colors['rsi'], width=2),
    showlegend=False
), row=3, col=1)

# RSI levels
fig.add_hline(y=70, line_dash="dash", line_color=colors['bearish'], 
              line_width=1, opacity=0.7, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color=colors['bullish'], 
              line_width=1, opacity=0.7, row=3, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="#787B86", 
              line_width=1, opacity=0.5, row=3, col=1)

# --- TradingView-Style Layout ---
fig.update_layout(
    plot_bgcolor='#FFFFFF',
    paper_bgcolor='#FFFFFF',
    font=dict(
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        size=12,
        color=colors['text']
    ),
    title=dict(
        text="",
        font=dict(size=16, color=colors['text'])
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(54, 58, 69, 0.9)",
        font_size=11,
        font_color="white"
    ),
    height=800,
    margin=dict(l=10, r=10, t=50, b=50),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.05,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)"
    ),
    uirevision='constant'
)

# Update axes styling
for i in range(1, 4):
    fig.update_xaxes(
        row=i, col=1,
        type='category',
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showline=True,
        linewidth=1,
        linecolor='#E1E4E8',
        tickfont=dict(size=10, color=colors['text']),
        tickangle=0 if i == 3 else None
    )
    
    fig.update_yaxes(
        row=i, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor=colors['grid'],
        showline=True,
        linewidth=1,
        linecolor='#E1E4E8',
        tickfont=dict(size=10, color=colors['text']),
        title_font=dict(size=11, color=colors['text'])
    )

# Specific y-axis titles and ranges
fig.update_yaxes(title_text="Price", tickformat=".1f", row=1, col=1)
fig.update_yaxes(title_text="Volume", tickformat=".0f", row=2, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], tickformat=".0f", row=3, col=1)

# Remove x-axis labels for top two charts
fig.update_xaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(showticklabels=False, row=2, col=1)

# Style subplot titles
fig.update_annotations(font_size=12, font_color=colors['text'])

# Display chart
st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ['pan2d', 'lasso2d', 'select2d'],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "lowess_rsi_chart",
        "height": 800,
        "width": 1200,
        "scale": 2
    }
})

# -------------------------------
# Current Statistics
# -------------------------------
st.markdown("### 📊 Market Statistics")

current_data = df_filtered.iloc[-1]
prev_data = df_filtered.iloc[-2] if len(df_filtered) > 1 else current_data

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    price_change = current_data['close'] - prev_data['close']
    price_change_pct = (price_change / prev_data['close']) * 100
    st.metric("Current Price", f"{current_data['close']:.2f}",
              delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

with col2:
    st.metric("High", f"{current_data['high']:.2f}")

with col3:
    st.metric("Low", f"{current_data['low']:.2f}")

with col4:
    st.metric("Volume", f"{current_data['Volume']:,.0f}")

with col5:
    current_rsi = current_data['RSI']
    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    st.metric("RSI", f"{current_rsi:.1f}", delta=rsi_signal)