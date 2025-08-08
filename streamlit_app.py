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
st.set_page_config(layout="wide")

# -------------------------------
# App Title
# -------------------------------
st.markdown("<h1 style='text-align: center;'>📊 LOWESS + RSI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>TradingView-style interactive charts</p>", unsafe_allow_html=True)

# -------------------------------
# Load and Clean Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        with open('Latest file.csv', 'r', encoding='latin1', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        st.error("❌ 'Latest file.csv' not found.")
        return None

    # Match rows with all 11 columns
    lines = re.findall(r'\d{2}-\d{2}-\d{4},\d{2}:\d{2}:\d{2}(?:,[\d.]+){9}', content)
    if not lines:
        st.error("❌ No valid data found. Check file format.")
        return None

    df = pd.read_csv(StringIO('\n'.join(lines)), header=None)
    df.columns = [
        'date', 'time',
        'open', 'high', 'low', 'close', 'Volume',
        'vix_open', 'vix_high', 'vix_low', 'vix_close'
    ]

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S')
    return df

df = load_data()
if df is None:
    st.stop()

# -------------------------------
# Date Filter
# -------------------------------
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    from_date = st.date_input("📅 From Date", min_date, min_value=min_date, max_value=max_date)
with col2:
    to_date = st.date_input("📅 To Date", max_date, min_value=min_date, max_value=max_date)

if from_date > to_date:
    st.error("❌ 'From Date' cannot be after 'To Date'")
    st.stop()

mask = (df['Datetime'] >= pd.Timestamp(from_date)) & (df['Datetime'] <= pd.Timestamp(to_date) + pd.Timedelta(days=1))
df_filtered = df[mask].copy()
if df_filtered.empty:
    st.warning(f"⚠️ No data found between {from_date} and {to_date}")
    st.stop()

df_filtered.sort_values('Datetime', inplace=True)

# Convert Datetime to string for category x-axis (removes gaps)
df_filtered['TimeLabel'] = df_filtered['Datetime'].dt.strftime('%Y-%m-%d %H:%M')

# -------------------------------
# RSI Calculation
# -------------------------------
delta = df_filtered['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df_filtered['RSI'] = 100 - (100 / (1 + rs))

# -------------------------------
# LOWESS Calculation
# -------------------------------
x = np.arange(len(df_filtered))
y = df_filtered['close'].values
df_filtered['LOWESS'] = lowess(y, x, frac=0.1, it=3)[:, 1]
residuals = y - df_filtered['LOWESS']

window = min(50, len(df_filtered))
rolling_std = pd.Series(residuals).rolling(window=window, center=True).std()

df_filtered['Upper_Band_1'] = df_filtered['LOWESS'] + rolling_std
df_filtered['Upper_Band_2'] = df_filtered['LOWESS'] + 2 * rolling_std
df_filtered['Lower_Band_1'] = df_filtered['LOWESS'] - rolling_std
df_filtered['Lower_Band_2'] = df_filtered['LOWESS'] - 2 * rolling_std

# -------------------------------
# Combined Chart (Price + Volume + RSI)
# -------------------------------
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.65, 0.15, 0.2],
    subplot_titles=(f"Price & LOWESS Channel ({from_date} to {to_date})", "Volume", "RSI (14)")
)

# --- Price Chart ---
fig.add_trace(go.Candlestick(
    x=df_filtered['TimeLabel'],
    open=df_filtered['open'],
    high=df_filtered['high'],
    low=df_filtered['low'],
    close=df_filtered['close'],
    name='Price',
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350',
    increasing_fillcolor='#26a69a',
    decreasing_fillcolor='#ef5350',
    opacity=0.9
), row=1, col=1)

fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['LOWESS'],
                         mode='lines', name='Trend', line=dict(color='#fdd835', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Upper_Band_1'], mode='lines', name='U1', line=dict(color='#ef5350', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Upper_Band_2'], mode='lines', name='U2', line=dict(color='#ef5350', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Lower_Band_1'], mode='lines', name='L1', line=dict(color='#26a69a', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_filtered['TimeLabel'], y=df_filtered['Lower_Band_2'], mode='lines', name='L2', line=dict(color='#26a69a', width=1, dash='dash')), row=1, col=1)

# --- Volume Bars ---
fig.add_trace(go.Bar(
    x=df_filtered['TimeLabel'],
    y=df_filtered['Volume'],
    name='Volume',
    marker_color=np.where(df_filtered['close'] >= df_filtered['open'], '#26a69a', '#ef5350'),
    opacity=0.6
), row=2, col=1)

# --- RSI Chart ---
fig.add_trace(go.Scatter(
    x=df_filtered['TimeLabel'], y=df_filtered['RSI'],
    mode='lines', name='RSI', line=dict(color='#ab47bc', width=2)
), row=3, col=1)

fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold", row=3, col=1)

# --- Layout ---
fig.update_layout(
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0.85)",
        font_size=12,
        font_color="white",
        align="left"
    ),
    height=950,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    ),
    margin=dict(l=10, r=10, t=50, b=40),
    template="plotly_dark",
    uirevision='constant'
)

# X-axis as category removes gaps
fig.update_xaxes(type='category')

# Show Chart
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
