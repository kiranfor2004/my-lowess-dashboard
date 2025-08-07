# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from io import StringIO
import plotly.graph_objs as go
import re
from datetime import datetime

# -------------------------------
# App Title
# -------------------------------
st.title("📊 LOWESS + RSI Dashboard")
st.write("Analyze price trend and momentum across a date range")

# -------------------------------
# Load and Clean Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        with open('Latest file.csv', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        st.error("❌ 'Latest file.csv' not found. Upload it to the same folder.")
        return None

    # Extract rows using regex: dd-dd-dddd,hh:mm:ss,...
    lines = re.findall(r'\d{2}-\d{2}-\d{4},\d{2}:\d{2}:\d{2},[\d.,]+', content)
    
    if not lines:
        st.error("❌ No valid data found. Check if the file has correct format.")
        return None

    # Convert to DataFrame
    df = pd.read_csv(StringIO('\n'.join(lines)), header=None)
    df.columns = [
        'date', 'time',
        'open', 'high', 'low', 'close', 'Volume',
        'vix_open', 'vix_high', 'vix_low', 'vix_close'
    ]

    # Clean close price
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)

    # Create full datetime
    df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S')
    df['Date'] = df['Datetime'].dt.date  # For filtering

    return df

df = load_data()

if df is None:
    st.stop()

# -------------------------------
# Sidebar: Date Range Filter
# -------------------------------
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()

st.sidebar.header("📅 Date Range Filter")
from_date = st.sidebar.date_input("From Date", min_date, min_value=min_date, max_value=max_date)
to_date = st.sidebar.date_input("To Date", max_date, min_value=min_date, max_value=max_date)

# Validate range
if from_date > to_date:
    st.error("❌ 'From Date' cannot be after 'To Date'")
    st.stop()

# Filter data
from_dt = pd.Timestamp(from_date)
to_dt = pd.Timestamp(to_date) + pd.Timedelta(days=1)  # Include full end date

mask = (df['Datetime'] >= from_dt) & (df['Datetime'] < to_dt)
df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning(f"⚠️ No data found between {from_date} and {to_date}")
    st.stop()

df_filtered.sort_values('Datetime', inplace=True)

st.write(f"📈 Showing data from **{from_date}** to **{to_date}**")

# -------------------------------
# Calculate RSI (14-period)
# -------------------------------
delta = df_filtered['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df_filtered['RSI'] = 100 - (100 / (1 + rs))

# -------------------------------
# Calculate LOWESS Channel
# -------------------------------
x = np.arange(len(df_filtered))
y = df_filtered['close'].values

df_filtered['LOWESS'] = lowess(y, x, frac=0.1, it=3)[:, 1]
residuals = y - df_filtered['LOWESS']
rolling_std = pd.Series(residuals).rolling(20, center=True).std()

# Bands
df_filtered['Upper_Band_1'] = df_filtered['LOWESS'] + 1.0 * rolling_std
df_filtered['Upper_Band_2'] = df_filtered['LOWESS'] + 2.0 * rolling_std
df_filtered['Lower_Band_1'] = df_filtered['LOWESS'] - 1.0 * rolling_std
df_filtered['Lower_Band_2'] = df_filtered['LOWESS'] - 2.0 * rolling_std

# -------------------------------
# Plot Charts (Larger Size)
# -------------------------------

# Price + LOWESS
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['close'], name='Price', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['LOWESS'], name='LOWESS', line=dict(color='orange')))
fig1.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Upper_Band_1'], name='Upper Band 1', line=dict(dash='dot')))
fig1.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Upper_Band_2'], name='Upper Band 2', line=dict(dash='dash')))
fig1.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Lower_Band_1'], name='Lower Band 1', line=dict(dash='dot')))
fig1.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Lower_Band_2'], name='Lower Band 2', line=dict(dash='dash')))

fig1.update_layout(
    title=f"Price & LOWESS Channel ({from_date} to {to_date})",
    xaxis_title="Date & Time",
    yaxis_title="Price",
    hovermode='x unified',
    height=600  # Taller chart
)
st.plotly_chart(fig1, use_container_width=True)

# RSI
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['RSI'], name='RSI', line=dict(color='purple')))
fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

fig2.update_layout(
    title="RSI (14)",
    xaxis_title="Date & Time",
    yaxis_title="RSI",
    hovermode='x unified',
    height=500  # Taller chart
)
st.plotly_chart(fig2, use_container_width=True)