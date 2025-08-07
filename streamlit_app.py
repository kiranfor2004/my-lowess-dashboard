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
# Title
# -------------------------------
st.title("📊 LOWESS + RSI Dashboard")
st.write("Analyze price trend and RSI over a date range (time-based x-axis)")

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

    # Extract rows using regex
    lines = re.findall(r'\d{2}-\d{2}-\d{4},\d{2}:\d{2}:\d{2},[\d.,]+', content)
    if not lines:
        st.error("❌ No valid data found. Check file format.")
        return None

    df = pd.read_csv(StringIO('\n'.join(lines)), header=None)
    df.columns = [
        'date', 'time',
        'open', 'high', 'low', 'close', 'Volume',
        'vix_open', 'vix_high', 'vix_low', 'vix_close'
    ]

    # Clean close price
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)

    # Create Datetime
    df['Datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S')
    df['Date'] = df['Datetime'].dt.date  # Just the date (no time)
    df['TimeOnly'] = df['Datetime'].dt.strftime('%H:%M')  # Only time

    return df

df = load_data()

if df is None:
    st.stop()

# -------------------------------
# Sidebar: Date Range Filter
# -------------------------------
min_date = df['Date'].min()
max_date = df['Date'].max()

st.sidebar.header("📅 Date Range Filter")
from_date = st.sidebar.date_input("From Date", min_date, min_value=min_date, max_value=max_date)
to_date = st.sidebar.date_input("To Date", max_date, min_value=min_date, max_value=max_date)

# Validate range
if from_date > to_date:
    st.error("❌ 'From Date' cannot be after 'To Date'")
    st.stop()

# Filter data by date range
mask = (df['Date'] >= from_date) & (df['Date'] <= to_date)
df_filtered = df[mask]

if df_filtered.empty:
    st.warning(f"⚠️ No data found between {from_date} and {to_date}")
    st.stop()

st.write(f"📈 Showing data from **{from_date}** to **{to_date}**")

# -------------------------------
# Group by TimeOnly for Aggregation (Optional)
# -------------------------------
# If you want to average across multiple days
grouped = df_filtered.groupby('TimeOnly').agg({
    'close': ['mean', 'min', 'max'],
    'Datetime': 'first'  # Keep one datetime for sorting
}).droplevel(1, axis=1).reset_index()

grouped.columns = ['TimeOnly', 'close_mean', 'close_min', 'close_max', 'Datetime']
grouped.sort_values('Datetime', inplace=True)

# Use averaged close for indicators
x = np.arange(len(grouped))
y = grouped['close_mean'].values

# -------------------------------
# Calculate RSI (14-period)
# -------------------------------
delta = pd.Series(y).diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi_values = 100 - (100 / (1 + rs))
grouped['RSI'] = rsi_values

# -------------------------------
# Calculate LOWESS Channel
# -------------------------------
grouped['LOWESS'] = lowess(y, x, frac=0.1, it=3)[:, 1]
residuals = y - grouped['LOWESS']
rolling_std = pd.Series(residuals).rolling(20, center=True).std()

# Bands
grouped['Upper_Band_1'] = grouped['LOWESS'] + 1.0 * rolling_std
grouped['Upper_Band_2'] = grouped['LOWESS'] + 2.0 * rolling_std
grouped['Lower_Band_1'] = grouped['LOWESS'] - 1.0 * rolling_std
grouped['Lower_Band_2'] = grouped['LOWESS'] - 2.0 * rolling_std

# -------------------------------
# Plot Charts
# -------------------------------

# Price + LOWESS
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['close_mean'], name='Avg Price', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['LOWESS'], name='LOWESS', line=dict(color='orange')))
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['Upper_Band_1'], name='Upper Band 1', line=dict(dash='dot')))
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['Upper_Band_2'], name='Upper Band 2', line=dict(dash='dash')))
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['Lower_Band_1'], name='Lower Band 1', line=dict(dash='dot')))
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['Lower_Band_2'], name='Lower Band 2', line=dict(dash='dash')))
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['close_min'], name='Min Price', line=dict(color='gray', width=1, dash='dot')))
fig1.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['close_max'], name='Max Price', line=dict(color='gray', width=1, dash='dot')))

fig1.update_layout(
    title=f"Average Price & LOWESS Channel ({from_date} to {to_date})",
    xaxis_title="Time of Day",
    yaxis_title="Price",
    hovermode='x unified'
)
st.plotly_chart(fig1)

# RSI
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=grouped['TimeOnly'], y=grouped['RSI'], name='RSI', line=dict(color='purple')))
fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

fig2.update_layout(
    title="RSI (14) - Average Across Days",
    xaxis_title="Time of Day",
    yaxis_title="RSI",
    hovermode='x unified'
)
st.plotly_chart(fig2)