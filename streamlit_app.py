# dash_lowess_rsi_app.py
# -----------------------------------------------------------
# A single-file Plotly Dash app that reproduces your
# Streamlit LOWESS + RSI dashboard without using Streamlit.
# -----------------------------------------------------------

from __future__ import annotations

import os
import re
from io import StringIO
from datetime import datetime, date

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# Config
# -------------------------------
# CSV location. Defaults to 'Latest file.csv' in the current folder.
CSV_PATH = os.environ.get("CSV_PATH", "Latest file.csv")
APP_TITLE = "📊 LOWESS + RSI Dashboard"
APP_SUBTITLE = "TradingView-style interactive charts"

# -------------------------------
# Data Loading & Cleaning
# -------------------------------

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Read the raw CSV, keep only lines that match the 11 expected columns,
    coerce types, and build a single Datetime column.
    Returns an empty DataFrame if the file is missing or invalid.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    with open(csv_path, "r", encoding="latin1", errors="ignore") as f:
        content = f.read()

    # Keep rows that match: dd-mm-YYYY,HH:MM:SS + 9 numeric commas (11 columns total)
    lines = re.findall(r"\d{2}-\d{2}-\d{4},\d{2}:\d{2}:\d{2}(?:,[\d.]+){9}", content)
    if not lines:
        return pd.DataFrame()

    df = pd.read_csv(StringIO("\n".join(lines)), header=None)
    df.columns = [
        "date", "time",
        "open", "high", "low", "close", "Volume",
        "vix_open", "vix_high", "vix_low", "vix_close",
    ]

    # Ensure numeric close and drop invalids
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)

    # Build datetime
    df["Datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)

    # Sort once
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI(14), LOWESS trend, and 1/2σ dynamic bands on the residuals.
    Expects columns: Datetime, open, high, low, close, Volume.
    """
    if df.empty:
        return df

    out = df.copy()

    # RSI(14) via simple rolling means (to match your original app)
    delta = out["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))

    # LOWESS trend on close
    x = np.arange(len(out))
    y = out["close"].values
    out["LOWESS"] = lowess(y, x, frac=0.1, it=3)[:, 1]
    residuals = y - out["LOWESS"]

    # Rolling std on residuals (centered window, capped at 50)
    window = min(50, len(out))
    rolling_std = pd.Series(residuals).rolling(window=window, center=True).std()

    out["Upper_Band_1"] = out["LOWESS"] + rolling_std
    out["Upper_Band_2"] = out["LOWESS"] + 2 * rolling_std
    out["Lower_Band_1"] = out["LOWESS"] - rolling_std
    out["Lower_Band_2"] = out["LOWESS"] - 2 * rolling_std

    # Label for category x-axis (removes gaps)
    out["TimeLabel"] = out["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
    return out


# -------------------------------
# Figure Builder
# -------------------------------

def build_figure(df: pd.DataFrame, start_dt: date, end_dt: date) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=650)
        fig.add_annotation(text="❌ No data loaded.", showarrow=False, font=dict(size=18))
        return fig

    # Filter by inclusive range; end date + 1 day to include all intraday rows
    mask = (df["Datetime"] >= pd.Timestamp(start_dt)) & (
        df["Datetime"] <= pd.Timestamp(end_dt) + pd.Timedelta(days=1)
    )
    dff = df.loc[mask].copy()

    if dff.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=650)
        fig.add_annotation(
            text=f"⚠️ No data found between {start_dt} and {end_dt}",
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Subplots: Price + Volume + RSI
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.65, 0.15, 0.2],
        subplot_titles=(
            f"Price & LOWESS Channel ({start_dt} to {end_dt})",
            "Volume",
            "RSI (14)",
        ),
    )

    # Price (candles) + bands
    fig.add_trace(
        go.Candlestick(
            x=dff["TimeLabel"],
            open=dff["open"],
            high=dff["high"],
            low=dff["low"],
            close=dff["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            opacity=0.9,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dff["TimeLabel"],
            y=dff["LOWESS"],
            mode="lines",
            name="Trend",
            line=dict(color="#fdd835", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=dff["TimeLabel"], y=dff["Upper_Band_1"], mode="lines", name="U1", line=dict(color="#ef5350", width=1, dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dff["TimeLabel"], y=dff["Upper_Band_2"], mode="lines", name="U2", line=dict(color="#ef5350", width=1, dash="dash")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dff["TimeLabel"], y=dff["Lower_Band_1"], mode="lines", name="L1", line=dict(color="#26a69a", width=1, dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dff["TimeLabel"], y=dff["Lower_Band_2"], mode="lines", name="L2", line=dict(color="#26a69a", width=1, dash="dash")),
        row=1,
        col=1,
    )

    # Volume
    fig.add_trace(
        go.Bar(
            x=dff["TimeLabel"],
            y=dff["Volume"],
            name="Volume",
            marker_color=np.where(dff["close"] >= dff["open"], "#26a69a", "#ef5350"),
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=dff["TimeLabel"],
            y=dff["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="#ab47bc", width=2),
        ),
        row=3,
        col=1,
    )

    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold", row=3, col=1)

    # Layout
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.85)",
            font_size=12,
            font_color="white",
            align="left",
        ),
        height=950,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(l=10, r=10, t=50, b=40),
        template="plotly_dark",
        uirevision="constant",
    )

    # Remove gaps by treating x as categorical
    fig.update_xaxes(type="category")

    return fig


# -------------------------------
# App Setup
# -------------------------------
raw_df = load_and_clean_data(CSV_PATH)
if not raw_df.empty:
    df = add_indicators(raw_df)
    min_date = df["Datetime"].min().date()
    max_date = df["Datetime"].max().date()
else:
    df = pd.DataFrame()
    # Fallback dates (today) so the picker renders
    today = date.today()
    min_date = today
    max_date = today

app = Dash(__name__)
app.title = "LOWESS + RSI"

# Simple dark styling
app.layout = html.Div(
    style={"backgroundColor": "#0E1117", "minHeight": "100vh", "padding": "20px"},
    children=[
        html.Div(
            [
                html.H1(APP_TITLE, style={"textAlign": "center", "color": "#FFFFFF", "marginBottom": "4px"}),
                html.P(APP_SUBTITLE, style={"textAlign": "center", "color": "#A3A3A3", "marginTop": 0}),
            ]
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(12, 1fr)",
                "gap": "12px",
                "alignItems": "center",
                "margin": "12px 0 20px 0",
            },
            children=[
                html.Div("From Date", style={"color": "#E0E0E0", "gridColumn": "span 1"}),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=min_date,
                    end_date=max_date,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    display_format="YYYY-MM-DD",
                    style={"gridColumn": "span 4", "backgroundColor": "#272B33"},
                ),
                html.Div(id="file-indicator", style={"color": "#A3A3A3", "gridColumn": "span 7", "textAlign": "right"},
                         children=f"Data source: {os.path.basename(CSV_PATH) if df is not None else '—'}"),
            ],
        ),
        dcc.Loading(
            id="loading-graph",
            type="dot",
            color="#FFFFFF",
            children=dcc.Graph(id="main-graph", config={"displayModeBar": True}),
        ),
        html.Div(id="message", style={"color": "#F5F5F5", "marginTop": "10px"}),
    ],
)


# -------------------------------
# Callbacks
# -------------------------------
@app.callback(
    Output("main-graph", "figure"),
    Output("message", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_chart(start_date_iso, end_date_iso):
    if df.empty:
        return build_figure(df, min_date, max_date), "❌ 'Latest file.csv' not found or no valid data."

    # Convert ISO strings to date objects
    try:
        start_dt = datetime.fromisoformat(start_date_iso).date() if start_date_iso else min_date
        end_dt = datetime.fromisoformat(end_date_iso).date() if end_date_iso else max_date
    except Exception:
        start_dt, end_dt = min_date, max_date

    if start_dt > end_dt:
        fig = build_figure(df, min_date, max_date)
        return fig, "❌ 'From Date' cannot be after 'To Date'"

    fig = build_figure(df, start_dt, end_dt)
    return fig, ""


# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    # Run the app
    # Visit http://127.0.0.1:8050/ in your browser
    app.run_server(debug=True)