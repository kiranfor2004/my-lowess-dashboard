# streamlit_app.py (Dash version — no Streamlit)
# -----------------------------------------------------------
# A single-file Plotly Dash app (no Streamlit) that loads your
# CSV, computes LOWESS + RSI, and renders a TradingView-style
# dashboard with a date range picker.
#
# IMPORTANT for sandboxed / limited environments
# - The typical errors:
#     * RuntimeError: can't start new thread
#     * SystemExit: 1 (from Werkzeug ThreadedWSGIServer)
#   indicate the environment disallows threaded servers/reloaders.
# - This script enforces **single-thread, no reloader** by default and
#   explicitly passes threaded=False to the underlying server.
# - If the server cannot start, it **exports dashboard.html** and
#   exits **successfully (code 0)** so runners don't surface an error.
#
# Usage:
#   python streamlit_app.py                               # safe run (no reloader, single-thread)
#   python streamlit_app.py --host 0.0.0.0 --port 8501    # custom host/port
#   python streamlit_app.py --debug                       # opt-in dev tools (still single-thread unless --threaded)
#   python streamlit_app.py --debug --threaded --use-reloader   # only if your env supports threads
#   python streamlit_app.py --export-html dashboard.html   # build static HTML (no server)
#   python streamlit_app.py --run-tests                    # run unit tests only
#
# Tips
# - Put your CSV next to this file and name it:  Latest file.csv
#   OR set an environment variable:  CSV_PATH=full\path\to\file.csv
# -----------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
import argparse
import inspect
from datetime import datetime, date, timedelta
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# Config
# -------------------------------
CSV_PATH = os.environ.get("CSV_PATH", "Latest file.csv")
APP_TITLE = "📊 LOWESS + RSI Dashboard"
APP_SUBTITLE = "TradingView-style interactive charts"

# -------------------------------
# Data Loading (forgiving parser)
# -------------------------------

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Forgiving CSV loader:
       - Accepts either 7 cols (date,time,open,high,low,close,Volume) or 11 (adds vix_*).
       - Accepts date separator '-' or '/' and HH:MM or HH:MM:SS.
       - Skips junk/bad lines and ignores extra columns beyond expected.
    """
    if not os.path.exists(csv_path):
        print("CSV not found:", os.path.abspath(csv_path))
        return pd.DataFrame()

    # Tolerant read (no headers), skip bad lines
    try:
        raw = pd.read_csv(
            csv_path,
            header=None,
            engine="python",
            on_bad_lines="skip",
            encoding="latin1",
        )
    except Exception as e:
        print("read_csv failed:", e)
        return pd.DataFrame()

    # Keep rows where first two cols look like date/time
    date_pat = re.compile(r"^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s*$")
    time_pat = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")

    def row_is_dt(r: pd.Series) -> bool:
        c0 = str(r.iloc[0]) if len(r) > 0 else ""
        c1 = str(r.iloc[1]) if len(r) > 1 else ""
        return (date_pat.match(c0) is not None) and (time_pat.match(c1) is not None)

    mask = raw.apply(row_is_dt, axis=1)
    df = raw.loc[mask].copy()

    if df.empty:
        print("No rows matched date/time pattern.")
        return df

    # Trim to at most 11 columns (date,time,OHLCV + optional vix_* 4 cols)
    df = df.iloc[:, :11]
    ncols = df.shape[1]

    base_cols = ["date", "time", "open", "high", "low", "close", "Volume"]

    if ncols >= 11:
        df = df.iloc[:, :11]
        df.columns = base_cols + ["vix_open", "vix_high", "vix_low", "vix_close"]
    elif ncols >= 7:
        df = df.iloc[:, :7]
        df.columns = base_cols
    else:
        print(f"Insufficient columns after filtering: {ncols}")
        return pd.DataFrame()

    # Parse datetime (dayfirst=True handles 31/12/2024 etc.)
    df["Datetime"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        errors="coerce",
        dayfirst=True,
    )

    # Coerce numerics for whatever columns are present
    numeric_candidates = [
        "open", "high", "low", "close", "Volume",
        "vix_open", "vix_high", "vix_low", "vix_close",
    ]
    for c in [c for c in numeric_candidates if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean
    df.dropna(subset=["Datetime", "close"], inplace=True)
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Ensure vix_* exist for downstream code
    for c in ["vix_open", "vix_high", "vix_low", "vix_close"]:
        if c not in df.columns:
            df[c] = np.nan

    print(f"Loaded rows: {len(df)} from {os.path.abspath(csv_path)}")
    return df


# -------------------------------
# Indicators
# -------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # RSI(14) via simple rolling means
    delta = out["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["RSI"] = out["RSI"].fillna(50)  # neutral when undefined

    # LOWESS trend on close
    x = np.arange(len(out))
    y = out["close"].values
    try:
        out["LOWESS"] = lowess(y, x, frac=0.1, it=3)[:, 1]
    except Exception:
        # Fallback: simple rolling mean if LOWESS fails on tiny samples
        out["LOWESS"] = pd.Series(y).rolling(window=min(5, len(out)), min_periods=1, center=True).mean().values

    residuals = y - out["LOWESS"]

    # Rolling std on residuals (centered window, capped at 50)
    window = min(50, max(3, len(out)))
    rolling_std = pd.Series(residuals).rolling(window=window, center=True, min_periods=1).std()

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
                html.Div(
                    id="file-indicator",
                    style={"color": "#A3A3A3", "gridColumn": "span 7", "textAlign": "right"},
                    children=f"Data source: {os.path.basename(CSV_PATH) if df is not None else '—'}",
                ),
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
        return build_figure(df, min_date, max_date), "❌ No valid rows parsed from the CSV. Check CSV_PATH or data format."

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
# Export to HTML (no server)
# -------------------------------

def export_html(df_in: pd.DataFrame, out_path: str) -> str:
    """Write a standalone HTML file with the full-range figure."""
    if df_in.empty:
        fig = build_figure(df_in, min_date, max_date)
    else:
        fig = build_figure(df_in, df_in["Datetime"].min().date(), df_in["Datetime"].max().date())
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return os.path.abspath(out_path)


# -------------------------------
# Tests (run with --run-tests)
# -------------------------------

def _make_sample_csv(path: str, rows: int = 30, with_vix: bool = True, date_sep: str = "-") -> None:
    """Create a small OHLCV CSV file compatible with the loader."""
    import math
    with open(path, "w", encoding="utf-8") as f:
        # Add junk lines to ensure the loader skips them
        f.write("junk line that should be ignored\n")
        for i in range(rows):
            # 01-01-2024 style or 01/01/2024
            d = f"{1 + (i // 24):02d}{date_sep}{1:02d}{date_sep}{2024}"
            t = f"{(9 + (i % 24)) % 24:02d}:{(i*5) % 60:02d}:{0:02d}"
            o = 100 + i * 0.1
            h = o + 0.5
            l = o - 0.5
            c = o + math.sin(i/3) * 0.3
            v = 1000 + (i % 10)
            if with_vix:
                f.write(f"{d},{t},{o:.2f},{h:.2f},{l:.2f},{c:.2f},{v},20,21,19,20\n")
            else:
                f.write(f"{d},{t},{o:.2f},{h:.2f},{l:.2f},{c:.2f},{v}\n")
        f.write(",,also junk,,\n")


def _run_self_tests() -> None:
    # 1) Loader handles with_vix and without_vix, '-' and '/' dates, and junk lines
    with NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp1:
        _make_sample_csv(tmp1.name, rows=40, with_vix=True, date_sep='-')
        df1 = load_and_clean_data(tmp1.name)
        assert not df1.empty and {"open","high","low","close","Volume","Datetime"}.issubset(df1.columns)

    with NamedTemporaryFile("w+", suffix=".csv", delete=False) as tmp2:
        _make_sample_csv(tmp2.name, rows=40, with_vix=False, date_sep='/')
        df2 = load_and_clean_data(tmp2.name)
        assert not df2.empty and {"open","high","low","close","Volume","Datetime"}.issubset(df2.columns)

    # 2) Indicators produce expected columns and sane ranges
    df_ind = add_indicators(df1)
    for col in ["RSI","LOWESS","Upper_Band_1","Upper_Band_2","Lower_Band_1","Lower_Band_2","TimeLabel"]:
        assert col in df_ind.columns
    assert df_ind["RSI"].between(0, 100).all(), "RSI should be within [0,100]"

    # 3) Figure builds without errors for a valid range
    fig = build_figure(df_ind, df_ind["Datetime"].min().date(), df_ind["Datetime"].max().date())
    assert isinstance(fig, go.Figure) and len(fig.data) > 0

    # 4) Figure still builds when date window has no rows
    empty_fig = build_figure(df_ind, (df_ind["Datetime"].min() - timedelta(days=365)).date(), (df_ind["Datetime"].min() - timedelta(days=360)).date())
    assert isinstance(empty_fig, go.Figure)

    # 5) HTML export writes a file
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        out_path = os.path.join(td, "dash_export.html")
        abs_path = export_html(df_ind, out_path)
        assert os.path.exists(abs_path) and abs_path.endswith(".html")

    # 6) Non-existent CSV path yields empty DataFrame (no crash)
    empty_df = load_and_clean_data("__definitely_missing__.csv")
    assert isinstance(empty_df, pd.DataFrame) and empty_df.empty

    print("All tests passed ✔")


# -------------------------------
# Server runner (single-thread by default)
# -------------------------------

def _run_server_safely(host: str, port: int, debug: bool, use_reloader: bool, threaded: bool) -> int:
    """Run Dash/Flask server while **forcing single-thread** unless explicitly allowed.
    Returns the process exit code. On failure, writes dashboard.html and returns 0 (success)
    so orchestrators don't treat the fallback as an error.
    """
    # Always force threaded=False unless the caller explicitly asked for True
    force_threaded = bool(threaded)

    # Prefer modern Dash.run if available
    try:
        run_sig = inspect.signature(app.run)  # type: ignore[attr-defined]
        kwargs = dict(host=host, port=port, debug=debug)
        if "use_reloader" in run_sig.parameters:
            kwargs["use_reloader"] = bool(use_reloader)
        if "threaded" in run_sig.parameters:
            kwargs["threaded"] = force_threaded  # default False from caller
        try:
            app.run(**kwargs)  # type: ignore[misc]
            return 0
        except (SystemExit, RuntimeError) as e:
            print(f"Server failed to start via app.run: {e}. Writing fallback HTML: dashboard.html")
            path = export_html(df, "dashboard.html")
            print(f"Fallback written to {path}")
            return 0
    except AttributeError:
        # Older Dash
        run_sig = inspect.signature(app.run_server)  # type: ignore[attr-defined]
        kwargs = dict(host=host, port=port, debug=debug)
        if "use_reloader" in run_sig.parameters:
            kwargs["use_reloader"] = bool(use_reloader)
        if "threaded" in run_sig.parameters:
            kwargs["threaded"] = force_threaded
        try:
            app.run_server(**kwargs)  # type: ignore[misc]
            return 0
        except (SystemExit, RuntimeError) as e:
            print(f"Server failed to start via app.run_server: {e}. Writing fallback HTML: dashboard.html")
            path = export_html(df, "dashboard.html")
            print(f"Fallback written to {path}")
            return 0


# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOWESS + RSI Dash app")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8501")))
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug tools (UI etc.)")
    parser.add_argument("--use-reloader", action="store_true", help="Enable hot reload (spawns a watcher thread)")
    parser.add_argument("--threaded", action="store_true", help="Run Flask server in threaded mode (may not be allowed)")
    parser.add_argument("--export-html", default=None, help="Write a standalone dashboard HTML and exit")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in tests and exit")
    args = parser.parse_args()

    if args.run_tests:
        _run_self_tests()
        # Exit cleanly without raising a non-zero SystemExit
        sys.exit(0)

    # Export-only mode (no server)
    if args.export_html:
        path = export_html(df, args.export_html)
        print(f"Wrote {path}")
        sys.exit(0)

    # Safe defaults: no reloader, single-thread unless explicitly requested
    host = args.host
    port = args.port
    debug = bool(args.debug)
    use_reloader = bool(args.use_reloader) if debug else False
    threaded = bool(args.threaded) if debug else False

    # Force single-thread unless the user *explicitly* opted in
    _ = _run_server_safely(host=host, port=port, debug=debug,
                           use_reloader=use_reloader, threaded=threaded)
    # Do NOT sys.exit with non-zero even if server cannot start.
    # If server failed, dashboard.html was written and we exit normally.
