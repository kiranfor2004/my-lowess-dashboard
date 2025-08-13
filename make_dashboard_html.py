# make_dashboard_html.py
import os, pandas as pd, numpy as np
from datetime import date
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSV_PATH = os.environ.get("CSV_PATH", "Latest file.csv")

# --- minimal loader (same rules as your app) ---
import re
from io import StringIO
def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path): return pd.DataFrame()
    content = open(csv_path, "r", encoding="latin1", errors="ignore").read()
    lines = re.findall(r"\d{2}-\d{2}-\d{4},\d{2}:\d{2}:\d{2}(?:,[\\d.]+){9}", content)
    if not lines: return pd.DataFrame()
    df = pd.read_csv(StringIO("\\n".join(lines)), header=None)
    df.columns = ["date","time","open","high","low","close","Volume","vix_open","vix_high","vix_low","vix_close"]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df["Datetime"] = pd.to_datetime(df["date"]+" "+df["time"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    delta = out["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))
    x = np.arange(len(out)); y = out["close"].values
    out["LOWESS"] = lowess(y, x, frac=0.1, it=3)[:, 1]
    resid = y - out["LOWESS"]
    win = min(50, len(out))
    rstd = pd.Series(resid).rolling(window=win, center=True).std()
    out["Upper_Band_1"] = out["LOWESS"] + rstd
    out["Upper_Band_2"] = out["LOWESS"] + 2*rstd
    out["Lower_Band_1"] = out["LOWESS"] - rstd
    out["Lower_Band_2"] = out["LOWESS"] - 2*rstd
    out["TimeLabel"] = out["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
    return out

def build_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure(); fig.update_layout(template="plotly_dark", height=650)
        fig.add_annotation(text="❌ No data loaded.", showarrow=False, font=dict(size=18)); return fig
    start_dt = df["Datetime"].min().date(); end_dt = df["Datetime"].max().date()
    dff = df.copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.65,0.15,0.2],
                        subplot_titles=(f"Price & LOWESS Channel ({start_dt} to {end_dt})","Volume","RSI (14)"))
    fig.add_trace(go.Candlestick(x=dff["TimeLabel"], open=dff["open"], high=dff["high"], low=dff["low"], close=dff["close"],
                                 name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                                 increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350", opacity=0.9), row=1, col=1)
    fig.add_trace(go.Scatter(x=dff["TimeLabel"], y=dff["LOWESS"], mode="lines", name="Trend",
                             line=dict(color="#fdd835", width=2)), row=1, col=1)
    for y, nm, style in [(dff["Upper_Band_1"],"U1","dot"),(dff["Upper_Band_2"],"U2","dash"),
                         (dff["Lower_Band_1"],"L1","dot"),(dff["Lower_Band_2"],"L2","dash")]:
        fig.add_trace(go.Scatter(x=dff["TimeLabel"], y=y, mode="lines", name=nm,
                                 line=dict(width=1, dash=style, color="#ef5350" if "U" in nm else "#26a69a")),
                      row=1, col=1)
    fig.add_trace(go.Bar(x=dff["TimeLabel"], y=dff["Volume"], name="Volume",
                         marker_color=np.where(dff["close"]>=dff["open"], "#26a69a", "#ef5350"), opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=dff["TimeLabel"], y=dff["RSI"], mode="lines", name="RSI",
                             line=dict(color="#ab47bc", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold", row=3, col=1)
    fig.update_layout(hovermode="x unified", template="plotly_dark", height=950,
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
                      margin=dict(l=10,r=10,t=50,b=40))
    fig.update_xaxes(type="category")
    return fig

if __name__ == "__main__":
    df = add_indicators(load_and_clean_data(CSV_PATH))
    fig = build_figure(df)
    fig.write_html("dashboard.html", include_plotlyjs="cdn", full_html=True)
    print("Wrote dashboard.html — open it in your browser.")
