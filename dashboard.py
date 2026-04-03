"""
dashboard.py
------------
Phase 5 — Real-Time Stock Analytics Dashboard

Streamlit app that visualizes the entire pipeline:
  - Live price charts with Bollinger Bands
  - RSI and MACD indicator charts
  - Anomaly flags overlaid on price
  - 7-day ARIMA vs Prophet forecast comparison
  - Summary metrics per ticker

Usage:
    streamlit run dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import text

from db.connection import get_engine
from config import TICKERS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analytics Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 12px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_prices(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ts::date AS date, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :ticker AND source = 'yfinance'
          AND ts >= NOW() - INTERVAL ':days days'
        ORDER BY ts ASC
    """.replace(":days days", f"{days} days"))
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_indicators(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ts::date AS date, rsi_14, macd, macd_signal, macd_hist,
               bb_upper, bb_middle, bb_lower
        FROM technical_indicators
        WHERE ticker = :ticker
          AND ts >= NOW() - INTERVAL ':days days'
        ORDER BY ts ASC
    """.replace(":days days", f"{days} days"))
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_anomalies(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ts::date AS date, close, zscore, flag
        FROM anomalies
        WHERE ticker = :ticker
          AND ts >= NOW() - INTERVAL ':days days'
        ORDER BY ts ASC
    """.replace(":days days", f"{days} days"))
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_forecasts(ticker: str) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT model, forecast_date, predicted_close, lower_bound, upper_bound
        FROM forecasts
        WHERE ticker = :ticker
        ORDER BY model, forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


@st.cache_data(ttl=300)
def load_anomaly_summary() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ticker,
               COUNT(*) FILTER (WHERE flag='HIGH') AS spikes,
               COUNT(*) FILTER (WHERE flag='LOW')  AS crashes,
               COUNT(*)                             AS total
        FROM anomalies
        GROUP BY ticker ORDER BY total DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# ── Charts ────────────────────────────────────────────────────────────────────

def price_chart(prices: pd.DataFrame, indicators: pd.DataFrame,
                anomalies: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick chart with Bollinger Bands and anomaly markers."""
    fig = go.Figure()

    # Bollinger Bands
    if not indicators.empty:
        fig.add_trace(go.Scatter(
            x=indicators["date"], y=indicators["bb_upper"],
            name="BB Upper", line=dict(color="rgba(100,100,255,0.4)", dash="dash"),
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=indicators["date"], y=indicators["bb_lower"],
            name="BB Lower", line=dict(color="rgba(100,100,255,0.4)", dash="dash"),
            fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=indicators["date"], y=indicators["bb_middle"],
            name="BB Middle (20 SMA)", line=dict(color="rgba(150,150,255,0.6)", dash="dot"),
        ))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=prices["date"],
        open=prices["open"], high=prices["high"],
        low=prices["low"],   close=prices["close"],
        name=ticker,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # Anomaly markers
    if not anomalies.empty:
        high_anom = anomalies[anomalies["flag"] == "HIGH"]
        low_anom  = anomalies[anomalies["flag"] == "LOW"]

        if not high_anom.empty:
            fig.add_trace(go.Scatter(
                x=high_anom["date"], y=high_anom["close"],
                mode="markers", name="Spike 🔺",
                marker=dict(symbol="triangle-up", size=12, color="#ff6b6b"),
            ))
        if not low_anom.empty:
            fig.add_trace(go.Scatter(
                x=low_anom["date"], y=low_anom["close"],
                mode="markers", name="Crash 🔻",
                marker=dict(symbol="triangle-down", size=12, color="#ffd93d"),
            ))

    fig.update_layout(
        title=f"{ticker} — Price + Bollinger Bands",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=500,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def rsi_macd_chart(indicators: pd.DataFrame, ticker: str) -> go.Figure:
    """RSI and MACD in a two-panel chart."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("RSI (14)", "MACD"),
        vertical_spacing=0.12, row_heights=[0.4, 0.6],
    )

    # RSI
    fig.add_trace(go.Scatter(
        x=indicators["date"], y=indicators["rsi_14"],
        name="RSI", line=dict(color="#7b61ff"),
    ), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="white", opacity=0.02, row=1, col=1)

    # MACD line + signal
    fig.add_trace(go.Scatter(
        x=indicators["date"], y=indicators["macd"],
        name="MACD", line=dict(color="#26c6da"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=indicators["date"], y=indicators["macd_signal"],
        name="Signal", line=dict(color="#ff7043"),
    ), row=2, col=1)

    # MACD histogram
    colors = ["#26a69a" if v >= 0 else "#ef5350"
              for v in indicators["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=indicators["date"], y=indicators["macd_hist"],
        name="Histogram", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=450,
        title=f"{ticker} — RSI & MACD Indicators",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="RSI", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    return fig


def forecast_chart(prices: pd.DataFrame, forecasts: pd.DataFrame, ticker: str) -> go.Figure:
    """Historical prices + ARIMA and Prophet forecasts with confidence bands."""
    fig = go.Figure()

    # Last 60 days of actual prices
    recent = prices.tail(60)
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["close"],
        name="Actual", line=dict(color="#ffffff", width=2),
    ))

    colors = {"arima": "#ff7043", "prophet": "#26c6da"}

    for model in ["arima", "prophet"]:
        mdf = forecasts[forecasts["model"] == model]
        if mdf.empty:
            continue

        # Confidence band
        band_color = "rgba(230,100,50,0.15)" if model == "arima" else "rgba(25,180,200,0.15)"
        fig.add_trace(go.Scatter(
            x=pd.concat([mdf["forecast_date"], mdf["forecast_date"].iloc[::-1]]),
            y=pd.concat([mdf["upper_bound"], mdf["lower_bound"].iloc[::-1]]),
            fill="toself",
            fillcolor=band_color,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{model.upper()} confidence",
            showlegend=True,
        ))

        fig.add_trace(go.Scatter(
            x=mdf["forecast_date"], y=mdf["predicted_close"],
            name=f"{model.upper()} forecast",
            line=dict(color=colors[model], width=2, dash="dash"),
            mode="lines+markers",
        ))

    fig.update_layout(
        title=f"{ticker} — 7-Day Forecast: ARIMA vs Prophet",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Stock Pipeline")
    st.caption("Real-time analytics dashboard")
    st.divider()

    ticker = st.selectbox("Select Ticker", TICKERS, index=0)
    days   = st.slider("Lookback (days)", min_value=30, max_value=500, value=180, step=30)
    st.divider()

    st.markdown("**Pipeline Phases**")
    st.success("✅ Phase 1 — Ingestion")
    st.success("✅ Phase 2 — Indicators")
    st.success("✅ Phase 3 — Anomalies")
    st.success("✅ Phase 4 — Forecasting")
    st.success("✅ Phase 5 — Dashboard")
    st.divider()
    st.caption("Data refreshes every 5 min")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────

st.title(f"📊 {ticker} — Stock Analytics Dashboard")

# Load data
prices     = load_prices(ticker, days)
indicators = load_indicators(ticker, days)
anomalies  = load_anomalies(ticker, days)
forecasts  = load_forecasts(ticker)

if prices.empty:
    st.error(f"No data found for {ticker}. Run seed_db.py first.")
    st.stop()

# ── Top metrics ───────────────────────────────────────────────────────────────
latest       = prices.iloc[-1]
prev         = prices.iloc[-2]
price_change = latest["close"] - prev["close"]
price_pct    = (price_change / prev["close"]) * 100

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Current Price", f"${latest['close']:.2f}",
              delta=f"{price_change:+.2f} ({price_pct:+.2f}%)")
with col2:
    st.metric("Day High", f"${latest['high']:.2f}")
with col3:
    st.metric("Day Low", f"${latest['low']:.2f}")
with col4:
    if not indicators.empty:
        rsi = indicators.iloc[-1]["rsi_14"]
        rsi_label = "🔴 Overbought" if rsi > 70 else ("🟢 Oversold" if rsi < 30 else "⚪ Neutral")
        st.metric("RSI (14)", f"{rsi:.1f}", delta=rsi_label)
with col5:
    anom_count = len(anomalies)
    st.metric("Anomalies (period)", anom_count,
              delta="flags detected", delta_color="off")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Bollinger Bands",
    "📉 RSI & MACD",
    "🎯 Forecasts",
    "🌍 Market Overview",
])

with tab1:
    st.plotly_chart(
        price_chart(prices, indicators, anomalies, ticker),
        use_container_width=True,
    )

    if not anomalies.empty:
        with st.expander(f"🚨 Anomaly Details ({len(anomalies)} events)"):
            display_anom = anomalies.copy()
            display_anom["close"]  = display_anom["close"].round(2)
            display_anom["zscore"] = display_anom["zscore"].round(3)
            st.dataframe(display_anom.sort_values("date", ascending=False),
                         use_container_width=True, hide_index=True)

with tab2:
    if indicators.empty:
        st.warning("No indicators found. Run: python indicators.py")
    else:
        st.plotly_chart(
            rsi_macd_chart(indicators, ticker),
            use_container_width=True,
        )

        # Signal interpretation
        latest_ind = indicators.iloc[-1]
        c1, c2 = st.columns(2)
        with c1:
            rsi = latest_ind["rsi_14"]
            if rsi > 70:
                st.error(f"🔴 RSI {rsi:.1f} — Overbought. Potential pullback ahead.")
            elif rsi < 30:
                st.success(f"🟢 RSI {rsi:.1f} — Oversold. Potential bounce ahead.")
            else:
                st.info(f"⚪ RSI {rsi:.1f} — Neutral zone (30–70).")
        with c2:
            hist = latest_ind["macd_hist"]
            if hist > 0:
                st.success(f"🟢 MACD Histogram {hist:.4f} — Bullish momentum.")
            else:
                st.error(f"🔴 MACD Histogram {hist:.4f} — Bearish momentum.")

with tab3:
    if forecasts.empty:
        st.warning("No forecasts found. Run: python forecasting.py")
    else:
        st.plotly_chart(
            forecast_chart(prices, forecasts, ticker),
            use_container_width=True,
        )

        # Side-by-side forecast table
        st.subheader("7-Day Forecast Numbers")
        arima_df   = forecasts[forecasts["model"] == "arima"][["forecast_date", "predicted_close", "lower_bound", "upper_bound"]].copy()
        prophet_df = forecasts[forecasts["model"] == "prophet"][["forecast_date", "predicted_close", "lower_bound", "upper_bound"]].copy()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ARIMA**")
            arima_df.columns = ["Date", "Forecast", "Lower", "Upper"]
            arima_df[["Forecast","Lower","Upper"]] = arima_df[["Forecast","Lower","Upper"]].round(2)
            st.dataframe(arima_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Prophet**")
            prophet_df.columns = ["Date", "Forecast", "Lower", "Upper"]
            prophet_df[["Forecast","Lower","Upper"]] = prophet_df[["Forecast","Lower","Upper"]].round(2)
            st.dataframe(prophet_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Anomaly Summary — All Tickers")
    summary = load_anomaly_summary()
    if not summary.empty:
        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary["ticker"], y=summary["spikes"],
                             name="Spikes", marker_color="#ff6b6b"))
        fig.add_trace(go.Bar(x=summary["ticker"], y=summary["crashes"],
                             name="Crashes", marker_color="#ffd93d"))
        fig.update_layout(
            barmode="group", template="plotly_dark",
            title="Anomaly Counts per Ticker",
            xaxis_title="Ticker", yaxis_title="Count", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Latest Prices — All Tickers")
    rows = []
    engine = get_engine()
    for t in TICKERS:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT close FROM raw_prices
                    WHERE ticker=:t AND source='yfinance'
                    ORDER BY ts DESC LIMIT 1
                """), {"t": t}).fetchone()
                if result:
                    rows.append({"Ticker": t, "Latest Close": round(float(result[0]), 2)})
        except Exception:
            pass
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)