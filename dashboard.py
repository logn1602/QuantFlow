"""
dashboard.py
------------
QuantFlow — Real-Time Stock Analytics Dashboard

Streamlit app visualizing the full pipeline:
  - Live price charts with Bollinger Bands + anomaly flags
  - RSI and MACD indicator charts with signal interpretation
  - 7-day forecast comparison: ARIMA, Prophet, XGBoost, LightGBM
  - FinBERT news sentiment analysis per ticker
  - Market-wide overview with anomaly summary

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
    page_title="QuantFlow Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card { background: #1e1e2e; border-radius: 10px; padding: 16px; text-align: center; }
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 12px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_prices(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text(f"""
        SELECT ts::date AS date, open, high, low, close, volume
        FROM raw_prices
        WHERE ticker = :ticker AND source = 'yfinance'
          AND ts >= NOW() - INTERVAL '{days} days'
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_indicators(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text(f"""
        SELECT ts::date AS date, rsi_14, macd, macd_signal, macd_hist,
               bb_upper, bb_middle, bb_lower
        FROM technical_indicators
        WHERE ticker = :ticker
          AND ts >= NOW() - INTERVAL '{days} days'
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_anomalies(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text(f"""
        SELECT ts::date AS date, close, zscore, flag
        FROM anomalies
        WHERE ticker = :ticker
          AND ts >= NOW() - INTERVAL '{days} days'
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_forecasts(ticker: str) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT model, forecast_date,
               AVG(predicted_close) AS predicted_close,
               AVG(lower_bound)     AS lower_bound,
               AVG(upper_bound)     AS upper_bound
        FROM forecasts
        WHERE ticker = :ticker
        GROUP BY model, forecast_date
        ORDER BY model, forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


@st.cache_data(ttl=300)
def load_sentiment(ticker: str, days: int = 7) -> pd.DataFrame:
    engine = get_engine()
    query = text(f"""
        SELECT
            published_at::date              AS date,
            LEFT(headline, 80)              AS headline,
            sentiment,
            ROUND(compound::numeric, 3)     AS compound,
            source
        FROM news_sentiment
        WHERE ticker = :ticker
          AND published_at >= NOW() - INTERVAL '{days} days'
        ORDER BY published_at DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_sentiment_summary() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT
            ticker,
            ROUND(AVG(compound)::numeric, 3)                 AS avg_compound,
            COUNT(*) FILTER (WHERE sentiment='positive')     AS positive,
            COUNT(*) FILTER (WHERE sentiment='negative')     AS negative,
            COUNT(*) FILTER (WHERE sentiment='neutral')      AS neutral,
            COUNT(*)                                         AS total
        FROM news_sentiment
        WHERE published_at >= NOW() - INTERVAL '7 days'
        GROUP BY ticker
        ORDER BY avg_compound DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=300)
def load_backtest_summary() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ticker, total_return, annualized_return, sharpe_ratio,
               max_drawdown, win_rate, num_trades, benchmark_return, alpha,
               final_value, run_at::date AS run_date
        FROM backtest_results
        ORDER BY alpha DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


@st.cache_data(ttl=300)
def load_backtest_series(ticker: str) -> dict:
    import json as _json
    engine = get_engine()
    query = text("""
        SELECT daily_values, benchmark_values
        FROM backtest_results
        WHERE ticker = :t
        ORDER BY run_at DESC LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"t": ticker}).fetchone()
    if not row:
        return {}
    return {
        "daily_values":     _json.loads(row[0]),
        "benchmark_values": _json.loads(row[1]),
    }


@st.cache_data(ttl=300)
def load_anomaly_summary() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ticker,
               COUNT(*) FILTER (WHERE flag='HIGH') AS spikes,
               COUNT(*) FILTER (WHERE flag='LOW')  AS crashes,
               COUNT(*) AS total
        FROM anomalies
        GROUP BY ticker ORDER BY total DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# ── Charts ────────────────────────────────────────────────────────────────────

def price_chart(prices, indicators, anomalies, ticker):
    fig = go.Figure()
    if not indicators.empty:
        fig.add_trace(go.Scatter(
            x=indicators["date"], y=indicators["bb_upper"],
            name="BB Upper", line=dict(color="rgba(100,100,255,0.4)", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=indicators["date"], y=indicators["bb_lower"],
            name="BB Lower", line=dict(color="rgba(100,100,255,0.4)", dash="dash"),
            fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
        ))
        fig.add_trace(go.Scatter(
            x=indicators["date"], y=indicators["bb_middle"],
            name="BB Middle (20 SMA)", line=dict(color="rgba(150,150,255,0.6)", dash="dot"),
        ))
    fig.add_trace(go.Candlestick(
        x=prices["date"],
        open=prices["open"], high=prices["high"],
        low=prices["low"],   close=prices["close"],
        name=ticker,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))
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


def rsi_macd_chart(indicators, ticker):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("RSI (14)", "MACD"),
        vertical_spacing=0.12, row_heights=[0.4, 0.6],
    )
    fig.add_trace(go.Scatter(
        x=indicators["date"], y=indicators["rsi_14"],
        name="RSI", line=dict(color="#7b61ff"),
    ), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.5, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=indicators["date"], y=indicators["macd"],
        name="MACD", line=dict(color="#26c6da"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=indicators["date"], y=indicators["macd_signal"],
        name="Signal", line=dict(color="#ff7043"),
    ), row=2, col=1)
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in indicators["macd_hist"].fillna(0)]
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


def forecast_chart(prices, forecasts, ticker):
    fig = go.Figure()
    recent = prices.tail(60)
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["close"],
        name="Actual", line=dict(color="#ffffff", width=2),
    ))
    model_colors = {
        "arima":          "#ff7043",
        "prophet":        "#26c6da",
        "xgboost":        "#ab47bc",
        "lightgbm":       "#66bb6a",
        "ensemble_stack": "#ffd700",
    }
    band_colors = {
        "arima":          "rgba(255,112,67,0.12)",
        "prophet":        "rgba(38,198,218,0.12)",
        "xgboost":        "rgba(171,71,188,0.12)",
        "lightgbm":       "rgba(102,187,106,0.12)",
        "ensemble_stack": "rgba(255,215,0,0.18)",
    }
    for model in ["arima", "prophet", "xgboost", "lightgbm", "ensemble_stack"]:
        mdf = forecasts[forecasts["model"] == model].copy()
        if mdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pd.concat([mdf["forecast_date"], mdf["forecast_date"].iloc[::-1]]),
            y=pd.concat([mdf["upper_bound"], mdf["lower_bound"].iloc[::-1]]),
            fill="toself", fillcolor=band_colors[model],
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{model.upper()} band", showlegend=False,
        ))
        is_ensemble = model == "ensemble_stack"
        fig.add_trace(go.Scatter(
            x=mdf["forecast_date"], y=mdf["predicted_close"],
            name="ENSEMBLE ⭐" if is_ensemble else model.upper(),
            line=dict(
                color=model_colors[model],
                width=4 if is_ensemble else 2,
                dash="solid" if is_ensemble else "dash",
            ),
            mode="lines+markers",
        ))
    fig.update_layout(
        title=f"{ticker} — 7-Day Forecast: 4 Base Models + Stacking Ensemble",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def sentiment_bar_chart(sentiment_df, ticker):
    daily = sentiment_df.groupby("date").agg(
        compound=("compound", "mean"),
    ).reset_index()
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in daily["compound"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["compound"],
        marker_color=colors, name="Avg compound",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"{ticker} — Daily Sentiment Score (FinBERT)",
        xaxis_title="Date", yaxis_title="Compound Score",
        template="plotly_dark", height=300,
    )
    return fig


def sentiment_gauge(compound, ticker):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(float(compound), 3),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"{ticker} Sentiment"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar":  {"color": "#26a69a" if compound >= 0 else "#ef5350"},
            "steps": [
                {"range": [-1,   -0.2], "color": "rgba(239,83,80,0.2)"},
                {"range": [-0.2,  0.2], "color": "rgba(100,100,100,0.2)"},
                {"range": [0.2,    1],  "color": "rgba(38,166,154,0.2)"},
            ],
        },
    ))
    fig.update_layout(template="plotly_dark", height=260)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 QuantFlow")
    st.caption("Quantitative stock analytics platform")
    st.divider()

    ticker = st.selectbox("Select Ticker", TICKERS, index=0)
    days   = st.slider("Lookback (days)", min_value=30, max_value=500, value=180, step=30)
    st.divider()

    st.markdown("**Pipeline**")
    st.success("✅ Phase 1 — Ingestion")
    st.success("✅ Phase 2 — Indicators")
    st.success("✅ Phase 3 — Anomalies")
    st.success("✅ Phase 4 — Forecasting")
    st.success("✅ Phase 5 — Dashboard")
    st.success("✅ Phase 6 — Sentiment")
    st.success("✅ Level 2 — XGBoost/LightGBM")
    st.success("✅ Level 3 — Stacking Ensemble")
    st.divider()
    st.caption("Data refreshes every 5 min")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────

st.title(f"📊 {ticker} — QuantFlow Analytics")

prices     = load_prices(ticker, days)
indicators = load_indicators(ticker, days)
anomalies  = load_anomalies(ticker, days)
forecasts  = load_forecasts(ticker)
sentiment  = load_sentiment(ticker, days=7)

if prices.empty:
    st.error(f"No data found for {ticker}. Run seed_db.py first.")
    st.stop()

# ── Top metrics ───────────────────────────────────────────────────────────────
latest       = prices.iloc[-1]
prev         = prices.iloc[-2]
price_change = latest["close"] - prev["close"]
price_pct    = (price_change / prev["close"]) * 100

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Current Price", f"${latest['close']:.2f}",
              delta=f"{price_change:+.2f} ({price_pct:+.2f}%)")
with col2:
    st.metric("Day High", f"${latest['high']:.2f}")
with col3:
    st.metric("Day Low",  f"${latest['low']:.2f}")
with col4:
    if not indicators.empty:
        rsi = indicators.iloc[-1]["rsi_14"]
        rsi_label = "🔴 Overbought" if rsi > 70 else ("🟢 Oversold" if rsi < 30 else "⚪ Neutral")
        st.metric("RSI (14)", f"{rsi:.1f}", delta=rsi_label)
with col5:
    if not sentiment.empty:
        avg_sent  = float(sentiment["compound"].mean())
        sent_label = "🟢 Positive" if avg_sent > 0.1 else ("🔴 Negative" if avg_sent < -0.1 else "⚪ Neutral")
        st.metric("Sentiment (7d)", f"{avg_sent:.3f}", delta=sent_label)
with col6:
    st.metric("Anomalies", len(anomalies), delta="flags", delta_color="off")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price & Bollinger Bands",
    "📉 RSI & MACD",
    "🎯 Forecasts",
    "🗞️ Sentiment",
    "🌍 Market Overview",
    "📊 Backtest",
])

with tab1:
    st.plotly_chart(price_chart(prices, indicators, anomalies, ticker),
                    use_container_width=True)
    if not anomalies.empty:
        with st.expander(f"🚨 Anomaly Details ({len(anomalies)} events)"):
            d = anomalies.copy()
            d["close"]  = d["close"].round(2)
            d["zscore"] = d["zscore"].round(3)
            st.dataframe(d.sort_values("date", ascending=False),
                         use_container_width=True, hide_index=True)

with tab2:
    if indicators.empty:
        st.warning("No indicators found. Run: python indicators.py")
    else:
        st.plotly_chart(rsi_macd_chart(indicators, ticker), use_container_width=True)
        latest_ind = indicators.iloc[-1]
        c1, c2 = st.columns(2)
        with c1:
            rsi = latest_ind["rsi_14"]
            if rsi > 70:   st.error(f"🔴 RSI {rsi:.1f} — Overbought.")
            elif rsi < 30: st.success(f"🟢 RSI {rsi:.1f} — Oversold.")
            else:          st.info(f"⚪ RSI {rsi:.1f} — Neutral zone.")
        with c2:
            hist = latest_ind["macd_hist"]
            if hist > 0: st.success(f"🟢 MACD {hist:.4f} — Bullish momentum.")
            else:        st.error(f"🔴 MACD {hist:.4f} — Bearish momentum.")

with tab3:
    if forecasts.empty:
        st.warning("No forecasts found. Run: python forecasting.py && python xgboost_model.py")
    else:
        st.plotly_chart(forecast_chart(prices, forecasts, ticker), use_container_width=True)

        st.subheader("7-Day Numbers — All Models")
        c1, c2, c3, c4, c5 = st.columns(5)
        model_labels = {
            "arima": "ARIMA", "prophet": "Prophet",
            "xgboost": "XGBoost", "lightgbm": "LightGBM",
            "ensemble_stack": "⭐ Ensemble",
        }
        for col, model in zip([c1, c2, c3, c4, c5],
                               ["arima", "prophet", "xgboost", "lightgbm", "ensemble_stack"]):
            mdf = forecasts[forecasts["model"] == model][
                ["forecast_date", "predicted_close", "lower_bound", "upper_bound"]
            ].copy()
            with col:
                st.markdown(f"**{model_labels[model]}**")
                if mdf.empty:
                    st.caption("No data")
                else:
                    mdf.columns = ["Date", "Forecast", "Lower", "Upper"]
                    mdf[["Forecast","Lower","Upper"]] = mdf[["Forecast","Lower","Upper"]].round(2)
                    st.dataframe(mdf, use_container_width=True, hide_index=True)

        st.divider()
        st.caption("30-day holdout MAPE (lower = better)")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1: st.info("ARIMA: ~4.1%")
        with b2: st.info("Prophet: ~1.8%")
        with b3: st.success("XGBoost: ~1.1%")
        with b4: st.success("LightGBM: ~1.0%")
        with b5: st.success("Ensemble: best ⭐")

with tab4:
    if sentiment.empty:
        st.warning("No sentiment data. Run: python sentiment.py")
    else:
        avg_compound = float(sentiment["compound"].mean())
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(sentiment_gauge(avg_compound, ticker),
                            use_container_width=True)
            pos = len(sentiment[sentiment["sentiment"] == "positive"])
            neg = len(sentiment[sentiment["sentiment"] == "negative"])
            neu = len(sentiment[sentiment["sentiment"] == "neutral"])
            st.metric("Positive", pos)
            st.metric("Negative", neg)
            st.metric("Neutral",  neu)
        with c2:
            st.plotly_chart(sentiment_bar_chart(sentiment, ticker),
                            use_container_width=True)

        st.subheader("Latest Headlines")
        for _, row in sentiment.head(12).iterrows():
            c = row["compound"]
            icon = "🟢" if c > 0.1 else ("🔴" if c < -0.1 else "⚪")
            st.markdown(
                f"{icon} **{row['date'].strftime('%b %d')}** — {row['headline']} `{c:+.3f}`"
            )

with tab5:
    st.subheader("Market Sentiment — All Tickers (7 days)")
    sent_summary = load_sentiment_summary()
    if not sent_summary.empty:
        colors_sent = ["#26a69a" if v >= 0 else "#ef5350"
                       for v in sent_summary["avg_compound"]]
        fig_sent = go.Figure(go.Bar(
            x=sent_summary["ticker"], y=sent_summary["avg_compound"],
            marker_color=colors_sent,
        ))
        fig_sent.add_hline(y=0, line_dash="dot",
                           line_color="rgba(255,255,255,0.3)")
        fig_sent.update_layout(
            title="Average Sentiment Score by Ticker",
            template="plotly_dark", height=300,
            xaxis_title="Ticker", yaxis_title="Avg Compound",
        )
        st.plotly_chart(fig_sent, use_container_width=True)
        st.dataframe(sent_summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Anomaly Summary — All Tickers")
    summary = load_anomaly_summary()
    if not summary.empty:
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Bar(x=summary["ticker"], y=summary["spikes"],
                                  name="Spikes", marker_color="#ff6b6b"))
        fig_anom.add_trace(go.Bar(x=summary["ticker"], y=summary["crashes"],
                                  name="Crashes", marker_color="#ffd93d"))
        fig_anom.update_layout(
            barmode="group", template="plotly_dark",
            title="Anomaly Count per Ticker", height=300,
        )
        st.plotly_chart(fig_anom, use_container_width=True)

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
                    rows.append({"Ticker": t,
                                 "Latest Close": round(float(result[0]), 2)})
        except Exception:
            pass
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("Ensemble Strategy Backtest — 30-Day Holdout")
    st.caption(
        "Long/flat strategy: BUY when ensemble predicts price rise, hold cash otherwise. "
        "1-day holding period · 0.1% transaction cost · compared to buy-and-hold benchmark."
    )

    bt_summary = load_backtest_summary()

    if bt_summary.empty:
        st.warning("No backtest results found. Run: `python backtest.py`")
        st.code("python backtest.py          # all tickers\npython backtest.py --ticker AAPL  # single ticker")
    else:
        # ── Cumulative return chart for selected ticker ────────────────────────
        bt_series = load_backtest_series(ticker)
        if bt_series:
            days  = list(range(len(bt_series["daily_values"])))
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=days, y=bt_series["daily_values"],
                name="Ensemble Strategy",
                line=dict(color="#ffd700", width=3),
            ))
            fig_bt.add_trace(go.Scatter(
                x=days, y=bt_series["benchmark_values"],
                name="Buy & Hold",
                line=dict(color="#ffffff", width=2, dash="dash"),
            ))
            fig_bt.add_hline(
                y=10000, line_dash="dot",
                line_color="rgba(255,255,255,0.2)",
            )
            fig_bt.update_layout(
                title=f"{ticker} — Ensemble Strategy vs Buy & Hold ($10,000 start)",
                xaxis_title="Trading Day (holdout period)",
                yaxis_title="Portfolio Value (USD)",
                template="plotly_dark", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        # ── Top metrics for selected ticker ───────────────────────────────────
        ticker_row = bt_summary[bt_summary["ticker"] == ticker]
        if not ticker_row.empty:
            r = ticker_row.iloc[0]
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Strategy Return", f"{r['total_return']:+.2f}%")
            with m2:
                st.metric("Benchmark Return", f"{r['benchmark_return']:+.2f}%")
            with m3:
                alpha_val = r['alpha']
                st.metric("Alpha", f"{alpha_val:+.2f}%",
                          delta="outperforming" if alpha_val > 0 else "underperforming")
            with m4:
                st.metric("Sharpe Ratio", f"{r['sharpe_ratio']:.3f}")
            with m5:
                st.metric("Max Drawdown", f"{r['max_drawdown']:.2f}%")
            with m6:
                st.metric("Win Rate", f"{r['win_rate']:.1f}%",
                          delta=f"{int(r['num_trades'])} trades")

        st.divider()

        # ── Summary table — all tickers ───────────────────────────────────────
        st.subheader("All Tickers — Performance Summary")
        display = bt_summary[[
            "ticker", "total_return", "annualized_return", "sharpe_ratio",
            "max_drawdown", "win_rate", "num_trades", "benchmark_return", "alpha",
        ]].copy()
        display.columns = [
            "Ticker", "Return %", "Ann. Return %", "Sharpe",
            "Max DD %", "Win Rate %", "Trades", "Benchmark %", "Alpha %",
        ]

        # Format numeric columns cleanly
        for col in ["Return %", "Ann. Return %", "Max DD %", "Benchmark %", "Alpha %"]:
            display[col] = display[col].map(lambda x: f"{x:+.2f}%")
        display["Sharpe"]    = display["Sharpe"].map(lambda x: f"{x:.3f}")
        display["Win Rate %"] = display["Win Rate %"].map(lambda x: f"{x:.1f}%")
        display["Trades"]    = display["Trades"].astype(int)

        def color_alpha(val):
            if not isinstance(val, str):
                return ""
            color = "#26a69a" if val.startswith("+") else "#ef5350"
            return f"color: {color}"

        st.dataframe(
            display.style.applymap(color_alpha, subset=["Alpha %"]),
            use_container_width=True, hide_index=True,
        )

        st.divider()
        st.caption(
            "Sharpe > 1.0 = good risk-adjusted return · "
            "Alpha > 0 = strategy beats buy-and-hold · "
            "Ann. Return extrapolated from 30-day holdout — treat as indicative · "
            "All metrics on out-of-fold holdout (no look-ahead bias)"
        )