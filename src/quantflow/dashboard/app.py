"""
dashboard/app.py
----------------
QuantFlow: stock analytics dashboard

Streamlit app visualizing the full pipeline:
  - Live price charts with Bollinger Bands + anomaly flags
  - RSI and MACD indicator charts with signal interpretation
  - 7-day forecasts: ARIMA, Prophet, XGBoost, LightGBM, the ensemble and
    "no change", with their measured accuracy (dashboard/accuracy.py)
  - FinBERT news sentiment analysis per ticker
  - Market-wide overview with anomaly summary

Usage:
    streamlit run dashboard.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import text

from quantflow.config import TICKERS
from quantflow.dashboard import accuracy, trending_view
from quantflow.dashboard.caching import CACHE_SECONDS, cached
from quantflow.db.connection import get_engine
from quantflow.db.prices import load_recent_ohlcv
from quantflow.features.news import NEWS_SOURCES
from quantflow.ingestion import trending_rules
from quantflow.utils.logger import get_logger

logger = get_logger("dashboard")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantFlow Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card { background: #1e1e2e; border-radius: 10px; padding: 16px; text-align: center; }
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 12px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Data loaders ──────────────────────────────────────────────────────────────


@cached
def load_prices(ticker: str, days: int = 180) -> pd.DataFrame:
    """The ticker's daily bars over the lookback, the largest read on the
    page: cached like the rest, where it was read again on every rerun."""
    return load_recent_ohlcv(ticker, days)


@cached
def load_latest_closes(tickers: tuple[str, ...]) -> pd.DataFrame:
    """The newest stored close of each ticker, over one connection. A ticker
    whose read fails is left out, as before."""
    query = text("""
        SELECT close FROM raw_prices
        WHERE ticker = :t AND source = 'yfinance'
        ORDER BY ts DESC LIMIT 1
    """)
    rows = []
    with get_engine().connect() as conn:
        for t in tickers:
            try:
                result = conn.execute(query, {"t": t}).fetchone()
            except Exception:
                # A failed statement aborts the transaction, which would fail
                # every ticker after it; start a fresh one.
                conn.rollback()
                continue
            if result:
                rows.append({"Ticker": t, "Latest Close": round(float(result[0]), 2)})
    return pd.DataFrame(rows)


# Indicator, anomaly and sentiment rows are dated by UTC calendar date, the
# date the price candles carry (db.prices.daily_bars). A bare ts::date follows
# the session's TimeZone instead: UTC on Supabase, but on a session west of UTC
# every indicator and anomaly stored at 00:00 UTC sat one day left of its
# candle. No regression test yet: this module cannot be imported in tests
# (Known problem #5 in docs/architecture.md).
@cached
def load_indicators(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT (ts AT TIME ZONE 'UTC')::date AS date,
               rsi_14, macd, macd_signal, macd_hist,
               bb_upper, bb_middle, bb_lower
        FROM technical_indicators
        WHERE ticker = :ticker
          AND ts >= NOW() - INTERVAL '1 day' * :days
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "days": days})
    df["date"] = pd.to_datetime(df["date"])
    return df


@cached
def load_anomalies(ticker: str, days: int = 180) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT (ts AT TIME ZONE 'UTC')::date AS date, close, zscore, flag
        FROM anomalies
        WHERE ticker = :ticker
          AND ts >= NOW() - INTERVAL '1 day' * :days
        ORDER BY ts ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "days": days})
    df["date"] = pd.to_datetime(df["date"])
    return df


@cached
def load_forecasts(ticker: str) -> pd.DataFrame:
    engine = get_engine()
    # Show only the most recent run per model. Averaging across runs blended
    # every forecast ever made for a date into one meaningless number.
    query = text("""
        SELECT f.model, f.forecast_date, f.predicted_close,
               f.lower_bound, f.upper_bound
        FROM forecasts f
        JOIN (
            SELECT model, MAX(run_at) AS run_at
            FROM forecasts
            WHERE ticker = :ticker
            GROUP BY model
        ) latest
          ON f.model = latest.model AND f.run_at = latest.run_at
        WHERE f.ticker = :ticker
        ORDER BY f.model, f.forecast_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker})
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


@cached
def load_sentiment(ticker: str, days: int = 7) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT
            (published_at AT TIME ZONE 'UTC')::date AS date,
            LEFT(headline, 80)              AS headline,
            sentiment,
            ROUND(compound::numeric, 3)     AS compound,
            source
        FROM news_sentiment
        WHERE ticker = :ticker
          AND source = ANY(:sources)
          AND published_at >= NOW() - INTERVAL '1 day' * :days
        ORDER BY published_at DESC
    """)
    # The news the models count (features.news.NEWS_SOURCES): RSS only.
    params = {"ticker": ticker, "days": days, "sources": list(NEWS_SOURCES)}
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


@cached
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
        WHERE source = ANY(:sources)
          AND published_at >= NOW() - INTERVAL '7 days'
        GROUP BY ticker
        ORDER BY avg_compound DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"sources": list(NEWS_SOURCES)})


@cached
def load_backtest_summary() -> pd.DataFrame:
    engine = get_engine()
    query = text("""
        SELECT ticker, total_return, annualized_return, sharpe_ratio,
               max_drawdown, win_rate, num_trades, benchmark_return, alpha,
               final_value, holdout_days, run_at::date AS run_date
        FROM backtest_results
        ORDER BY alpha DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


@cached
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
        "daily_values": _json.loads(row[0]),
        "benchmark_values": _json.loads(row[1]),
    }


@cached
def load_trending_table() -> pd.DataFrame:
    """The Trending tab's table, one grouped query (dashboard/trending_view.py)."""
    with get_engine().connect() as conn:
        return trending_view.load_table(conn)


@cached
def load_trending_lists() -> pd.DataFrame:
    """Each trending list's last success, for the line above the table."""
    with get_engine().connect() as conn:
        return trending_view.load_list_status(conn)


@cached
def load_trending_chart(symbol: str) -> tuple:
    """One trending symbol's chart: its latest bars with their indicators
    and anomaly flags (trending_view.load_chart), computed once an hour
    rather than on every rerun; () with no bars, which is not kept."""
    with get_engine().connect() as conn:
        return trending_view.load_chart(conn, symbol)


@cached
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
        fig.add_trace(
            go.Scatter(
                x=indicators["date"],
                y=indicators["bb_upper"],
                name="BB Upper",
                line={"color": "rgba(100,100,255,0.4)", "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=indicators["date"],
                y=indicators["bb_lower"],
                name="BB Lower",
                line={"color": "rgba(100,100,255,0.4)", "dash": "dash"},
                fill="tonexty",
                fillcolor="rgba(100,100,255,0.05)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=indicators["date"],
                y=indicators["bb_middle"],
                name="BB Middle (20 SMA)",
                line={"color": "rgba(150,150,255,0.6)", "dash": "dot"},
            )
        )
    fig.add_trace(
        go.Candlestick(
            x=prices["date"],
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )
    if not anomalies.empty:
        high_anom = anomalies[anomalies["flag"] == "HIGH"]
        low_anom = anomalies[anomalies["flag"] == "LOW"]
        if not high_anom.empty:
            fig.add_trace(
                go.Scatter(
                    x=high_anom["date"],
                    y=high_anom["close"],
                    mode="markers",
                    name="Spike 🔺",
                    marker={"symbol": "triangle-up", "size": 12, "color": "#ff6b6b"},
                )
            )
        if not low_anom.empty:
            fig.add_trace(
                go.Scatter(
                    x=low_anom["date"],
                    y=low_anom["close"],
                    mode="markers",
                    name="Crash 🔻",
                    marker={"symbol": "triangle-down", "size": 12, "color": "#ffd93d"},
                )
            )
    fig.update_layout(
        title=f"{ticker} — Price + Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def rsi_macd_chart(indicators, ticker):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("RSI (14)", "MACD"),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.6],
    )
    fig.add_trace(
        go.Scatter(
            x=indicators["date"],
            y=indicators["rsi_14"],
            name="RSI",
            line={"color": "#7b61ff"},
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=indicators["date"],
            y=indicators["macd"],
            name="MACD",
            line={"color": "#26c6da"},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=indicators["date"],
            y=indicators["macd_signal"],
            name="Signal",
            line={"color": "#ff7043"},
        ),
        row=2,
        col=1,
    )
    colors = [
        "#26a69a" if v >= 0 else "#ef5350" for v in indicators["macd_hist"].fillna(0)
    ]
    fig.add_trace(
        go.Bar(
            x=indicators["date"],
            y=indicators["macd_hist"],
            name="Histogram",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title=f"{ticker} — RSI & MACD Indicators",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    fig.update_yaxes(title_text="RSI", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    return fig


def forecast_chart(prices, forecasts, ticker):
    fig = go.Figure()
    recent = prices.tail(60)
    fig.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["close"],
            name="Actual",
            line={"color": "#ffffff", "width": 2},
        )
    )
    model_colors = {
        "arima": "#ff7043",
        "prophet": "#26c6da",
        "xgboost": "#ab47bc",
        "lightgbm": "#66bb6a",
        "ensemble_stack": "#ffd700",
    }
    band_colors = {
        "arima": "rgba(255,112,67,0.12)",
        "prophet": "rgba(38,198,218,0.12)",
        "xgboost": "rgba(171,71,188,0.12)",
        "lightgbm": "rgba(102,187,106,0.12)",
        "ensemble_stack": "rgba(255,215,0,0.12)",
    }
    for model in ["arima", "prophet", "xgboost", "lightgbm", "ensemble_stack"]:
        mdf = forecasts[forecasts["model"] == model].copy()
        if mdf.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.concat([mdf["forecast_date"], mdf["forecast_date"].iloc[::-1]]),
                y=pd.concat([mdf["upper_bound"], mdf["lower_bound"].iloc[::-1]]),
                fill="toself",
                fillcolor=band_colors[model],
                line={"color": "rgba(0,0,0,0)"},
                name=f"{model.upper()} band",
                showlegend=False,
            )
        )
        # Every model drawn alike: none is crowned. The measured 7-day error
        # below marks the lowest per window, often "no change".
        fig.add_trace(
            go.Scatter(
                x=mdf["forecast_date"],
                y=mdf["predicted_close"],
                name=accuracy.SHOWN[model].upper(),
                line={"color": model_colors[model], "width": 2, "dash": "dash"},
                mode="lines+markers",
            )
        )
    # The yardstick every model is measured against: the latest close, held
    # for all 7 days (dashboard/accuracy.py has how they compare).
    no_change = accuracy.no_change_forecast(recent, forecasts)
    if not no_change.empty:
        fig.add_trace(
            go.Scatter(
                x=no_change["date"],
                y=no_change["close"],
                name="No change (latest close)",
                line={"color": "#bdbdbd", "width": 2, "dash": "dot"},
                mode="lines",
            )
        )
    fig.update_layout(
        title=f"{ticker}: 7-day forecast, 4 base models, the ensemble and no change",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=450,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def sentiment_bar_chart(sentiment_df, ticker):
    daily = (
        sentiment_df.groupby("date")
        .agg(
            compound=("compound", "mean"),
        )
        .reset_index()
    )
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in daily["compound"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["compound"],
            marker_color=colors,
            name="Avg compound",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"{ticker} — Daily Sentiment Score (FinBERT)",
        xaxis_title="Date",
        yaxis_title="Compound Score",
        template="plotly_dark",
        height=300,
    )
    return fig


def sentiment_gauge(compound, ticker):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(float(compound), 3),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"{ticker} Sentiment"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#26a69a" if compound >= 0 else "#ef5350"},
                "steps": [
                    {"range": [-1, -0.2], "color": "rgba(239,83,80,0.2)"},
                    {"range": [-0.2, 0.2], "color": "rgba(100,100,100,0.2)"},
                    {"range": [0.2, 1], "color": "rgba(38,166,154,0.2)"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=260)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 QuantFlow")
    st.caption("Quantitative stock analytics platform")
    st.divider()

    ticker = st.selectbox("Select Ticker", TICKERS, index=0)
    days = st.slider("Lookback (days)", min_value=30, max_value=500, value=180, step=30)
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
    # No refresh button: it cleared the cache for every visitor, forcing fresh
    # reads of the shared database. The cache lifetime picks up each refresh.
    st.caption(
        "Data refreshes once a day, after the US close; this page shows it "
        f"within {CACHE_SECONDS // 60} minutes"
    )


# ── Main layout ───────────────────────────────────────────────────────────────

st.title(f"📊 {ticker} — QuantFlow Analytics")

prices = load_prices(ticker, days)
indicators = load_indicators(ticker, days)
anomalies = load_anomalies(ticker, days)
forecasts = load_forecasts(ticker)
sentiment = load_sentiment(ticker, days=7)

if prices.empty:
    st.error(f"No data found for {ticker}. Run: make seed")
    st.stop()

# ── Top metrics ───────────────────────────────────────────────────────────────
latest = prices.iloc[-1]
prev = prices.iloc[-2]
price_change = latest["close"] - prev["close"]
price_pct = (price_change / prev["close"]) * 100

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric(
        "Current Price",
        f"${latest['close']:.2f}",
        delta=f"{price_change:+.2f} ({price_pct:+.2f}%)",
    )
with col2:
    st.metric("Day High", f"${latest['high']:.2f}")
with col3:
    st.metric("Day Low", f"${latest['low']:.2f}")
with col4:
    if not indicators.empty:
        rsi = indicators.iloc[-1]["rsi_14"]
        rsi_label = (
            "🔴 Overbought"
            if rsi > 70
            else ("🟢 Oversold" if rsi < 30 else "⚪ Neutral")
        )
        st.metric("RSI (14)", f"{rsi:.1f}", delta=rsi_label)
with col5:
    if not sentiment.empty:
        avg_sent = float(sentiment["compound"].mean())
        sent_label = (
            "🟢 Positive"
            if avg_sent > 0.1
            else ("🔴 Negative" if avg_sent < -0.1 else "⚪ Neutral")
        )
        st.metric("Sentiment (7d)", f"{avg_sent:.3f}", delta=sent_label)
with col6:
    st.metric("Anomalies", len(anomalies), delta="flags", delta_color="off")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📈 Price & Bollinger Bands",
        "📉 RSI & MACD",
        "🎯 Forecasts",
        "🗞️ Sentiment",
        "🌍 Market Overview",
        "📊 Backtest",
        "🔥 Trending",
    ]
)

with tab1:
    st.plotly_chart(
        price_chart(prices, indicators, anomalies, ticker), use_container_width=True
    )
    if not anomalies.empty:
        with st.expander(f"🚨 Anomaly Details ({len(anomalies)} events)"):
            d = anomalies.copy()
            d["close"] = d["close"].round(2)
            d["zscore"] = d["zscore"].round(3)
            st.dataframe(
                d.sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

with tab2:
    if indicators.empty:
        st.warning("No indicators found. Run: make indicators")
    else:
        st.plotly_chart(rsi_macd_chart(indicators, ticker), use_container_width=True)
        latest_ind = indicators.iloc[-1]
        c1, c2 = st.columns(2)
        with c1:
            rsi = latest_ind["rsi_14"]
            if rsi > 70:
                st.error(f"🔴 RSI {rsi:.1f} — Overbought.")
            elif rsi < 30:
                st.success(f"🟢 RSI {rsi:.1f} — Oversold.")
            else:
                st.info(f"⚪ RSI {rsi:.1f} — Neutral zone.")
        with c2:
            hist = latest_ind["macd_hist"]
            if hist > 0:
                st.success(f"🟢 MACD {hist:.4f} — Bullish momentum.")
            else:
                st.error(f"🔴 MACD {hist:.4f} — Bearish momentum.")

with tab3:
    # The 7-day error as the rolling-origin evaluation measured it, from the
    # published table (dashboard/accuracy.py). The daily run cannot measure
    # it: it keeps no forecast history.
    published = accuracy.load_published()
    if published is not None:
        st.info(accuracy.baseline_note(published))
    if forecasts.empty:
        st.warning("No forecasts found. Run: make models")
    else:
        st.plotly_chart(
            forecast_chart(prices, forecasts, ticker), use_container_width=True
        )
        if published is not None:
            st.caption(accuracy.band_caption(published))

        st.subheader("7-day numbers, all models")
        model_labels = accuracy.SHOWN
        # Its first row is the latest close itself, not a forecast.
        no_change = accuracy.no_change_forecast(prices, forecasts).iloc[1:]
        for col, (model, label) in zip(
            st.columns(len(model_labels)), model_labels.items(), strict=True
        ):
            if model == accuracy.BASELINE:
                mdf = pd.DataFrame(
                    {"Date": no_change["date"], "Forecast": no_change["close"].round(2)}
                )
            else:
                mdf = forecasts[forecasts["model"] == model][
                    ["forecast_date", "predicted_close", "lower_bound", "upper_bound"]
                ].copy()
                mdf.columns = ["Date", "Forecast", "Lower", "Upper"]
                mdf[["Forecast", "Lower", "Upper"]] = mdf[
                    ["Forecast", "Lower", "Upper"]
                ].round(2)
            with col:
                st.markdown(f"**{label}**")
                if mdf.empty:
                    st.caption("No data")
                else:
                    st.dataframe(mdf, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"Measured 7-day accuracy, {ticker}")
    if published is None:
        st.info(
            "No published evaluation found. Run: python -m "
            "quantflow.pipelines.evaluate publish --validation <run> --test <run>"
        )
    elif not accuracy.covers(published, ticker):
        st.info(f"The published evaluation did not score {ticker}.")
        st.caption(accuracy.provenance(published))
    else:
        st.dataframe(
            accuracy.accuracy_rows(published, ticker),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(accuracy.provenance(published))

with tab4:
    if sentiment.empty:
        st.warning("No sentiment data. Run: make sentiment")
    else:
        avg_compound = float(sentiment["compound"].mean())
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(
                sentiment_gauge(avg_compound, ticker), use_container_width=True
            )
            pos = len(sentiment[sentiment["sentiment"] == "positive"])
            neg = len(sentiment[sentiment["sentiment"] == "negative"])
            neu = len(sentiment[sentiment["sentiment"] == "neutral"])
            st.metric("Positive", pos)
            st.metric("Negative", neg)
            st.metric("Neutral", neu)
        with c2:
            st.plotly_chart(
                sentiment_bar_chart(sentiment, ticker), use_container_width=True
            )

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
        colors_sent = [
            "#26a69a" if v >= 0 else "#ef5350" for v in sent_summary["avg_compound"]
        ]
        fig_sent = go.Figure(
            go.Bar(
                x=sent_summary["ticker"],
                y=sent_summary["avg_compound"],
                marker_color=colors_sent,
            )
        )
        fig_sent.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_sent.update_layout(
            title="Average Sentiment Score by Ticker",
            template="plotly_dark",
            height=300,
            xaxis_title="Ticker",
            yaxis_title="Avg Compound",
        )
        st.plotly_chart(fig_sent, use_container_width=True)
        st.dataframe(sent_summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Anomaly Summary — All Tickers")
    summary = load_anomaly_summary()
    if not summary.empty:
        fig_anom = go.Figure()
        fig_anom.add_trace(
            go.Bar(
                x=summary["ticker"],
                y=summary["spikes"],
                name="Spikes",
                marker_color="#ff6b6b",
            )
        )
        fig_anom.add_trace(
            go.Bar(
                x=summary["ticker"],
                y=summary["crashes"],
                name="Crashes",
                marker_color="#ffd93d",
            )
        )
        fig_anom.update_layout(
            barmode="group",
            template="plotly_dark",
            title="Anomaly Count per Ticker",
            height=300,
        )
        st.plotly_chart(fig_anom, use_container_width=True)

    st.divider()
    st.subheader("Latest Prices — All Tickers")
    latest_closes = load_latest_closes(tuple(TICKERS))
    if not latest_closes.empty:
        st.dataframe(latest_closes, use_container_width=True, hide_index=True)

with tab6:
    bt_summary = load_backtest_summary()

    # Window length is whatever the backtest actually evaluated on — the
    # out-of-fold window, which is shorter than the 30-day holdout.
    _windows = (
        sorted({int(d) for d in bt_summary["holdout_days"].dropna()})
        if not bt_summary.empty
        else []
    )
    if len(_windows) == 1:
        st.subheader(
            f"Ensemble Strategy Backtest — {_windows[0]}-Day Out-of-Fold Window"
        )
    else:
        st.subheader("Ensemble Strategy Backtest — Out-of-Fold Window")

    st.caption(
        "Long/flat strategy: BUY when ensemble predicts price rise, hold cash otherwise. "
        "1-day holding period · 0.1% transaction cost · compared to buy-and-hold benchmark. "
        "It demonstrates the evaluation harness, not a trading edge: the window "
        "is short and holds only a handful of trades."
    )

    if bt_summary.empty:
        st.warning("No backtest results found. Run: `make backtest`")
        st.code(
            "make backtest  # all tickers\n"
            "python -m quantflow.pipelines.backtest --ticker AAPL  # one ticker"
        )
    else:
        # ── Cumulative return chart for selected ticker ────────────────────────
        bt_series = load_backtest_series(ticker)
        if bt_series:
            day_indices = list(range(len(bt_series["daily_values"])))
            fig_bt = go.Figure()
            fig_bt.add_trace(
                go.Scatter(
                    x=day_indices,
                    y=bt_series["daily_values"],
                    name="Ensemble Strategy",
                    line={"color": "#ffd700", "width": 3},
                )
            )
            fig_bt.add_trace(
                go.Scatter(
                    x=day_indices,
                    y=bt_series["benchmark_values"],
                    name="Buy & Hold",
                    line={"color": "#ffffff", "width": 2, "dash": "dash"},
                )
            )
            fig_bt.add_hline(
                y=10000,
                line_dash="dot",
                line_color="rgba(255,255,255,0.2)",
            )
            fig_bt.update_layout(
                title=f"{ticker} — Ensemble Strategy vs Buy & Hold ($10,000 start)",
                xaxis_title="Trading Day (holdout period)",
                yaxis_title="Portfolio Value (USD)",
                template="plotly_dark",
                height=420,
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
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
                alpha_val = r["alpha"]
                st.metric(
                    "Alpha",
                    f"{alpha_val:+.2f}%",
                    delta="outperforming" if alpha_val > 0 else "underperforming",
                )
            with m4:
                st.metric("Sharpe Ratio", f"{r['sharpe_ratio']:.3f}")
            with m5:
                st.metric("Max Drawdown", f"{r['max_drawdown']:.2f}%")
            with m6:
                st.metric(
                    "Win Rate",
                    f"{r['win_rate']:.1f}%",
                    delta=f"{int(r['num_trades'])} trades",
                )

        st.divider()

        # ── Summary table — all tickers ───────────────────────────────────────
        st.subheader("All Tickers — Performance Summary")
        summary_table = bt_summary[
            [
                "ticker",
                "total_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "num_trades",
                "benchmark_return",
                "alpha",
            ]
        ].copy()
        summary_table.columns = [
            "Ticker",
            "Return %",
            "Ann. Return %",
            "Sharpe",
            "Max DD %",
            "Win Rate %",
            "Trades",
            "Benchmark %",
            "Alpha %",
        ]

        # Format numeric columns cleanly
        percent_columns = [
            "Return %",
            "Ann. Return %",
            "Max DD %",
            "Benchmark %",
            "Alpha %",
        ]
        for column_name in percent_columns:
            summary_table[column_name] = summary_table[column_name].map(
                lambda x: f"{x:+.2f}%"
            )
        summary_table["Sharpe"] = summary_table["Sharpe"].map(lambda x: f"{x:.3f}")
        summary_table["Win Rate %"] = summary_table["Win Rate %"].map(
            lambda x: f"{x:.1f}%"
        )
        summary_table["Trades"] = summary_table["Trades"].astype(int)

        def color_alpha(val):
            if not isinstance(val, str):
                return ""
            color = "#26a69a" if val.startswith("+") else "#ef5350"
            return f"color: {color}"

        st.dataframe(
            summary_table.style.applymap(color_alpha, subset=["Alpha %"]),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        _win_txt = f"{_windows[0]}-day" if len(_windows) == 1 else "the"
        st.caption(
            "Sharpe > 1.0 = good risk-adjusted return · "
            "Alpha > 0 = strategy beats buy-and-hold · "
            f"Ann. Return extrapolated from {_win_txt} out-of-fold window "
            "(×252/n) — treat as indicative · "  # noqa: RUF001
            "Base models and the ensemble meta-learner are both evaluated "
            "out-of-fold, so no prediction saw its own actual"
        )

with tab7:
    # The trending tier (docs/trending.md, section 10): discovery only, from
    # its own tables. Its data loading and shaping live in trending_view.py,
    # where they are tested; symbols and names are shown as text, never HTML.
    st.caption(trending_view.CAPTION)
    trending_settings: trending_rules.TrendingSettings | None
    try:
        trending_settings = trending_rules.settings(TICKERS)
    except trending_rules.TrendingConfigError as error:
        trending_settings = None
        logger.warning(f"The trending settings cannot be read: {error}")
        st.warning(trending_view.SETTINGS_WARNING)

    if trending_settings is not None:
        try:
            trending_table = load_trending_table()
            trending_lists = load_trending_lists()
            if trending_table.empty and trending_lists.empty:
                st.info(trending_view.EMPTY)
            else:
                st.caption(
                    trending_view.list_caption(
                        trending_view.list_lines(
                            trending_lists, trending_settings.lists
                        )
                    )
                )
                shown = trending_view.table_rows(
                    trending_table,
                    settings=trending_settings,
                    today=trending_rules.session_date(pd.Timestamp.now(tz="UTC")),
                )
                if shown.empty:
                    st.info(trending_view.EMPTY)
                else:
                    st.dataframe(
                        shown,
                        use_container_width=True,
                        hide_index=True,
                        column_config=trending_view.table_column_config(),
                    )
                    chosen = st.selectbox(
                        "Chart", list(shown["Symbol"]), key="trending_symbol"
                    )
                    chart = load_trending_chart(chosen)
                    if not chart:
                        st.info(f"No stored bars for {chosen} yet.")
                    else:
                        bars_t, indicators_t, flags_t = chart
                        st.plotly_chart(
                            price_chart(bars_t, indicators_t, flags_t, chosen),
                            use_container_width=True,
                        )
                        if not indicators_t.empty:
                            st.plotly_chart(
                                rsi_macd_chart(indicators_t, chosen),
                                use_container_width=True,
                            )
        except Exception:
            # The last tab: a fault here must not show a traceback, with the
            # database's details, to every visitor.
            logger.exception("The Trending tab failed")
            st.error("The trending data could not be loaded right now.")
