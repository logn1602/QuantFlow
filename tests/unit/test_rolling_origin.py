"""tests/unit/test_rolling_origin.py

Covers the rolling-origin loop and its baselines with fake forecasters: no
model is trained. The spy forecaster is the leakage check: whatever it is
given at an origin must predate that origin's cut.
"""

import numpy as np
import pandas as pd
import pytest

from quantflow.evaluation import rolling_origin as ro
from quantflow.utils.trading_calendar import TRADING_DAY, news_cutoff

DATES = pd.date_range("2026-01-05", periods=60, freq=TRADING_DAY)
H = 7


def _bars(seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(DATES))))
    return pd.DataFrame(
        {"date": DATES, "open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1e6}
    )  # fmt: skip


def _articles():
    """One article per day, stored the evening it was published."""
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "source": "rss",
            "published_at": [
                pd.Timestamp(f"{d.date()} 20:00", tz="UTC") for d in DATES
            ],
            "inserted_at": [pd.Timestamp(f"{d.date()} 22:40", tz="UTC") for d in DATES],
        }
    )


ORIGINS = [(30, "warmup"), (35, "scored"), (40, "scored")]


class Spy:
    def __init__(self):
        self.seen = []

    def __call__(self, data):
        self.seen.append(
            (data.origin, data.prices["date"].max(), data.articles["inserted_at"].max())
        )
        return ro.naive_forecast(data)


def test_a_forecaster_never_sees_past_its_origin():
    spy = Spy()
    ro.run_origins({"AAPL": _bars()}, _articles(), ORIGINS, {"spy": spy})
    assert len(spy.seen) == len(ORIGINS)
    for origin, last_price, last_article in spy.seen:
        assert last_price == origin
        assert last_article <= news_cutoff(origin)


def test_a_forecaster_gets_every_tickers_fetch_times_up_to_the_cutoff():
    """Decision D6 needs to know whether any fetch ran, not only whether
    this ticker had news."""
    seen = []

    def spy(data):
        seen.append((data.origin, data.fetched_at))
        return ro.naive_forecast(data)

    origin = DATES[35]
    other = pd.DataFrame(
        {"ticker": ["MSFT", "MSFT"],
         "inserted_at": [news_cutoff(origin) - pd.Timedelta(hours=1),
                         news_cutoff(origin) + pd.Timedelta(hours=1)]}
    )  # fmt: skip
    articles = pd.concat([_articles(), other], ignore_index=True)
    ro.run_origins({"AAPL": _bars()}, articles, [(35, "scored")], {"spy": spy})
    ((at, fetched),) = seen
    assert at == origin
    assert fetched.max() <= news_cutoff(origin)
    assert news_cutoff(origin) - pd.Timedelta(hours=1) in set(fetched)


def test_cells_are_scored_against_the_next_trading_days():
    bars = _bars()
    cells = ro.run_origins(
        {"AAPL": bars}, _articles(), ORIGINS, {"naive": ro.naive_forecast}
    )
    at_35 = cells[cells["origin"] == DATES[35]]
    assert at_35["target_date"].tolist() == list(DATES[36 : 36 + H])
    assert np.allclose(at_35["actual"], bars["close"].iloc[36 : 36 + H])
    assert (at_35["origin_close"] == bars["close"].iloc[35]).all()
    assert set(cells["role"]) == {"warmup", "scored"}
    assert set(cells["window"]) == {"validation"}
    assert cells.groupby("origin").size().eq(H).all()


def test_a_gap_in_the_data_fails_the_run_instead_of_shifting_the_targets():
    bars = _bars().drop(index=38).reset_index(drop=True)
    with pytest.raises(ro.CalendarMismatchError):
        ro.run_origins(
            {"AAPL": bars}, _articles(), [(35, "scored")], {"naive": ro.naive_forecast}
        )


def test_a_failing_model_gives_failed_cells_and_the_others_carry_on():
    def broken(data):
        raise ValueError("did not converge")

    cells = ro.run_origins(
        {"AAPL": _bars()},
        _articles(),
        ORIGINS,
        {"broken": broken, "naive": ro.naive_forecast},
    )
    failed = cells[cells["model"] == "broken"]
    assert failed["status"].eq("failed").all() and failed["yhat"].isna().all()
    assert failed["error"].str.contains("did not converge").all()
    assert cells[cells["model"] == "naive"]["status"].eq("ok").all()


def test_a_non_finite_forecast_is_a_failed_cell():
    def nan_on_day_3(data):
        out = ro.naive_forecast(data)
        out.loc[2, "yhat"] = np.nan
        return out

    cells = ro.run_origins(
        {"AAPL": _bars()}, _articles(), [(35, "scored")], {"m": nan_on_day_3}
    )
    assert cells["status"].eq("failed").all()
    assert cells["error"].eq("non-finite forecast").all()


def test_the_wrong_number_of_rows_is_a_bug_not_a_failure():
    with pytest.raises(ro.ForecasterContractError):
        ro.run_origins(
            {"AAPL": _bars()}, _articles(), [(35, "scored")],
            {"short": lambda data: ro.naive_forecast(data).iloc[:5]},
        )  # fmt: skip


def test_a_combiner_runs_at_scored_origins_with_their_base_forecasts_and_earlier_cells():
    """Warmup origins only supply base cells for the combiners."""
    calls = []

    def combiner(data, base, earlier):
        calls.append((data.origin, sorted(base), sorted(set(earlier["origin"]))))
        return base["naive"]

    ro.run_origins(
        {"AAPL": _bars()}, _articles(), ORIGINS,
        {"naive": ro.naive_forecast, "drift": ro.drift_forecast}, {"ens": combiner},
    )  # fmt: skip
    assert [c[0] for c in calls] == [DATES[35], DATES[40]]
    assert all(c[1] == ["drift", "naive"] for c in calls)
    assert calls[0][2] == [DATES[30]] and calls[1][2] == [DATES[30], DATES[35]]


def test_a_combiner_is_not_given_a_failed_base_forecast():
    seen = []

    def combiner(data, base, earlier):
        seen.append(sorted(base))
        return base["naive"]

    def broken(data):
        raise RuntimeError("x")

    ro.run_origins(
        {"AAPL": _bars()}, _articles(), [(35, "scored")],
        {"naive": ro.naive_forecast, "broken": broken}, {"ens": combiner},
    )  # fmt: skip
    assert seen == [["naive"]]


def test_naive_repeats_the_close_and_drift_uses_only_the_history():
    bars = _bars()
    data, _ = ro.origin_data("AAPL", bars, _articles(), 35, H)
    assert np.allclose(ro.naive_forecast(data)["yhat"], bars["close"].iloc[35])
    drift = np.mean(np.diff(np.log(bars["close"].iloc[:36].to_numpy())))
    expected = bars["close"].iloc[35] * np.exp(drift * np.arange(1, H + 1))
    assert np.allclose(ro.drift_forecast(data)["yhat"], expected)


def test_extra_forecast_columns_travel_with_the_cells():
    def flagged(data):
        return ro.naive_forecast(data).assign(yearly_seasonality=True)

    cells = ro.run_origins(
        {"AAPL": _bars()}, _articles(), [(35, "scored")], {"p": flagged}
    )
    assert cells["yearly_seasonality"].all()


def test_a_model_failing_over_ten_percent_fails_the_run():
    cells = pd.DataFrame(
        {"model": ["a"] * 10 + ["b"] * 10,
         "status": ["failed"] * 1 + ["ok"] * 9 + ["failed"] * 2 + ["ok"] * 8}
    )  # fmt: skip
    with pytest.raises(ro.FailureRateError, match="b 20%"):
        ro.check_failure_rate(cells)
    ro.check_failure_rate(cells[cells["model"] == "a"])  # exactly 10%: allowed


# ── Worker processes ──────────────────────────────────────────────────────────
# Module-level, so a spawned worker can import them.


def _fails_at_the_second_origin(data):
    if data.origin == DATES[35]:
        raise RuntimeError("no forecast today")
    return ro.drift_forecast(data)


def _halfway_to_earlier_naive(data, base, earlier):
    """A combiner that reads earlier origins' cells, as the rolling
    ensemble does."""
    past = earlier.loc[earlier["model"] == "naive", "yhat"]
    level = float(past.mean()) if len(past) else data.close
    return pd.DataFrame({"yhat": np.full(len(data.dates), (level + data.close) / 2)})


def test_worker_processes_give_the_sequential_cells():
    prices = {t: _bars(seed) for seed, t in enumerate(["AAPL", "AMZN", "MSFT", "NVDA"])}
    forecasters = {**ro.BASELINES, "flaky": _fails_at_the_second_origin}
    combiners = {"halfway": _halfway_to_earlier_naive}
    run = [prices, _articles(), ORIGINS, forecasters, combiners]
    sequential = ro.run_origins(*run)
    parallel = ro.run_origins(*run, workers=3)
    assert sequential["status"].eq("failed").any()
    assert set(sequential["role"]) == {"warmup", "scored"}
    # Everything but how long each call took.
    pd.testing.assert_frame_equal(
        parallel.drop(columns="seconds"), sequential.drop(columns="seconds")
    )


def test_more_workers_than_the_cap_are_refused():
    with pytest.raises(ValueError, match="workers must be 1 to 3"):
        ro.run_origins({"AAPL": _bars()}, _articles(), ORIGINS, ro.BASELINES, workers=4)


def test_a_contract_error_in_a_worker_still_raises():
    with pytest.raises(ro.ForecasterContractError):
        ro.run_origins(
            {"AAPL": _bars(1), "MSFT": _bars(2)},
            _articles(),
            ORIGINS,
            {"short": _three_rows},
            workers=2,
        )


def _three_rows(data):
    return ro.naive_forecast(data).iloc[:3]


def _whose_process(data):
    import os

    return ro.naive_forecast(data).assign(pid=os.getpid())


def test_the_tickers_really_run_in_other_processes():
    import os

    prices = {t: _bars(seed) for seed, t in enumerate(["AAPL", "AMZN", "MSFT"])}
    cells = ro.run_origins(
        prices, _articles(), ORIGINS, {"whose": _whose_process}, workers=3
    )
    pids = set(cells["pid"])
    assert os.getpid() not in pids and len(pids) > 1
