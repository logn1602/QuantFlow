"""
tests/test_backtest.py
----------------------
Covers the two pure functions in backtest.py: simulate_strategy and
compute_metrics. No DB, no network.
"""

import numpy as np
import pytest

from backtest import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    compute_metrics,
    simulate_strategy,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _preds_for(signals, prev_prices):
    """Build ensemble_preds that produce exactly the requested signals.

    simulate_strategy derives signals as (ensemble_preds > prev_prices),
    so nudging each prediction above/below the previous close controls it.
    """
    return np.where(np.asarray(signals) == 1, prev_prices + 1.0, prev_prices - 1.0)


def _rising(n=10, start=100.0, step=1.0):
    """A strictly rising close series of length n + 1."""
    return start + step * np.arange(n + 1, dtype=float)


# ── simulate_strategy ─────────────────────────────────────────────────────────


def test_rising_market_always_long_gives_positive_return():
    prices = _rising(n=10)
    prev_prices = prices[:-1]
    actual_prices = prices[1:]
    preds = _preds_for([1] * 10, prev_prices)

    sim = simulate_strategy(prev_prices, actual_prices, preds)

    assert all(s == 1 for s in sim["signals"])
    assert sim["daily_values"][-1] > INITIAL_CAPITAL
    assert sim["win_rate"] == 100.0


def test_always_flat_returns_exactly_zero_and_no_trades():
    prices = _rising(n=10)
    prev_prices = prices[:-1]
    actual_prices = prices[1:]
    preds = _preds_for([0] * 10, prev_prices)

    sim = simulate_strategy(prev_prices, actual_prices, preds)

    assert sim["num_trades"] == 0
    assert all(s == 0 for s in sim["signals"])
    assert np.allclose(sim["strategy_returns"], 0.0)
    # Never invested, so the portfolio never moves off the starting capital.
    assert sim["daily_values"][-1] == pytest.approx(INITIAL_CAPITAL)
    assert compute_metrics(sim)["total_return"] == pytest.approx(0.0)
    # No long days at all -> win rate must be the 0.0 fallback, not NaN.
    assert sim["win_rate"] == 0.0


def test_num_trades_counts_only_zero_to_one_transitions():
    # 0->1 at index 0, stays long, exits, re-enters at index 4, and one more
    # entry at index 7  =>  exactly 3 entries.
    signals = [1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    prices = _rising(n=10)
    prev_prices = prices[:-1]
    actual_prices = prices[1:]

    sim = simulate_strategy(
        prev_prices, actual_prices, _preds_for(signals, prev_prices)
    )

    assert sim["signals"] == signals
    assert sim["num_trades"] == 3


def test_transaction_cost_charged_on_entry_days_only():
    signals = [1, 1, 0, 1]
    prices = _rising(n=4)
    prev_prices = prices[:-1]
    actual_prices = prices[1:]

    sim = simulate_strategy(
        prev_prices, actual_prices, _preds_for(signals, prev_prices)
    )

    daily_returns = (actual_prices - prev_prices) / prev_prices
    gross = np.asarray(signals) * daily_returns
    charged = gross - np.asarray(sim["strategy_returns"])

    # Entries are at index 0 (0->1) and index 3 (0->1). Index 1 holds, so no
    # new cost; index 2 is flat.
    expected = np.array([TRANSACTION_COST, 0.0, 0.0, TRANSACTION_COST])
    assert np.allclose(charged, expected)


def test_holding_long_across_days_is_charged_once():
    prices = _rising(n=5)
    prev_prices = prices[:-1]
    actual_prices = prices[1:]
    preds = _preds_for([1] * 5, prev_prices)

    sim = simulate_strategy(prev_prices, actual_prices, preds)

    assert sim["num_trades"] == 1


# ── compute_metrics ───────────────────────────────────────────────────────────


def _sim_from_returns(strategy_returns, benchmark_returns=None):
    """Build the minimal sim dict compute_metrics consumes."""
    sr = np.asarray(strategy_returns, dtype=float)
    br = np.asarray(
        benchmark_returns if benchmark_returns is not None else strategy_returns,
        dtype=float,
    )
    return {
        "strategy_returns": sr,
        "benchmark_returns": br,
        "daily_values": np.concatenate(
            [[INITIAL_CAPITAL], INITIAL_CAPITAL * np.cumprod(1 + sr)]
        ).tolist(),
        "benchmark_values": np.concatenate(
            [[INITIAL_CAPITAL], INITIAL_CAPITAL * np.cumprod(1 + br)]
        ).tolist(),
        "win_rate": 0.0,
        "num_trades": 0,
    }


def test_total_return_matches_hand_computed_value():
    # +10% then +10% compounds to +21%, not +20%.
    sim = _sim_from_returns([0.10, 0.10])
    m = compute_metrics(sim)

    assert m["total_return"] == pytest.approx(21.0, abs=1e-9)
    assert m["final_value"] == pytest.approx(INITIAL_CAPITAL * 1.21, abs=1e-6)


def test_max_drawdown_is_never_positive():
    for returns in (
        [0.05, 0.05, 0.05],  # monotonic up
        [-0.05, 0.02, -0.03],  # choppy
        [0.0, 0.0, 0.0],
    ):  # flat
        m = compute_metrics(_sim_from_returns(returns))
        assert m["max_drawdown"] <= 0.0


def test_max_drawdown_matches_hand_computed_value():
    # 100 -> 110 -> 88: peak 110, trough 88  =>  -20%
    sim = _sim_from_returns([0.10, -0.20])
    assert compute_metrics(sim)["max_drawdown"] == pytest.approx(-20.0, abs=1e-6)


@pytest.mark.parametrize("constant", [0.0, -0.002, 0.01, 0.001, 0.0025, -0.0075])
def test_sharpe_is_zero_when_returns_have_no_variance(constant):
    """Guards the zero-variance branch in compute_metrics — a constant return
    series has no variance and must not divide by ~zero.

    This is the path a 0-trade backtest takes (every strategy return is
    exactly 0.0), which is how JPM reports Sharpe 0.000.

    0.001 is here deliberately: `0.001 - RISK_FREE_RATE/252` repeated leaves
    ~1e-19 of float residue in std(), which the original `> 0` guard let
    through and turned into a Sharpe of ~1.2e17.
    """
    m = compute_metrics(_sim_from_returns([constant] * 5))
    assert m["sharpe_ratio"] == 0.0


def test_sharpe_is_still_computed_for_genuinely_small_variance():
    """The zero-variance tolerance must not swallow real signal. Daily-return
    variance is orders of magnitude above ZERO_VARIANCE_TOL."""
    m = compute_metrics(_sim_from_returns([0.001, 0.0011, 0.0009, 0.00105]))
    assert m["sharpe_ratio"] != 0.0
    assert np.isfinite(m["sharpe_ratio"])
    assert abs(m["sharpe_ratio"]) < 1e4, "implausible Sharpe from tiny variance"


def test_alpha_is_annualised_difference_vs_benchmark():
    sim = _sim_from_returns([0.02, 0.02], benchmark_returns=[0.01, 0.01])
    m = compute_metrics(sim)

    assert m["alpha"] == pytest.approx(
        m["annualised_return"] - _bench_annual(sim), abs=0.01
    )
    assert m["alpha"] > 0


def _bench_annual(sim):
    bv = np.asarray(sim["benchmark_values"])
    n = len(sim["strategy_returns"])
    return float(((bv[-1] / INITIAL_CAPITAL) ** (252 / n) - 1) * 100)


def test_metrics_are_all_finite():
    m = compute_metrics(_sim_from_returns([0.03, -0.01, 0.02, 0.0]))
    for key, value in m.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
