import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from peak_trading_agent import (
    make_synth, FeatureStore, DonchianBreakout20,
    atr_sizing, fractional_kelly, risk_parity_weights,
    kpis, simulate_intrabar
)


def test_make_synth_deterministic():
    np.random.seed(42)
    df1 = make_synth(n=10, price0=100)
    np.random.seed(42)
    df2 = make_synth(n=10, price0=100)
    pd.testing.assert_frame_equal(df1, df2)


def test_feature_store_compute_columns():
    np.random.seed(0)
    df = make_synth(n=100, price0=100).set_index("timestamp")
    feats = FeatureStore().compute(df)
    expected_cols = {"atr14", "rsi2", "ma_fast", "ma_slow", "donchian_upper", "bb_up"}
    assert expected_cols.issubset(feats.columns)
    assert len(feats) > 0
    assert not feats.isna().any().any()


def test_strategy_donchian_breakout():
    strat = DonchianBreakout20({"donchian": 20})
    row = {"donchian_upper": 100, "close": 101}
    sig = strat.generate(row)
    assert sig.side == "buy"
    assert sig.reason == "breakout"


def test_sizing_functions():
    size, stop_dist, dollar_risk = atr_sizing(10000, 50, 2, 0.01)
    assert size == 1
    assert stop_dist == 100
    assert dollar_risk == 100

    k = fractional_kelly(0.55, 1.5, 1.0, cap=0.02)
    assert k == 0.02


def test_risk_parity_weights():
    cov = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=["A", "B"], columns=["A", "B"])
    w = risk_parity_weights(cov, iters=50)
    assert np.isclose(w.sum(), 1.0)
    assert w["A"] > w["B"]


def test_reporting_kpis():
    df = pd.DataFrame({
        "net_R": [1.0, -0.5, 1.5],
        "equity": [1.0, 0.5, 2.0],
        "pnl_R": [1.0, -0.5, 1.5],
        "fee_R": [0.1, 0.2, 0.0],
        "slip_R": [0.05, 0.05, 0.05],
    })
    metrics = kpis(df)
    assert np.isclose(metrics["pf"], 5.0)
    assert np.isclose(metrics["hit"], 2/3)
    assert np.isclose(metrics["avgR"], 2/3)
    assert np.isclose(metrics["maxDD"], 0.25)
    assert np.isclose(metrics["cost_share"], 0.15)


def test_simulate_intrabar_paths():
    # Buy side: take-profit only
    prob_tp, prob_sl = simulate_intrabar(100, 105, 99, 104, 98, 103, "buy")
    assert np.isclose(prob_tp, 1.0) and np.isclose(prob_sl, 0.0)
    assert prob_tp + prob_sl <= 1.0

    # Buy side: stop-loss only
    prob_tp, prob_sl = simulate_intrabar(100, 101, 95, 97, 96, 103, "buy")
    assert np.isclose(prob_tp, 0.0) and np.isclose(prob_sl, 1.0)
    assert prob_tp + prob_sl <= 1.0

    # Buy side: both barriers depending on path
    prob_tp, prob_sl = simulate_intrabar(100, 105, 95, 102, 96, 103, "buy")
    assert np.isclose(prob_tp, 0.4) and np.isclose(prob_sl, 0.6)
    assert prob_tp + prob_sl <= 1.0

    # Sell side: take-profit only
    prob_tp, prob_sl = simulate_intrabar(100, 101, 94, 96, 105, 95, "sell")
    assert np.isclose(prob_tp, 1.0) and np.isclose(prob_sl, 0.0)
    assert prob_tp + prob_sl <= 1.0

    # Sell side: stop-loss only
    prob_tp, prob_sl = simulate_intrabar(100, 106, 98, 104, 105, 95, "sell")
    assert np.isclose(prob_tp, 0.0) and np.isclose(prob_sl, 1.0)
    assert prob_tp + prob_sl <= 1.0

    # Sell side: both barriers depending on path
    prob_tp, prob_sl = simulate_intrabar(100, 110, 90, 96, 105, 95, "sell")
    assert np.isclose(prob_tp, 0.6) and np.isclose(prob_sl, 0.4)
    assert prob_tp + prob_sl <= 1.0

    # Edge case: open equals tp equals stop
    prob_tp, prob_sl = simulate_intrabar(100, 100, 100, 100, 100, 100, "buy")
    assert np.isclose(prob_tp, 1.0) and np.isclose(prob_sl, 0.0)
    assert prob_tp + prob_sl <= 1.0

    prob_tp, prob_sl = simulate_intrabar(100, 100, 100, 100, 100, 100, "sell")
    assert np.isclose(prob_tp, 1.0) and np.isclose(prob_sl, 0.0)
    assert prob_tp + prob_sl <= 1.0

