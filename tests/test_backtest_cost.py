import pandas as pd
import pytest



from src.backtest.engine import BacktestEngine  # adjust import to your module path


def make_df(prices, signals):
    return pd.DataFrame({"Close": prices, "signal": signals})


def test_round_trip_costs_two_trades():
    # Flat price, long for 3 bars then flat -> one entry + one exit.
    df = make_df([100.0] * 6, [0, 1, 1, 1, 0, 0])
    out = BacktestEngine(transaction_cost_bps=10, slippage_bps=0).run(df)
    assert out["strategy_return"].sum() == pytest.approx(-2 * 10 / 10_000)


def test_flip_long_to_short_costs_double():
    # Positions: 0, +1, +1, -1 -> trades 0, 1, 0, -2 => 3 units traded.
    df = make_df([100.0] * 4, [1, 1, -1, -1])
    out = BacktestEngine(transaction_cost_bps=10, slippage_bps=0).run(df)
    assert out["strategy_return"].sum() == pytest.approx(-3 * 10 / 10_000)


def test_cost_independent_of_share_price():
    # Regression test: cost must not scale with the stock's price level.
    sig = [0, 1, 1, 0]
    cheap = BacktestEngine().run(make_df([10.0] * 4, sig))
    pricey = BacktestEngine().run(make_df([1000.0] * 4, sig))
    assert cheap["cost"].sum() == pytest.approx(pricey["cost"].sum())
    assert cheap["cost"].sum() > 0