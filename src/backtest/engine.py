"""
Backtest engine (cost-fixed version).

Given a DataFrame with a `signal` column (1 long, -1 short, 0 flat),
simulate trading it, including transaction costs and slippage.
Strategy-agnostic: any strategy that produces `signal` works.
"""

import pandas as pd
import numpy as np


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100_000,
        transaction_cost_bps: float = 5,
        slippage_bps: float = 2,
        position_size_pct: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.position_size_pct = position_size_pct

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df must contain 'Close' and 'signal' columns.
        Adds: position (fraction of capital, signed), trade (change in
        position), cost (dollars), strategy_return (net, after costs),
        equity (cumulative curve).
        """
        out = df.copy()

        # Signal decided at close of bar t; position held over bar t+1.
        out["position"] = out["signal"].shift(1).fillna(0) * self.position_size_pct
        out["trade"] = out["position"].diff().fillna(out["position"])

        # `trade` is a FRACTION of capital, so notional traded is
        # |trade| * capital (NOT * Close, which is a per-share price).
        bps_total = (self.transaction_cost_bps + self.slippage_bps) / 10_000
        out["cost"] = out["trade"].abs() * self.initial_capital * bps_total

        daily_returns = out["Close"].pct_change().fillna(0)
        gross_pnl = out["position"] * daily_returns * self.initial_capital
        out["strategy_return"] = (gross_pnl - out["cost"]) / self.initial_capital
        out["equity"] = self.initial_capital * (1 + out["strategy_return"]).cumprod()
        return out