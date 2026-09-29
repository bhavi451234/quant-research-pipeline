"""
Backtest engine.

Responsibility: given a DataFrame that already has a `signal` column
(1 = long, -1 = short, 0 = flat), simulate what actually trading that
signal would have done — including costs. This is the layer that
turns "the signal predicted returns" into "you would have made money
after realistic frictions," which are two very different claims.

Deliberately signal-agnostic: this file has no idea what SMA or RSI
are. Strategies live elsewhere and just need to produce a `signal`
column. That separation is what lets you swap strategies without
touching the backtester.
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
        Returns a DataFrame with the added columns:
          position       - shares/units held (signed)
          trade          - change in position this bar (nonzero = a trade happened)
          cost            - dollar cost of trading this bar (transaction cost + slippage)
          strategy_return - net daily return of the strategy after costs
          equity          - cumulative equity curve starting at initial_capital
        """
        out = df.copy()

        # Signal is decided using info available at the close of bar t;
        # the position it implies can only be entered at bar t+1's open in reality.
        # We approximate by shifting the signal one bar forward to avoid lookahead.
        out["position"] = out["signal"].shift(1).fillna(0) * self.position_size_pct

        out["trade"] = out["position"].diff().fillna(out["position"])

        # Cost is charged on the notional value traded, in both transaction cost and slippage.
        bps_total = (self.transaction_cost_bps + self.slippage_bps) / 10_000
        out["cost"] = out["trade"].abs() * out["Close"] * bps_total

        daily_returns = out["Close"].pct_change().fillna(0)
        gross_pnl = out["position"] * daily_returns * self.initial_capital
        out["strategy_return"] = (gross_pnl - out["cost"]) / self.initial_capital

        out["equity"] = self.initial_capital * (1 + out["strategy_return"]).cumprod()

        return out 