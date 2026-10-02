import pandas as pd


def combine_portfolio_equity(ticker_backtests: dict) -> pd.DataFrame:
    """
    Combines each ticker's own backtest result into one portfolio equity
    curve, by summing equity across tickers, day by day.

    Returns 'equity', 'strategy_return' (the portfolio's own day-to-day
    return, computed from the combined equity, NOT summed from each
    ticker's individual strategy_return), and 'window'.

    Deliberately does NOT attempt to produce position, trade, or Close at
    the portfolio level - none of those have a single well-defined meaning
    for a multi-asset portfolio (a portfolio can be simultaneously long one
    stock and short another; there is no one "position"). This means
    metrics that need those columns (win_rate, turnover,
    direction_accuracy) cannot be computed on the portfolio result, only
    on each ticker's own backtest individually. sharpe, sortino,
    max_drawdown, cagr, and total_return all work fine here, since they
    only need equity / strategy_return.
    """
    tickers = list(ticker_backtests.keys())
    if len(tickers) == 0:
        raise ValueError("combine_portfolio_equity needs at least one ticker's backtest result.")

    reference_ticker = tickers[0]
    reference_dates = ticker_backtests[reference_ticker].index
    reference_windows = ticker_backtests[reference_ticker]["window"]

    for ticker in tickers[1:]:
        other_dates = ticker_backtests[ticker].index
        if not reference_dates.equals(other_dates):
            raise ValueError(
                f"Date mismatch: '{reference_ticker}' and '{ticker}' have different "
                f"walk-forward dates. All tickers must use the same train_size/test_size "
                f"and date range so their windows line up on the same calendar dates."
            )
        other_windows = ticker_backtests[ticker]["window"]
        if not reference_windows.equals(other_windows):
            raise ValueError(
                f"Window mismatch: '{reference_ticker}' and '{ticker}' have different "
                f"window numbering despite matching dates."
            )

    equity_columns = pd.concat(
        [ticker_backtests[t]["equity"] for t in tickers], axis=1, keys=tickers
    )
    portfolio_equity = equity_columns.sum(axis=1)
    portfolio_return = portfolio_equity.pct_change().fillna(0)

    return pd.DataFrame({
        "equity": portfolio_equity,
        "strategy_return": portfolio_return,
        "window": reference_windows,
    })