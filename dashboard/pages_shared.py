"""
Shared config-building logic for every dashboard page. Defined ONCE here
so Page 1, Page 2, and Page 3 can never drift out of sync with each other
(exactly the kind of mismatch that caused confusing, hard-to-diagnose
differences between the dashboard and the CLI earlier in this project).
"""

STRATEGY_PRESETS = {
    "sma_crossover": {
        "feature_cfg": lambda p: [
            {"name": "sma", "params": {"window": p["fast_window"]}},
            {"name": "sma", "params": {"window": p["slow_window"]}},
        ],
        "params": {"fast_window": 20, "slow_window": 50},
    },
    "ema_crossover": {
        "feature_cfg": lambda p: [
            {"name": "ema", "params": {"span": p["fast_span"]}},
            {"name": "ema", "params": {"span": p["slow_span"]}},
        ],
        "params": {"fast_span": 20, "slow_span": 50},
    },
    "momentum_strategy": {
        "feature_cfg": lambda p: [
            {"name": "momentum", "params": {"window": p["momentum_window"]}},
        ],
        "params": {"momentum_window": 10, "threshold": 0.02},
    },
    "bollinger_reversion": {
        "feature_cfg": lambda p: [
            {"name": "bollinger_upper", "params": {"window": p["window"], "num_std": p["num_std"]}},
            {"name": "bollinger_lower", "params": {"window": p["window"], "num_std": p["num_std"]}},
        ],
        "params": {"window": 20, "num_std": 2.0},
    },
}

ML_FEATURE_CFG = [
    {"name": "price_to_sma", "params": {"window": 10}},
    {"name": "price_to_sma", "params": {"window": 20}},
    {"name": "price_to_sma", "params": {"window": 50}},
    {"name": "price_to_ema", "params": {"span": 10}},
    {"name": "price_to_ema", "params": {"span": 20}},
    {"name": "price_to_ema", "params": {"span": 50}},
    {"name": "rsi", "params": {"window": 14}},
    {"name": "volatility", "params": {"window": 10}},
    {"name": "volatility", "params": {"window": 20}},
    {"name": "momentum", "params": {"window": 5}},
    {"name": "momentum", "params": {"window": 10}},
    {"name": "momentum", "params": {"window": 20}},
    {"name": "bollinger_position", "params": {"window": 20, "num_std": 2.0}},
]
ML_FEATURE_COLUMNS = [
    "price_to_sma_window10", "price_to_sma_window20", "price_to_sma_window50",
    "price_to_ema_span10", "price_to_ema_span20", "price_to_ema_span50",
    "rsi_window14", "volatility_window10", "volatility_window20",
    "momentum_window5", "momentum_window10", "momentum_window20",
    "bollinger_position_window20_num_std2.0",
]


def build_base_config(train_size, test_size, initial_capital):
    """The parts of a config every page needs, regardless of strategy."""
    return {
        "data": {"start_date": "2018-01-01", "end_date": "2024-12-31"},
        "walk_forward": {"train_size": train_size, "test_size": test_size},
        "backtest": {
            "initial_capital": initial_capital,
            "transaction_cost_bps": 5,
            "slippage_bps": 2,
            "position_size_pct": 1.0,
        },
        "evaluation": {
            "risk_free_rate_annual": 0.05,
            "metrics": ["sharpe", "sortino", "max_drawdown", "win_rate", "turnover", "cagr",
                       "total_return", "direction_accuracy"],
        },
    }


def union_feature_cfg(strategy_keys):
    """
    Builds the combined feature list needed to run ALL the given strategies
    together - required for the portfolio's best-per-ticker mode, since
    run_portfolio_best_per_ticker computes features ONCE from a single
    shared config.features list and then every ticker's own strategy reads
    from that same computed table. If ticker A uses sma_crossover and
    ticker B uses an ML model, the features list must include both SMA
    AND all the ML features, or whichever ticker's strategy needs a
    column that isn't there will fail.
    """
    seen = set()
    combined = []

    def add(entry):
        key = (entry["name"], tuple(sorted(entry["params"].items())))
        if key not in seen:
            seen.add(key)
            combined.append(entry)

    needs_ml = False
    for key in strategy_keys:
        if key.startswith("ml_"):
            needs_ml = True
        else:
            preset = STRATEGY_PRESETS[key]
            for entry in preset["feature_cfg"](preset["params"]):
                add(entry)

    if needs_ml:
        for entry in ML_FEATURE_CFG:
            add(entry)

    return combined