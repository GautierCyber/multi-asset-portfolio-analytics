import pandas as pd
import pytest

from multi_asset_portfolio.backtest import (
    BacktestConfig,
    BacktestError,
    run_strategy_suite,
    run_walk_forward_backtest,
)


def make_prices(periods: int = 100) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods)
    returns = pd.DataFrame(
        {
            "A": [0.0010] * (periods - 1),
            "B": [0.0004] * (periods - 1),
        },
        index=index[1:],
    )
    prices = pd.concat(
        [
            pd.DataFrame({"A": [100.0], "B": [100.0]}, index=index[:1]),
            100.0 * (1.0 + returns).cumprod(),
        ]
    )
    return prices


def test_start_date_delays_first_rebalance_and_restarts_from_cash() -> None:
    prices = make_prices()
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=10.0,
    )

    native = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )
    requested_start = native.rebalance_dates[1]

    delayed = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
        start_date=requested_start,
    )

    assert delayed.rebalance_dates[0] == requested_start
    assert delayed.net_returns.index[0] == requested_start
    assert delayed.gross_returns.iloc[0] == pytest.approx(0.0)
    assert delayed.turnover.iloc[0] == pytest.approx(1.0)
    assert delayed.transaction_cost_fractions.iloc[0] == pytest.approx(0.001)


def test_start_date_uses_first_eligible_month_end_on_or_after_request() -> None:
    prices = make_prices()
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=0.0,
    )
    native = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )

    requested_start = native.rebalance_dates[0] + pd.Timedelta(days=1)
    delayed = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
        start_date=requested_start,
    )

    assert delayed.rebalance_dates[0] == native.rebalance_dates[1]


def test_start_date_after_last_eligible_rebalance_is_rejected() -> None:
    prices = make_prices()
    config = BacktestConfig(estimation_window=20)

    with pytest.raises(BacktestError, match="on or after start_date"):
        run_walk_forward_backtest(
            prices,
            strategy="equal_weight",
            config=config,
            start_date="2030-01-01",
        )


def test_strategy_suite_respects_shared_explicit_start_date() -> None:
    prices = make_prices()
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=0.0,
    )
    native = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )
    requested_start = native.rebalance_dates[1]

    suite = run_strategy_suite(
        prices,
        config=config,
        strategies=("equal_weight", "gmv"),
        start_date=requested_start,
    )

    for result in suite.values():
        assert result.rebalance_dates[0] == requested_start
        assert result.net_returns.index[0] == requested_start
