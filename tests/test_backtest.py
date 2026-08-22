import numpy as np
import pandas as pd
import pytest

from multi_asset_portfolio.backtest import (
    BacktestConfig,
    BacktestError,
    monthly_rebalance_dates,
    run_strategy_suite,
    run_walk_forward_backtest,
)
from multi_asset_portfolio.portfolio import PortfolioConstraints


def make_prices(
    *,
    periods: int = 90,
    seed: int = 123,
    columns: tuple[str, ...] = ("A", "B", "C"),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(
        loc=np.array([0.0004, 0.0002, 0.0001])[: len(columns)],
        scale=np.array([0.0100, 0.0060, 0.0040])[: len(columns)],
        size=(periods - 1, len(columns)),
    )
    returns = np.clip(returns, -0.20, 0.20)
    growth = np.vstack(
        [
            np.ones(len(columns)),
            np.cumprod(1.0 + returns, axis=0),
        ]
    )
    index = pd.bdate_range("2024-01-02", periods=periods)
    return pd.DataFrame(
        100.0 * growth,
        index=index,
        columns=list(columns),
    )


def test_config_rejects_invalid_estimation_window() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        BacktestConfig(estimation_window=1)


def test_config_rejects_negative_transaction_cost() -> None:
    with pytest.raises(ValueError, match="between 0 and 10,000"):
        BacktestConfig(transaction_cost_bps=-1.0)


def test_monthly_rebalance_dates_are_month_end_and_exclude_terminal_date() -> None:
    index = pd.bdate_range("2024-01-02", periods=80)

    result = monthly_rebalance_dates(
        index,
        estimation_window=20,
    )

    expected = pd.DatetimeIndex(
        [
            "2024-01-31",
            "2024-02-29",
            "2024-03-29",
        ]
    )

    assert result.equals(expected)
    assert result[-1] < index[-1]


def test_monthly_rebalance_dates_require_future_oos_observation() -> None:
    index = pd.bdate_range("2024-01-02", periods=20)

    with pytest.raises(BacktestError, match="too short"):
        monthly_rebalance_dates(
            index,
            estimation_window=20,
        )


def test_equal_weight_backtest_uses_exact_estimation_window() -> None:
    prices = make_prices(periods=90)
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=0.0,
    )

    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )

    assert (
        result.rebalance_log["estimation_observations"]
        == 20
    ).all()

    assert (
        result.rebalance_log["estimation_end"]
        < result.rebalance_log.index
    ).all()

    all_returns = prices.pct_change(fill_method=None).iloc[1:]

    for date, row in result.rebalance_log.iterrows():
        expected_sample = all_returns.loc[
            all_returns.index < date
        ].tail(20)
        assert len(expected_sample) == 20
        assert row["estimation_start"] == expected_sample.index[0]
        assert row["estimation_end"] == expected_sample.index[-1]


def test_initial_target_is_applied_only_after_first_rebalance_close() -> None:
    prices = make_prices(periods=90)
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=0.0,
    )

    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )

    first_date = result.rebalance_dates[0]
    assert result.gross_returns.loc[first_date] == pytest.approx(0.0)
    assert result.net_returns.loc[first_date] == pytest.approx(0.0)

    first_target = result.target_weights.loc[first_date]
    assert result.end_weights.loc[first_date].to_numpy() == pytest.approx(
        first_target.to_numpy()
    )

    next_date = result.net_returns.index[
        result.net_returns.index.get_loc(first_date) + 1
    ]
    assert result.beginning_weights.loc[next_date].to_numpy() == pytest.approx(
        first_target.to_numpy()
    )


def test_initial_transaction_cost_is_charged_from_cash() -> None:
    prices = make_prices(periods=90)
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=10.0,
    )

    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )

    first_date = result.rebalance_dates[0]

    assert result.turnover.loc[first_date] == pytest.approx(1.0)
    assert result.transaction_cost_fractions.loc[first_date] == pytest.approx(
        0.001
    )
    assert result.transaction_cost_amounts.loc[first_date] == pytest.approx(
        0.001
    )
    assert result.net_returns.loc[first_date] == pytest.approx(-0.001)
    assert result.wealth.loc[first_date] == pytest.approx(0.999)


def test_initial_transaction_cost_can_be_disabled() -> None:
    prices = make_prices(periods=90)
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=10.0,
        charge_initial_transaction_cost=False,
    )

    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )

    first_date = result.rebalance_dates[0]
    assert result.turnover.loc[first_date] == pytest.approx(1.0)
    assert result.transaction_cost_fractions.loc[first_date] == pytest.approx(
        0.0
    )
    assert result.wealth.loc[first_date] == pytest.approx(1.0)


def test_weights_drift_between_rebalances_and_rebalance_turnover_is_exact() -> None:
    index = pd.bdate_range("2024-01-02", periods=65)
    returns = pd.DataFrame(
        {
            "A": np.full(64, 0.01),
            "B": np.zeros(64),
        },
        index=index[1:],
    )
    prices = pd.concat(
        [
            pd.DataFrame(
                {"A": [100.0], "B": [100.0]},
                index=index[:1],
            ),
            100.0 * (1.0 + returns).cumprod(),
        ]
    )

    config = BacktestConfig(
        estimation_window=10,
        transaction_cost_bps=0.0,
    )

    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )

    second_rebalance = result.rebalance_dates[1]
    pretrade = result.pretrade_weights.loc[second_rebalance]
    target = result.target_weights.loc[second_rebalance]

    assert pretrade["A"] > 0.5
    assert pretrade["B"] < 0.5

    expected_turnover = 0.5 * float((target - pretrade).abs().sum())
    assert result.turnover.loc[second_rebalance] == pytest.approx(
        expected_turnover
    )


def test_transaction_cost_is_multiplicative_after_rebalance_day_return() -> None:
    index = pd.bdate_range("2024-01-02", periods=65)
    returns = pd.DataFrame(
        {
            "A": np.full(64, 0.01),
            "B": np.zeros(64),
        },
        index=index[1:],
    )
    prices = pd.concat(
        [
            pd.DataFrame(
                {"A": [100.0], "B": [100.0]},
                index=index[:1],
            ),
            100.0 * (1.0 + returns).cumprod(),
        ]
    )

    config = BacktestConfig(
        estimation_window=10,
        transaction_cost_bps=100.0,
        charge_initial_transaction_cost=False,
    )

    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
    )

    second_rebalance = result.rebalance_dates[1]
    gross = result.gross_returns.loc[second_rebalance]
    cost = result.transaction_cost_fractions.loc[second_rebalance]
    expected_net = (1.0 + gross) * (1.0 - cost) - 1.0

    assert result.net_returns.loc[second_rebalance] == pytest.approx(
        expected_net
    )


def test_end_weights_sum_to_one_every_day() -> None:
    prices = make_prices(periods=90)
    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=BacktestConfig(
            estimation_window=20,
            transaction_cost_bps=0.0,
        ),
    )

    np.testing.assert_allclose(
        result.end_weights.sum(axis=1).to_numpy(),
        np.ones(len(result.end_weights)),
        atol=1e-10,
    )


def test_target_weights_respect_portfolio_constraints() -> None:
    prices = make_prices(periods=90)
    constraints = PortfolioConstraints(
        lower_bound=0.10,
        upper_bound=0.60,
    )

    result = run_walk_forward_backtest(
        prices,
        strategy="gmv",
        config=BacktestConfig(
            estimation_window=20,
            transaction_cost_bps=0.0,
            constraints=constraints,
        ),
    )

    assert (result.target_weights >= 0.10 - 1e-8).all().all()
    assert (result.target_weights <= 0.60 + 1e-8).all().all()
    np.testing.assert_allclose(
        result.target_weights.sum(axis=1).to_numpy(),
        np.ones(len(result.target_weights)),
        atol=1e-8,
    )


def test_rebalance_close_and_future_prices_do_not_change_first_gmv_target() -> None:
    prices = make_prices(periods=90, seed=123)
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=0.0,
    )

    original = run_walk_forward_backtest(
        prices,
        strategy="gmv",
        config=config,
    )

    first_rebalance = original.rebalance_dates[0]
    modified = prices.copy()

    modified.loc[first_rebalance, "A"] *= 1.25

    future_mask = modified.index > first_rebalance
    modified.loc[future_mask, "A"] *= np.linspace(
        1.0,
        1.5,
        future_mask.sum(),
    )

    rerun = run_walk_forward_backtest(
        modified,
        strategy="gmv",
        config=config,
    )

    np.testing.assert_allclose(
        original.target_weights.loc[first_rebalance].to_numpy(),
        rerun.target_weights.loc[first_rebalance].to_numpy(),
        atol=1e-10,
    )


def test_wealth_matches_compounded_net_returns() -> None:
    prices = make_prices(periods=90)
    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=BacktestConfig(
            estimation_window=20,
            transaction_cost_bps=10.0,
        ),
    )

    expected = (1.0 + result.net_returns).cumprod()
    np.testing.assert_allclose(
        result.wealth.to_numpy(),
        expected.to_numpy(),
        atol=1e-12,
    )


def test_total_transaction_cost_matches_cost_amount_series() -> None:
    prices = make_prices(periods=90)
    result = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=BacktestConfig(
            estimation_window=20,
            transaction_cost_bps=10.0,
        ),
    )

    assert result.total_transaction_cost == pytest.approx(
        result.transaction_cost_amounts.sum()
    )


def test_strategy_suite_uses_identical_oos_and_rebalance_dates() -> None:
    prices = make_prices(periods=75)
    config = BacktestConfig(
        estimation_window=20,
        transaction_cost_bps=0.0,
    )

    results = run_strategy_suite(
        prices,
        config=config,
    )

    assert set(results) == {
        "equal_weight",
        "gmv",
        "max_sharpe",
        "erc",
    }

    reference = results["equal_weight"]
    for result in results.values():
        assert result.net_returns.index.equals(
            reference.net_returns.index
        )
        assert result.rebalance_dates.equals(
            reference.rebalance_dates
        )


def test_invalid_strategy_is_rejected() -> None:
    prices = make_prices(periods=90)

    with pytest.raises(ValueError, match="strategy must be one of"):
        run_walk_forward_backtest(
            prices,
            strategy="invalid",  # type: ignore[arg-type]
            config=BacktestConfig(estimation_window=20),
        )


def test_erc_rejects_short_selling_constraints() -> None:
    prices = make_prices(periods=90)

    with pytest.raises(Exception, match="non-negative"):
        run_walk_forward_backtest(
            prices,
            strategy="erc",
            config=BacktestConfig(
                estimation_window=20,
                constraints=PortfolioConstraints(
                    lower_bound=-0.20,
                    upper_bound=1.20,
                ),
            ),
        )
