import numpy as np
import pandas as pd
import pytest

from multi_asset_portfolio.backtest import (
    BacktestConfig,
    run_strategy_suite,
    run_walk_forward_backtest,
)
from multi_asset_portfolio.backtest_analytics import (
    BacktestAnalyticsError,
    allocation_stability_summary,
    calendar_year_returns,
    compare_backtests,
    drawdown_episodes,
    drawdown_series,
    gross_wealth,
    rolling_backtest_metrics,
    summarize_backtest,
    transaction_cost_summary,
    validate_backtest_result,
)


def make_prices(periods: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    index = pd.bdate_range("2023-01-03", periods=periods)
    simulated = rng.normal(
        loc=np.array([0.00045, 0.00020, 0.00010]),
        scale=np.array([0.0100, 0.0060, 0.0040]),
        size=(periods - 1, 3),
    )
    simulated = np.clip(simulated, -0.15, 0.15)
    growth = np.vstack(
        [
            np.ones(3),
            np.cumprod(1.0 + simulated, axis=0),
        ]
    )
    return pd.DataFrame(
        100.0 * growth,
        index=index,
        columns=["A", "B", "C"],
    )


def make_result(transaction_cost_bps: float = 10.0):
    return run_walk_forward_backtest(
        make_prices(),
        strategy="equal_weight",
        config=BacktestConfig(
            estimation_window=60,
            transaction_cost_bps=transaction_cost_bps,
        ),
    )


def test_validate_backtest_result_accepts_engine_output() -> None:
    result = make_result()
    assert validate_backtest_result(result) is result


def test_gross_wealth_matches_compounded_gross_returns() -> None:
    result = make_result()
    actual = gross_wealth(result)
    expected = (1.0 + result.gross_returns).cumprod()
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())


def test_drawdown_is_never_positive_and_matches_summary_minimum() -> None:
    result = make_result()
    drawdown = drawdown_series(result)
    assert (drawdown <= 1e-12).all()
    summary = summarize_backtest(result)
    assert summary["maximum_drawdown"] == pytest.approx(float(drawdown.min()))


def test_drawdown_episodes_match_worst_drawdown_when_episodes_exist() -> None:
    result = make_result()
    drawdown = drawdown_series(result)
    episodes = drawdown_episodes(result)

    expected_columns = {
        "drawdown_start",
        "peak_date",
        "trough_date",
        "recovery_date",
        "max_drawdown",
        "days_to_trough",
        "days_underwater",
        "recovered",
    }
    assert set(episodes.columns) == expected_columns
    if not episodes.empty:
        assert episodes["max_drawdown"].min() == pytest.approx(
            float(drawdown.min())
        )


def test_calendar_year_returns_compound_daily_returns() -> None:
    result = make_result()
    annual = calendar_year_returns(result)
    first_year = annual.index[0]
    mask = result.net_returns.index.year == first_year
    expected = float((1.0 + result.net_returns.loc[mask]).prod() - 1.0)
    assert annual.loc[first_year, "net_return"] == pytest.approx(expected)
    assert annual.loc[first_year, "observations"] == int(mask.sum())


def test_rolling_metrics_use_exact_window_and_expected_columns() -> None:
    result = make_result()
    metrics = rolling_backtest_metrics(result, window=40)

    assert metrics.index[0] == result.net_returns.index[39]
    assert list(metrics.columns) == [
        "annualized_compound_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    ]
    assert len(metrics) == len(result.net_returns) - 40 + 1


def test_transaction_cost_summary_reconciles_terminal_wealth() -> None:
    result = make_result(transaction_cost_bps=25.0)
    summary = transaction_cost_summary(result)

    assert summary["total_turnover"] == pytest.approx(
        float(result.turnover.loc[result.rebalance_dates].sum())
    )
    assert summary["net_final_wealth"] == pytest.approx(
        float(result.wealth.iloc[-1])
    )
    assert summary["gross_final_wealth"] >= summary["net_final_wealth"]
    assert summary["terminal_wealth_cost_drag"] >= 0.0


def test_equal_weight_allocation_stability_is_exact() -> None:
    result = make_result(transaction_cost_bps=0.0)
    summary = allocation_stability_summary(result)

    assert summary["average_max_target_weight"] == pytest.approx(1.0 / 3.0)
    assert summary["maximum_target_weight"] == pytest.approx(1.0 / 3.0)
    assert summary["average_hhi"] == pytest.approx(1.0 / 3.0)
    assert summary["average_effective_number_of_assets"] == pytest.approx(3.0)


def test_summary_contains_performance_cost_and_concentration_metrics() -> None:
    result = make_result()
    summary = summarize_backtest(result)

    required = {
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "total_turnover",
        "transaction_cost_amount_paid",
        "terminal_wealth_cost_drag",
        "maximum_target_weight",
        "average_effective_number_of_assets",
    }
    assert required.issubset(summary.index)


def test_compare_backtests_requires_common_calendar_by_default() -> None:
    prices = make_prices()
    config = BacktestConfig(estimation_window=60, transaction_cost_bps=0.0)
    suite = run_strategy_suite(
        prices,
        config=config,
        strategies=("equal_weight", "gmv"),
    )
    comparison = compare_backtests(suite)
    assert list(comparison.index) == ["equal_weight", "gmv"]

    delayed = run_walk_forward_backtest(
        prices,
        strategy="equal_weight",
        config=config,
        start_date=suite["equal_weight"].rebalance_dates[1],
    )
    with pytest.raises(BacktestAnalyticsError, match="identical OOS"):
        compare_backtests(
            {
                "native": suite["equal_weight"],
                "delayed": delayed,
            }
        )
