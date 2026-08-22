import numpy as np
import pandas as pd
import pytest

from multi_asset_portfolio.robustness import (
    RobustnessError,
    RobustnessScenario,
    common_robustness_start_date,
    metric_pivot,
    run_robustness_analysis,
    standard_robustness_scenarios,
)


def make_prices(periods: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2024-01-02", periods=periods)
    simulated = rng.normal(
        loc=np.array([0.0004, 0.0002, 0.0001]),
        scale=np.array([0.010, 0.006, 0.004]),
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


def test_standard_scenarios_are_unique_and_one_factor_at_a_time() -> None:
    scenarios = standard_robustness_scenarios()
    names = [scenario.name for scenario in scenarios]

    assert len(scenarios) == 7
    assert len(set(names)) == 7
    base = scenarios[0]
    assert base.name == "base"

    for scenario in scenarios[1:]:
        changes = sum(
            [
                scenario.estimation_window != base.estimation_window,
                scenario.transaction_cost_bps != base.transaction_cost_bps,
                scenario.upper_bound != base.upper_bound,
            ]
        )
        assert changes == 1


def test_common_start_is_latest_native_first_rebalance() -> None:
    prices = make_prices()
    scenarios = (
        RobustnessScenario("short", 20, 0.0, 0.60),
        RobustnessScenario("long", 40, 0.0, 0.60),
    )

    common_start = common_robustness_start_date(prices, scenarios)

    result = run_robustness_analysis(
        prices,
        scenarios=scenarios,
        strategies=("equal_weight",),
    )
    assert result.common_start_date == common_start


def test_run_robustness_uses_identical_cash_initialised_oos_calendar() -> None:
    prices = make_prices()
    scenarios = (
        RobustnessScenario("window_20", 20, 10.0, 0.60),
        RobustnessScenario("window_40", 40, 10.0, 0.60),
    )

    result = run_robustness_analysis(
        prices,
        scenarios=scenarios,
        strategies=("equal_weight", "gmv"),
    )

    reference_index = None
    for suite in result.backtests.values():
        for backtest in suite.values():
            if reference_index is None:
                reference_index = backtest.net_returns.index
            else:
                assert backtest.net_returns.index.equals(reference_index)
            assert backtest.turnover.iloc[0] == pytest.approx(1.0)
            assert backtest.gross_returns.iloc[0] == pytest.approx(0.0)

    assert reference_index is not None
    assert result.common_start_date == reference_index[0]
    assert result.common_end_date == reference_index[-1]


def test_robustness_summary_has_scenario_strategy_multiindex() -> None:
    prices = make_prices()
    scenarios = (
        RobustnessScenario("base", 30, 10.0, 0.60),
        RobustnessScenario("cost", 30, 25.0, 0.60),
    )
    result = run_robustness_analysis(
        prices,
        scenarios=scenarios,
        strategies=("equal_weight", "gmv"),
    )

    assert result.summary.index.names == ["scenario", "strategy"]
    assert len(result.summary) == 4
    assert "cagr" in result.summary.columns
    assert "sharpe_ratio" in result.summary.columns
    assert "upper_bound" in result.summary.columns


def test_metric_pivot_returns_scenario_by_strategy_table() -> None:
    result = run_robustness_analysis(
        make_prices(),
        scenarios=(
            RobustnessScenario("base", 30, 10.0, 0.60),
            RobustnessScenario("cost", 30, 25.0, 0.60),
        ),
        strategies=("equal_weight", "gmv"),
    )

    pivot = metric_pivot(result, metric="sharpe_ratio")
    assert set(pivot.index) == {"base", "cost"}
    assert set(pivot.columns) == {"equal_weight", "gmv"}


def test_duplicate_scenario_names_are_rejected() -> None:
    scenarios = (
        RobustnessScenario("same", 20, 0.0, 0.60),
        RobustnessScenario("same", 40, 0.0, 0.60),
    )

    with pytest.raises(ValueError, match="unique"):
        run_robustness_analysis(
            make_prices(),
            scenarios=scenarios,
            strategies=("equal_weight",),
        )


def test_infeasible_upper_bound_is_rejected_before_backtesting() -> None:
    scenario = RobustnessScenario(
        "infeasible",
        20,
        0.0,
        0.30,
    )

    with pytest.raises(RobustnessError, match="infeasible upper"):
        run_robustness_analysis(
            make_prices(),
            scenarios=(scenario,),
            strategies=("equal_weight",),
        )
