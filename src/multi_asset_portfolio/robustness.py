"""One-factor-at-a-time robustness analysis for portfolio backtests.

The standard scenario set varies estimation-window length, transaction costs
and the per-asset upper weight bound around a single base configuration. All
scenarios are re-initialised from cash on the same rebalance date so performance
comparisons are not contaminated by different OOS start dates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable

import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestResult,
    StrategyName,
    SUPPORTED_STRATEGIES,
    monthly_rebalance_dates,
    run_strategy_suite,
)
from .backtest_analytics import compare_backtests
from .portfolio import PortfolioConstraints
from .returns import TRADING_DAYS_PER_YEAR, simple_returns_from_prices, validate_price_panel


class RobustnessError(ValueError):
    """Raised when robustness scenarios cannot be compared fairly."""


@dataclass(frozen=True)
class RobustnessScenario:
    """A single walk-forward robustness configuration."""

    name: str
    estimation_window: int
    transaction_cost_bps: float
    upper_bound: float
    lower_bound: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Scenario name must be a non-empty string.")
        if not isinstance(self.estimation_window, int):
            raise TypeError("estimation_window must be an integer.")
        if self.estimation_window < 2:
            raise ValueError("estimation_window must be at least 2.")

        transaction_cost = _finite_real(
            self.transaction_cost_bps,
            name="transaction_cost_bps",
        )
        if not 0.0 <= transaction_cost < 10_000.0:
            raise ValueError(
                "transaction_cost_bps must be between 0 and 10,000."
            )

        lower = _finite_real(self.lower_bound, name="lower_bound")
        upper = _finite_real(self.upper_bound, name="upper_bound")
        if lower < 0.0:
            raise ValueError(
                "Standard robustness analysis requires non-negative weights."
            )
        if upper <= lower:
            raise ValueError("upper_bound must be greater than lower_bound.")


@dataclass(frozen=True)
class RobustnessResult:
    """Complete scenario outputs and their common-period summary."""

    scenarios: tuple[RobustnessScenario, ...]
    common_start_date: pd.Timestamp
    common_end_date: pd.Timestamp
    summary: pd.DataFrame
    backtests: dict[str, dict[StrategyName, BacktestResult]]


def _finite_real(value: Real, *, name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def standard_robustness_scenarios() -> tuple[RobustnessScenario, ...]:
    """Return the seven canonical one-factor sensitivity scenarios.

    The base case is 3-year estimation / 10 bps / 40% cap. The alternative
    scenarios change exactly one of those three design choices at a time.
    """
    return (
        RobustnessScenario(
            name="base",
            estimation_window=756,
            transaction_cost_bps=10.0,
            upper_bound=0.40,
        ),
        RobustnessScenario(
            name="window_2y",
            estimation_window=504,
            transaction_cost_bps=10.0,
            upper_bound=0.40,
        ),
        RobustnessScenario(
            name="window_4y",
            estimation_window=1008,
            transaction_cost_bps=10.0,
            upper_bound=0.40,
        ),
        RobustnessScenario(
            name="cost_0bps",
            estimation_window=756,
            transaction_cost_bps=0.0,
            upper_bound=0.40,
        ),
        RobustnessScenario(
            name="cost_25bps",
            estimation_window=756,
            transaction_cost_bps=25.0,
            upper_bound=0.40,
        ),
        RobustnessScenario(
            name="cap_30pct",
            estimation_window=756,
            transaction_cost_bps=10.0,
            upper_bound=0.30,
        ),
        RobustnessScenario(
            name="cap_50pct",
            estimation_window=756,
            transaction_cost_bps=10.0,
            upper_bound=0.50,
        ),
    )


def _validate_scenarios(
    scenarios: Iterable[RobustnessScenario],
    *,
    number_of_assets: int,
) -> tuple[RobustnessScenario, ...]:
    scenario_tuple = tuple(scenarios)
    if len(scenario_tuple) == 0:
        raise ValueError("scenarios must contain at least one scenario.")
    if not all(isinstance(item, RobustnessScenario) for item in scenario_tuple):
        raise TypeError(
            "Every scenario must be a RobustnessScenario instance."
        )

    names = [scenario.name for scenario in scenario_tuple]
    if len(set(names)) != len(names):
        raise ValueError("Scenario names must be unique.")

    for scenario in scenario_tuple:
        constraints = PortfolioConstraints(
            lower_bound=scenario.lower_bound,
            upper_bound=scenario.upper_bound,
        )
        lower = constraints.lower_bound
        upper = constraints.upper_bound
        if number_of_assets * lower > 1.0 + 1e-12:
            raise RobustnessError(
                f"Scenario {scenario.name!r} has infeasible lower bounds."
            )
        if number_of_assets * upper < 1.0 - 1e-12:
            raise RobustnessError(
                f"Scenario {scenario.name!r} has infeasible upper bounds."
            )

    return scenario_tuple


def _validate_strategies(
    strategies: tuple[StrategyName, ...],
) -> tuple[StrategyName, ...]:
    if not isinstance(strategies, tuple):
        raise TypeError("strategies must be a tuple.")
    if len(strategies) == 0:
        raise ValueError("strategies must contain at least one strategy.")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates.")
    unsupported = [
        strategy
        for strategy in strategies
        if strategy not in SUPPORTED_STRATEGIES
    ]
    if unsupported:
        raise ValueError(
            "Unsupported strategies: " + ", ".join(unsupported) + "."
        )
    return strategies


def common_robustness_start_date(
    prices: pd.DataFrame,
    scenarios: Iterable[RobustnessScenario],
) -> pd.Timestamp:
    """Return the latest native first-rebalance date across scenarios."""
    numeric_prices = validate_price_panel(prices, min_observations=3)
    scenario_tuple = _validate_scenarios(
        scenarios,
        number_of_assets=numeric_prices.shape[1],
    )
    returns = simple_returns_from_prices(numeric_prices)

    first_dates = []
    for scenario in scenario_tuple:
        dates = monthly_rebalance_dates(
            returns.index,
            estimation_window=scenario.estimation_window,
        )
        first_dates.append(dates[0])

    return max(first_dates)


def run_robustness_analysis(
    prices: pd.DataFrame,
    *,
    scenarios: Iterable[RobustnessScenario] | None = None,
    strategies: tuple[StrategyName, ...] = SUPPORTED_STRATEGIES,
    risk_free_rate: Real = 0.0,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
    charge_initial_transaction_cost: bool = True,
    confidence_level: Real = 0.95,
) -> RobustnessResult:
    """Run all scenarios from one common cash-initialised OOS start date."""
    numeric_prices = validate_price_panel(prices, min_observations=3)
    scenario_tuple = _validate_scenarios(
        standard_robustness_scenarios() if scenarios is None else scenarios,
        number_of_assets=numeric_prices.shape[1],
    )
    strategy_tuple = _validate_strategies(strategies)

    risk_free = _finite_real(risk_free_rate, name="risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -100%.")
    periods = _finite_real(periods_per_year, name="periods_per_year")
    if periods <= 0.0:
        raise ValueError("periods_per_year must be strictly positive.")
    if not isinstance(charge_initial_transaction_cost, bool):
        raise TypeError(
            "charge_initial_transaction_cost must be a boolean."
        )

    common_start = common_robustness_start_date(
        numeric_prices,
        scenario_tuple,
    )

    all_backtests: dict[
        str,
        dict[StrategyName, BacktestResult],
    ] = {}
    summary_rows: list[pd.DataFrame] = []
    reference_index: pd.DatetimeIndex | None = None

    for scenario in scenario_tuple:
        config = BacktestConfig(
            estimation_window=scenario.estimation_window,
            transaction_cost_bps=scenario.transaction_cost_bps,
            risk_free_rate=risk_free,
            periods_per_year=periods,
            constraints=PortfolioConstraints(
                lower_bound=scenario.lower_bound,
                upper_bound=scenario.upper_bound,
            ),
            charge_initial_transaction_cost=charge_initial_transaction_cost,
        )
        suite = run_strategy_suite(
            numeric_prices,
            config=config,
            strategies=strategy_tuple,
            start_date=common_start,
        )

        scenario_index = next(iter(suite.values())).net_returns.index
        if reference_index is None:
            reference_index = scenario_index
        elif not scenario_index.equals(reference_index):
            raise RobustnessError(
                "Robustness scenarios do not share the same OOS calendar."
            )

        comparison = compare_backtests(
            suite,
            confidence_level=confidence_level,
            require_common_calendar=True,
        )
        comparison.insert(0, "scenario", scenario.name)
        comparison.insert(1, "estimation_window", scenario.estimation_window)
        comparison.insert(
            2,
            "transaction_cost_bps",
            scenario.transaction_cost_bps,
        )
        comparison.insert(3, "lower_bound", scenario.lower_bound)
        comparison.insert(4, "upper_bound", scenario.upper_bound)
        comparison = comparison.reset_index().set_index(
            ["scenario", "strategy"]
        )
        summary_rows.append(comparison)
        all_backtests[scenario.name] = suite

    if reference_index is None:
        raise RobustnessError("Robustness analysis produced no results.")

    summary = pd.concat(summary_rows, axis=0).sort_index()
    return RobustnessResult(
        scenarios=scenario_tuple,
        common_start_date=reference_index[0],
        common_end_date=reference_index[-1],
        summary=summary,
        backtests=all_backtests,
    )


def metric_pivot(
    robustness_result: RobustnessResult,
    *,
    metric: str,
) -> pd.DataFrame:
    """Pivot one summary metric to scenario-by-strategy form."""
    if not isinstance(robustness_result, RobustnessResult):
        raise TypeError(
            "robustness_result must be a RobustnessResult instance."
        )
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string.")
    if metric not in robustness_result.summary.columns:
        raise KeyError(f"Unknown robustness metric: {metric}.")

    return robustness_result.summary[metric].unstack("strategy")
