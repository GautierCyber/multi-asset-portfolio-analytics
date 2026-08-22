"""Walk-forward out-of-sample backtesting for portfolio strategies.

The engine uses close-to-close simple returns and rebalances at the close of
selected trading days. A target portfolio executed at the close of rebalance
date ``t`` is estimated strictly from returns whose timestamps are earlier than
``t``. The return ending at ``t`` is therefore never used to determine weights
executed at that same close. Target weights apply only from ``t`` to the next
trading date.

Between rebalances, portfolio weights drift with realised asset returns.
Transaction costs are charged at rebalance closes as a configurable cost per
unit of one-way turnover. No price or return observation is filled or
interpolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from .portfolio import (
    PortfolioConstraints,
    equal_risk_contribution_weights,
    equal_weight_weights,
    estimate_annualized_moments,
    global_minimum_variance_weights,
    maximum_sharpe_weights,
)
from .returns import (
    TRADING_DAYS_PER_YEAR,
    simple_returns_from_prices,
    validate_price_panel,
    validate_return_panel,
)


StrategyName = Literal[
    "equal_weight",
    "gmv",
    "max_sharpe",
    "erc",
]

SUPPORTED_STRATEGIES: tuple[StrategyName, ...] = (
    "equal_weight",
    "gmv",
    "max_sharpe",
    "erc",
)

WEIGHT_SUM_TOLERANCE: float = 1e-8
TURNOVER_TOLERANCE: float = 1e-12


class BacktestError(ValueError):
    """Raised when walk-forward backtesting inputs or state are invalid."""


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration shared by all strategies in a fair OOS comparison.

    ``estimation_window`` counts realised daily returns. Rebalancing is
    monthly at the final observed trading date of each calendar month.

    ``transaction_cost_bps`` is applied to one-way turnover. For an existing
    fully invested portfolio, one-way turnover is half the L1 distance between
    pre-trade and target weights. The initial allocation from cash uses the
    gross asset notional ``sum(abs(target_weights))`` and therefore equals
    100% for a conventional long-only fully invested portfolio.
    """

    estimation_window: int = 756
    transaction_cost_bps: float = 10.0
    risk_free_rate: float = 0.0
    periods_per_year: float = TRADING_DAYS_PER_YEAR
    constraints: PortfolioConstraints = field(
        default_factory=PortfolioConstraints
    )
    charge_initial_transaction_cost: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.estimation_window, int):
            raise TypeError("estimation_window must be an integer.")
        if self.estimation_window < 2:
            raise ValueError(
                "estimation_window must contain at least 2 returns."
            )

        transaction_cost_bps = _finite_real(
            self.transaction_cost_bps,
            name="transaction_cost_bps",
        )
        if not 0.0 <= transaction_cost_bps < 10_000.0:
            raise ValueError(
                "transaction_cost_bps must be between 0 and 10,000."
            )

        risk_free_rate = _finite_real(
            self.risk_free_rate,
            name="risk_free_rate",
        )
        if risk_free_rate <= -1.0:
            raise ValueError(
                "risk_free_rate must be greater than -100%."
            )

        periods_per_year = _finite_real(
            self.periods_per_year,
            name="periods_per_year",
        )
        if periods_per_year <= 0.0:
            raise ValueError(
                "periods_per_year must be strictly positive."
            )

        if not isinstance(self.constraints, PortfolioConstraints):
            raise TypeError(
                "constraints must be a PortfolioConstraints instance."
            )

        if not isinstance(self.charge_initial_transaction_cost, bool):
            raise TypeError(
                "charge_initial_transaction_cost must be a boolean."
            )


@dataclass(frozen=True)
class BacktestResult:
    """Complete audit trail for one walk-forward strategy backtest."""

    strategy: StrategyName
    config: BacktestConfig
    gross_returns: pd.Series
    net_returns: pd.Series
    wealth: pd.Series
    turnover: pd.Series
    transaction_cost_fractions: pd.Series
    transaction_cost_amounts: pd.Series
    beginning_weights: pd.DataFrame
    end_weights: pd.DataFrame
    target_weights: pd.DataFrame
    pretrade_weights: pd.DataFrame
    rebalance_log: pd.DataFrame

    @property
    def rebalance_dates(self) -> pd.DatetimeIndex:
        """Return the strategy's rebalance dates."""
        return pd.DatetimeIndex(self.target_weights.index)

    @property
    def total_transaction_cost(self) -> float:
        """Return cumulative transaction-cost amount in initial-wealth units."""
        return float(self.transaction_cost_amounts.sum())


def _finite_real(value: Real, *, name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_strategy(strategy: str) -> StrategyName:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            "strategy must be one of: "
            + ", ".join(SUPPORTED_STRATEGIES)
            + "."
        )
    return strategy  # type: ignore[return-value]


def _validate_return_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise BacktestError(
            "Return panel must use a pandas DatetimeIndex."
        )
    if index.hasnans:
        raise BacktestError("Return index contains NaT values.")
    if index.has_duplicates:
        raise BacktestError("Return index contains duplicate dates.")
    if not index.is_monotonic_increasing:
        raise BacktestError(
            "Return index must be sorted in ascending order."
        )
    return index


def monthly_rebalance_dates(
    return_index: pd.DatetimeIndex,
    *,
    estimation_window: int,
) -> pd.DatetimeIndex:
    """Return eligible month-end trading dates with a subsequent OOS period.

    The first rebalance is the first month-end for which at least
    ``estimation_window`` realised returns are available strictly before the
    rebalance close. A terminal date with no subsequent return observation is
    excluded so the engine never pays for a rebalance that cannot affect future
    performance.
    """
    index = _validate_return_index(return_index)

    if not isinstance(estimation_window, int):
        raise TypeError("estimation_window must be an integer.")
    if estimation_window < 2:
        raise ValueError(
            "estimation_window must contain at least 2 returns."
        )
    if len(index) <= estimation_window:
        raise BacktestError(
            "Return history is too short to form an estimation window "
            "and retain an out-of-sample observation."
        )

    earliest_eligible = index[estimation_window]
    eligible = index[index >= earliest_eligible]

    grouped = pd.Series(
        eligible,
        index=eligible,
    ).groupby(
        eligible.to_period("M"),
        sort=True,
    ).max()

    dates = pd.DatetimeIndex(grouped.to_numpy())
    dates = dates[dates < index[-1]]

    if len(dates) == 0:
        raise BacktestError(
            "No eligible monthly rebalance date has a subsequent return."
        )

    return dates


def _estimation_sample(
    returns: pd.DataFrame,
    *,
    rebalance_date: pd.Timestamp,
    estimation_window: int,
) -> pd.DataFrame:
    sample = returns.loc[returns.index < rebalance_date].tail(
        estimation_window
    )

    if len(sample) != estimation_window:
        raise BacktestError(
            "Could not construct the requested pre-rebalance estimation "
            f"window on {rebalance_date.date()}."
        )
    if sample.index[-1] >= rebalance_date:
        raise BacktestError(
            "Estimation sample must end strictly before the rebalance date."
        )

    return sample


def _target_weights(
    strategy: StrategyName,
    estimation_returns: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.Series:
    assets = estimation_returns.columns

    if strategy == "equal_weight":
        return equal_weight_weights(
            assets,
            constraints=config.constraints,
        )

    expected_returns, covariance = estimate_annualized_moments(
        estimation_returns,
        periods_per_year=config.periods_per_year,
    )

    if strategy == "gmv":
        return global_minimum_variance_weights(
            covariance,
            constraints=config.constraints,
        )

    if strategy == "max_sharpe":
        return maximum_sharpe_weights(
            expected_returns,
            covariance,
            risk_free_rate=config.risk_free_rate,
            periods_per_year=config.periods_per_year,
            constraints=config.constraints,
        )

    if strategy == "erc":
        return equal_risk_contribution_weights(
            covariance,
            constraints=config.constraints,
        )

    raise BacktestError(f"Unsupported strategy: {strategy}.")


def _validate_unconstrained_weights(
    weights: pd.Series,
    *,
    assets: pd.Index,
    context: str,
) -> pd.Series:
    if not isinstance(weights, pd.Series):
        raise BacktestError(f"{context} must be a pandas Series.")
    if not weights.index.equals(assets):
        raise BacktestError(
            f"{context} index must exactly match the asset universe."
        )

    numeric = pd.to_numeric(weights, errors="coerce")
    values = numeric.to_numpy(dtype=float, copy=False)
    if numeric.isna().any() or not np.isfinite(values).all():
        raise BacktestError(
            f"{context} contains missing or non-finite weights."
        )
    if not math.isclose(
        float(values.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=WEIGHT_SUM_TOLERANCE,
    ):
        raise BacktestError(f"{context} weights must sum to 1.")

    return numeric.astype(float)


def _portfolio_gross_return(
    weights: pd.Series,
    asset_returns: pd.Series,
) -> float:
    return float(
        weights.to_numpy(dtype=float, copy=False)
        @ asset_returns.to_numpy(dtype=float, copy=False)
    )


def _drift_weights(
    weights: pd.Series,
    asset_returns: pd.Series,
) -> tuple[float, pd.Series]:
    gross_return = _portfolio_gross_return(weights, asset_returns)
    denominator = 1.0 + gross_return

    if denominator <= 0.0:
        raise BacktestError(
            "Portfolio wealth reached zero or became negative; "
            "weight drift is undefined."
        )

    drifted = (
        weights
        * (1.0 + asset_returns)
        / denominator
    )
    drifted = _validate_unconstrained_weights(
        drifted,
        assets=weights.index,
        context="Drifted portfolio",
    )
    return gross_return, drifted


def _one_way_turnover(
    pretrade_weights: pd.Series,
    target_weights: pd.Series,
    *,
    initial_allocation: bool,
) -> float:
    if initial_allocation:
        turnover = float(target_weights.abs().sum())
    else:
        turnover = 0.5 * float(
            (target_weights - pretrade_weights).abs().sum()
        )

    if turnover < -TURNOVER_TOLERANCE:
        raise BacktestError("Calculated turnover cannot be negative.")

    return max(turnover, 0.0)


def _transaction_cost_fraction(
    turnover: float,
    *,
    transaction_cost_bps: float,
) -> float:
    cost_fraction = (
        turnover
        * transaction_cost_bps
        / 10_000.0
    )
    if cost_fraction < 0.0:
        raise BacktestError(
            "Transaction-cost fraction cannot be negative."
        )
    if cost_fraction >= 1.0:
        raise BacktestError(
            "Transaction costs would consume 100% or more of portfolio value."
        )
    return cost_fraction


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    *,
    strategy: StrategyName,
    config: BacktestConfig = BacktestConfig(),
    start_date: str | date | pd.Timestamp | None = None,
) -> BacktestResult:
    """Run one monthly close-to-close walk-forward OOS backtest.

    Timing on an ordinary rebalance date ``t`` is:

    1. target weights for the close of ``t`` are estimated from exactly the
       latest ``estimation_window`` returns with timestamps strictly before
       ``t``;
    2. weights held from ``t-1`` realise the asset return ending at ``t``;
    3. those existing weights drift with the realised return;
    4. turnover and transaction cost are charged at the close of ``t`` when
       moving from pre-trade weights to the pre-computed target weights;
    5. target weights are applied only to the next return period.

    The first rebalance begins from cash, so its gross market return is zero.
    If configured, the initial allocation cost is charged immediately. This
    convention makes the rebalance decision implementable without using the
    closing return of ``t`` to determine weights executed at that same close.

    When ``start_date`` is supplied, the engine starts from cash at the first
    otherwise-eligible monthly rebalance date on or after that date. Estimation
    history before ``start_date`` remains available. This is useful for fair
    robustness comparisons between different estimation-window lengths.
    """
    strategy = _validate_strategy(strategy)
    if not isinstance(config, BacktestConfig):
        raise TypeError("config must be a BacktestConfig instance.")

    numeric_prices = validate_price_panel(
        prices,
        min_observations=config.estimation_window + 2,
    )
    returns = validate_return_panel(
        simple_returns_from_prices(numeric_prices),
        min_observations=config.estimation_window + 1,
    )

    rebalance_dates = monthly_rebalance_dates(
        returns.index,
        estimation_window=config.estimation_window,
    )

    if start_date is not None:
        try:
            requested_start = pd.Timestamp(start_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "start_date must be convertible to a pandas Timestamp."
            ) from exc

        if pd.isna(requested_start):
            raise ValueError("start_date must not be NaT.")
        if requested_start.tz is not None:
            requested_start = requested_start.tz_localize(None)
        requested_start = requested_start.normalize()

        rebalance_dates = rebalance_dates[
            rebalance_dates >= requested_start
        ]
        if len(rebalance_dates) == 0:
            raise BacktestError(
                "No eligible rebalance date exists on or after start_date."
            )

    rebalance_set = set(rebalance_dates)
    first_rebalance = rebalance_dates[0]
    performance_index = returns.index[
        returns.index >= first_rebalance
    ]

    assets = returns.columns
    zero_weights = pd.Series(0.0, index=assets, dtype=float)

    gross_returns = pd.Series(
        0.0,
        index=performance_index,
        name=f"{strategy}_gross_return",
        dtype=float,
    )
    net_returns = pd.Series(
        0.0,
        index=performance_index,
        name=f"{strategy}_net_return",
        dtype=float,
    )
    turnover = pd.Series(
        0.0,
        index=performance_index,
        name=f"{strategy}_turnover",
        dtype=float,
    )
    transaction_cost_fractions = pd.Series(
        0.0,
        index=performance_index,
        name=f"{strategy}_transaction_cost_fraction",
        dtype=float,
    )
    transaction_cost_amounts = pd.Series(
        0.0,
        index=performance_index,
        name=f"{strategy}_transaction_cost_amount",
        dtype=float,
    )
    wealth = pd.Series(
        np.nan,
        index=performance_index,
        name=f"{strategy}_wealth",
        dtype=float,
    )

    beginning_weights = pd.DataFrame(
        np.nan,
        index=performance_index,
        columns=assets,
        dtype=float,
    )
    end_weights = pd.DataFrame(
        np.nan,
        index=performance_index,
        columns=assets,
        dtype=float,
    )

    target_rows: dict[pd.Timestamp, pd.Series] = {}
    pretrade_rows: dict[pd.Timestamp, pd.Series] = {}
    log_rows: list[dict[str, object]] = []

    current_weights: pd.Series | None = None
    previous_wealth = 1.0

    for date in performance_index:
        asset_returns = returns.loc[date]

        if current_weights is None:
            if date != first_rebalance:
                raise BacktestError(
                    "Internal state error before the first rebalance."
                )

            beginning = zero_weights.copy()
            gross_return = 0.0
            pretrade = zero_weights.copy()
        else:
            beginning = current_weights.copy()
            gross_return, pretrade = _drift_weights(
                beginning,
                asset_returns,
            )

        beginning_weights.loc[date] = beginning
        gross_returns.loc[date] = gross_return

        cost_fraction = 0.0
        cost_amount = 0.0
        day_turnover = 0.0

        if date in rebalance_set:
            estimation_returns = _estimation_sample(
                returns,
                rebalance_date=date,
                estimation_window=config.estimation_window,
            )
            target = _target_weights(
                strategy,
                estimation_returns,
                config=config,
            )

            initial_allocation = current_weights is None
            day_turnover = _one_way_turnover(
                pretrade,
                target,
                initial_allocation=initial_allocation,
            )

            if (
                initial_allocation
                and not config.charge_initial_transaction_cost
            ):
                cost_fraction = 0.0
            else:
                cost_fraction = _transaction_cost_fraction(
                    day_turnover,
                    transaction_cost_bps=config.transaction_cost_bps,
                )

            gross_wealth = previous_wealth * (1.0 + gross_return)
            cost_amount = gross_wealth * cost_fraction
            current_wealth = gross_wealth - cost_amount

            if current_wealth <= 0.0:
                raise BacktestError(
                    "Portfolio wealth became non-positive after transaction costs."
                )

            net_return = current_wealth / previous_wealth - 1.0
            current_weights = target.copy()

            target_rows[date] = target.copy()
            pretrade_rows[date] = pretrade.copy()
            log_rows.append(
                {
                    "date": date,
                    "estimation_start": estimation_returns.index[0],
                    "estimation_end": estimation_returns.index[-1],
                    "estimation_observations": len(estimation_returns),
                    "turnover": day_turnover,
                    "transaction_cost_fraction": cost_fraction,
                    "transaction_cost_amount": cost_amount,
                    "wealth_before_trade": gross_wealth,
                    "wealth_after_trade": current_wealth,
                }
            )
        else:
            if current_weights is None:
                raise BacktestError(
                    "Internal state error: no portfolio exists outside rebalance."
                )
            current_wealth = previous_wealth * (1.0 + gross_return)
            if current_wealth <= 0.0:
                raise BacktestError("Portfolio wealth became non-positive.")
            net_return = gross_return
            current_weights = pretrade

        turnover.loc[date] = day_turnover
        transaction_cost_fractions.loc[date] = cost_fraction
        transaction_cost_amounts.loc[date] = cost_amount
        net_returns.loc[date] = net_return
        wealth.loc[date] = current_wealth
        end_weights.loc[date] = current_weights

        previous_wealth = current_wealth

    target_weights = pd.DataFrame.from_dict(
        target_rows,
        orient="index",
    ).reindex(columns=assets)
    target_weights.index = pd.DatetimeIndex(target_weights.index)
    target_weights.index.name = "Date"

    pretrade_weights = pd.DataFrame.from_dict(
        pretrade_rows,
        orient="index",
    ).reindex(columns=assets)
    pretrade_weights.index = pd.DatetimeIndex(pretrade_weights.index)
    pretrade_weights.index.name = "Date"

    rebalance_log = pd.DataFrame(log_rows).set_index("date")
    rebalance_log.index = pd.DatetimeIndex(rebalance_log.index)
    rebalance_log.index.name = "Date"

    for frame in (beginning_weights, end_weights):
        frame.index.name = "Date"
    for series in (
        gross_returns,
        net_returns,
        wealth,
        turnover,
        transaction_cost_fractions,
        transaction_cost_amounts,
    ):
        series.index.name = "Date"

    if not target_weights.index.equals(rebalance_dates):
        raise BacktestError(
            "Target-weight audit trail does not match rebalance schedule."
        )
    if net_returns.isna().any() or wealth.isna().any():
        raise BacktestError("Backtest produced missing performance values.")
    if not np.isfinite(net_returns.to_numpy()).all():
        raise BacktestError("Backtest produced non-finite net returns.")
    if not np.isfinite(wealth.to_numpy()).all():
        raise BacktestError("Backtest produced non-finite wealth values.")
    if (wealth <= 0.0).any():
        raise BacktestError("Backtest wealth must remain strictly positive.")

    return BacktestResult(
        strategy=strategy,
        config=config,
        gross_returns=gross_returns,
        net_returns=net_returns,
        wealth=wealth,
        turnover=turnover,
        transaction_cost_fractions=transaction_cost_fractions,
        transaction_cost_amounts=transaction_cost_amounts,
        beginning_weights=beginning_weights,
        end_weights=end_weights,
        target_weights=target_weights,
        pretrade_weights=pretrade_weights,
        rebalance_log=rebalance_log,
    )


def run_strategy_suite(
    prices: pd.DataFrame,
    *,
    config: BacktestConfig = BacktestConfig(),
    strategies: tuple[StrategyName, ...] = SUPPORTED_STRATEGIES,
    start_date: str | date | pd.Timestamp | None = None,
) -> dict[StrategyName, BacktestResult]:
    """Run multiple strategies under one identical OOS configuration.

    ``start_date`` is forwarded unchanged to every strategy so each portfolio
    is initialised from cash on the same eligible rebalance date.
    """
    if not isinstance(strategies, tuple):
        raise TypeError("strategies must be a tuple.")
    if len(strategies) == 0:
        raise ValueError("strategies must contain at least one strategy.")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates.")

    validated = tuple(_validate_strategy(strategy) for strategy in strategies)
    results: dict[StrategyName, BacktestResult] = {}

    reference_index: pd.DatetimeIndex | None = None
    reference_rebalances: pd.DatetimeIndex | None = None

    for strategy in validated:
        result = run_walk_forward_backtest(
            prices,
            strategy=strategy,
            config=config,
            start_date=start_date,
        )

        if reference_index is None:
            reference_index = result.net_returns.index
            reference_rebalances = result.rebalance_dates
        else:
            if not result.net_returns.index.equals(reference_index):
                raise BacktestError(
                    "Strategy results do not share the same OOS return dates."
                )
            if not result.rebalance_dates.equals(reference_rebalances):
                raise BacktestError(
                    "Strategy results do not share the same rebalance dates."
                )

        results[strategy] = result

    return results
