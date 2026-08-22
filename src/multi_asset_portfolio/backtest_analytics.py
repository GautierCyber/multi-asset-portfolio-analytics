"""Performance, drawdown, cost and allocation analytics for OOS backtests.

The functions in this module operate on :class:`BacktestResult` objects and
preserve the backtest engine's net-of-cost and gross-of-cost distinction.
Transaction-cost amounts are reported separately from compounded terminal
wealth drag because those two quantities are not economically identical: a
cost paid earlier also loses its subsequent compounding opportunity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd

from .analytics import (
    annual_rate_to_period_rate,
    annualized_volatility,
    historical_cvar,
    historical_var,
    sharpe_ratio,
    sortino_ratio,
)
from .backtest import BacktestResult
from .returns import CALENDAR_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR


ANALYTICS_TOLERANCE: float = 1e-10


class BacktestAnalyticsError(ValueError):
    """Raised when a backtest result is inconsistent or analytically invalid."""


def _finite_real(value: Real, *, name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _validate_confidence_level(confidence_level: Real) -> float:
    confidence = _finite_real(
        confidence_level,
        name="confidence_level",
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )
    return confidence


def _validate_window(window: int, *, observations: int) -> int:
    if not isinstance(window, int):
        raise TypeError("window must be an integer.")
    if window < 2:
        raise ValueError("window must contain at least 2 observations.")
    if window > observations:
        raise BacktestAnalyticsError(
            "window cannot exceed the number of backtest observations."
        )
    return window


def validate_backtest_result(result: BacktestResult) -> BacktestResult:
    """Validate the internal alignment and accounting identity of a result."""
    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult instance.")

    index = result.net_returns.index
    if not isinstance(index, pd.DatetimeIndex):
        raise BacktestAnalyticsError(
            "Backtest returns must use a DatetimeIndex."
        )
    if len(index) < 2:
        raise BacktestAnalyticsError(
            "At least two OOS observations are required for analytics."
        )
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise BacktestAnalyticsError(
            "Backtest return dates must be unique, valid and sorted."
        )

    aligned_series = (
        result.gross_returns,
        result.wealth,
        result.turnover,
        result.transaction_cost_fractions,
        result.transaction_cost_amounts,
    )
    for series in aligned_series:
        if not isinstance(series, pd.Series) or not series.index.equals(index):
            raise BacktestAnalyticsError(
                "Backtest performance series must share the same OOS index."
            )
        values = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        if not np.isfinite(values).all():
            raise BacktestAnalyticsError(
                "Backtest performance series contain non-finite values."
            )

    net_values = pd.to_numeric(result.net_returns, errors="coerce").to_numpy(
        dtype=float,
        copy=False,
    )
    if not np.isfinite(net_values).all():
        raise BacktestAnalyticsError(
            "Backtest net returns contain non-finite values."
        )
    if (net_values <= -1.0).any():
        raise BacktestAnalyticsError(
            "Backtest net returns must remain strictly above -100%."
        )
    if (result.wealth <= 0.0).any():
        raise BacktestAnalyticsError(
            "Backtest wealth must remain strictly positive."
        )
    if (result.turnover < -ANALYTICS_TOLERANCE).any():
        raise BacktestAnalyticsError("Turnover cannot be negative.")
    if (result.transaction_cost_fractions < -ANALYTICS_TOLERANCE).any():
        raise BacktestAnalyticsError(
            "Transaction-cost fractions cannot be negative."
        )
    if (result.transaction_cost_amounts < -ANALYTICS_TOLERANCE).any():
        raise BacktestAnalyticsError(
            "Transaction-cost amounts cannot be negative."
        )

    expected_wealth = (1.0 + result.net_returns).cumprod()
    if not np.allclose(
        expected_wealth.to_numpy(dtype=float, copy=False),
        result.wealth.to_numpy(dtype=float, copy=False),
        rtol=0.0,
        atol=ANALYTICS_TOLERANCE,
    ):
        raise BacktestAnalyticsError(
            "Backtest wealth does not match compounded net returns."
        )

    if not result.target_weights.index.equals(result.rebalance_dates):
        raise BacktestAnalyticsError(
            "Target-weight dates do not match rebalance dates."
        )
    if not result.pretrade_weights.index.equals(result.rebalance_dates):
        raise BacktestAnalyticsError(
            "Pre-trade weight dates do not match rebalance dates."
        )

    return result


def _calendar_years(index: pd.DatetimeIndex) -> float:
    elapsed_days = (index[-1] - index[0]).total_seconds() / 86_400.0
    years = elapsed_days / CALENDAR_DAYS_PER_YEAR
    if years <= 0.0:
        raise BacktestAnalyticsError(
            "Backtest calendar span must be strictly positive."
        )
    return years


def _cagr_from_final_wealth(final_wealth: float, years: float) -> float:
    if final_wealth <= 0.0:
        raise BacktestAnalyticsError(
            "Final wealth must be strictly positive for CAGR."
        )
    return final_wealth ** (1.0 / years) - 1.0


def gross_wealth(result: BacktestResult) -> pd.Series:
    """Return the gross-of-transaction-cost compounded wealth path."""
    result = validate_backtest_result(result)
    wealth = (1.0 + result.gross_returns).cumprod()
    wealth.name = f"{result.strategy}_gross_wealth"
    return wealth


def drawdown_series(result: BacktestResult) -> pd.Series:
    """Return net drawdown relative to the running peak and initial capital."""
    result = validate_backtest_result(result)
    running_peak = result.wealth.cummax().clip(lower=1.0)
    drawdown = result.wealth / running_peak - 1.0
    drawdown.name = f"{result.strategy}_drawdown"
    return drawdown


def drawdown_episodes(result: BacktestResult) -> pd.DataFrame:
    """Extract contiguous underwater episodes from the net wealth path.

    ``peak_date`` is ``NaT`` only when the backtest starts below initial
    capital because of the initial transaction cost; in that case the peak is
    the virtual initial wealth of 1.0 immediately before the first OOS row.
    """
    result = validate_backtest_result(result)
    drawdown = drawdown_series(result)
    underwater = drawdown < -ANALYTICS_TOLERANCE
    index = drawdown.index

    episodes: list[dict[str, object]] = []
    position = 0

    while position < len(index):
        if not bool(underwater.iloc[position]):
            position += 1
            continue

        start_position = position
        while position + 1 < len(index) and bool(
            underwater.iloc[position + 1]
        ):
            position += 1
        end_position = position

        segment = drawdown.iloc[start_position : end_position + 1]
        trough_date = segment.idxmin()
        trough_position = index.get_loc(trough_date)

        if start_position == 0:
            peak_date = pd.NaT
            days_to_trough = (
                trough_date - index[start_position]
            ).days
        else:
            peak_date = index[start_position - 1]
            days_to_trough = (trough_date - peak_date).days

        if end_position + 1 < len(index):
            recovery_date: pd.Timestamp | pd.NaTType = index[end_position + 1]
            recovered = True
            duration_start = (
                peak_date if not pd.isna(peak_date) else index[start_position]
            )
            days_underwater = (recovery_date - duration_start).days
        else:
            recovery_date = pd.NaT
            recovered = False
            duration_start = (
                peak_date if not pd.isna(peak_date) else index[start_position]
            )
            days_underwater = (index[-1] - duration_start).days

        episodes.append(
            {
                "drawdown_start": index[start_position],
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_date": recovery_date,
                "max_drawdown": float(segment.min()),
                "days_to_trough": int(days_to_trough),
                "days_underwater": int(days_underwater),
                "recovered": recovered,
            }
        )
        position += 1

    columns = [
        "drawdown_start",
        "peak_date",
        "trough_date",
        "recovery_date",
        "max_drawdown",
        "days_to_trough",
        "days_underwater",
        "recovered",
    ]
    return pd.DataFrame(episodes, columns=columns)


def calendar_year_returns(result: BacktestResult) -> pd.DataFrame:
    """Return gross and net compounded returns for each calendar year."""
    result = validate_backtest_result(result)
    frame = pd.DataFrame(
        {
            "gross_return": result.gross_returns,
            "net_return": result.net_returns,
        }
    )
    years = frame.index.year
    grouped = frame.groupby(years, sort=True)
    annual = grouped.agg(
        gross_return=("gross_return", lambda x: float((1.0 + x).prod() - 1.0)),
        net_return=("net_return", lambda x: float((1.0 + x).prod() - 1.0)),
    )
    annual["observations"] = grouped.size().astype(int)
    annual.index.name = "year"
    return annual


def rolling_backtest_metrics(
    result: BacktestResult,
    *,
    window: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: Real | None = None,
    periods_per_year: Real | None = None,
) -> pd.DataFrame:
    """Calculate fixed-observation rolling net performance metrics.

    The rolling compounded return is annualised as
    ``prod(1+r) ** (periods_per_year/window) - 1``. It is deliberately named
    ``annualized_compound_return`` rather than CAGR because the annualisation
    is observation-count based, not calendar-time based.
    """
    result = validate_backtest_result(result)
    window = _validate_window(window, observations=len(result.net_returns))

    periods = (
        result.config.periods_per_year
        if periods_per_year is None
        else _finite_real(periods_per_year, name="periods_per_year")
    )
    if periods <= 0.0:
        raise ValueError("periods_per_year must be strictly positive.")

    annual_rf = (
        result.config.risk_free_rate
        if risk_free_rate is None
        else _finite_real(risk_free_rate, name="risk_free_rate")
    )
    if annual_rf <= -1.0:
        raise ValueError("risk_free_rate must be greater than -100%.")
    period_rf = annual_rate_to_period_rate(
        annual_rf,
        periods_per_year=periods,
    )

    returns = result.net_returns
    rolling = returns.rolling(window=window, min_periods=window)

    compound = rolling.apply(
        lambda values: float(np.prod(1.0 + values)),
        raw=True,
    )
    annualized_compound = compound.pow(periods / window) - 1.0

    volatility = rolling.std(ddof=1) * math.sqrt(periods)
    excess_mean = (returns - period_rf).rolling(
        window=window,
        min_periods=window,
    ).mean()
    sharpe = excess_mean / rolling.std(ddof=1) * math.sqrt(periods)
    sharpe = sharpe.where(rolling.std(ddof=1) > 0.0)

    def _window_drawdown(values: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + values)
        peaks = np.maximum.accumulate(np.maximum(wealth, 1.0))
        return float(np.min(wealth / peaks - 1.0))

    max_drawdown = rolling.apply(_window_drawdown, raw=True)

    metrics = pd.DataFrame(
        {
            "annualized_compound_return": annualized_compound,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "maximum_drawdown": max_drawdown,
        }
    ).dropna(how="all")
    metrics.index.name = result.net_returns.index.name
    return metrics


def transaction_cost_summary(result: BacktestResult) -> pd.Series:
    """Summarise turnover, paid costs and compounded terminal cost drag."""
    result = validate_backtest_result(result)
    years = _calendar_years(result.net_returns.index)

    rebalances = result.rebalance_dates
    rebalance_turnover = result.turnover.loc[rebalances]
    recurring_turnover = rebalance_turnover.iloc[1:]

    gross_path = gross_wealth(result)
    gross_final = float(gross_path.iloc[-1])
    net_final = float(result.wealth.iloc[-1])
    gross_cagr = _cagr_from_final_wealth(gross_final, years)
    net_cagr = _cagr_from_final_wealth(net_final, years)

    summary = pd.Series(
        {
            "initial_turnover": float(rebalance_turnover.iloc[0]),
            "recurring_turnover": float(recurring_turnover.sum()),
            "total_turnover": float(rebalance_turnover.sum()),
            "average_recurring_rebalance_turnover": (
                float(recurring_turnover.mean())
                if len(recurring_turnover) > 0
                else float("nan")
            ),
            "annualized_recurring_turnover": float(
                recurring_turnover.sum() / years
            ),
            "transaction_cost_amount_paid": float(
                result.transaction_cost_amounts.sum()
            ),
            "gross_final_wealth": gross_final,
            "net_final_wealth": net_final,
            "terminal_wealth_cost_drag": gross_final - net_final,
            "gross_cagr": gross_cagr,
            "net_cagr": net_cagr,
            "cagr_cost_drag": gross_cagr - net_cagr,
        },
        name=result.strategy,
        dtype=float,
    )
    return summary


def allocation_stability_summary(result: BacktestResult) -> pd.Series:
    """Summarise concentration, effective breadth and bound activity."""
    result = validate_backtest_result(result)
    weights = result.target_weights
    if weights.empty:
        raise BacktestAnalyticsError("Target-weight history is empty.")

    hhi = weights.pow(2).sum(axis=1)
    effective_assets = 1.0 / hhi
    max_weight = weights.max(axis=1)
    lower = result.config.constraints.lower_bound
    upper = result.config.constraints.upper_bound

    lower_hits = np.isclose(
        weights.to_numpy(dtype=float, copy=False),
        lower,
        rtol=0.0,
        atol=1e-8,
    )
    upper_hits = np.isclose(
        weights.to_numpy(dtype=float, copy=False),
        upper,
        rtol=0.0,
        atol=1e-8,
    )

    return pd.Series(
        {
            "average_max_target_weight": float(max_weight.mean()),
            "maximum_target_weight": float(max_weight.max()),
            "average_hhi": float(hhi.mean()),
            "average_effective_number_of_assets": float(
                effective_assets.mean()
            ),
            "mean_asset_weight_std": float(weights.std(ddof=1).mean()),
            "lower_bound_hit_fraction": float(lower_hits.mean()),
            "upper_bound_hit_fraction": float(upper_hits.mean()),
        },
        name=result.strategy,
        dtype=float,
    )


def summarize_backtest(
    result: BacktestResult,
    *,
    confidence_level: Real = 0.95,
) -> pd.Series:
    """Build a comprehensive one-row summary for one OOS strategy."""
    result = validate_backtest_result(result)
    confidence = _validate_confidence_level(confidence_level)
    years = _calendar_years(result.net_returns.index)

    net_frame = result.net_returns.to_frame(name=result.strategy)
    final_wealth = float(result.wealth.iloc[-1])
    cagr = _cagr_from_final_wealth(final_wealth, years)
    drawdown = drawdown_series(result)
    cost = transaction_cost_summary(result)
    allocation = allocation_stability_summary(result)

    volatility = float(
        annualized_volatility(
            net_frame,
            periods_per_year=result.config.periods_per_year,
        ).iloc[0]
    )
    sharpe = float(
        sharpe_ratio(
            net_frame,
            risk_free_rate=result.config.risk_free_rate,
            periods_per_year=result.config.periods_per_year,
        ).iloc[0]
    )
    sortino = float(
        sortino_ratio(
            net_frame,
            minimum_acceptable_return=0.0,
            periods_per_year=result.config.periods_per_year,
        ).iloc[0]
    )
    var = float(
        historical_var(
            net_frame,
            confidence_level=confidence,
        ).iloc[0]
    )
    cvar = float(
        historical_cvar(
            net_frame,
            confidence_level=confidence,
        ).iloc[0]
    )

    summary = pd.Series(
        {
            "start_date": result.net_returns.index[0],
            "end_date": result.net_returns.index[-1],
            "observations": len(result.net_returns),
            "rebalances": len(result.rebalance_dates),
            "final_wealth": final_wealth,
            "total_return": final_wealth - 1.0,
            "cagr": cagr,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "maximum_drawdown": float(drawdown.min()),
            f"historical_var_{confidence:.2f}_daily": var,
            f"historical_cvar_{confidence:.2f}_daily": cvar,
            "total_turnover": cost["total_turnover"],
            "average_recurring_rebalance_turnover": cost[
                "average_recurring_rebalance_turnover"
            ],
            "annualized_recurring_turnover": cost[
                "annualized_recurring_turnover"
            ],
            "transaction_cost_amount_paid": cost[
                "transaction_cost_amount_paid"
            ],
            "terminal_wealth_cost_drag": cost[
                "terminal_wealth_cost_drag"
            ],
            "cagr_cost_drag": cost["cagr_cost_drag"],
            "average_max_target_weight": allocation[
                "average_max_target_weight"
            ],
            "maximum_target_weight": allocation[
                "maximum_target_weight"
            ],
            "average_effective_number_of_assets": allocation[
                "average_effective_number_of_assets"
            ],
            "lower_bound_hit_fraction": allocation[
                "lower_bound_hit_fraction"
            ],
            "upper_bound_hit_fraction": allocation[
                "upper_bound_hit_fraction"
            ],
        },
        name=result.strategy,
    )
    return summary


def compare_backtests(
    results: Mapping[str, BacktestResult],
    *,
    confidence_level: Real = 0.95,
    require_common_calendar: bool = True,
) -> pd.DataFrame:
    """Compare multiple backtests under a common OOS calendar by default."""
    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping of names to BacktestResult.")
    if len(results) == 0:
        raise ValueError("results must contain at least one backtest.")
    if not isinstance(require_common_calendar, bool):
        raise TypeError("require_common_calendar must be a boolean.")

    validated: dict[str, BacktestResult] = {}
    reference_index: pd.DatetimeIndex | None = None

    for name, result in results.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Backtest comparison names must be non-empty strings.")
        validated_result = validate_backtest_result(result)
        if require_common_calendar:
            if reference_index is None:
                reference_index = validated_result.net_returns.index
            elif not validated_result.net_returns.index.equals(reference_index):
                raise BacktestAnalyticsError(
                    "Backtests do not share an identical OOS return calendar."
                )
        validated[name] = validated_result

    rows = {
        name: summarize_backtest(
            result,
            confidence_level=confidence_level,
        )
        for name, result in validated.items()
    }
    comparison = pd.DataFrame.from_dict(rows, orient="index")
    comparison.index.name = "strategy"
    return comparison
