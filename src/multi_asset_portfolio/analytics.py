"""Performance, statistical and risk analytics for multi-asset returns.

Two annualisation concepts are kept separate deliberately:

* arithmetic annualisation of periodic returns for expected-return and risk
  estimators (252 trading periods per year by default);
* calendar-time CAGR for realised long-horizon performance from adjusted-close
  prices.

Historical VaR and Expected Shortfall use a positive-loss convention: 0.02
means a 2% loss. Negative values are retained if the empirical tail is itself
profitable; risk estimates are never clipped.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd

from .returns import (
    TRADING_DAYS_PER_YEAR,
    calendar_span_years,
    cumulative_wealth,
    simple_returns_from_prices,
    validate_price_panel,
    validate_return_panel,
)


__all__ = [
    "annual_rate_to_period_rate",
    "annualized_mean_return",
    "cagr_from_prices",
    "annualized_volatility",
    "annualized_downside_deviation",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown_from_prices",
    "drawdown_from_returns",
    "maximum_drawdown_from_prices",
    "historical_var",
    "historical_expected_shortfall",
    "historical_cvar",
    "annualized_covariance_matrix",
    "correlation_matrix",
    "summary_statistics",
]


def _validate_periods_per_year(
    periods_per_year: Real,
) -> float:
    """Validate an annualisation factor."""
    if not isinstance(
        periods_per_year,
        Real,
    ):
        raise TypeError(
            "periods_per_year must be numeric."
        )

    periods = float(
        periods_per_year
    )

    if not math.isfinite(
        periods
    ):
        raise ValueError(
            "periods_per_year must be finite."
        )

    if periods <= 0.0:
        raise ValueError(
            "periods_per_year must be strictly positive."
        )

    return periods


def _validate_annual_rate(
    annual_rate: Real,
    *,
    name: str,
) -> float:
    """Validate an effective annual rate."""
    if not isinstance(
        annual_rate,
        Real,
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    rate = float(
        annual_rate
    )

    if not math.isfinite(
        rate
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if rate <= -1.0:
        raise ValueError(
            f"{name} must be greater than -100%."
        )

    return rate


def _validate_confidence_level(
    confidence_level: Real,
) -> float:
    """Validate a VaR / Expected-Shortfall confidence level."""
    if not isinstance(
        confidence_level,
        Real,
    ):
        raise TypeError(
            "confidence_level must be numeric."
        )

    confidence = float(
        confidence_level
    )

    if not math.isfinite(
        confidence
    ):
        raise ValueError(
            "confidence_level must be finite."
        )

    if not (
        0.0
        < confidence
        < 1.0
    ):
        raise ValueError(
            "confidence_level must lie strictly "
            "between 0 and 1."
        )

    return confidence


def annual_rate_to_period_rate(
    annual_rate: Real,
    *,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> float:
    """Convert an effective annual rate to an equivalent periodic rate."""
    rate = _validate_annual_rate(
        annual_rate,
        name="annual_rate",
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    return (
        (1.0 + rate)
        ** (1.0 / periods)
        - 1.0
    )


def annualized_mean_return(
    returns: pd.DataFrame,
    *,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualise the arithmetic mean of periodic simple returns."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=1,
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    result = (
        numeric_returns.mean()
        * periods
    )

    result.name = (
        "annualized_mean_return"
    )

    return result


def cagr_from_prices(
    prices: pd.DataFrame,
) -> pd.Series:
    """Calculate realised CAGR using actual calendar time between prices."""
    numeric_prices = validate_price_panel(
        prices,
        min_observations=2,
    )

    years = calendar_span_years(
        numeric_prices.index
    )

    growth_factor = (
        numeric_prices.iloc[-1]
        / numeric_prices.iloc[0]
    )

    result = (
        growth_factor
        ** (1.0 / years)
        - 1.0
    )

    result.name = "cagr"

    return result


def annualized_volatility(
    returns: pd.DataFrame,
    *,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate sample volatility annualised by square-root-of-time."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=2,
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    result = (
        numeric_returns.std(
            ddof=1
        )
        * math.sqrt(
            periods
        )
    )

    result.name = (
        "annualized_volatility"
    )

    return result


def annualized_downside_deviation(
    returns: pd.DataFrame,
    *,
    minimum_acceptable_return: Real = 0.0,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate downside deviation relative to an effective annual MAR.

    The denominator uses all observations:

    sqrt(mean(min(r_t - MAR_period, 0)^2)) * sqrt(periods_per_year).
    """
    numeric_returns = validate_return_panel(
        returns,
        min_observations=1,
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    mar = _validate_annual_rate(
        minimum_acceptable_return,
        name="minimum_acceptable_return",
    )

    mar_period = annual_rate_to_period_rate(
        mar,
        periods_per_year=periods,
    )

    downside = (
        numeric_returns
        - mar_period
    ).clip(
        upper=0.0
    )

    result = (
        downside.pow(2)
        .mean()
        .pow(0.5)
        * math.sqrt(
            periods
        )
    )

    result.name = (
        "annualized_downside_deviation"
    )

    return result


def sharpe_ratio(
    returns: pd.DataFrame,
    *,
    risk_free_rate: Real = 0.0,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate the annualised Sharpe ratio from periodic simple returns."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=2,
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    risk_free = _validate_annual_rate(
        risk_free_rate,
        name="risk_free_rate",
    )

    risk_free_period = (
        annual_rate_to_period_rate(
            risk_free,
            periods_per_year=periods,
        )
    )

    mean_excess_return = (
        numeric_returns
        - risk_free_period
    ).mean()

    periodic_volatility = (
        numeric_returns.std(
            ddof=1
        )
        .replace(
            0.0,
            np.nan,
        )
    )

    result = (
        mean_excess_return
        / periodic_volatility
        * math.sqrt(
            periods
        )
    )

    result.name = (
        "sharpe_ratio"
    )

    return result


def sortino_ratio(
    returns: pd.DataFrame,
    *,
    minimum_acceptable_return: Real = 0.0,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate the annualised Sortino ratio relative to an annual MAR."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=1,
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    mar = _validate_annual_rate(
        minimum_acceptable_return,
        name="minimum_acceptable_return",
    )

    mar_period = annual_rate_to_period_rate(
        mar,
        periods_per_year=periods,
    )

    annualized_excess_return = (
        (
            numeric_returns
            - mar_period
        )
        .mean()
        * periods
    )

    downside_deviation = (
        annualized_downside_deviation(
            numeric_returns,
            minimum_acceptable_return=mar,
            periods_per_year=periods,
        )
        .replace(
            0.0,
            np.nan,
        )
    )

    result = (
        annualized_excess_return
        / downside_deviation
    )

    result.name = (
        "sortino_ratio"
    )

    return result


def drawdown_from_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate drawdown directly from adjusted-close total-return prices."""
    numeric_prices = validate_price_panel(
        prices,
        min_observations=1,
    )

    drawdown = (
        numeric_prices
        / numeric_prices.cummax()
        - 1.0
    )

    drawdown.index.name = (
        prices.index.name
    )

    return drawdown


def drawdown_from_returns(
    returns: pd.DataFrame,
    *,
    initial_value: Real = 1.0,
) -> pd.DataFrame:
    """Calculate drawdown from returns, including initial capital as a peak."""
    wealth = cumulative_wealth(
        returns,
        initial_value=initial_value,
    )

    initial = float(
        initial_value
    )

    running_peak = (
        wealth.cummax()
        .clip(
            lower=initial
        )
    )

    drawdown = (
        wealth
        / running_peak
        - 1.0
    )

    drawdown.index.name = (
        returns.index.name
    )

    return drawdown


def maximum_drawdown_from_prices(
    prices: pd.DataFrame,
) -> pd.Series:
    """Return the most negative drawdown observed for each asset."""
    result = (
        drawdown_from_prices(
            prices
        )
        .min()
    )

    result.name = (
        "maximum_drawdown"
    )

    return result


def historical_var(
    returns: pd.DataFrame,
    *,
    confidence_level: Real = 0.95,
) -> pd.Series:
    """Calculate historical VaR using a positive-loss convention."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=1,
    )

    confidence = _validate_confidence_level(
        confidence_level
    )

    return_quantile = (
        numeric_returns.quantile(
            q=1.0 - confidence,
            axis=0,
            numeric_only=True,
            interpolation="linear",
            method="single",
        )
    )

    result = (
        -return_quantile
    )

    result.name = (
        "historical_var"
    )

    return result


def historical_expected_shortfall(
    returns: pd.DataFrame,
    *,
    confidence_level: Real = 0.95,
) -> pd.Series:
    """Calculate empirical Expected Shortfall with fractional tail weighting.

    For N observations and tail probability p = 1-confidence, exactly N*p
    observations of empirical probability mass are averaged. If N*p is not an
    integer, the next order statistic receives the required fractional weight.
    """
    numeric_returns = validate_return_panel(
        returns,
        min_observations=1,
    )

    confidence = _validate_confidence_level(
        confidence_level
    )

    tail_mass = (
        len(
            numeric_returns
        )
        * (
            1.0
            - confidence
        )
    )

    whole_observations = int(
        math.floor(
            tail_mass
        )
    )

    fractional_observation = (
        tail_mass
        - whole_observations
    )

    ordered = np.sort(
        numeric_returns.to_numpy(
            dtype=float,
            copy=True,
        ),
        axis=0,
    )

    tail_sum = np.zeros(
        numeric_returns.shape[1],
        dtype=float,
    )

    if whole_observations > 0:
        tail_sum += (
            ordered[
                :whole_observations,
                :,
            ]
            .sum(
                axis=0
            )
        )

    if fractional_observation > 0.0:
        tail_sum += (
            fractional_observation
            * ordered[
                whole_observations,
                :,
            ]
        )

    result = pd.Series(
        -(
            tail_sum
            / tail_mass
        ),
        index=numeric_returns.columns,
        name="historical_expected_shortfall",
        dtype=float,
    )

    return result


def historical_cvar(
    returns: pd.DataFrame,
    *,
    confidence_level: Real = 0.95,
) -> pd.Series:
    """Return historical Expected Shortfall under the CVaR naming convention."""
    result = (
        historical_expected_shortfall(
            returns,
            confidence_level=confidence_level,
        )
        .copy()
    )

    result.name = (
        "historical_cvar"
    )

    return result


def annualized_covariance_matrix(
    returns: pd.DataFrame,
    *,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate the complete-sample covariance matrix and annualise linearly."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=2,
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    covariance = (
        numeric_returns.cov(
            min_periods=len(
                numeric_returns
            ),
            ddof=1,
            numeric_only=True,
        )
    )

    return (
        covariance
        * periods
    )


def correlation_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the Pearson correlation matrix of complete return series."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=2,
    )

    return numeric_returns.corr(
        method="pearson",
        min_periods=len(
            numeric_returns
        ),
        numeric_only=True,
    )


def summary_statistics(
    prices: pd.DataFrame,
    *,
    risk_free_rate: Real = 0.0,
    minimum_acceptable_return: Real = 0.0,
    confidence_level: Real = 0.95,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Build a per-asset performance and risk statistics table."""
    numeric_prices = validate_price_panel(
        prices,
        min_observations=3,
    )

    periods = _validate_periods_per_year(
        periods_per_year
    )

    confidence = _validate_confidence_level(
        confidence_level
    )

    risk_free = _validate_annual_rate(
        risk_free_rate,
        name="risk_free_rate",
    )

    mar = _validate_annual_rate(
        minimum_acceptable_return,
        name="minimum_acceptable_return",
    )

    returns = simple_returns_from_prices(
        numeric_prices
    )

    confidence_label = (
        f"{confidence:.4f}"
        .rstrip("0")
        .rstrip(".")
    )

    var_column = (
        f"historical_var_"
        f"{confidence_label}_daily"
    )

    cvar_column = (
        f"historical_cvar_"
        f"{confidence_label}_daily"
    )

    result = pd.DataFrame(
        {
            "observations": (
                returns.count()
                .astype(int)
            ),
            "total_return": (
                numeric_prices.iloc[-1]
                / numeric_prices.iloc[0]
                - 1.0
            ),
            "cagr": (
                cagr_from_prices(
                    numeric_prices
                )
            ),
            "annualized_mean_return": (
                annualized_mean_return(
                    returns,
                    periods_per_year=periods,
                )
            ),
            "annualized_volatility": (
                annualized_volatility(
                    returns,
                    periods_per_year=periods,
                )
            ),
            "sharpe_ratio": (
                sharpe_ratio(
                    returns,
                    risk_free_rate=risk_free,
                    periods_per_year=periods,
                )
            ),
            "sortino_ratio": (
                sortino_ratio(
                    returns,
                    minimum_acceptable_return=mar,
                    periods_per_year=periods,
                )
            ),
            "maximum_drawdown": (
                maximum_drawdown_from_prices(
                    numeric_prices
                )
            ),
            var_column: (
                historical_var(
                    returns,
                    confidence_level=confidence,
                )
            ),
            cvar_column: (
                historical_cvar(
                    returns,
                    confidence_level=confidence,
                )
            ),
            "worst_daily_return": (
                returns.min()
            ),
            "best_daily_return": (
                returns.max()
            ),
            "skewness": (
                returns.skew()
            ),
            "excess_kurtosis": (
                returns.kurt()
            ),
        }
    )

    result.index.name = (
        "ticker"
    )

    return result