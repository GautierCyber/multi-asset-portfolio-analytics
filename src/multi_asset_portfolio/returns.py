"""Return construction utilities for the multi-asset portfolio project.

This module transforms a validated adjusted-close price panel into simple
returns, log returns and compounded wealth series. It never forward-fills,
backward-fills, interpolates or otherwise imputes missing observations.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR: int = 252
CALENDAR_DAYS_PER_YEAR: float = 365.2425


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "CALENDAR_DAYS_PER_YEAR",
    "ReturnCalculationError",
    "validate_price_panel",
    "validate_return_panel",
    "simple_returns_from_prices",
    "log_returns_from_prices",
    "cumulative_wealth",
    "total_return",
    "calendar_span_years",
]


class ReturnCalculationError(ValueError):
    """Raised when prices or returns are unsuitable for return analytics."""


def _validate_datetime_index(
    index: pd.Index,
    *,
    context: str,
) -> None:
    """Validate the chronological axis of an analytics panel."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ReturnCalculationError(
            f"{context} must use a pandas DatetimeIndex."
        )

    if index.hasnans:
        raise ReturnCalculationError(
            f"{context} contains NaT values in its index."
        )

    if index.has_duplicates:
        raise ReturnCalculationError(
            f"{context} contains duplicate dates."
        )

    if not index.is_monotonic_increasing:
        raise ReturnCalculationError(
            f"{context} index must be sorted in ascending order."
        )


def _validate_columns(
    columns: pd.Index,
    *,
    context: str,
) -> None:
    """Validate the asset axis of an analytics panel."""
    if len(columns) == 0:
        raise ReturnCalculationError(
            f"{context} must contain at least one asset column."
        )

    if columns.has_duplicates:
        raise ReturnCalculationError(
            f"{context} contains duplicate asset columns."
        )

    if any(
        not isinstance(column, str)
        for column in columns
    ):
        raise ReturnCalculationError(
            f"{context} asset columns must be strings."
        )


def _coerce_numeric_frame(
    frame: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Convert a complete panel to floats without hiding invalid data."""
    numeric = frame.apply(
        pd.to_numeric,
        errors="coerce",
    )

    introduced_missing = (
        frame.notna()
        & numeric.isna()
    )

    if introduced_missing.any().any():
        raise ReturnCalculationError(
            f"{context} contains non-numeric values."
        )

    if numeric.isna().any().any():
        raise ReturnCalculationError(
            f"{context} contains missing values."
        )

    values = numeric.to_numpy(
        dtype=float,
        copy=False,
    )

    if not np.isfinite(values).all():
        raise ReturnCalculationError(
            f"{context} contains non-finite values."
        )

    return numeric.astype(
        float
    )


def _validate_min_observations(
    min_observations: int,
) -> None:
    """Validate a minimum-observation requirement."""
    if not isinstance(
        min_observations,
        int,
    ):
        raise TypeError(
            "min_observations must be an integer."
        )

    if min_observations < 1:
        raise ValueError(
            "min_observations must be at least 1."
        )


def validate_price_panel(
    prices: pd.DataFrame,
    *,
    min_observations: int = 2,
) -> pd.DataFrame:
    """Validate a complete panel of strictly positive adjusted-close prices."""
    if not isinstance(
        prices,
        pd.DataFrame,
    ):
        raise TypeError(
            "prices must be a pandas DataFrame."
        )

    _validate_min_observations(
        min_observations
    )

    if len(prices) < min_observations:
        raise ReturnCalculationError(
            f"Price panel contains only {len(prices)} observations; "
            f"at least {min_observations} are required."
        )

    _validate_datetime_index(
        prices.index,
        context="Price panel",
    )

    _validate_columns(
        prices.columns,
        context="Price panel",
    )

    numeric = _coerce_numeric_frame(
        prices,
        context="Price panel",
    )

    non_positive = (
        numeric <= 0.0
    )

    if non_positive.any().any():
        row_position, column_position = (
            np.argwhere(
                non_positive.to_numpy()
            )[0]
        )

        bad_date = numeric.index[
            int(row_position)
        ]

        bad_asset = numeric.columns[
            int(column_position)
        ]

        raise ReturnCalculationError(
            "Price panel contains a non-positive adjusted price for "
            f"{bad_asset} on {bad_date.date()}."
        )

    return numeric


def validate_return_panel(
    returns: pd.DataFrame,
    *,
    min_observations: int = 1,
) -> pd.DataFrame:
    """Validate a complete panel of simple returns.

    Simple returns below -100% are impossible for an unlevered wealth process
    and are rejected. A return of exactly -100% is mathematically valid.
    """
    if not isinstance(
        returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "returns must be a pandas DataFrame."
        )

    _validate_min_observations(
        min_observations
    )

    if len(returns) < min_observations:
        raise ReturnCalculationError(
            f"Return panel contains only {len(returns)} observations; "
            f"at least {min_observations} are required."
        )

    _validate_datetime_index(
        returns.index,
        context="Return panel",
    )

    _validate_columns(
        returns.columns,
        context="Return panel",
    )

    numeric = _coerce_numeric_frame(
        returns,
        context="Return panel",
    )

    below_total_loss = (
        numeric < -1.0
    )

    if below_total_loss.any().any():
        row_position, column_position = (
            np.argwhere(
                below_total_loss.to_numpy()
            )[0]
        )

        bad_date = numeric.index[
            int(row_position)
        ]

        bad_asset = numeric.columns[
            int(column_position)
        ]

        raise ReturnCalculationError(
            "Return panel contains a simple return below -100% for "
            f"{bad_asset} on {bad_date.date()}."
        )

    return numeric


def simple_returns_from_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate one-period simple returns from adjusted-close prices."""
    numeric_prices = validate_price_panel(
        prices,
        min_observations=2,
    )

    returns = (
        numeric_prices
        .pct_change(
            periods=1,
            fill_method=None,
        )
        .iloc[1:]
        .copy()
    )

    returns.index.name = (
        prices.index.name
    )

    return validate_return_panel(
        returns,
        min_observations=1,
    )


def log_returns_from_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate continuously compounded one-period log returns."""
    numeric_prices = validate_price_panel(
        prices,
        min_observations=2,
    )

    log_returns = (
        np.log(
            numeric_prices
            / numeric_prices.shift(1)
        )
        .iloc[1:]
        .copy()
    )

    values = log_returns.to_numpy(
        dtype=float,
        copy=False,
    )

    if (
        log_returns.isna().any().any()
        or not np.isfinite(values).all()
    ):
        raise ReturnCalculationError(
            "Log-return calculation produced missing "
            "or non-finite values."
        )

    log_returns.index.name = (
        prices.index.name
    )

    return log_returns


def cumulative_wealth(
    returns: pd.DataFrame,
    *,
    initial_value: Real = 1.0,
) -> pd.DataFrame:
    """Compound simple returns into a wealth index."""
    numeric_returns = validate_return_panel(
        returns,
        min_observations=1,
    )

    if not isinstance(
        initial_value,
        Real,
    ):
        raise TypeError(
            "initial_value must be numeric."
        )

    initial = float(
        initial_value
    )

    if not math.isfinite(
        initial
    ):
        raise ValueError(
            "initial_value must be finite."
        )

    if initial <= 0.0:
        raise ValueError(
            "initial_value must be strictly positive."
        )

    wealth = (
        (1.0 + numeric_returns)
        .cumprod()
        * initial
    )

    wealth.index.name = (
        returns.index.name
    )

    return wealth


def total_return(
    returns: pd.DataFrame,
) -> pd.Series:
    """Return the compounded total simple return for every asset column."""
    result = (
        cumulative_wealth(
            returns,
            initial_value=1.0,
        )
        .iloc[-1]
        - 1.0
    )

    result.name = (
        "total_return"
    )

    return result


def calendar_span_years(
    index: pd.DatetimeIndex,
) -> float:
    """Return an ordered time-index span in tropical calendar years."""
    _validate_datetime_index(
        index,
        context="Time index",
    )

    if len(index) < 2:
        raise ReturnCalculationError(
            "At least two dates are required "
            "to calculate a calendar span."
        )

    elapsed_days = (
        (
            index[-1]
            - index[0]
        )
        .total_seconds()
        / 86_400.0
    )

    if elapsed_days <= 0.0:
        raise ReturnCalculationError(
            "Calendar span must be strictly positive."
        )

    return (
        elapsed_days
        / CALENDAR_DAYS_PER_YEAR
    )