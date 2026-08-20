"""Market-data validation utilities for the multi-asset portfolio project.

This module centralises the quality-control rules applied to raw market data
before any return calculation, portfolio optimisation or backtesting step.

The validation layer is deliberately independent from the market-data provider.
It never silently fills, interpolates or manufactures missing price
observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .universe import AssetSpec


REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }
)

MAX_ABS_DAILY_RETURN: float = 0.35
MAX_MISSING_FRACTION: float = 0.005
MIN_COMMON_HISTORY_YEARS: float = 10.0

CALENDAR_DAYS_PER_YEAR: float = 365.2425


class DataValidationError(ValueError):
    """Raised when market data fail a mandatory validation rule."""


@dataclass(frozen=True)
class PriceFrameValidationReport:
    """Quality-control summary for one raw market-data series."""

    ticker: str
    rows: int
    start_date: str
    end_date: str
    missing_fraction_by_column: dict[str, float]
    max_abs_daily_return: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the report."""
        return asdict(self)


def _asset_label(ticker: str | None) -> str:
    """Return a readable ticker suffix for diagnostic messages."""
    return f" for {ticker}" if ticker else ""


def _require_valid_datetime_index(
    index: pd.Index,
    *,
    context: str,
) -> None:
    """Validate the chronological structure of a market-data index."""
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError(
            f"{context} must use a pandas DatetimeIndex."
        )

    if index.hasnans:
        raise DataValidationError(
            f"{context} contains NaT values in its index."
        )

    if index.has_duplicates:
        duplicate_dates = index[index.duplicated()].unique()

        preview = ", ".join(
            timestamp.isoformat()
            for timestamp in duplicate_dates[:3]
        )

        raise DataValidationError(
            f"{context} contains duplicate dates: {preview}."
        )

    if not index.is_monotonic_increasing:
        raise DataValidationError(
            f"{context} index must be sorted in ascending order."
        )


def _numeric_series(
    series: pd.Series,
    *,
    column_name: str,
    ticker: str | None,
) -> pd.Series:
    """Convert a required column to numeric data without hiding bad values."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    introduced_missing_values = (
        series.notna()
        & numeric.isna()
    )

    if introduced_missing_values.any():
        raise DataValidationError(
            f"Column '{column_name}'{_asset_label(ticker)} "
            "contains non-numeric values."
        )

    non_finite_values = (
        numeric.notna()
        & ~np.isfinite(numeric)
    )

    if non_finite_values.any():
        raise DataValidationError(
            f"Column '{column_name}'{_asset_label(ticker)} "
            "contains non-finite values."
        )

    return numeric


def validate_price_frame(
    frame: pd.DataFrame,
    *,
    ticker: str | None = None,
    max_abs_daily_return: float = MAX_ABS_DAILY_RETURN,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> PriceFrameValidationReport:
    """Validate one raw OHLCV market-data DataFrame.

    The validation is non-mutating. No missing price is filled or repaired.

    Parameters
    ----------
    frame:
        Raw OHLCV market-data DataFrame.
    ticker:
        Optional ticker used in diagnostic messages.
    max_abs_daily_return:
        Maximum tolerated absolute one-day adjusted-close return.
    max_missing_fraction:
        Maximum tolerated missing fraction for each required field.

    Returns
    -------
    PriceFrameValidationReport
        Structured quality-control statistics for the asset.

    Raises
    ------
    DataValidationError
        If any mandatory quality-control rule fails.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError(
            "max_missing_fraction must be between 0 and 1."
        )

    if max_abs_daily_return <= 0.0:
        raise ValueError(
            "max_abs_daily_return must be strictly positive."
        )

    if frame.empty:
        raise DataValidationError(
            f"Market-data frame{_asset_label(ticker)} is empty."
        )

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(frame.columns)
    )

    if missing_columns:
        raise DataValidationError(
            f"Market-data frame{_asset_label(ticker)} is missing "
            f"required columns: {missing_columns}."
        )

    _require_valid_datetime_index(
        frame.index,
        context=f"Market-data frame{_asset_label(ticker)}",
    )

    numeric_columns: dict[str, pd.Series] = {}

    for column in sorted(REQUIRED_COLUMNS):
        numeric_columns[column] = _numeric_series(
            frame[column],
            column_name=column,
            ticker=ticker,
        )

    adjusted_close = numeric_columns["Adj Close"]

    if adjusted_close.notna().sum() == 0:
        raise DataValidationError(
            f"'Adj Close'{_asset_label(ticker)} is entirely missing."
        )

    required_frame = frame.loc[
        :,
        sorted(REQUIRED_COLUMNS),
    ]

    missing_fractions = required_frame.isna().mean()

    excessive_missing = missing_fractions[
        missing_fractions > max_missing_fraction
    ]

    if not excessive_missing.empty:
        details = ", ".join(
            f"{column}={fraction:.2%}"
            for column, fraction in excessive_missing.items()
        )

        raise DataValidationError(
            f"Market-data frame{_asset_label(ticker)} exceeds the "
            f"missing-data limit of {max_missing_fraction:.2%}: "
            f"{details}."
        )

    price_columns = (
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    )

    for column in price_columns:
        series = numeric_columns[column]

        non_positive = (
            series.notna()
            & (series <= 0.0)
        )

        if non_positive.any():
            first_bad_date = frame.index[non_positive][0]

            raise DataValidationError(
                f"Column '{column}'{_asset_label(ticker)} contains "
                f"a non-positive value on "
                f"{first_bad_date.date()}."
            )

    volume = numeric_columns["Volume"]

    negative_volume = (
        volume.notna()
        & (volume < 0.0)
    )

    if negative_volume.any():
        first_bad_date = frame.index[negative_volume][0]

        raise DataValidationError(
            f"'Volume'{_asset_label(ticker)} contains a negative "
            f"value on {first_bad_date.date()}."
        )

    open_price = numeric_columns["Open"]
    high_price = numeric_columns["High"]
    low_price = numeric_columns["Low"]
    close_price = numeric_columns["Close"]

    complete_ohlc = (
        open_price.notna()
        & high_price.notna()
        & low_price.notna()
        & close_price.notna()
    )

    invalid_high = (
        complete_ohlc
        & (
            (high_price < open_price)
            | (high_price < low_price)
            | (high_price < close_price)
        )
    )

    if invalid_high.any():
        first_bad_date = frame.index[invalid_high][0]

        raise DataValidationError(
            f"'High'{_asset_label(ticker)} is inconsistent with OHLC "
            f"prices on {first_bad_date.date()}."
        )

    invalid_low = (
        complete_ohlc
        & (
            (low_price > open_price)
            | (low_price > high_price)
            | (low_price > close_price)
        )
    )

    if invalid_low.any():
        first_bad_date = frame.index[invalid_low][0]

        raise DataValidationError(
            f"'Low'{_asset_label(ticker)} is inconsistent with OHLC "
            f"prices on {first_bad_date.date()}."
        )

    daily_returns = adjusted_close.pct_change(
        fill_method=None
    )

    absolute_daily_returns = daily_returns.abs()

    suspicious_returns = (
        absolute_daily_returns > max_abs_daily_return
    )

    if suspicious_returns.any():
        first_bad_date = suspicious_returns[
            suspicious_returns
        ].index[0]

        bad_return = float(
            daily_returns.loc[first_bad_date]
        )

        raise DataValidationError(
            f"'Adj Close'{_asset_label(ticker)} contains a suspicious "
            f"daily return of {bad_return:.2%} on "
            f"{first_bad_date.date()}, above the "
            f"{max_abs_daily_return:.0%} validation limit."
        )

    if absolute_daily_returns.notna().any():
        observed_max_abs_return = float(
            absolute_daily_returns.max()
        )
    else:
        observed_max_abs_return = 0.0

    return PriceFrameValidationReport(
        ticker=ticker or "",
        rows=len(frame),
        start_date=frame.index[0].date().isoformat(),
        end_date=frame.index[-1].date().isoformat(),
        missing_fraction_by_column={
            column: float(
                missing_fractions[column]
            )
            for column in sorted(REQUIRED_COLUMNS)
        },
        max_abs_daily_return=observed_max_abs_return,
    )


def validate_asset_frame(
    frame: pd.DataFrame,
    *,
    asset: AssetSpec,
    max_abs_daily_return: float = MAX_ABS_DAILY_RETURN,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> PriceFrameValidationReport:
    """Validate raw market data for one asset in the investment universe.

    Asset metadata are provided through AssetSpec, while the numerical and
    structural quality-control logic remains centralised in
    validate_price_frame().
    """
    if not isinstance(asset, AssetSpec):
        raise TypeError(
            "asset must be an AssetSpec instance."
        )

    return validate_price_frame(
        frame=frame,
        ticker=asset.ticker,
        max_abs_daily_return=max_abs_daily_return,
        max_missing_fraction=max_missing_fraction,
    )


def build_common_price_panel(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the common adjusted-close panel across all assets.

    The individual adjusted-close series are first aligned on the union of
    their observed dates. This makes missing observations measurable.

    The final investable panel then retains only dates for which every asset
    has a valid adjusted-close observation.

    No forward-fill, backward-fill or interpolation is performed.

    Parameters
    ----------
    frames:
        Mapping from ticker or asset identifier to raw market-data DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        The common adjusted-close panel and, for each asset, the number of
        missing adjusted-close observations observed on the aligned union
        calendar before restricting the dataset to shared dates.
    """
    if not isinstance(frames, Mapping):
        raise TypeError(
            "frames must be a mapping of asset identifiers "
            "to pandas DataFrames."
        )

    if not frames:
        raise DataValidationError(
            "No market-data frames were supplied."
        )

    adjusted_close_series: list[pd.Series] = []

    for ticker, frame in frames.items():
        if not isinstance(ticker, str):
            raise TypeError(
                "Every market-data mapping key must be a string."
            )

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"Market data for {ticker} must be a pandas DataFrame."
            )

        if frame.empty:
            raise DataValidationError(
                f"Market-data frame for {ticker} is empty."
            )

        if "Adj Close" not in frame.columns:
            raise DataValidationError(
                f"Market-data frame for {ticker} is missing "
                "'Adj Close'."
            )

        _require_valid_datetime_index(
            frame.index,
            context=f"Market-data frame for {ticker}",
        )

        adjusted_close = _numeric_series(
            frame["Adj Close"],
            column_name="Adj Close",
            ticker=ticker,
        )

        if adjusted_close.notna().sum() == 0:
            raise DataValidationError(
                f"'Adj Close' for {ticker} is entirely missing."
            )

        non_positive = (
            adjusted_close.notna()
            & (adjusted_close <= 0.0)
        )

        if non_positive.any():
            first_bad_date = frame.index[non_positive][0]

            raise DataValidationError(
                f"'Adj Close' for {ticker} contains a non-positive "
                f"value on {first_bad_date.date()}."
            )

        adjusted_close_series.append(
            adjusted_close.rename(ticker)
        )

    aligned_panel = pd.concat(
        adjusted_close_series,
        axis=1,
        join="outer",
    ).sort_index()

    missing_observations = {
        ticker: int(
            aligned_panel[ticker].isna().sum()
        )
        for ticker in aligned_panel.columns
    }

    common_panel = aligned_panel.dropna(
        axis=0,
        how="any",
    )

    if common_panel.empty:
        raise DataValidationError(
            "The assets have no common dates with valid "
            "adjusted-close observations."
        )

    _require_valid_datetime_index(
        common_panel.index,
        context="Common adjusted-close panel",
    )

    return common_panel, missing_observations


def build_common_adjusted_close_panel(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return only the common adjusted-close panel.

    This convenience function delegates construction to
    build_common_price_panel() and discards the missing-observation
    diagnostics.
    """
    panel, _ = build_common_price_panel(
        frames
    )

    return panel


def common_history_years(
    panel: pd.DataFrame,
) -> float:
    """Return the calendar span of a common price panel in years."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(
            "panel must be a pandas DataFrame."
        )

    if panel.empty:
        return 0.0

    _require_valid_datetime_index(
        panel.index,
        context="Common adjusted-close panel",
    )

    if len(panel.index) < 2:
        return 0.0

    history_days = (
        panel.index[-1] - panel.index[0]
    ).total_seconds() / 86_400.0

    return history_days / CALENDAR_DAYS_PER_YEAR


def validate_common_history(
    panel: pd.DataFrame,
    *,
    min_common_history_years: float = MIN_COMMON_HISTORY_YEARS,
) -> float:
    """Validate the usable common history of the investment universe.

    Parameters
    ----------
    panel:
        Common adjusted-close price panel.
    min_common_history_years:
        Minimum accepted common calendar history.

    Returns
    -------
    float
        Observed common-history length in calendar years.

    Raises
    ------
    DataValidationError
        If the panel contains invalid prices, missing values or an
        insufficient common history.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(
            "panel must be a pandas DataFrame."
        )

    if min_common_history_years <= 0.0:
        raise ValueError(
            "min_common_history_years must be strictly positive."
        )

    if panel.empty:
        raise DataValidationError(
            "Common adjusted-close panel is empty."
        )

    _require_valid_datetime_index(
        panel.index,
        context="Common adjusted-close panel",
    )

    numeric_panel = panel.apply(
        pd.to_numeric,
        errors="coerce",
    )

    introduced_missing = (
        panel.notna()
        & numeric_panel.isna()
    )

    if introduced_missing.any().any():
        raise DataValidationError(
            "Common adjusted-close panel contains "
            "non-numeric values."
        )

    non_finite = (
        numeric_panel.notna()
        & ~np.isfinite(numeric_panel)
    )

    if non_finite.any().any():
        raise DataValidationError(
            "Common adjusted-close panel contains "
            "non-finite values."
        )

    if numeric_panel.isna().any().any():
        raise DataValidationError(
            "Common adjusted-close panel contains missing values."
        )

    if (numeric_panel <= 0.0).any().any():
        raise DataValidationError(
            "Common adjusted-close panel contains "
            "non-positive prices."
        )

    history_years = common_history_years(
        numeric_panel
    )

    if history_years < min_common_history_years:
        raise DataValidationError(
            f"Common history spans only {history_years:.2f} years; "
            f"at least {min_common_history_years:.2f} years "
            "are required."
        )

    return history_years