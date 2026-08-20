import pandas as pd
import pytest

from multi_asset_portfolio.universe import ASSET_UNIVERSE
from multi_asset_portfolio.validation import (
    DataValidationError,
    build_common_price_panel,
    validate_asset_frame,
)


def make_valid_frame(
    start: str = "2010-01-01",
    periods: int = 4000,
) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)

    prices = pd.Series(
        [100.0 * (1.0001 ** i) for i in range(periods)],
        index=dates,
    )

    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.001,
            "Low": prices * 0.999,
            "Close": prices,
            "Adj Close": prices,
            "Volume": 1_000_000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


def test_valid_asset_frame_passes() -> None:
    frame = make_valid_frame()

    asset = ASSET_UNIVERSE["DEVELOPED_EQUITY"]

    validate_asset_frame(
        frame=frame,
        asset=asset,
    )


def test_duplicate_dates_are_rejected() -> None:
    frame = make_valid_frame(periods=20)

    duplicated = pd.concat(
        [
            frame,
            frame.iloc[[5]],
        ]
    ).sort_index()

    asset = ASSET_UNIVERSE["DEVELOPED_EQUITY"]

    with pytest.raises(
        DataValidationError,
        match="duplicate dates",
    ):
        validate_asset_frame(
            frame=duplicated,
            asset=asset,
        )


def test_non_positive_adjusted_price_is_rejected() -> None:
    frame = make_valid_frame(periods=20)

    frame.iloc[5, frame.columns.get_loc("Adj Close")] = 0.0

    asset = ASSET_UNIVERSE["DEVELOPED_EQUITY"]

    with pytest.raises(
        DataValidationError,
        match="non-positive",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_extreme_daily_return_is_rejected() -> None:
    frame = make_valid_frame(periods=20)

    frame.iloc[10, frame.columns.get_loc("Adj Close")] *= 2.0

    asset = ASSET_UNIVERSE["DEVELOPED_EQUITY"]

    with pytest.raises(
        DataValidationError,
        match="suspicious daily",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_common_panel_uses_only_shared_dates() -> None:
    first = make_valid_frame(
        start="2010-01-01",
        periods=4000,
    )

    second = make_valid_frame(
        start="2010-01-04",
        periods=4000,
    )

    panel, missing = build_common_price_panel(
        {
            "FIRST": first,
            "SECOND": second,
        }
    )

    assert panel.index.min() >= second.index.min()
    assert panel.notna().all().all()
    assert set(panel.columns) == {"FIRST", "SECOND"}
    assert set(missing) == {"FIRST", "SECOND"}