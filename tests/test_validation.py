import pandas as pd
import pytest

from multi_asset_portfolio.universe import ASSET_UNIVERSE
from multi_asset_portfolio.validation import (
    DataValidationError,
    build_common_price_panel,
    validate_asset_frame,
    validate_common_history,
)


def make_valid_frame(
    start: str = "2010-01-01",
    periods: int = 4000,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        start=start,
        periods=periods,
    )

    prices = pd.Series(
        [
            100.0
            * (
                1.0001 ** i
            )
            for i in range(
                periods
            )
        ],
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

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    report = validate_asset_frame(
        frame=frame,
        asset=asset,
    )

    assert (
        report.ticker
        == asset.ticker
    )

    assert (
        report
        .zero_ohlc_count_by_column
        == {
            "Open": 0,
            "High": 0,
            "Low": 0,
            "Close": 0,
        }
    )

    assert (
        report
        .inconsistent_ohlc_row_count
        == 0
    )


def test_missing_required_column_is_rejected() -> None:
    frame = (
        make_valid_frame(
            periods=20
        )
        .drop(
            columns=[
                "Adj Close"
            ]
        )
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="missing required columns",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_duplicate_dates_are_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    duplicated = pd.concat(
        [
            frame,
            frame.iloc[[5]],
        ]
    ).sort_index()

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="duplicate dates",
    ):
        validate_asset_frame(
            frame=duplicated,
            asset=asset,
        )


def test_unsorted_dates_are_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    unsorted = frame.iloc[
        ::-1
    ]

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="ascending order",
    ):
        validate_asset_frame(
            frame=unsorted,
            asset=asset,
        )


def test_excessive_missing_data_are_rejected() -> None:
    frame = make_valid_frame(
        periods=100
    )

    frame.loc[
        frame.index[:2],
        "Adj Close",
    ] = float(
        "nan"
    )

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="missing-data limit",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_non_positive_adjusted_price_is_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    frame.iloc[
        5,
        frame.columns.get_loc(
            "Adj Close"
        ),
    ] = 0.0

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="non-positive",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_negative_raw_ohlc_price_is_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    frame.iloc[
        5,
        frame.columns.get_loc(
            "Open"
        ),
    ] = -1.0

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="negative value",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_negative_volume_is_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    frame.iloc[
        5,
        frame.columns.get_loc(
            "Volume"
        ),
    ] = -1

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="negative value",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_zero_raw_ohlc_is_reported_but_not_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    warning_date = frame.index[
        5
    ]

    frame.loc[
        warning_date,
        "Open",
    ] = 0.0

    asset = ASSET_UNIVERSE[
        "EURO_GOVERNMENT_BONDS"
    ]

    report = validate_asset_frame(
        frame=frame,
        asset=asset,
    )

    assert (
        report
        .zero_ohlc_count_by_column[
            "Open"
        ]
        == 1
    )

    assert (
        report
        .zero_ohlc_dates_by_column[
            "Open"
        ]
        == [
            warning_date
            .date()
            .isoformat()
        ]
    )

    assert (
        report
        .zero_ohlc_count_by_column[
            "High"
        ]
        == 0
    )


def test_inconsistent_ohlc_is_reported_but_not_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    warning_date = frame.index[
        5
    ]

    frame.loc[
        warning_date,
        "High",
    ] = (
        frame.loc[
            warning_date,
            "Close",
        ]
        * 0.99
    )

    asset = ASSET_UNIVERSE[
        "EURO_GOVERNMENT_BONDS"
    ]

    report = validate_asset_frame(
        frame=frame,
        asset=asset,
    )

    assert (
        report
        .inconsistent_ohlc_row_count
        == 1
    )

    assert (
        report
        .inconsistent_ohlc_dates
        == [
            warning_date
            .date()
            .isoformat()
        ]
    )


def test_extreme_daily_return_is_rejected() -> None:
    frame = make_valid_frame(
        periods=20
    )

    frame.iloc[
        10,
        frame.columns.get_loc(
            "Adj Close"
        ),
    ] *= 2.0

    asset = ASSET_UNIVERSE[
        "DEVELOPED_EQUITY"
    ]

    with pytest.raises(
        DataValidationError,
        match="suspicious daily",
    ):
        validate_asset_frame(
            frame=frame,
            asset=asset,
        )


def test_common_panel_uses_only_shared_dates_without_penalising_inception() -> None:
    first = make_valid_frame(
        start="2010-01-01",
        periods=4000,
    )

    second = make_valid_frame(
        start="2010-01-04",
        periods=4000,
    )

    panel, missing = (
        build_common_price_panel(
            {
                "FIRST": first,
                "SECOND": second,
            }
        )
    )

    assert (
        panel.index.min()
        >= second.index.min()
    )

    assert (
        panel.notna()
        .all()
        .all()
    )

    assert set(
        panel.columns
    ) == {
        "FIRST",
        "SECOND",
    }

    assert missing == {
        "FIRST": 0,
        "SECOND": 0,
    }


def test_common_panel_counts_internal_missing_observations() -> None:
    first = make_valid_frame(
        periods=100
    )

    second = make_valid_frame(
        periods=100
    )

    missing_date = (
        second.index[50]
    )

    second.loc[
        missing_date,
        "Adj Close",
    ] = float(
        "nan"
    )

    panel, missing = (
        build_common_price_panel(
            {
                "FIRST": first,
                "SECOND": second,
            }
        )
    )

    assert (
        missing_date
        not in panel.index
    )

    assert (
        missing[
            "FIRST"
        ]
        == 0
    )

    assert (
        missing[
            "SECOND"
        ]
        == 1
    )


def test_common_panel_does_not_forward_fill_missing_prices() -> None:
    first = make_valid_frame(
        periods=100
    )

    second = make_valid_frame(
        periods=100
    )

    missing_date = (
        second.index[50]
    )

    original_previous_price = (
        second.loc[
            second.index[49],
            "Adj Close",
        ]
    )

    second.loc[
        missing_date,
        "Adj Close",
    ] = float(
        "nan"
    )

    panel, _ = (
        build_common_price_panel(
            {
                "FIRST": first,
                "SECOND": second,
            }
        )
    )

    assert (
        missing_date
        not in panel.index
    )

    assert (
        original_previous_price
        == second.loc[
            second.index[49],
            "Adj Close",
        ]
    )


def test_common_history_rejects_insufficient_span() -> None:
    frame = make_valid_frame(
        periods=1000
    )

    panel = (
        frame[
            [
                "Adj Close"
            ]
        ]
        .rename(
            columns={
                "Adj Close": "ONLY"
            }
        )
    )

    with pytest.raises(
        DataValidationError,
        match="Common history spans",
    ):
        validate_common_history(
            panel
        )


def test_common_history_accepts_sufficient_span() -> None:
    frame = make_valid_frame(
        periods=4000
    )

    panel = (
        frame[
            [
                "Adj Close"
            ]
        ]
        .rename(
            columns={
                "Adj Close": "ONLY"
            }
        )
    )

    history_years = (
        validate_common_history(
            panel
        )
    )

    assert (
        history_years
        >= 10.0
    )