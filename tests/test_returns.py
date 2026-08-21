import numpy as np
import pandas as pd
import pytest

from multi_asset_portfolio.returns import (
    ReturnCalculationError,
    calendar_span_years,
    cumulative_wealth,
    log_returns_from_prices,
    simple_returns_from_prices,
    total_return,
    validate_price_panel,
    validate_return_panel,
)


def make_prices(
    values: list[float],
    *,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    index = pd.bdate_range(
        start=start,
        periods=len(values),
    )

    return pd.DataFrame(
        {
            "A": values,
        },
        index=index,
    )


def make_returns(
    values: list[float],
    *,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    index = pd.bdate_range(
        start=start,
        periods=len(values),
    )

    return pd.DataFrame(
        {
            "A": values,
        },
        index=index,
    )


def test_simple_returns_are_exact() -> None:
    prices = make_prices(
        [
            100.0,
            110.0,
            99.0,
        ]
    )

    returns = simple_returns_from_prices(
        prices
    )

    np.testing.assert_allclose(
        returns["A"].to_numpy(),
        [
            0.10,
            -0.10,
        ],
    )

    assert returns.index.equals(
        prices.index[1:]
    )


def test_log_returns_are_exact() -> None:
    prices = make_prices(
        [
            100.0,
            110.0,
            99.0,
        ]
    )

    returns = log_returns_from_prices(
        prices
    )

    expected = np.log(
        np.array(
            [
                1.10,
                0.90,
            ]
        )
    )

    np.testing.assert_allclose(
        returns["A"].to_numpy(),
        expected,
    )


def test_missing_price_is_rejected() -> None:
    prices = make_prices(
        [
            100.0,
            110.0,
            99.0,
        ]
    )

    prices.iloc[
        1,
        0,
    ] = np.nan

    with pytest.raises(
        ReturnCalculationError,
        match="missing values",
    ):
        validate_price_panel(
            prices
        )


def test_non_positive_price_is_rejected() -> None:
    prices = make_prices(
        [
            100.0,
            0.0,
            99.0,
        ]
    )

    with pytest.raises(
        ReturnCalculationError,
        match="non-positive",
    ):
        validate_price_panel(
            prices
        )


def test_duplicate_dates_are_rejected() -> None:
    prices = make_prices(
        [
            100.0,
            110.0,
            99.0,
        ]
    )

    prices.index = pd.DatetimeIndex(
        [
            prices.index[0],
            prices.index[0],
            prices.index[2],
        ]
    )

    with pytest.raises(
        ReturnCalculationError,
        match="duplicate",
    ):
        validate_price_panel(
            prices
        )


def test_unsorted_dates_are_rejected() -> None:
    prices = (
        make_prices(
            [
                100.0,
                110.0,
                99.0,
            ]
        )
        .iloc[::-1]
    )

    with pytest.raises(
        ReturnCalculationError,
        match="ascending",
    ):
        validate_price_panel(
            prices
        )


def test_return_below_minus_one_is_rejected() -> None:
    returns = make_returns(
        [
            0.10,
            -1.01,
        ]
    )

    with pytest.raises(
        ReturnCalculationError,
        match="below -100%",
    ):
        validate_return_panel(
            returns
        )


def test_minus_one_return_is_allowed() -> None:
    returns = make_returns(
        [
            0.10,
            -1.00,
        ]
    )

    validated = validate_return_panel(
        returns
    )

    assert (
        validated.iloc[-1, 0]
        == -1.0
    )


def test_cumulative_wealth_is_exact() -> None:
    returns = make_returns(
        [
            -0.10,
            0.20,
        ]
    )

    wealth = cumulative_wealth(
        returns,
        initial_value=100.0,
    )

    np.testing.assert_allclose(
        wealth["A"].to_numpy(),
        [
            90.0,
            108.0,
        ],
    )


def test_total_return_is_compounded() -> None:
    returns = make_returns(
        [
            0.10,
            -0.10,
        ]
    )

    result = total_return(
        returns
    )

    assert result["A"] == pytest.approx(
        -0.01
    )


def test_calendar_span_years_uses_calendar_time() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2021-01-01",
        ]
    )

    result = calendar_span_years(
        index
    )

    assert result == pytest.approx(
        366.0 / 365.2425
    )