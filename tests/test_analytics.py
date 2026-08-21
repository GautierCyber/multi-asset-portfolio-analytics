import math

import numpy as np
import pandas as pd
import pytest

from multi_asset_portfolio.analytics import (
    annual_rate_to_period_rate,
    annualized_covariance_matrix,
    annualized_downside_deviation,
    annualized_mean_return,
    annualized_volatility,
    cagr_from_prices,
    correlation_matrix,
    drawdown_from_prices,
    drawdown_from_returns,
    historical_cvar,
    historical_expected_shortfall,
    historical_var,
    maximum_drawdown_from_prices,
    sharpe_ratio,
    sortino_ratio,
    summary_statistics,
)


def make_returns(
    first: list[float],
    second: list[float] | None = None,
) -> pd.DataFrame:
    index = pd.bdate_range(
        start="2024-01-02",
        periods=len(first),
    )

    data: dict[
        str,
        list[float],
    ] = {
        "A": first,
    }

    if second is not None:
        data["B"] = second

    return pd.DataFrame(
        data,
        index=index,
    )


def make_prices(
    first: list[float],
    second: list[float] | None = None,
) -> pd.DataFrame:
    index = pd.bdate_range(
        start="2024-01-01",
        periods=len(first),
    )

    data: dict[
        str,
        list[float],
    ] = {
        "A": first,
    }

    if second is not None:
        data["B"] = second

    return pd.DataFrame(
        data,
        index=index,
    )


def test_annual_rate_conversion_compounds_back() -> None:
    daily = annual_rate_to_period_rate(
        0.05,
        periods_per_year=252,
    )

    reconstructed = (
        (1.0 + daily) ** 252
        - 1.0
    )

    assert reconstructed == pytest.approx(
        0.05
    )


def test_annualized_mean_return_is_exact() -> None:
    values = [
        0.01,
        0.02,
        -0.01,
    ]

    returns = make_returns(
        values
    )

    result = annualized_mean_return(
        returns,
        periods_per_year=12,
    )

    expected = (
        np.mean(values)
        * 12
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_cagr_uses_calendar_span() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2022-01-01",
        ]
    )

    prices = pd.DataFrame(
        {
            "A": [
                100.0,
                121.0,
            ],
        },
        index=index,
    )

    years = (
        731.0
        / 365.2425
    )

    expected = (
        1.21 ** (1.0 / years)
        - 1.0
    )

    result = cagr_from_prices(
        prices
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_annualized_volatility_uses_sample_std() -> None:
    values = [
        0.01,
        0.02,
        -0.01,
    ]

    returns = make_returns(
        values
    )

    result = annualized_volatility(
        returns,
        periods_per_year=12,
    )

    expected = (
        np.std(
            values,
            ddof=1,
        )
        * math.sqrt(
            12
        )
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_downside_deviation_uses_all_observations() -> None:
    returns = make_returns(
        [
            0.02,
            -0.01,
            -0.03,
            0.04,
        ]
    )

    result = annualized_downside_deviation(
        returns,
        periods_per_year=1,
    )

    expected = math.sqrt(
        (
            0.00**2
            + 0.01**2
            + 0.03**2
            + 0.00**2
        )
        / 4.0
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_sharpe_ratio_with_zero_risk_free_rate() -> None:
    values = [
        0.01,
        0.02,
        -0.01,
        0.00,
    ]

    returns = make_returns(
        values
    )

    result = sharpe_ratio(
        returns,
        periods_per_year=12,
    )

    expected = (
        np.mean(values)
        / np.std(
            values,
            ddof=1,
        )
        * math.sqrt(
            12
        )
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_sharpe_ratio_is_nan_when_volatility_is_zero() -> None:
    returns = make_returns(
        [
            0.01,
            0.01,
            0.01,
        ]
    )

    result = sharpe_ratio(
        returns
    )

    assert math.isnan(
        result["A"]
    )


def test_sortino_ratio_is_exact_with_zero_mar() -> None:
    values = [
        0.02,
        -0.01,
        -0.03,
        0.04,
    ]

    returns = make_returns(
        values
    )

    downside = math.sqrt(
        (
            0.00**2
            + 0.01**2
            + 0.03**2
            + 0.00**2
        )
        / 4.0
    )

    expected = (
        np.mean(values)
        / downside
    )

    result = sortino_ratio(
        returns,
        periods_per_year=1,
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_drawdown_from_prices_is_exact() -> None:
    prices = make_prices(
        [
            100.0,
            120.0,
            90.0,
            108.0,
            130.0,
        ]
    )

    drawdown = drawdown_from_prices(
        prices
    )

    assert drawdown["A"].tolist() == pytest.approx(
        [
            0.00,
            0.00,
            -0.25,
            -0.10,
            0.00,
        ]
    )

    maximum = maximum_drawdown_from_prices(
        prices
    )

    assert maximum["A"] == pytest.approx(
        -0.25
    )


def test_drawdown_from_returns_includes_initial_capital_peak() -> None:
    returns = make_returns(
        [
            -0.10,
            0.20,
        ]
    )

    drawdown = drawdown_from_returns(
        returns
    )

    assert drawdown["A"].iloc[0] == pytest.approx(
        -0.10
    )

    assert drawdown["A"].iloc[1] == pytest.approx(
        0.00
    )


def test_historical_var_uses_linear_quantile() -> None:
    values = [
        -0.05,
        -0.02,
        0.00,
        0.01,
        0.03,
    ]

    returns = make_returns(
        values
    )

    expected = -pd.Series(
        values
    ).quantile(
        0.20,
        interpolation="linear",
    )

    result = historical_var(
        returns,
        confidence_level=0.80,
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_expected_shortfall_with_integer_tail_mass() -> None:
    returns = make_returns(
        [
            -0.10,
            -0.05,
            0.00,
            0.01,
            0.02,
        ]
    )

    result = historical_expected_shortfall(
        returns,
        confidence_level=0.60,
    )

    assert result["A"] == pytest.approx(
        0.075
    )

    cvar = historical_cvar(
        returns,
        confidence_level=0.60,
    )

    assert cvar["A"] == pytest.approx(
        0.075
    )


def test_expected_shortfall_with_fractional_tail_mass() -> None:
    returns = make_returns(
        [
            -0.10,
            -0.05,
            0.00,
            0.01,
        ]
    )

    result = historical_expected_shortfall(
        returns,
        confidence_level=0.625,
    )

    expected = (
        -(
            -0.10
            + 0.5 * -0.05
        )
        / 1.5
    )

    assert result["A"] == pytest.approx(
        expected
    )


def test_covariance_and_correlation_are_exact() -> None:
    first = [
        0.01,
        0.02,
        -0.01,
        0.00,
    ]

    second = [
        0.02,
        0.04,
        -0.02,
        0.00,
    ]

    returns = make_returns(
        first,
        second,
    )

    covariance = annualized_covariance_matrix(
        returns,
        periods_per_year=12,
    )

    expected_covariance = (
        np.cov(
            np.array(
                [
                    first,
                    second,
                ]
            ),
            ddof=1,
        )
        * 12
    )

    np.testing.assert_allclose(
        covariance.to_numpy(),
        expected_covariance,
    )

    correlation = correlation_matrix(
        returns
    )

    assert correlation.loc[
        "A",
        "B",
    ] == pytest.approx(
        1.0
    )


def test_summary_statistics_has_expected_structure() -> None:
    prices = make_prices(
        [
            100.0,
            101.0,
            99.0,
            103.0,
            104.0,
        ],
        [
            100.0,
            99.0,
            100.0,
            98.0,
            101.0,
        ],
    )

    result = summary_statistics(
        prices,
        periods_per_year=252,
        confidence_level=0.95,
    )

    expected_columns = {
        "observations",
        "total_return",
        "cagr",
        "annualized_mean_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "historical_var_0.95_daily",
        "historical_cvar_0.95_daily",
        "worst_daily_return",
        "best_daily_return",
        "skewness",
        "excess_kurtosis",
    }

    assert list(
        result.index
    ) == [
        "A",
        "B",
    ]

    assert (
        result.index.name
        == "ticker"
    )

    assert set(
        result.columns
    ) == expected_columns

    assert (
        result["observations"]
        == 4
    ).all()


def test_invalid_confidence_level_is_rejected() -> None:
    returns = make_returns(
        [
            0.01,
            0.02,
            -0.01,
        ]
    )

    with pytest.raises(
        ValueError,
        match="strictly between",
    ):
        historical_var(
            returns,
            confidence_level=1.0,
        )