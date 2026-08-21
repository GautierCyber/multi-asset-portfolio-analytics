
import math

import numpy as np
import pandas as pd
import pytest

from multi_asset_portfolio.portfolio import (
    PortfolioConstraints,
    PortfolioConstructionError,
    component_risk_contribution,
    equal_risk_contribution_weights,
    equal_weight_weights,
    estimate_annualized_moments,
    global_minimum_variance_weights,
    marginal_risk_contribution,
    maximum_sharpe_weights,
    percentage_risk_contribution,
    portfolio_expected_return,
    portfolio_sharpe_ratio,
    portfolio_variance,
    portfolio_volatility,
    risk_budgeting_weights,
    validate_covariance_matrix,
    validate_expected_returns,
    validate_weights,
)


def diagonal_covariance() -> pd.DataFrame:
    assets = pd.Index(["A", "B", "C"])
    return pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]),
        index=assets,
        columns=assets,
    )


def expected_returns() -> pd.Series:
    return pd.Series(
        [0.08, 0.12, 0.10],
        index=pd.Index(["A", "B", "C"]),
        dtype=float,
    )


def test_constraints_reject_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        PortfolioConstraints(lower_bound=0.5, upper_bound=0.1)


def test_infeasible_bounds_are_rejected() -> None:
    covariance = diagonal_covariance()
    constraints = PortfolioConstraints(lower_bound=0.4, upper_bound=1.0)
    with pytest.raises(PortfolioConstructionError, match="infeasible"):
        global_minimum_variance_weights(
            covariance,
            constraints=constraints,
        )


def test_non_symmetric_covariance_is_rejected() -> None:
    covariance = diagonal_covariance()
    covariance.loc["A", "B"] = 0.01
    with pytest.raises(PortfolioConstructionError, match="symmetric"):
        validate_covariance_matrix(covariance)


def test_non_psd_covariance_is_rejected() -> None:
    covariance = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    with pytest.raises(
        PortfolioConstructionError,
        match="positive semidefinite",
    ):
        validate_covariance_matrix(covariance)


def test_expected_return_labels_must_match() -> None:
    covariance = diagonal_covariance()
    returns = expected_returns().reindex(["B", "A", "C"])
    with pytest.raises(PortfolioConstructionError, match="exactly match"):
        validate_expected_returns(
            returns,
            assets=covariance.index,
        )


def test_weights_must_be_fully_invested() -> None:
    assets = diagonal_covariance().index
    weights = pd.Series([0.2, 0.2, 0.2], index=assets)
    with pytest.raises(PortfolioConstructionError, match="sum to 1"):
        validate_weights(weights, assets=assets)


def test_equal_weight_is_exact() -> None:
    assets = diagonal_covariance().index
    weights = equal_weight_weights(assets)
    np.testing.assert_allclose(
        weights.to_numpy(),
        np.full(3, 1.0 / 3.0),
    )
    assert weights.sum() == pytest.approx(1.0)


def test_estimate_annualized_moments_is_exact() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.00],
            "B": [0.02, 0.04, -0.02, 0.00],
        },
        index=index,
    )
    means, covariance = estimate_annualized_moments(
        returns,
        periods_per_year=12,
    )
    np.testing.assert_allclose(
        means.to_numpy(),
        returns.mean().to_numpy() * 12,
    )
    np.testing.assert_allclose(
        covariance.to_numpy(),
        returns.cov().to_numpy() * 12,
    )


def test_gmv_matches_inverse_variance_solution_for_diagonal_covariance() -> None:
    covariance = diagonal_covariance()
    weights = global_minimum_variance_weights(covariance)

    inverse_variances = 1.0 / np.diag(covariance.to_numpy())
    expected = inverse_variances / inverse_variances.sum()

    np.testing.assert_allclose(
        weights.to_numpy(),
        expected,
        atol=1e-6,
    )


def test_gmv_respects_upper_bound() -> None:
    covariance = diagonal_covariance()
    constraints = PortfolioConstraints(lower_bound=0.0, upper_bound=0.5)
    weights = global_minimum_variance_weights(
        covariance,
        constraints=constraints,
    )
    assert weights.max() <= 0.5 + 1e-8
    assert weights.sum() == pytest.approx(1.0)


def test_maximum_sharpe_matches_tangency_solution_for_diagonal_covariance() -> None:
    covariance = diagonal_covariance()
    means = expected_returns()
    weights = maximum_sharpe_weights(
        means,
        covariance,
        risk_free_rate=0.0,
    )

    raw = means.to_numpy() / np.diag(covariance.to_numpy())
    expected = raw / raw.sum()

    np.testing.assert_allclose(
        weights.to_numpy(),
        expected,
        atol=1e-6,
    )


def test_risk_budgeting_matches_target_contributions() -> None:
    covariance = diagonal_covariance()
    budgets = pd.Series(
        [0.50, 0.30, 0.20],
        index=covariance.index,
    )
    weights = risk_budgeting_weights(covariance, budgets)
    contributions = percentage_risk_contribution(
        weights,
        covariance,
    )
    np.testing.assert_allclose(
        contributions.to_numpy(),
        budgets.to_numpy(),
        atol=1e-6,
    )


def test_invalid_risk_budgets_are_rejected() -> None:
    covariance = diagonal_covariance()
    budgets = pd.Series(
        [0.50, 0.30, 0.30],
        index=covariance.index,
    )
    with pytest.raises(PortfolioConstructionError, match="sum to 1"):
        risk_budgeting_weights(covariance, budgets)


def test_erc_equalizes_percentage_risk_contributions() -> None:
    covariance = diagonal_covariance()
    weights = equal_risk_contribution_weights(covariance)
    contributions = percentage_risk_contribution(
        weights,
        covariance,
    )
    np.testing.assert_allclose(
        contributions.to_numpy(),
        np.full(3, 1.0 / 3.0),
        atol=1e-6,
    )


def test_erc_diagonal_solution_is_inverse_volatility() -> None:
    covariance = diagonal_covariance()
    weights = equal_risk_contribution_weights(covariance)

    inverse_volatilities = 1.0 / np.sqrt(
        np.diag(covariance.to_numpy())
    )
    expected = inverse_volatilities / inverse_volatilities.sum()

    np.testing.assert_allclose(
        weights.to_numpy(),
        expected,
        atol=1e-6,
    )


def test_portfolio_return_variance_and_volatility_are_exact() -> None:
    covariance = diagonal_covariance()
    means = expected_returns()
    weights = pd.Series(
        [0.5, 0.3, 0.2],
        index=covariance.index,
    )

    expected_return = float(weights @ means)
    expected_variance = float(
        weights.to_numpy()
        @ covariance.to_numpy()
        @ weights.to_numpy()
    )

    assert portfolio_expected_return(
        weights,
        means,
    ) == pytest.approx(expected_return)

    assert portfolio_variance(
        weights,
        covariance,
    ) == pytest.approx(expected_variance)

    assert portfolio_volatility(
        weights,
        covariance,
    ) == pytest.approx(math.sqrt(expected_variance))


def test_zero_rate_portfolio_sharpe_is_return_over_volatility() -> None:
    covariance = diagonal_covariance()
    means = expected_returns()
    weights = equal_weight_weights(covariance.index)

    result = portfolio_sharpe_ratio(
        weights,
        means,
        covariance,
        risk_free_rate=0.0,
    )

    expected = (
        portfolio_expected_return(weights, means)
        / portfolio_volatility(weights, covariance)
    )

    assert result == pytest.approx(expected)


def test_component_risk_contributions_sum_to_volatility() -> None:
    covariance = diagonal_covariance()
    weights = pd.Series(
        [0.5, 0.3, 0.2],
        index=covariance.index,
    )

    marginal = marginal_risk_contribution(
        weights,
        covariance,
    )
    component = component_risk_contribution(
        weights,
        covariance,
    )
    volatility = portfolio_volatility(
        weights,
        covariance,
    )

    np.testing.assert_allclose(
        component.to_numpy(),
        weights.to_numpy() * marginal.to_numpy(),
    )
    assert component.sum() == pytest.approx(volatility)


def test_percentage_risk_contributions_sum_to_one() -> None:
    covariance = diagonal_covariance()
    weights = pd.Series(
        [0.5, 0.3, 0.2],
        index=covariance.index,
    )
    contributions = percentage_risk_contribution(
        weights,
        covariance,
    )
    assert contributions.sum() == pytest.approx(1.0)


def test_optimizer_handles_positive_lower_bound() -> None:
    covariance = diagonal_covariance()
    constraints = PortfolioConstraints(lower_bound=0.2, upper_bound=0.8)
    weights = global_minimum_variance_weights(
        covariance,
        constraints=constraints,
    )
    assert weights.min() >= 0.2 - 1e-8
    assert weights.max() <= 0.8 + 1e-8
    assert weights.sum() == pytest.approx(1.0)


def test_risk_budgeting_rejects_short_selling_bounds() -> None:
    covariance = diagonal_covariance()
    budgets = pd.Series(
        [1.0 / 3.0] * 3,
        index=covariance.index,
    )
    constraints = PortfolioConstraints(lower_bound=-0.2, upper_bound=1.2)
    with pytest.raises(
        PortfolioConstructionError,
        match="non-negative",
    ):
        risk_budgeting_weights(
            covariance,
            budgets,
            constraints=constraints,
        )
