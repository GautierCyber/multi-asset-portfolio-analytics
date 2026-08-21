
"""Portfolio construction and risk-contribution utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, OptimizeResult, minimize

from .analytics import (
    annual_rate_to_period_rate,
    annualized_covariance_matrix,
    annualized_mean_return,
)
from .returns import TRADING_DAYS_PER_YEAR, validate_return_panel


WEIGHT_TOLERANCE = 1e-8
COVARIANCE_SYMMETRY_TOLERANCE = 1e-10
COVARIANCE_PSD_TOLERANCE = 1e-10
OPTIMIZER_FTOL = 1e-12
OPTIMIZER_MAX_ITERATIONS = 2_000


class PortfolioConstructionError(ValueError):
    """Raised when portfolio inputs, constraints or optimisation fail."""


@dataclass(frozen=True)
class PortfolioConstraints:
    """Uniform box bounds for a fully invested portfolio."""

    lower_bound: float = 0.0
    upper_bound: float = 1.0

    def __post_init__(self) -> None:
        lower = _finite_real(self.lower_bound, name="lower_bound")
        upper = _finite_real(self.upper_bound, name="upper_bound")
        if lower > upper:
            raise ValueError(
                "lower_bound must be less than or equal to upper_bound."
            )


def _finite_real(value: Real, *, name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _asset_index(values: pd.Index, *, context: str) -> pd.Index:
    assets = values if isinstance(values, pd.Index) else pd.Index(values)
    if len(assets) == 0:
        raise PortfolioConstructionError(
            f"{context} must contain at least one asset."
        )
    if assets.has_duplicates:
        raise PortfolioConstructionError(
            f"{context} contains duplicate asset labels."
        )
    if any(not isinstance(asset, str) for asset in assets):
        raise PortfolioConstructionError(
            f"{context} asset labels must be strings."
        )
    return assets


def _constraint_bounds(
    constraints: PortfolioConstraints,
    number_of_assets: int,
) -> tuple[float, float]:
    if not isinstance(constraints, PortfolioConstraints):
        raise TypeError(
            "constraints must be a PortfolioConstraints instance."
        )

    lower = float(constraints.lower_bound)
    upper = float(constraints.upper_bound)

    if (
        number_of_assets * lower > 1.0 + WEIGHT_TOLERANCE
        or number_of_assets * upper < 1.0 - WEIGHT_TOLERANCE
    ):
        raise PortfolioConstructionError(
            "Weight bounds are infeasible for a fully invested portfolio: "
            f"{number_of_assets} assets with bounds "
            f"[{lower:.6g}, {upper:.6g}] cannot sum to 1."
        )
    return lower, upper


def validate_covariance_matrix(covariance: pd.DataFrame) -> pd.DataFrame:
    """Validate a finite symmetric positive-semidefinite covariance matrix."""
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame.")
    if covariance.empty:
        raise PortfolioConstructionError("Covariance matrix is empty.")
    if covariance.shape[0] != covariance.shape[1]:
        raise PortfolioConstructionError("Covariance matrix must be square.")

    assets = _asset_index(covariance.index, context="Covariance matrix")
    if not covariance.columns.equals(assets):
        raise PortfolioConstructionError(
            "Covariance matrix columns must exactly match its index "
            "in the same order."
        )

    numeric = covariance.apply(pd.to_numeric, errors="coerce")
    if (covariance.notna() & numeric.isna()).any().any():
        raise PortfolioConstructionError(
            "Covariance matrix contains non-numeric values."
        )
    if numeric.isna().any().any():
        raise PortfolioConstructionError(
            "Covariance matrix contains missing values."
        )

    values = numeric.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise PortfolioConstructionError(
            "Covariance matrix contains non-finite values."
        )
    if not np.allclose(
        values,
        values.T,
        rtol=0.0,
        atol=COVARIANCE_SYMMETRY_TOLERANCE,
    ):
        raise PortfolioConstructionError(
            "Covariance matrix must be symmetric."
        )
    if (np.diag(values) <= 0.0).any():
        raise PortfolioConstructionError(
            "Covariance matrix diagonal must be strictly positive."
        )

    eigenvalues = np.linalg.eigvalsh(values)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if eigenvalues.min() < -COVARIANCE_PSD_TOLERANCE * scale:
        raise PortfolioConstructionError(
            "Covariance matrix must be positive semidefinite."
        )
    return numeric.astype(float)


def validate_expected_returns(
    expected_returns: pd.Series,
    *,
    assets: pd.Index,
) -> pd.Series:
    """Validate expected returns against an exact asset ordering."""
    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a pandas Series.")

    asset_index = _asset_index(assets, context="Expected-return reference")
    if not expected_returns.index.equals(asset_index):
        raise PortfolioConstructionError(
            "Expected-return index must exactly match the asset index "
            "in the same order."
        )

    numeric = pd.to_numeric(expected_returns, errors="coerce")
    if (expected_returns.notna() & numeric.isna()).any():
        raise PortfolioConstructionError(
            "Expected returns contain non-numeric values."
        )
    if numeric.isna().any() or not np.isfinite(
        numeric.to_numpy(dtype=float, copy=False)
    ).all():
        raise PortfolioConstructionError(
            "Expected returns contain missing or non-finite values."
        )
    return numeric.astype(float)


def validate_weights(
    weights: pd.Series,
    *,
    assets: pd.Index,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Validate finite labelled fully invested portfolio weights."""
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series.")

    asset_index = _asset_index(assets, context="Weight reference")
    if not weights.index.equals(asset_index):
        raise PortfolioConstructionError(
            "Weight index must exactly match the asset index "
            "in the same order."
        )

    lower, upper = _constraint_bounds(constraints, len(asset_index))
    numeric = pd.to_numeric(weights, errors="coerce")
    if (weights.notna() & numeric.isna()).any():
        raise PortfolioConstructionError("Weights contain non-numeric values.")

    values = numeric.to_numpy(dtype=float, copy=False)
    if numeric.isna().any() or not np.isfinite(values).all():
        raise PortfolioConstructionError(
            "Weights contain missing or non-finite values."
        )
    if (values < lower - WEIGHT_TOLERANCE).any():
        raise PortfolioConstructionError(
            "At least one weight is below the lower bound."
        )
    if (values > upper + WEIGHT_TOLERANCE).any():
        raise PortfolioConstructionError(
            "At least one weight is above the upper bound."
        )
    if not math.isclose(
        float(values.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=WEIGHT_TOLERANCE,
    ):
        raise PortfolioConstructionError(
            "Portfolio weights must sum to 1."
        )
    return numeric.astype(float)


def estimate_annualized_moments(
    returns: pd.DataFrame,
    *,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate annualised arithmetic means and sample covariance."""
    validated_returns = validate_return_panel(returns, min_observations=2)
    expected_returns = annualized_mean_return(
        validated_returns,
        periods_per_year=periods_per_year,
    )
    covariance = validate_covariance_matrix(
        annualized_covariance_matrix(
            validated_returns,
            periods_per_year=periods_per_year,
        )
    )
    expected_returns = validate_expected_returns(
        expected_returns,
        assets=covariance.index,
    )
    return expected_returns, covariance


def equal_weight_weights(
    assets: pd.Index,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Return a fully invested equal-weight allocation."""
    asset_index = _asset_index(
        pd.Index(assets),
        context="Equal-weight universe",
    )
    lower, upper = _constraint_bounds(constraints, len(asset_index))
    weight = 1.0 / len(asset_index)

    if (
        weight < lower - WEIGHT_TOLERANCE
        or weight > upper + WEIGHT_TOLERANCE
    ):
        raise PortfolioConstructionError(
            "Equal weighting violates the supplied weight bounds."
        )

    result = pd.Series(
        weight,
        index=asset_index,
        name="equal_weight",
        dtype=float,
    )
    return validate_weights(
        result,
        assets=asset_index,
        constraints=constraints,
    )


def _tilted_start(
    base: np.ndarray,
    *,
    target_index: int,
    target_value: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    candidate = base.copy()
    difference = target_value - candidate[target_index]
    candidate[target_index] = target_value
    others = [i for i in range(len(candidate)) if i != target_index]

    if difference > 0.0:
        remaining = difference
        for index in others:
            adjustment = min(candidate[index] - lower, remaining)
            candidate[index] -= adjustment
            remaining -= adjustment
            if remaining <= WEIGHT_TOLERANCE:
                break
    elif difference < 0.0:
        remaining = -difference
        for index in others:
            adjustment = min(upper - candidate[index], remaining)
            candidate[index] += adjustment
            remaining -= adjustment
            if remaining <= WEIGHT_TOLERANCE:
                break

    if not math.isclose(
        float(candidate.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=WEIGHT_TOLERANCE,
    ):
        raise PortfolioConstructionError(
            "Could not construct a feasible optimiser starting point."
        )
    return candidate


def _starting_points(
    number_of_assets: int,
    *,
    lower: float,
    upper: float,
) -> list[np.ndarray]:
    base = np.full(number_of_assets, 1.0 / number_of_assets, dtype=float)
    starts = [base]

    feasible_target_lower = max(
        lower,
        1.0 - (number_of_assets - 1) * upper,
    )
    feasible_target_upper = min(
        upper,
        1.0 - (number_of_assets - 1) * lower,
    )

    for target_index in range(number_of_assets):
        for target_value in (feasible_target_lower, feasible_target_upper):
            candidate = _tilted_start(
                base,
                target_index=target_index,
                target_value=target_value,
                lower=lower,
                upper=upper,
            )
            if not any(
                np.allclose(
                    candidate,
                    existing,
                    rtol=0.0,
                    atol=WEIGHT_TOLERANCE,
                )
                for existing in starts
            ):
                starts.append(candidate)
    return starts


def _solution_is_feasible(
    weights: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> bool:
    return (
        np.isfinite(weights).all()
        and not (weights < lower - WEIGHT_TOLERANCE).any()
        and not (weights > upper + WEIGHT_TOLERANCE).any()
        and math.isclose(
            float(weights.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_TOLERANCE,
        )
    )


def _solve_slsqp(
    objective: Callable[[np.ndarray], float],
    *,
    assets: pd.Index,
    constraints: PortfolioConstraints,
) -> pd.Series:
    lower, upper = _constraint_bounds(constraints, len(assets))
    bounds = Bounds(
        np.full(len(assets), lower),
        np.full(len(assets), upper),
    )
    equality = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
    }

    successes: list[OptimizeResult] = []
    failures: list[str] = []

    for start in _starting_points(
        len(assets),
        lower=lower,
        upper=upper,
    ):
        result = minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=(equality,),
            options={
                "ftol": OPTIMIZER_FTOL,
                "maxiter": OPTIMIZER_MAX_ITERATIONS,
                "disp": False,
            },
        )
        if (
            result.success
            and math.isfinite(float(result.fun))
            and _solution_is_feasible(
                result.x,
                lower=lower,
                upper=upper,
            )
        ):
            successes.append(result)
        else:
            failures.append(str(result.message))

    if not successes:
        details = "; ".join(dict.fromkeys(failures))
        raise PortfolioConstructionError(
            "Portfolio optimisation failed: "
            f"{details or 'no feasible finite solution was returned'}."
        )

    best = min(successes, key=lambda result: float(result.fun))
    weights = pd.Series(best.x, index=assets, dtype=float)
    return validate_weights(
        weights,
        assets=assets,
        constraints=constraints,
    )


def global_minimum_variance_weights(
    covariance: pd.DataFrame,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Optimise a bounded fully invested global minimum-variance portfolio."""
    covariance = validate_covariance_matrix(covariance)
    matrix = covariance.to_numpy(dtype=float, copy=False)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ matrix @ weights)

    result = _solve_slsqp(
        objective,
        assets=covariance.index,
        constraints=constraints,
    )
    result.name = "global_minimum_variance"
    return result


def _annualized_arithmetic_risk_free(
    risk_free_rate: Real,
    periods_per_year: Real,
) -> float:
    periods = _finite_real(periods_per_year, name="periods_per_year")
    if periods <= 0.0:
        raise ValueError("periods_per_year must be strictly positive.")

    risk_free = _finite_real(risk_free_rate, name="risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -100%.")

    periodic = annual_rate_to_period_rate(
        risk_free,
        periods_per_year=periods,
    )
    return periodic * periods


def maximum_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: Real = 0.0,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Optimise the bounded fully invested ex-ante Sharpe ratio."""
    covariance = validate_covariance_matrix(covariance)
    expected_returns = validate_expected_returns(
        expected_returns,
        assets=covariance.index,
    )
    risk_free = _annualized_arithmetic_risk_free(
        risk_free_rate,
        periods_per_year,
    )
    means = expected_returns.to_numpy(dtype=float, copy=False)
    matrix = covariance.to_numpy(dtype=float, copy=False)

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ matrix @ weights)
        if variance <= 0.0:
            return float("inf")
        return -float(
            (weights @ means - risk_free) / math.sqrt(variance)
        )

    result = _solve_slsqp(
        objective,
        assets=covariance.index,
        constraints=constraints,
    )
    result.name = "maximum_sharpe"
    return result


def _validate_risk_budgets(
    risk_budgets: pd.Series,
    *,
    assets: pd.Index,
) -> pd.Series:
    if not isinstance(risk_budgets, pd.Series):
        raise TypeError("risk_budgets must be a pandas Series.")
    if not risk_budgets.index.equals(assets):
        raise PortfolioConstructionError(
            "Risk-budget index must exactly match the asset index "
            "in the same order."
        )

    numeric = pd.to_numeric(risk_budgets, errors="coerce")
    values = numeric.to_numpy(dtype=float, copy=False)
    if (
        numeric.isna().any()
        or not np.isfinite(values).all()
        or (values <= 0.0).any()
    ):
        raise PortfolioConstructionError(
            "Risk budgets must be finite, numeric and strictly positive."
        )
    if not math.isclose(
        float(values.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=WEIGHT_TOLERANCE,
    ):
        raise PortfolioConstructionError("Risk budgets must sum to 1.")
    return numeric.astype(float)


def risk_budgeting_weights(
    covariance: pd.DataFrame,
    risk_budgets: pd.Series,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Optimise weights toward target percentage volatility contributions."""
    if constraints.lower_bound < 0.0:
        raise PortfolioConstructionError(
            "Risk budgeting requires non-negative weight bounds."
        )

    covariance = validate_covariance_matrix(covariance)
    budgets = _validate_risk_budgets(
        risk_budgets,
        assets=covariance.index,
    )
    matrix = covariance.to_numpy(dtype=float, copy=False)
    target = budgets.to_numpy(dtype=float, copy=False)

    def objective(weights: np.ndarray) -> float:
        marginal_variance = matrix @ weights
        variance = float(weights @ marginal_variance)
        if variance <= 0.0:
            return float("inf")
        contributions = weights * marginal_variance / variance
        residuals = contributions - target
        return float(residuals @ residuals)

    result = _solve_slsqp(
        objective,
        assets=covariance.index,
        constraints=constraints,
    )
    result.name = "risk_budgeting"
    return result


def equal_risk_contribution_weights(
    covariance: pd.DataFrame,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Optimise an Equal Risk Contribution / Risk Parity portfolio."""
    covariance = validate_covariance_matrix(covariance)
    budgets = pd.Series(
        1.0 / len(covariance),
        index=covariance.index,
        dtype=float,
    )
    result = risk_budgeting_weights(
        covariance,
        budgets,
        constraints=constraints,
    )
    result.name = "equal_risk_contribution"
    return result


def portfolio_expected_return(
    weights: pd.Series,
    expected_returns: pd.Series,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> float:
    """Calculate annualised expected portfolio return."""
    assets = _asset_index(expected_returns.index, context="Expected returns")
    expected_returns = validate_expected_returns(
        expected_returns,
        assets=assets,
    )
    weights = validate_weights(
        weights,
        assets=assets,
        constraints=constraints,
    )
    return float(
        weights.to_numpy(dtype=float, copy=False)
        @ expected_returns.to_numpy(dtype=float, copy=False)
    )


def portfolio_variance(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> float:
    """Calculate annualised portfolio variance."""
    covariance = validate_covariance_matrix(covariance)
    weights = validate_weights(
        weights,
        assets=covariance.index,
        constraints=constraints,
    )
    w = weights.to_numpy(dtype=float, copy=False)
    matrix = covariance.to_numpy(dtype=float, copy=False)
    variance = float(w @ matrix @ w)

    if variance < -COVARIANCE_PSD_TOLERANCE:
        raise PortfolioConstructionError(
            "Calculated portfolio variance is negative."
        )
    return max(variance, 0.0)


def portfolio_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> float:
    """Calculate annualised portfolio volatility."""
    return math.sqrt(
        portfolio_variance(
            weights,
            covariance,
            constraints=constraints,
        )
    )


def portfolio_sharpe_ratio(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: Real = 0.0,
    periods_per_year: Real = TRADING_DAYS_PER_YEAR,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> float:
    """Calculate ex-ante annualised Sharpe ratio for annualised moments."""
    covariance = validate_covariance_matrix(covariance)
    expected_returns = validate_expected_returns(
        expected_returns,
        assets=covariance.index,
    )
    weights = validate_weights(
        weights,
        assets=covariance.index,
        constraints=constraints,
    )
    risk_free = _annualized_arithmetic_risk_free(
        risk_free_rate,
        periods_per_year,
    )
    expected = portfolio_expected_return(
        weights,
        expected_returns,
        constraints=constraints,
    )
    volatility = portfolio_volatility(
        weights,
        covariance,
        constraints=constraints,
    )
    if volatility <= 0.0:
        return float("nan")
    return (expected - risk_free) / volatility


def marginal_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Calculate marginal contributions to portfolio volatility."""
    covariance = validate_covariance_matrix(covariance)
    weights = validate_weights(
        weights,
        assets=covariance.index,
        constraints=constraints,
    )
    volatility = portfolio_volatility(
        weights,
        covariance,
        constraints=constraints,
    )
    if volatility <= 0.0:
        raise PortfolioConstructionError(
            "Risk contributions are undefined for zero-volatility portfolios."
        )

    marginal = (
        covariance.to_numpy(dtype=float, copy=False)
        @ weights.to_numpy(dtype=float, copy=False)
        / volatility
    )
    return pd.Series(
        marginal,
        index=covariance.index,
        name="marginal_risk_contribution",
        dtype=float,
    )


def component_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Calculate additive component contributions to portfolio volatility."""
    covariance = validate_covariance_matrix(covariance)
    weights = validate_weights(
        weights,
        assets=covariance.index,
        constraints=constraints,
    )
    result = weights * marginal_risk_contribution(
        weights,
        covariance,
        constraints=constraints,
    )
    result.name = "component_risk_contribution"
    return result


def percentage_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> pd.Series:
    """Calculate each asset's share of total portfolio volatility."""
    covariance = validate_covariance_matrix(covariance)
    weights = validate_weights(
        weights,
        assets=covariance.index,
        constraints=constraints,
    )
    volatility = portfolio_volatility(
        weights,
        covariance,
        constraints=constraints,
    )
    if volatility <= 0.0:
        raise PortfolioConstructionError(
            "Percentage risk contributions are undefined "
            "for zero-volatility portfolios."
        )

    result = component_risk_contribution(
        weights,
        covariance,
        constraints=constraints,
    ) / volatility
    result.name = "percentage_risk_contribution"
    return result
