"""Portfolio optimizer.

Mixed-integer linear programming (MILP) portfolio optimiser with pluggable
solver backends: ``scipy`` (HiGHS via ``scipy.optimize.milp``) and ``mosek``
(MOSEK Fusion API).  The formulation maximises expected *active* return
(net of transaction costs) subject to fully-invested, position-count,
trade-count, ticker-count, and attribute-exposure constraints.

Objective::

    max  (μ - c - ε)ᵀ t⁺  -  μᵀ t⁻

Decision variables (stacked vector ``x`` of length 5n + K)::

    w[i]   - continuous weight for asset i              (indices 0   … n-1)
    ρ[i]   - binary indicator: 1 if asset i is held     (indices n   … 2n-1)
    t⁺[i]  - buy-trade variable:  (w_i - h_i)⁺          (indices 2n  … 3n-1)
    t⁻[i]  - sell-trade variable: (h_i - w_i)⁺          (indices 3n  … 4n-1)
    τ[i]   - binary indicator: 1 if asset i is traded   (indices 4n  … 5n-1)
    π[k]   - binary indicator: 1 if ticker k is held    (indices 5n  … 5n+K-1)

Notation::

    μ_i   - i-th bond's gross expected return
    w_i   - target portfolio weight for i-th bond
    M     - mask matrix indicating which group of a category bond i belongs to
    D     - duration vector
    S     - spread (OAS) vector
    h_i   - initial portfolio holding weight for i-th bond
    t⁺_i  - traded weight of buy order: (w_i - h_i)⁺
    t⁻_i  - traded weight of sell order (positivised): (h_i - w_i)⁺
    ρ_i   - binary indicator of bond position
    b_i   - benchmark weight for i-th bond
    c_i   - bid/ask cost for i-th bond
    ε_i   - estimation error in gross expected return for i-th bond
    τ_i   - binary indicator of trade
    π_k   - binary indicator of ticker position for k-th ticker
    M_k   - match matrix between k-th ticker and i-th bond
    ω_k   - aggregate weight in k-th ticker: Σ_{i∈M_k} w_i
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from archipelago.portfolio.constraints import (
    AttributeConstraint,
    ConstraintViolation,
    EntityCapPolicy,
    EntityWeightCapRule,
    MILPOptimizerConfig,
    Multiplier,
    OptimizerAttempt,
    Tolerance,
    build_attribute_constraints,
    build_milp_constraints,
    diagnose_portfolio_feasibility,
    make_attribute_value_mask,
)
from archipelago.portfolio.solver_base import SolverResult, solve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-screening
# ---------------------------------------------------------------------------


def shrink_buy_candidates(
    mu: np.ndarray,
    costs: np.ndarray,
    estimation_error: np.ndarray,
    holdings: np.ndarray,
    return_threshold: float | None = None,
) -> np.ndarray:
    """Identify bonds ineligible for purchase based on net return.

    Bonds not currently held (h_i = 0) whose net return ``μ_i - c_i - ε_i``
    falls below *return_threshold* are excluded from the buy candidate set.
    Excluded bonds have *w*, *t⁺*, and *τ* upper bounds fixed to zero by the
    caller; *t⁻* remains free so existing positions can still be sold.

    Parameters
    ----------
    mu : np.ndarray
        Gross expected return vector *μ* (length *n*).
    costs : np.ndarray
        Bid/ask cost per asset *c* (length *n*).
    estimation_error : np.ndarray
        Return estimation error per asset *ε* (length *n*).
    holdings : np.ndarray
        Current portfolio weight vector *h* (length *n*).
    return_threshold : float, optional
        Net-return floor.  Bonds with ``μ_i - c_i - ε_i < threshold``
        and ``h_i = 0`` are excluded.  Defaults to ``max(μ[h > 0])`` —
        the best gross return among current holdings.  Set to ``-np.inf``
        to disable screening entirely.

    Returns
    -------
    np.ndarray
        Boolean mask of length *n*.  ``True`` where the bond is excluded.
    """
    net_return = mu - costs - estimation_error
    held_mask = holdings > 0

    if return_threshold is None:
        if held_mask.any():
            return_threshold = float(mu[held_mask].max())
        else:
            return np.zeros(len(mu), dtype=bool)

    not_held = ~held_mask
    below_threshold = net_return < return_threshold
    exclude = not_held & below_threshold

    logger.debug(
        "shrink_buy_candidates: threshold=%.6f  excluded=%d / %d unheld bonds",
        return_threshold,
        int(exclude.sum()),
        int(not_held.sum()),
    )
    return exclude


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    """Container for optimisation output.

    Attributes
    ----------
    weights : np.ndarray
        Optimal portfolio weights *w* (length *n*).
    active : np.ndarray
        Active weights *w - b* (length *n*).
    positions : np.ndarray
        Binary position indicators *ρ* (length *n*); 1 where held.
    trades : np.ndarray
        Binary trade indicators *τ* (length *n*); 1 where traded.
    buy_trades : np.ndarray
        Buy-trade amounts *t⁺ = (w - h)⁺* (length *n*).
    sell_trades : np.ndarray
        Sell-trade amounts *t⁻ = (h - w)⁺* (length *n*).
    n_positions : int
        Number of held positions ``Σρ``.
    n_trades : int
        Number of traded assets ``Στ``.
    gross_return_pre : float
        Expected return of pre-trade portfolio ``μᵀh``.
    gross_return_post : float
        Expected return of post-trade portfolio ``μᵀw``.
    transaction_cost : float
        Pure bid/ask transaction cost on buys ``cᵀ t⁺``.
    estimation_error_cost : float
        Estimation-error cost on buys ``εᵀ t⁺``.
    sell_cost : float
        Sell opportunity cost ``μᵀ t⁻``.
    net_return : float
        Net expected return:
        ``gross_return_post - transaction_cost - estimation_error_cost - sell_cost``.
    avg_gross_return_sells : float
        Weighted-average gross return of sold bonds ``(μᵀ t⁻) / (1ᵀ t⁻)``.
    avg_net_return_buys : float
        Weighted-average net return of bought bonds, net of transaction
        cost only: ``((μ - c)ᵀ t⁺) / (1ᵀ t⁺)``.
    avg_net_return_buys_incl_ee : float
        Weighted-average net return of bought bonds, net of transaction
        cost and estimation error: ``((μ - c - ε)ᵀ t⁺) / (1ᵀ t⁺)``.
    success : bool
        Whether the solver converged.
    message : str
        Solver status message.
    n_tickers : int
        Number of distinct tickers held ``Σπ``.  Default 0.
    cash_weight : float
        Optimal scalar cash weight ``c`` (zero return, zero duration).
        ``weights.sum() + cash_weight ≈ 1`` always.  Default 0.0.
    """

    weights: np.ndarray
    active: np.ndarray
    positions: np.ndarray
    trades: np.ndarray
    buy_trades: np.ndarray
    sell_trades: np.ndarray
    n_positions: int
    n_trades: int
    gross_return_pre: float
    gross_return_post: float
    transaction_cost: float
    estimation_error_cost: float
    sell_cost: float
    net_return: float
    avg_gross_return_sells: float
    avg_net_return_buys: float
    avg_net_return_buys_incl_ee: float
    success: bool
    message: str
    n_tickers: int = 0
    cash_weight: float = 0.0
    feasibility_violations: list[ConstraintViolation] | None = None
    attempt_index: int = 0
    attempt_label: str = "primary"


# ---------------------------------------------------------------------------
# Entity weight caps
# ---------------------------------------------------------------------------


def compute_entity_weight_caps(
    entity_signals: np.ndarray,
    holdings: np.ndarray,
    bm_weights: np.ndarray,
    rules: list[EntityWeightCapRule],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bond weight upper bounds from entity-level signal rules.

    Rules are evaluated in descending threshold order (strictest first).
    The first matching rule per bond determines its cap.

    Parameters
    ----------
    entity_signals : np.ndarray
        Per-bond signal values (length *n*).  ``NaN`` means no rule applies.
    holdings : np.ndarray
        Current portfolio weight vector *h* (length *n*).
    bm_weights : np.ndarray
        Benchmark weight vector *b* (length *n*).
    rules : list[EntityWeightCapRule]
        Threshold-based rules, applied in descending threshold order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(w_ub_cap, tp_ub_cap, restricted_mask)`` where:

        - ``w_ub_cap``: per-bond weight upper bound (``np.inf`` = no cap).
        - ``tp_ub_cap``: per-bond buy-trade upper bound (``np.inf`` = no cap).
        - ``restricted_mask``: boolean mask (True where any rule applies).
    """
    n = len(entity_signals)
    w_ub_cap = np.full(n, np.inf)
    tp_ub_cap = np.full(n, np.inf)
    restricted_mask = np.zeros(n, dtype=bool)

    # Sort rules by descending threshold (strictest first)
    sorted_rules = sorted(rules, key=lambda r: r.threshold, reverse=True)

    signals = np.asarray(entity_signals, dtype=float)

    for rule in sorted_rules:
        mask = signals >= rule.threshold
        # Only apply to bonds not already capped by a stricter rule
        new_mask = mask & ~restricted_mask

        if not new_mask.any():
            continue

        restricted_mask |= new_mask

        if rule.policy == EntityCapPolicy.ZERO_WEIGHT:
            w_ub_cap[new_mask] = 0.0
            tp_ub_cap[new_mask] = 0.0
        elif rule.policy == EntityCapPolicy.HOLD_WEIGHT:
            w_ub_cap[new_mask] = holdings[new_mask]
            tp_ub_cap[new_mask] = 0.0
        elif rule.policy == EntityCapPolicy.MARKET_WEIGHT:
            w_ub_cap[new_mask] = bm_weights[new_mask]
            tp_ub_cap[new_mask] = np.maximum(
                bm_weights[new_mask] - holdings[new_mask], 0.0,
            )

    n_restricted = int(restricted_mask.sum())
    if n_restricted > 0:
        n_zero = int((w_ub_cap == 0.0).sum())
        n_hold = int(((w_ub_cap > 0) & (tp_ub_cap == 0.0) & restricted_mask).sum())
        n_mkt = n_restricted - n_zero - n_hold
        logger.info(
            "Entity caps: %d restricted bonds (zero=%d, hold=%d, market_wt=%d)",
            n_restricted, n_zero, n_hold, n_mkt,
        )

    return w_ub_cap, tp_ub_cap, restricted_mask


# ---------------------------------------------------------------------------
# MILP optimizer
# ---------------------------------------------------------------------------


def milp_portfolio_opt(
    mu: np.ndarray,
    bm_weights: np.ndarray,
    config: MILPOptimizerConfig | None = None,
    holdings: np.ndarray | None = None,
    costs: np.ndarray | None = None,
    estimation_error: np.ndarray | None = None,
    return_threshold: float | None = None,
    return_threshold_percentile: float = 50.0,
    ticker_labels: np.ndarray | None = None,
    illiquid_mask: np.ndarray | None = None,
    entity_weight_caps: tuple[np.ndarray, np.ndarray] | None = None,
) -> PortfolioResult:
    """Maximise net active return via a mixed-integer linear programme.

    The formulation is::

        max   (μ - c - ε)ᵀ t⁺  -  μᵀ t⁻

        s.t.  1ᵀw + c = 1                                    (fully invested incl. cash)
              cash_min ≤ c ≤ cash_max                        (scalar cash bounds)
              l ≤ Mᵀ(w - b) ≤ u                              (WEIGHT exposure)
              l ≤ (M·D)ᵀ(w - b) ≤ u                          (OAD exposure)
              l ≤ (M·D·S)ᵀ(w - b) ≤ u                        (DTS exposure)
              wᵢ ∈ [0, w_max]
              w_min_i·ρᵢ ≤ wᵢ ≤ w_max·ρᵢ                     (position linking)
                  where w_min_i = min(w_min, hᵢ) if hᵢ > 0 else w_min
                  (per-bond floor grandfathers drifted holdings)
              ρᵢ ∈ {0, 1}
              ρ_L ≤ 1ᵀρ ≤ ρ_U                                (position count)
              t⁺ᵢ - t⁻ᵢ = wᵢ - hᵢ                            (trade balance)
              τᵢ ∈ {0, 1}
              t⁺ᵢ + t⁻ᵢ ≥ t_min · τᵢ                         (minimum trade size)
              t⁺ᵢ + t⁻ᵢ ≤ max(hᵢ, w_max - hᵢ) · τᵢ           (trade linking)
              τ_L ≤ 1ᵀτ ≤ τ_U                                (trade count)
              πₖ ∈ {0, 1}                                       
              ρᵢ ≤ π_{k(i)}                                  (bond→ticker)
              πₖ ≤ Σ_{i∈Iₖ} ρᵢ                                (ticker→bond)
              π_L ≤ 1ᵀπ ≤ π_U                                (ticker count)
              ωₖ := Σ_{i∈Mₖ} wᵢ
              ωₖ ≤ ω_max · πₖ                                 (ticker weight upper)
              ωₖ ≥ ω_min · πₖ                                 (ticker weight lower)

    Parameters
    ----------
    mu : np.ndarray
        Gross expected return vector *μ* (length *n*).
    bm_weights : np.ndarray
        Benchmark weight vector *b* (length *n*).
    config : MILPOptimizerConfig, optional
        Solver configuration.  Uses defaults when ``None``.
    holdings : np.ndarray, optional
        Current portfolio weight vector *h* (length *n*).
        Defaults to zeros (new portfolio — all trades are buys).
    costs : np.ndarray, optional
        Bid/ask cost per asset *c* (length *n*).  Defaults to zeros.
    estimation_error : np.ndarray, optional
        Return estimation error per asset *ε* (length *n*).  Defaults to zeros.
    return_threshold : float, optional
        Net-return floor for :func:`shrink_buy_candidates`.  Defaults to
        ``max(μ[h > 0])``.  Set to ``-np.inf`` to disable screening.
    ticker_labels : np.ndarray, optional
        Categorical vector of length *n* assigning each bond to a ticker.
        When provided, ticker-level binary indicators *πₖ* and associated
        constraints (10–14) are added to the formulation.

    Returns
    -------
    PortfolioResult
    """
    if config is None:
        config = MILPOptimizerConfig()

    mu = np.asarray(mu, dtype=float).ravel()
    bm_weights = np.asarray(bm_weights, dtype=float).ravel()
    n = mu.shape[0]

    # Default optional arrays to zeros
    holdings = np.zeros(n) if holdings is None else np.asarray(holdings, dtype=float).ravel()
    costs = np.zeros(n) if costs is None else np.asarray(costs, dtype=float).ravel()
    estimation_error = (
        np.zeros(n) if estimation_error is None
        else np.asarray(estimation_error, dtype=float).ravel()
    )

    _validate_inputs(mu, bm_weights, holdings, costs, estimation_error, config, n)

    raw_costs = costs.copy()  # pure bid/ask before merge
    costs = costs + estimation_error  # c + ε  (used in MILP objective)

    # -- ticker mapping -----------------------------------------------------
    if ticker_labels is not None:
        ticker_labels = np.asarray(ticker_labels)
        if ticker_labels.shape[0] != n:
            raise ValueError(
                f"ticker_labels length {ticker_labels.shape[0]} != n={n}"
            )
        unique_tickers, ticker_map = np.unique(ticker_labels, return_inverse=True)
        K = len(unique_tickers)
    else:
        ticker_map = None
        K = 0

    # -- pre-screen: exclude unattractive unheld bonds ----------------------
    if return_threshold is None and (holdings > 0).any():
        return_threshold = float(np.percentile(mu[holdings > 0], return_threshold_percentile))
    exclude = shrink_buy_candidates(
        mu, raw_costs, estimation_error, holdings, return_threshold,
    )

    # -- variable bounds -------------------------------------------
    w_ub = np.full(n, config.w_max)

    # Per-bond active overweight cap: w_i ≤ min(w_max, b_i + δ)
    if config.max_active_overweight is not None:
        w_ub = np.minimum(w_ub, bm_weights + config.max_active_overweight)
        # Ensure lower bound ≤ upper bound (w_ub ≥ 0)
        w_ub = np.maximum(w_ub, 0.0)

    tp_ub = np.full(n, config.w_max)

    # Block purchases for excluded bonds (t⁺=0).
    # w, ρ, t⁻, τ remain free so poor return positions can still be held or sold.
    tp_ub[exclude] = 0.0

    # Block purchases for illiquid bonds (liquidity model gate).
    if illiquid_mask is not None:
        tp_ub[illiquid_mask] = 0.0

    # Apply entity-level weight caps (structural credit signal).
    if entity_weight_caps is not None:
        ewc_w_ub, ewc_tp_ub = entity_weight_caps
        w_ub = np.minimum(w_ub, ewc_w_ub)
        w_ub = np.maximum(w_ub, 0.0)
        tp_ub = np.minimum(tp_ub, ewc_tp_ub)
        tp_ub = np.maximum(tp_ub, 0.0)

    # -- dispatch to solver backend -----------------------------------------
    solver_result = solve(
        n=n, K=K, mu=mu, costs=costs, holdings=holdings,
        config=config, w_ub=w_ub, tp_ub=tp_ub,
        exclude=exclude, ticker_map=ticker_map,
    )

    if not solver_result.success:
        logger.warning("Solver failed: %s", solver_result.message)
        violations = diagnose_portfolio_feasibility(
            holdings, config.attribute_constraints,
        )
        if violations:
            logger.warning(
                "Pre-trade holdings violate %d attribute constraint(s): %s",
                len(violations),
                ", ".join(
                    f"{v.name} (lhs={v.lhs:.4g}, bounds=[{v.lower:.4g}, {v.upper:.4g}])"
                    for v in violations[:5]
                ) + (f", ... (+{len(violations) - 5} more)" if len(violations) > 5 else ""),
            )
        return PortfolioResult(
            weights=np.zeros(n), active=np.zeros(n),
            positions=np.zeros(n, dtype=int), trades=np.zeros(n, dtype=int),
            buy_trades=np.zeros(n), sell_trades=np.zeros(n),
            n_positions=0, n_trades=0,
            gross_return_pre=0.0, gross_return_post=0.0,
            transaction_cost=0.0, estimation_error_cost=0.0,
            sell_cost=0.0, net_return=0.0,
            avg_gross_return_sells=0.0, avg_net_return_buys=0.0,
            avg_net_return_buys_incl_ee=0.0,
            success=False, message=solver_result.message,
            feasibility_violations=violations,
        )

    w = solver_result.w
    rho = solver_result.rho
    buy_trades = solver_result.buy_trades
    sell_trades = solver_result.sell_trades
    tau = solver_result.tau
    n_tickers_held = solver_result.n_tickers_held

    active_wts = w - bm_weights
    gross_ret_pre = float(mu @ holdings)
    gross_ret_post = float(mu @ w)
    pure_tx_cost = float(raw_costs @ buy_trades)
    ee_cost = float(estimation_error @ buy_trades)
    s_cost = float(mu @ sell_trades)

    # Per-trade weighted averages
    total_buy = float(buy_trades.sum())
    total_sell = float(sell_trades.sum())
    avg_gross_sells = float(mu @ sell_trades) / total_sell if total_sell > 1e-12 else 0.0
    avg_net_buys = (
        float((mu - raw_costs) @ buy_trades) / total_buy
        if total_buy > 1e-12 else 0.0
    )
    avg_net_buys_ee = (
        float((mu - raw_costs - estimation_error) @ buy_trades) / total_buy
        if total_buy > 1e-12 else 0.0
    )

    logger.debug(
        "Result: positions=%d  sum(w)=%.6f  max|w-h|=%.2e  "
        "sum(t+)=%.6f  sum(t-)=%.6f  gross_pre=%.6f  gross_post=%.6f  "
        "tx_cost=%.6f  ee_cost=%.6f  sell_cost=%.6f",
        int(rho.sum()), w.sum(), float(np.abs(w - holdings).max()),
        buy_trades.sum(), sell_trades.sum(),
        gross_ret_pre, gross_ret_post, pure_tx_cost, ee_cost, s_cost,
    )

    return PortfolioResult(
        weights=w,
        active=active_wts,
        positions=rho,
        trades=tau,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
        n_positions=int(rho.sum()),
        n_trades=int(tau.sum()),
        gross_return_pre=gross_ret_pre,
        gross_return_post=gross_ret_post,
        transaction_cost=pure_tx_cost,
        estimation_error_cost=ee_cost,
        sell_cost=s_cost,
        net_return=gross_ret_post - pure_tx_cost - ee_cost - s_cost,
        avg_gross_return_sells=avg_gross_sells,
        avg_net_return_buys=avg_net_buys,
        avg_net_return_buys_incl_ee=avg_net_buys_ee,
        success=True,
        message=solver_result.message,
        n_tickers=n_tickers_held,
        cash_weight=solver_result.cash_weight,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    mu: np.ndarray,
    bm_weights: np.ndarray,
    holdings: np.ndarray,
    costs: np.ndarray,
    estimation_error: np.ndarray,
    config: MILPOptimizerConfig,
    n: int,
) -> None:
    """Raise ``ValueError`` on obviously invalid inputs."""
    if n == 0:
        raise ValueError("Expected-return vector `mu` must be non-empty.")
    for name, arr in [("bm_weights", bm_weights), ("holdings", holdings),
                      ("costs", costs), ("estimation_error", estimation_error)]:
        if arr.shape[0] != n:
            raise ValueError(f"{name} length {arr.shape[0]} != n={n}")
    if config.w_min <= 0:
        raise ValueError(f"w_min must be > 0, got {config.w_min}")
    if config.w_max < config.w_min:
        raise ValueError(
            f"w_max ({config.w_max}) must be >= w_min ({config.w_min})"
        )
    if config.min_positions < 0:
        raise ValueError(f"min_positions must be >= 0, got {config.min_positions}")
    if config.max_positions < config.min_positions:
        raise ValueError(
            f"max_positions ({config.max_positions}) must be >= "
            f"min_positions ({config.min_positions})"
        )
    if config.min_trades < 0:
        raise ValueError(f"min_trades must be >= 0, got {config.min_trades}")
    if config.max_trades < config.min_trades:
        raise ValueError(
            f"max_trades ({config.max_trades}) must be >= "
            f"min_trades ({config.min_trades})"
        )
    if config.min_tickers < 0:
        raise ValueError(f"min_tickers must be >= 0, got {config.min_tickers}")
    if config.max_tickers < config.min_tickers:
        raise ValueError(
            f"max_tickers ({config.max_tickers}) must be >= "
            f"min_tickers ({config.min_tickers})"
        )
    if config.cash_min < 0:
        raise ValueError(f"cash_min must be >= 0, got {config.cash_min}")
    if config.cash_max < config.cash_min:
        raise ValueError(
            f"cash_max ({config.cash_max}) must be >= cash_min ({config.cash_min})"
        )
    if config.cash_max > 1.0:
        raise ValueError(f"cash_max must be <= 1.0, got {config.cash_max}")
    for sc in config.attribute_constraints:
        if sc.asset_mask.shape[0] != n:
            raise ValueError(
                f"Constraint '{sc.multiplier.value}:{sc.attribute}:{sc.value}' "
                f"mask length {sc.asset_mask.shape[0]} != n={n}"
            )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_exposures_report(
    attribute_constraints: list[AttributeConstraint],
    result: PortfolioResult,
    bm_weights: np.ndarray,
) -> pd.DataFrame:
    """Build a DataFrame summarising constraint exposures.

    Columns: multiplier, attribute, value, n_held, bm_exposure,
    ptf_exposure, active, lower, upper.
    """
    rows: list[dict] = []
    for sc in attribute_constraints:
        mask = np.asarray(sc.asset_mask)
        is_boolean = mask.dtype == bool
        if is_boolean:
            ptf_exp = float(result.weights[mask].sum())
            bm_exp = float(bm_weights[mask].sum())
            n_held = int((result.positions[mask] > 0).sum()) if result.success else 0
        else:
            ptf_exp = float((result.weights * mask).sum())
            bm_exp = float((bm_weights * mask).sum())
            n_held = int((result.positions[mask > 0] > 0).sum()) if result.success else 0

        rows.append({
            "multiplier": sc.multiplier.value,
            "attribute": sc.attribute,
            "value": sc.value,
            "n_held": n_held,
            "bm_exposure": bm_exp,
            "ptf_exposure": ptf_exp,
            "active": ptf_exp - bm_exp,
            "lower": sc.lower,
            "upper": sc.upper,
        })

    return pd.DataFrame(rows)


def compute_constraint_exposures(
    attribute_constraints: list[AttributeConstraint],
    weights: np.ndarray,
    bm_weights: np.ndarray,
) -> pd.DataFrame:
    """Compute constraint exposures for an arbitrary weight vector.

    Unlike :func:`build_exposures_report`, this accepts raw weight/bm arrays
    rather than a :class:`PortfolioResult`, making it suitable for computing
    *pre-rebalance* exposures from holdings.

    Columns: ``multiplier``, ``attribute``, ``value``, ``bm_exposure``,
    ``ptf_exposure``, ``active``, ``lower``, ``upper``.
    """
    rows: list[dict] = []
    for sc in attribute_constraints:
        mask = np.asarray(sc.asset_mask)
        if mask.dtype == bool:
            ptf_exp = float(weights[mask].sum())
            bm_exp = float(bm_weights[mask].sum())
        else:
            ptf_exp = float((weights * mask).sum())
            bm_exp = float((bm_weights * mask).sum())

        rows.append({
            "multiplier": sc.multiplier.value,
            "attribute": sc.attribute,
            "value": sc.value,
            "bm_exposure": bm_exp,
            "ptf_exposure": ptf_exp,
            "active": ptf_exp - bm_exp,
            "lower": sc.lower,
            "upper": sc.upper,
            "inner_lower": sc.inner_lower,
            "inner_upper": sc.inner_upper,
            "state_lower": sc.state_lower,
            "state_upper": sc.state_upper,
            "buffer_lower": sc.buffer_lower,
            "buffer_upper": sc.buffer_upper,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-period bridge (DataFrame → numpy → PortfolioResult)
# ---------------------------------------------------------------------------


def run_single_period(
    bonds: pd.DataFrame,
    bm_weights: np.ndarray,
    mu: np.ndarray,
    costs: np.ndarray,
    estimation_error: np.ndarray,
    holdings: np.ndarray,
    constraint_specs: list[tuple],
    optimizer_config: MILPOptimizerConfig | None = None,
    return_threshold: float | None = None,
    return_threshold_percentile: float = 50.0,
    illiquid_mask: np.ndarray | None = None,
    entity_weight_caps: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[PortfolioResult, list[AttributeConstraint]]:
    """Run a single-period MILP optimisation from DataFrame inputs.

    This function bridges the DataFrame-level universe (with named columns
    for constraint building) to the numpy-level :func:`milp_portfolio_opt`.

    Parameters
    ----------
    bonds : pd.DataFrame
        Cleaned universe DataFrame (from :func:`build_universe`).  Must
        contain columns referenced in *constraint_specs* plus ``OAS`` and
        ``EFFECTIVE_DURATION`` for DTS/OAD multipliers.
    bm_weights : np.ndarray
        Normalized benchmark weight vector (length ``len(bonds)``).
    mu : np.ndarray
        Gross expected return vector (length *n*).
    costs : np.ndarray
        Bid/ask cost per asset (length *n*).
    estimation_error : np.ndarray
        Return estimation error per asset (length *n*).
    holdings : np.ndarray
        Current portfolio weight vector (length *n*).
    constraint_specs : list[tuple[Multiplier, str | list[str] | None, Tolerance]]
        Constraint specifications passed to
        :func:`build_attribute_constraints`.
    optimizer_config : MILPOptimizerConfig | None
        Base solver configuration (without ``attribute_constraints`` —
        those are built from *constraint_specs*).  Uses defaults when
        ``None``.
    return_threshold : float | None
        Net-return floor for :func:`shrink_buy_candidates`.

    Returns
    -------
    tuple[PortfolioResult, list[AttributeConstraint]]
        The optimisation result and the attribute constraints used.
    """
    if optimizer_config is None:
        optimizer_config = MILPOptimizerConfig()

    attribute_constraints = build_attribute_constraints(
        bonds, bm_weights, constraint_specs, holdings=holdings,
    )
    base_config = replace(optimizer_config, attribute_constraints=attribute_constraints)

    ticker_labels = (
        bonds["TICKER"].to_numpy() if "TICKER" in bonds.columns else None
    )

    base_options = dict(base_config.options)
    base_time_limit = base_options.get("time_limit")

    # Attempt 0 = "primary" (call-site args + base options).  Each subsequent
    # entry in retry_attempts is invoked only if the prior attempt failed.
    primary = OptimizerAttempt(
        label="primary",
        return_threshold_percentile=return_threshold_percentile,
        return_threshold=return_threshold,
        time_limit_extra=0.0,
    )
    attempts: list[OptimizerAttempt] = [primary, *base_config.retry_attempts]

    result: PortfolioResult | None = None
    for idx, attempt in enumerate(attempts):
        eff_threshold = (
            attempt.return_threshold if attempt.return_threshold is not None
            else return_threshold
        )
        eff_pct = (
            attempt.return_threshold_percentile
            if attempt.return_threshold_percentile is not None
            else return_threshold_percentile
        )
        if attempt.time_limit_extra and base_time_limit is not None:
            options = {**base_options, "time_limit": float(base_time_limit) + float(attempt.time_limit_extra)}
            attempt_config = replace(base_config, options=options)
        else:
            attempt_config = base_config

        if idx > 0:
            logger.warning(
                "Optimizer primary attempt failed; retrying with attempt %d ('%s') "
                "[return_threshold_percentile=%s, time_limit_extra=%.1f]",
                idx, attempt.label, eff_pct, attempt.time_limit_extra,
            )

        result = milp_portfolio_opt(
            mu,
            bm_weights,
            attempt_config,
            holdings=holdings,
            costs=costs,
            estimation_error=estimation_error,
            return_threshold=eff_threshold,
            return_threshold_percentile=eff_pct,
            ticker_labels=ticker_labels,
            illiquid_mask=illiquid_mask,
            entity_weight_caps=entity_weight_caps,
        )
        result.attempt_index = idx
        result.attempt_label = attempt.label or f"attempt_{idx}"

        if result.success:
            if idx > 0:
                logger.info(
                    "Optimizer attempt %d ('%s') succeeded.", idx, result.attempt_label,
                )
            return result, attribute_constraints

    if len(attempts) > 1:
        logger.warning(
            "All %d optimizer attempts failed; returning last result ('%s').",
            len(attempts), result.attempt_label,
        )
    return result, attribute_constraints


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a small demo optimisation and print results."""

    from datetime import datetime
    from pathlib import Path
    from archipelago.settings import load_snowflake_config
    from archipelago.data.queries.goblin import GoblinQuery

    project_root = Path(__file__).resolve().parents[3]  # src/archipelago/portfolio -> root
    input_dir = project_root / "input"
    output_dir = project_root / "output"

    sf_cfg = load_snowflake_config()
    goblin = GoblinQuery(sf_cfg)
    index_date = datetime(2024, 2, 29)
    bonds = goblin.get_index_for_date(index_date)

    bonds = bonds[bonds["ISIN"].notna()].copy()
    bonds["WEIGHT"] = pd.to_numeric(bonds["WEIGHT"], errors="coerce")

    bonds["OAS"] = pd.to_numeric(bonds["OAS"], errors="coerce").astype(float)
    bonds["EFFECTIVE_DURATION"] = pd.to_numeric(bonds["EFFECTIVE_DURATION"], errors="coerce").astype(float)
    bonds = bonds[bonds["WEIGHT"].notna()].reset_index(drop=True)

    n = len(bonds)
    bm_weights = bonds["WEIGHT"].to_numpy(dtype=float)

    # Diagnostic: check benchmark weight scale
    print(f"\nbm_weights: n={n}, sum={bm_weights.sum():.6f}, min={bm_weights.min():.8f}, max={bm_weights.max():.8f}")

    # Normalise to sum to 1 if needed
    bm_sum = bm_weights.sum()
    if not np.isclose(bm_sum, 1.0, atol=1e-3):
        logger.warning("Benchmark weights sum to %.6f — rescaling to 1.0", bm_sum)
        bm_weights = bm_weights / bm_sum

    # Synthetic expected returns
    rng = np.random.default_rng()


    # ------------------------------------------------------------------
    # Initial portfolio h_i from prior optimisation
    # ------------------------------------------------------------------
    Rampup= False
    if Rampup == True:
         holdings = np.zeros(n)
         mu = rng.normal(loc=0.02, scale=0.02, size=n)
         costs = rng.uniform(0.003, 0.01, size=n)
         estimation_error = rng.uniform(0.01, 0.04, size=n)
    else:
        holdings_path = input_dir / "portfolio_holdings.csv"
        holdings_csv = pd.read_csv(holdings_path)
        isin_to_weight = dict(zip(holdings_csv["ISIN"], holdings_csv["ptf_weight"]))
        holdings = bonds["ISIN"].map(isin_to_weight).fillna(0.0).to_numpy(dtype=float)

        isin_to_mu = dict(zip(holdings_csv["ISIN"], holdings_csv["mu"]))
        isin_to_cost = dict(zip(holdings_csv["ISIN"], holdings_csv["cost"]))
        isin_to_ee = dict(zip(holdings_csv["ISIN"], holdings_csv["estimation_error"]))

        mu = bonds["ISIN"].map(isin_to_mu).fillna(0.0).to_numpy(dtype=float) + rng.normal(loc=0.005, scale=0.01, size=n)
        costs = bonds["ISIN"].map(isin_to_cost).fillna(0.005).to_numpy(dtype=float) + rng.normal(loc=0.0, scale=0.005, size=n)
        estimation_error = bonds["ISIN"].map(isin_to_ee).fillna(0.02).to_numpy(dtype=float) + rng.normal(loc=0.0, scale=0.005, size=n)

        # Clip costs and estimation_error to remain non-negative
        costs = np.clip(costs, 0.0, None)
        estimation_error = np.clip(estimation_error, 0.0, None)

        logger.debug(
            "Loaded initial portfolio: %d ISINs mapped, h sum=%.6f",
            int((holdings > 0).sum()),
            holdings.sum(),
        )


    # ------------------------------------------------------------------
    # Simulated transaction costs c_i and estimation error ε_i
    # ------------------------------------------------------------------

    # Summary statistics for return/cost vectors
    net_mu = mu - costs - estimation_error
    print("\nReturn / cost summary:")
    print(f"{'':>12s} {'mu':>12s} {'costs':>12s} {'est_error':>12s} {'net_mu':>12s}")
    for label, pct in [("mean", None), ("25%", 25), ("50%", 50), ("75%", 75)]:
        vals = [
            arr.mean() if pct is None else np.percentile(arr, pct)
            for arr in (mu, costs, estimation_error, net_mu)
        ]
        print(f"{label:>12s} {vals[0]:>12.6f} {vals[1]:>12.6f} {vals[2]:>12.6f} {vals[3]:>12.6f}")
    print()

    # ------------------------------------------------------------------
    # Constraint specifications: (Multiplier, column, tolerance)
    # ------------------------------------------------------------------
    constraint_specs: list[tuple[Multiplier, str | None, Tolerance]] = [
        (Multiplier.OAD,    None,                   1e-2),
        (Multiplier.DTS,    None,                   1e-2),
        (Multiplier.WEIGHT, "INDUSTRY_LVL_3_DESC",  0.01),
        (Multiplier.WEIGHT, "INDUSTRY_LVL_4_DESC",  0.02),
        (Multiplier.WEIGHT, "RATING",               0.01),
        (Multiplier.WEIGHT, "SUBORDINATION_TYPE",   0.01),
        (Multiplier.WEIGHT, "CURRENCY",             0.01),
        (Multiplier.DTS,    "INDUSTRY_LVL_2_DESC",  10),
        (Multiplier.DTS,    "INDUSTRY_LVL_3_DESC",  20),
        (Multiplier.DTS,    "INDUSTRY_LVL_4_DESC",  30),
        (Multiplier.DTS,    "RATING",               200),
        (Multiplier.DTS,    "SUBORDINATION_TYPE",   200),
        (Multiplier.WEIGHT, "TICKER",               (None, 0.02)),    # upper-only: max 2% overweight per issuer
    ]

    attribute_constraints = build_attribute_constraints(bonds, bm_weights, constraint_specs, holdings=holdings)
    print(f"Attribute Constraints: {len(attribute_constraints)}")

    # Pre-screen summary
    exclude = shrink_buy_candidates(mu, costs, estimation_error, holdings)
    n_eligible = n - int(exclude.sum())
    print(f"Buy candidates: {n_eligible} / {n} bonds (excluded {int(exclude.sum())})")

    config = MILPOptimizerConfig(
        w_min=1e-4,
        w_max=0.10,
        min_positions=1000,
        max_positions=10000,
        attribute_constraints=attribute_constraints,
        options={"disp": False, "time_limit": 200.0, "mip_rel_gap": 1e-6, "mip_abs_gap": 1e-5, "node_limit": 10_000, "presolve": True},
    )

    result = milp_portfolio_opt(
        mu, bm_weights, config,
        holdings=holdings, costs=costs, estimation_error=estimation_error,
        ticker_labels=bonds["TICKER"].to_numpy() if "TICKER" in bonds.columns else None,
    )

    print(f"\nSuccess      : {result.success}")
    print(f"Message      : {result.message}")
    print(f"Positions    : {result.n_positions}")
    print(f"Tickers      : {result.n_tickers}")
    print(f"Trades       : {result.n_trades}")
    print(f"Gross pre    : {result.gross_return_pre:.6f}")
    print(f"Gross post   : {result.gross_return_post:.6f}")
    print(f"Tx cost      : {result.transaction_cost:.6f}")
    print(f"EE cost      : {result.estimation_error_cost:.6f}")
    print(f"Sell cost    : {result.sell_cost:.6f}")
    print(f"Net return   : {result.net_return:.6f}")
    print(f"Sum weights  : {result.weights.sum():.6f}")

    # Trade summary
    n_buys = int((result.buy_trades > 1e-4).sum())
    n_sells = int((result.sell_trades > 1e-4).sum())
    turnover = 0.5*float(result.buy_trades.sum() + result.sell_trades.sum())
    net_mu = mu - costs - estimation_error
    weighted_net_return_increase = float(
        net_mu @ result.buy_trades - mu @ result.sell_trades
    )
    print(f"\nBuys         : {n_buys}")
    print(f"Sells        : {n_sells}")
    print(f"Turnover     : {turnover:.6f}")
    print(f"Wtd net Δret : {weighted_net_return_increase:.6f}")
    print()

    # Build and export exposures report
    exposures_df = build_exposures_report(attribute_constraints, result, bm_weights)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exposures_path = output_dir / f"{timestamp}_portfolio_exposures.csv"
    exposures_df.to_csv(exposures_path, index=False, float_format="%.6f")
    logger.info("Wrote %d constraint exposures to %s", len(exposures_df), exposures_path)
    print(f"\nExported {len(exposures_df)} constraint exposures to {exposures_path}")

    # Portfolio holdings (ISIN + weight)
    holdings_df = bonds[["ISIN"]].copy()
    holdings_df["bm_weight"] = bm_weights
    holdings_df["init_holding"] = holdings
    holdings_df["ptf_weight"] = result.weights
    holdings_df["active_weight"] = result.active
    holdings_df["buy_trade"] = result.buy_trades
    holdings_df["sell_trade"] = result.sell_trades
    holdings_df["mu"] = mu
    holdings_df["cost"] = costs
    holdings_df["estimation_error"] = estimation_error
    holdings_df = holdings_df.sort_values("ptf_weight", ascending=False)
    holdings_path = output_dir / f"{timestamp}_portfolio_holdings.csv"
    holdings_df.to_csv(holdings_path, index=False, float_format="%.6f")
    logger.info("Wrote %d rows to %s", len(holdings_df), holdings_path)
    print(f"Exported {len(holdings_df)} rows to {holdings_path}")


if __name__ == "__main__":
    main()
