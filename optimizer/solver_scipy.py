"""Scipy (HiGHS) backend for the MILP portfolio optimiser.

Solves the mixed-integer linear programme via ``scipy.optimize.milp``
which delegates to the HiGHS solver.

See :mod:`archipelago.portfolio.solver_base` for the shared interface.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, milp

from archipelago.portfolio.constraints import build_milp_constraints
from archipelago.portfolio.solver_base import SolverResult

if TYPE_CHECKING:
    from archipelago.portfolio.constraints import MILPOptimizerConfig

logger = logging.getLogger(__name__)


def solve(
    n: int,
    K: int,
    mu: np.ndarray,
    costs: np.ndarray,
    holdings: np.ndarray,
    config: MILPOptimizerConfig,
    w_ub: np.ndarray,
    tp_ub: np.ndarray,
    exclude: np.ndarray,
    ticker_map: np.ndarray | None,
) -> SolverResult:
    """Solve the MILP portfolio problem using scipy / HiGHS.

    Parameters match :func:`archipelago.portfolio.solver_base.solve`.

    Returns
    -------
    SolverResult
    """
    # -- objective (5n + K + 1): minimise -(μ−c)ᵀt⁺ + μᵀt⁻ -----------------
    # Cash variable contributes 0 to the objective (zero return).
    c_obj = np.concatenate([
        np.zeros(n),             # w
        np.zeros(n),             # ρ
        -(mu - costs),           # t⁺ (negated for min)
        mu,                      # t⁻
        np.zeros(n),             # τ
        np.zeros(K),             # π
        np.zeros(1),             # c
    ])

    # -- variable bounds (5n + K + 1) --------------------------------------
    rho_ub = np.ones(n)
    tm_ub = np.full(n, config.w_max)
    tau_ub = np.ones(n)

    bounds = Bounds(
        lb=np.concatenate([
            np.zeros(5 * n + K), np.array([float(config.cash_min)]),
        ]),
        ub=np.concatenate([
            w_ub, rho_ub, tp_ub, tm_ub, tau_ub, np.ones(K),
            np.array([float(config.cash_max)]),
        ]),
    )

    # -- integrality (5n + K + 1): 0 = continuous, 1 = integer -------------
    integrality = np.concatenate([
        np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.ones(n),
        np.ones(K),
        np.zeros(1),             # c is continuous
    ])

    # -- constraints -------------------------------------------------------
    constraints = build_milp_constraints(
        n, config, holdings, ticker_map=ticker_map, n_tickers=K,
    )

    # -- solve -------------------------------------------------------------
    logger.debug(
        "Launching scipy MILP solver  n=%d  K=%d  vars=%d  excluded=%d  "
        "positions=[%d, %d]",
        n, K, 5 * n + K + 1, int(exclude.sum()),
        config.min_positions, config.max_positions,
    )

    # SciPy's milp wrapper keeps a hardcoded allowlist of recognised option
    # keys and emits a RuntimeWarning for any others (e.g. mip_abs_gap), even
    # though it forwards them verbatim to HiGHS — and HiGHS does honour them.
    # Suppress that one specific warning so the backtest log stays clean.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected.*",
            category=RuntimeWarning,
        )
        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options=config.options,
        )

    logger.debug(
        "Solver finished: success=%s  message='%s'  obj=%.8f",
        result.success, result.message,
        result.fun if result.fun is not None else float("nan"),
    )

    if not result.success:
        return SolverResult(
            success=False,
            message=result.message,
            w=np.zeros(n),
            rho=np.zeros(n, dtype=int),
            buy_trades=np.zeros(n),
            sell_trades=np.zeros(n),
            tau=np.zeros(n, dtype=int),
            n_tickers_held=0,
        )

    # -- extract solution --------------------------------------------------
    w = result.x[:n]
    rho = np.round(result.x[n : 2 * n]).astype(int)
    buy_trades = result.x[2 * n : 3 * n]
    sell_trades = result.x[3 * n : 4 * n]
    tau = np.round(result.x[4 * n : 5 * n]).astype(int)
    n_tickers_held = (
        int(np.round(result.x[5 * n : 5 * n + K]).sum()) if K > 0 else 0
    )
    cash_weight = float(result.x[5 * n + K])

    return SolverResult(
        success=True,
        message=result.message,
        w=w,
        rho=rho,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
        tau=tau,
        n_tickers_held=n_tickers_held,
        cash_weight=cash_weight,
    )
