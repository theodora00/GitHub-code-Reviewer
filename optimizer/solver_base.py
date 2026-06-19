"""Solver backend interface for the MILP portfolio optimiser.

Defines the shared :class:`SolverResult` returned by all solver backends
and the :func:`solve` dispatcher that routes to the configured backend.

Backends
--------
- ``"scipy"`` — :func:`archipelago.portfolio.solver_scipy.solve`
- ``"mosek"`` — :func:`archipelago.portfolio.solver_mosek.solve`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from archipelago.portfolio.constraints import MILPOptimizerConfig

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    """Normalised output from any MILP solver backend.

    Attributes
    ----------
    success : bool
        Whether the solver found a feasible solution.
    message : str
        Status message from the solver.
    w : np.ndarray
        Optimal portfolio weights (length *n*).
    rho : np.ndarray
        Binary position indicators (length *n*).
    buy_trades : np.ndarray
        Buy-trade amounts *t⁺* (length *n*).
    sell_trades : np.ndarray
        Sell-trade amounts *t⁻* (length *n*).
    tau : np.ndarray
        Binary trade indicators (length *n*).
    n_tickers_held : int
        Number of distinct tickers held.
    cash_weight : float
        Optimal scalar cash weight ``c`` (the residual ``1 − 1ᵀw``).
        Always 0.0 when ``cash_max == 0`` (default).
    """

    success: bool
    message: str
    w: np.ndarray
    rho: np.ndarray
    buy_trades: np.ndarray
    sell_trades: np.ndarray
    tau: np.ndarray
    n_tickers_held: int
    cash_weight: float = 0.0


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
    """Dispatch to the solver backend specified by ``config.solver``.

    All backends share the same parameter contract and return a
    :class:`SolverResult`.

    Parameters
    ----------
    n : int
        Number of assets.
    K : int
        Number of unique tickers (0 when ticker constraints are unused).
    mu : np.ndarray
        Gross expected return vector *μ* (length *n*).
    costs : np.ndarray
        Combined cost vector *c + ε* (length *n*).
    holdings : np.ndarray
        Current portfolio weight vector *h* (length *n*).
    config : MILPOptimizerConfig
        Solver configuration (includes ``solver``, ``options``,
        ``attribute_constraints``, etc.).
    w_ub : np.ndarray
        Per-bond upper bounds on *w* (length *n*).
    tp_ub : np.ndarray
        Per-bond upper bounds on *t⁺* (length *n*); 0 for excluded bonds.
    exclude : np.ndarray
        Boolean mask of excluded (non-purchasable) bonds (length *n*).
    ticker_map : np.ndarray | None
        Integer array mapping each bond to a ticker index ``0…K-1``.

    Returns
    -------
    SolverResult
    """
    solver_name = config.solver

    if solver_name == "scipy":
        from archipelago.portfolio.solver_scipy import solve as _solve
    elif solver_name == "mosek":
        from archipelago.settings import bootstrap_mosek
        bootstrap_mosek()
        from archipelago.portfolio.solver_mosek import solve as _solve
    else:
        raise ValueError(
            f"Unknown solver '{solver_name}'. Supported: 'scipy', 'mosek'."
        )

    logger.debug("Dispatching to '%s' solver  n=%d  K=%d", solver_name, n, K)
    return _solve(
        n=n, K=K, mu=mu, costs=costs, holdings=holdings,
        config=config, w_ub=w_ub, tp_ub=tp_ub, exclude=exclude,
        ticker_map=ticker_map,
    )
