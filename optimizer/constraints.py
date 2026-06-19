"""Portfolio constraints.

Provides all constraint-related types and builders for the MILP portfolio
optimiser:

* :class:`Multiplier` — exposure type enum (WEIGHT, DTS, OAD).
* :class:`AttributeConstraint` — single benchmark-relative constraint spec.
* :class:`MILPOptimizerConfig` — full solver configuration.
* :func:`make_attribute_value_mask` — per-category mask builder.
* :func:`build_attribute_constraints` — benchmark-relative constraint factory.
* :func:`build_milp_constraints` — assembles scipy ``LinearConstraint`` objects
  for the (5n + K)-variable MILP decision vector.

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
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint
from scipy.sparse import csc_array, diags as spdiags, eye as speye, hstack as sphstack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class Multiplier(Enum):
    """Type of exposure measured by a constraint."""

    WEIGHT = "WEIGHT"
    DTS = "DTS"
    OAD = "OAD"


class EntityCapPolicy(str, Enum):
    """Policy for capping entity-level weight in the optimizer.

    - ``ZERO_WEIGHT``: force full liquidation (w_ub = 0 for all entity bonds).
    - ``HOLD_WEIGHT``: freeze position (w_ub = current holding, block buys).
    - ``MARKET_WEIGHT``: cap at benchmark weight (cannot overweight vs index).
    """

    ZERO_WEIGHT = "zero_weight"
    HOLD_WEIGHT = "hold_weight"
    MARKET_WEIGHT = "market_weight"


@dataclass
class EntityWeightCapRule:
    """A threshold-based entity weight cap rule.

    When an entity's signal value meets or exceeds ``threshold``, the
    ``policy`` is applied to all bonds of that entity.

    Rules are evaluated in descending threshold order; the first
    (strictest) matching rule wins.

    Parameters
    ----------
    threshold : float
        Signal value at or above which this rule triggers.
    policy : EntityCapPolicy
        Weight cap policy to apply.
    """

    threshold: float
    policy: EntityCapPolicy


# Type alias for constraint tolerances.
# ``float``          → symmetric: [bm − tol, bm + tol]
# ``(lo, hi)``       → asymmetric: [bm − lo, bm + hi]
# ``(None, hi)``     → upper-only: [0, bm + hi]
# ``(lo, None)``     → lower-only: [bm − lo, +∞]
Tolerance = Union[float, Tuple[Optional[float], Optional[float]]]

# Type alias for the per-constraint hysteresis buffer (deadband width).
# Same shape as :data:`Tolerance`.  ``0.0`` (default) ⇒ no hysteresis,
# behaviour identical to a pure tolerance band.  When non-zero, the
# *outer* band is widened by *buffer* on each applicable side; if the
# current portfolio exposure ``lhs_h`` falls inside the (inner, outer]
# zone the bound on that side is *pinned* to ``lhs_h`` so the optimiser
# does not have to trade to satisfy it; if ``lhs_h`` is past the outer
# edge the bound snaps back to the inner tolerance, forcing a trade.
# Set to ``None`` (or omit in YAML) to disable on a side.
Buffer = Union[float, Tuple[Optional[float], Optional[float]]]

# Type alias for a constraint column specifier.
# ``None``           → aggregate (whole-portfolio) constraint
# ``str``            → single categorical column (per-category constraint)
# ``list[str]``      → compound key: Cartesian categories of the listed columns
ColumnSpec = Union[str, List[str], None]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AttributeConstraint:
    """Constraint on the aggregate exposure of a single category value.

    Parameters
    ----------
    multiplier : Multiplier
        Type of exposure:

        - WEIGHT: ``l ≤ Mᵀ(w − b) ≤ u``
        - OAD:    ``l ≤ (M·D)ᵀ(w − b) ≤ u``
        - DTS:    ``l ≤ (M·D·S)ᵀ(w − b) ≤ u``
    attribute : str
        Column name in the universe DataFrame (e.g. ``INDUSTRY_LVL_2_DESC``),
        or ``"AGGREGATE"`` for whole-portfolio constraints.
    value : str
        Category label within the attribute (e.g. ``Banking``),
        or ``"ALL"`` for aggregate constraints.
    asset_mask : np.ndarray
        Boolean (WEIGHT) or float (DTS/OAD) vector of length *n*.  For WEIGHT
        constraints this is *M* (boolean indicator); for OAD it is *M·D*;
        for DTS it is *M·D·S*.
    lower : float
        Minimum permissible aggregate exposure.  Default 0.
    upper : float
        Maximum permissible aggregate exposure.  Default 1.
    """

    multiplier: Multiplier
    attribute: str
    value: str
    asset_mask: np.ndarray
    lower: float = 0.0
    upper: float = 1.0
    # Hysteresis bookkeeping (purely informational — the LP only sees lower/upper).
    # ``inner_lower`` / ``inner_upper`` are the tolerance-only bounds (i.e. what
    # ``lower`` / ``upper`` would be if buffer were 0).
    # ``state_lower`` / ``state_upper`` ∈ {"inner", "pinned", "forced"}:
    #   - "inner":  lhs_h inside inner band, bound = inner.
    #   - "pinned": lhs_h in deadband, bound = lhs_h (no trade required).
    #   - "forced": lhs_h past outer edge, bound = inner (force back).
    # ``buffer_lower`` / ``buffer_upper`` are the per-side absolute buffer widths.
    inner_lower: float = 0.0
    inner_upper: float = 1.0
    state_lower: str = "inner"
    state_upper: str = "inner"
    buffer_lower: float = 0.0
    buffer_upper: float = 0.0


@dataclass
class OptimizerAttempt:
    """Override applied to a retry attempt of :func:`run_single_period`.

    The first ("primary") attempt uses the call-site arguments and the base
    :class:`MILPOptimizerConfig`.  Each entry in
    :attr:`MILPOptimizerConfig.retry_attempts` is then applied in order, only
    if the previous attempt did not return ``success``.

    Fields that are ``None`` inherit the base / call-site value.
    """

    label: str = ""
    return_threshold_percentile: float | None = None
    return_threshold: float | None = None
    time_limit_extra: float = 0.0


@dataclass
class MILPOptimizerConfig:
    """Configuration for the MILP portfolio optimiser.

    Parameters
    ----------
    w_min : float
        Minimum weight for a *held* position (must be > 0 so that the binary
        indicator ρ_i is meaningful).  Default 0.005 (50 bp).
    w_max : float
        Maximum weight for any single position.  Default 0.10 (10 %).
    t_min : float
        Minimum total trade size ``t⁺_i + t⁻_i`` when a trade occurs
        (τ_i = 1).  Prevents dust trades.  Default 0.0005 (5 bp).
    min_positions : int
        Lower bound on position count ρ_L.  Default 10.
    max_positions : int
        Upper bound on position count ρ_U.  Default 50.
    min_trades : int
        Lower bound on trade count τ_L.  Default 0.
    max_trades : float
        Upper bound on trade count τ_U.  Default ``np.inf``
        (unconstrained).
    min_tickers : int
        Lower bound on ticker count π_L.  Default 0.
    max_tickers : float
        Upper bound on ticker count π_U.  Default ``np.inf``
        (unconstrained).
    ticker_w_min : float
        Minimum aggregate weight per ticker ω_min when held (π_k = 1).
        Default 0.0 (no lower bound).
    ticker_w_max : float
        Maximum aggregate weight per ticker ω_max when held (π_k = 1).
        Default 1.0 (no effective upper bound).
    max_active_overweight : float | None
        Maximum per-bond active overweight ``w_i − b_i``.  When set,
        each bond's upper bound is tightened to
        ``min(w_max, b_i + max_active_overweight)``.  Applied via
        variable bounds (free to the solver) rather than linear
        constraints.  Default ``None`` (no per-bond cap).
    cash_min : float
        Lower bound on the scalar cash weight ``c``.  Default 0.0
        (cash disabled; budget collapses to ``1ᵀw = 1``).
    cash_max : float
        Upper bound on the scalar cash weight ``c``.  Default 0.0
        (cash disabled).  When ``cash_max > 0`` the budget becomes
        ``1ᵀw + c = 1`` with ``cash_min ≤ c ≤ cash_max``.  Cash is
        structurally absent from every other constraint (attribute,
        position-count, trade-count, ticker, KRD).
    attribute_constraints : list[AttributeConstraint]
        Benchmark-relative exposure bounds built by
        :func:`build_attribute_constraints`.  Empty list → no attribute
        constraints.
    solver : {"scipy", "mosek"}
        Which MILP solver backend to use.  ``"scipy"`` delegates to
        ``scipy.optimize.milp`` (HiGHS).  ``"mosek"`` uses the MOSEK
        Fusion API.  Default ``"scipy"``.
    options : dict
        Extra keyword arguments forwarded to the solver backend (e.g.
        ``time_limit``, ``mip_rel_gap``, ``mip_abs_gap`` for scipy;
        ``mioMaxTime``, ``mioTolRelGap``, ``mioTolAbsGap`` for MOSEK).
    """

    w_min: float = 0.005
    w_max: float = 0.10
    t_min: float = 0.0005
    min_positions: int = 10
    max_positions: int = 50
    min_trades: int = 0
    max_trades: float = np.inf
    min_tickers: int = 0
    max_tickers: float = np.inf
    ticker_w_min: float = 0.0
    ticker_w_max: float = 1.0
    max_active_overweight: float | None = None
    cash_min: float = 0.0
    cash_max: float = 0.0
    attribute_constraints: list[AttributeConstraint] = field(default_factory=list)
    solver: Literal["scipy", "mosek"] = "scipy"
    options: dict = field(default_factory=dict)
    retry_attempts: list[OptimizerAttempt] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mask builder
# ---------------------------------------------------------------------------


def make_attribute_value_mask(
    labels: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series | None = None,
) -> dict[str, np.ndarray]:
    """Build per-category masks from a categorical vector.

    When *values* is ``None`` the masks are boolean (``True`` where the asset
    belongs to the category).  When *values* is supplied (e.g. OAS × duration),
    each mask entry is the corresponding value where the asset belongs to
    that category and ``0.0`` elsewhere.

    Null / NaN entries in *labels* are silently excluded from all masks.

    Parameters
    ----------
    labels : np.ndarray | pd.Series
        Categorical vector of length *n*.
    values : np.ndarray | pd.Series | None
        Optional continuous vector of length *n*.  When provided the returned
        arrays are float-valued (category value or 0).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping ``category_label → mask array`` of length *n*.

    Raises
    ------
    ValueError
        If *values* is provided but has a different length to *labels*.
    """
    labels = np.asarray(labels)

    not_null = np.array(pd.notna(labels), dtype=bool)
    unique_labels = np.unique(labels[not_null])

    if values is None:
        return {
            str(label): np.asarray((labels == label) & not_null, dtype=bool)
            for label in unique_labels
        }

    values = np.asarray(values, dtype=float)
    if values.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != values length {values.shape[0]}"
        )
    return {
        str(label): np.where((labels == label) & not_null, values, 0.0)
        for label in unique_labels
    }


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------


def _split_sides(
    spec: Tolerance | Buffer | None,
) -> tuple[Optional[float], Optional[float]]:
    """Normalise a Tolerance/Buffer spec into ``(lo_side, hi_side)``.

    Returns ``(None, None)`` when *spec* is ``None``.  A scalar ``x`` becomes
    ``(x, x)``.  A 2-tuple is returned verbatim (each side may be ``None``).
    Negative values are rejected.
    """
    if spec is None:
        return (None, None)
    if isinstance(spec, (int, float)):
        v = float(spec)
        if v < 0.0:
            raise ValueError(f"tolerance/buffer must be non-negative, got {v}")
        return (v, v)
    lo, hi = spec
    lo_f = None if lo is None else float(lo)
    hi_f = None if hi is None else float(hi)
    if lo_f is not None and lo_f < 0.0:
        raise ValueError(f"tolerance/buffer lower must be non-negative, got {lo_f}")
    if hi_f is not None and hi_f < 0.0:
        raise ValueError(f"tolerance/buffer upper must be non-negative, got {hi_f}")
    return (lo_f, hi_f)


def _resolve_bounds(
    bm_exposure: float,
    tol: Tolerance,
    buffer: Buffer | None = None,
    lhs_h: float | None = None,
) -> tuple[float, float, float, float, str, str, float, float]:
    """Convert tolerance + optional hysteresis buffer into applied bounds.

    The applied bound on each side is the result of the three-zone rule:

    * **inner zone** — ``lhs_h`` inside the tolerance band → bound = inner
      (the standard tolerance-only behaviour).
    * **deadband / pinned** — ``lhs_h`` in ``(inner, inner+buffer]`` (or
      mirrored on the lower side) → bound = ``lhs_h``.  No trade is
      required to satisfy the constraint, but the bound prevents further
      drift in the same direction.
    * **outside outer / forced** — ``lhs_h`` strictly past the outer edge
      → bound = inner (force the optimiser to pull the exposure back).

    When ``buffer`` is ``None``/``0`` or ``lhs_h`` is ``None`` the inner band
    is returned, exactly reproducing the legacy two-argument behaviour.

    Parameters
    ----------
    bm_exposure : float
        Benchmark exposure for the category / aggregate.
    tol : Tolerance
        Symmetric float or asymmetric ``(lower_tol, upper_tol)`` tuple;
        ``None`` on either side means that side is unbounded.
    buffer : Buffer | None
        Symmetric float or asymmetric ``(lower_buf, upper_buf)`` tuple.
        ``None`` (or omitted) ⇒ no hysteresis.  A side ``None`` ⇒ no
        hysteresis on that side; a side ``0.0`` is also legitimate and
        identical in effect.
    lhs_h : float | None
        Current portfolio exposure ``mask · holdings`` (matching the
        masking convention used to compute *bm_exposure*).  ``None`` ⇒
        no hysteresis (e.g. the caller has not threaded holdings).

    Returns
    -------
    tuple
        ``(lower, upper, inner_lower, inner_upper, state_lower,
        state_upper, buf_lo, buf_hi)``.
        ``state_*`` ∈ ``{"inner", "pinned", "forced"}``.
    """
    lo_tol, hi_tol = _split_sides(tol)
    buf_lo, buf_hi = _split_sides(buffer)

    inner_lower = 0.0 if lo_tol is None else max(0.0, bm_exposure - lo_tol)
    inner_upper = np.inf if hi_tol is None else bm_exposure + hi_tol

    # Default per-side buffer is 0 (no hysteresis on that side).  A buffer
    # side is only honoured when the corresponding tolerance side is finite —
    # there is no "outer" of an unbounded side.
    eff_buf_lo = 0.0 if (buf_lo is None or lo_tol is None) else buf_lo
    eff_buf_hi = 0.0 if (buf_hi is None or hi_tol is None) else buf_hi

    state_lower = "inner"
    state_upper = "inner"
    lower = inner_lower
    upper = inner_upper

    if lhs_h is not None:
        lhs = float(lhs_h)

        # Upper side: only active when there is a finite inner upper.
        if hi_tol is not None and eff_buf_hi > 0.0:
            outer_upper = inner_upper + eff_buf_hi
            if lhs <= inner_upper:
                upper = inner_upper          # inner zone (normal)
                state_upper = "inner"
            elif lhs <= outer_upper:
                upper = lhs                  # deadband: pin to current
                state_upper = "pinned"
            else:
                upper = inner_upper          # past outer: force back
                state_upper = "forced"

        # Lower side: only active when there is a finite inner lower.
        if lo_tol is not None and eff_buf_lo > 0.0:
            # Inner lower may be clipped at 0 above; mirror that for the
            # outer edge so the deadband never wraps past 0.
            outer_lower = max(0.0, bm_exposure - lo_tol - eff_buf_lo)
            if lhs >= inner_lower:
                lower = inner_lower
                state_lower = "inner"
            elif lhs >= outer_lower:
                lower = lhs                  # pin: lhs reachable by holding
                state_lower = "pinned"
            else:
                lower = inner_lower
                state_lower = "forced"

    return (
        lower, upper,
        inner_lower, inner_upper,
        state_lower, state_upper,
        eff_buf_lo, eff_buf_hi,
    )


# ---------------------------------------------------------------------------
# Constraint factory
# ---------------------------------------------------------------------------


def _resolve_label_vector(
    bonds: pd.DataFrame, col: str | Sequence[str]
) -> tuple[pd.Series, str]:
    """Resolve a constraint column specifier into a label vector and key.

    For a single column name the labels are taken verbatim.  For a compound
    key (sequence of column names) the labels are the ``"|"``-joined string
    cast of each listed column, with any row having a null in *any* component
    marked ``NaN`` so it is excluded from every group (consistent with the
    single-column null handling in :func:`make_attribute_value_mask`).

    Parameters
    ----------
    bonds : pd.DataFrame
        Universe DataFrame.
    col : str | Sequence[str]
        Single column name or sequence of column names.

    Returns
    -------
    tuple[pd.Series, str]
        ``(label_vector, attribute_key)`` where *attribute_key* is the column
        name (single) or ``"|"``-joined names (compound).
    """
    if isinstance(col, str):
        return bonds[col], col

    cols = list(col)
    sub = bonds[cols]
    not_null_all = sub.notna().all(axis=1)
    parts = [sub[c].astype(str) for c in cols]
    composite = parts[0]
    for p in parts[1:]:
        composite = composite.str.cat(p, sep="|")
    composite = composite.where(not_null_all, other=np.nan)
    return composite, "|".join(cols)


def _unpack_spec(
    spec: tuple,
) -> tuple[Multiplier, ColumnSpec, Tolerance, Buffer | None]:
    """Accept either 3-tuple ``(mult, col, tol)`` or 4-tuple ``(mult, col, tol, buf)``."""
    if len(spec) == 3:
        mult, col, tol = spec
        return mult, col, tol, None
    if len(spec) == 4:
        return spec  # type: ignore[return-value]
    raise ValueError(
        f"constraint spec must be a 3- or 4-tuple, got length {len(spec)}: {spec!r}"
    )


def build_attribute_constraints(
    bonds: pd.DataFrame,
    bm_weights: np.ndarray,
    constraint_specs: Sequence[tuple],
    holdings: np.ndarray | None = None,
) -> list[AttributeConstraint]:
    """Build benchmark-relative :class:`AttributeConstraint` objects.

    For each spec the function enumerates every unique category in *column*,
    computes the benchmark exposure, and creates bounds via
    :func:`_resolve_bounds`.  When a *buffer* is supplied alongside a non-zero
    *holdings* vector the bounds are hysteresis-aware (see
    :func:`_resolve_bounds`).

    When *column* is ``None`` the constraint is applied at the aggregate
    (whole-portfolio) level — a single row with the multiplier applied to
    every asset.

    When *column* is a sequence of column names (e.g.
    ``["CURRENCY", "MATURITY_BUCKET_10"]``) the categories are the compound
    key formed by ``"|"``-joining the listed columns; any asset with a null in
    any component column is excluded from every group.

    Parameters
    ----------
    bonds : pd.DataFrame
        Universe DataFrame.  Must contain each column referenced in
        *constraint_specs*, plus ``OAS`` and ``EFFECTIVE_DURATION`` for DTS/OAD
        multipliers.
    bm_weights : np.ndarray
        Benchmark weight vector of length ``len(bonds)``.
    constraint_specs : sequence of tuples
        Each element is ``(multiplier, column, tolerance)`` or
        ``(multiplier, column, tolerance, buffer)``.  *column* may be a
        single column name, a sequence of column names (compound key), or
        ``None`` (aggregate).  *tolerance* and *buffer* may be a symmetric
        ``float`` or an asymmetric ``(lower, upper)`` tuple (see
        :data:`Tolerance`, :data:`Buffer`).
    holdings : np.ndarray | None
        Current portfolio weight vector, length ``len(bonds)``, aligned to
        the *bonds* universe.  Required for hysteresis to take effect; when
        ``None`` (or all zeros) every constraint is built with the inner
        tolerance band only, identical to the legacy behaviour.

    Returns
    -------
    list[AttributeConstraint]
    """
    constraints: list[AttributeConstraint] = []
    n = len(bonds)
    h = None if holdings is None else np.asarray(holdings, dtype=float).ravel()
    if h is not None and h.shape[0] != n:
        raise ValueError(
            f"holdings length {h.shape[0]} != bonds length {n}"
        )

    def _bm_and_lhs(mask: np.ndarray) -> tuple[float, float | None]:
        if mask.dtype == bool:
            bm = float(bm_weights[mask].sum())
            lhs = None if h is None else float(h[mask].sum())
        else:
            bm = float((bm_weights * mask).sum())
            lhs = None if h is None else float((h * mask).sum())
        return bm, lhs

    for spec in constraint_specs:
        multiplier, col, tol, buf = _unpack_spec(tuple(spec))

        # --------------------------------------------------------------
        # Aggregate (whole-portfolio) constraint: col is None
        # --------------------------------------------------------------
        if col is None:
            if multiplier == Multiplier.WEIGHT:
                mask = np.ones(n, dtype=bool)
            elif multiplier == Multiplier.DTS:
                mask = (bonds["OAS"] * bonds["EFFECTIVE_DURATION"]).to_numpy(dtype=float)
            elif multiplier == Multiplier.OAD:
                mask = bonds["EFFECTIVE_DURATION"].to_numpy(dtype=float)
            else:
                raise ValueError(f"Unknown multiplier: {multiplier}")

            bm_exposure, lhs_h = _bm_and_lhs(mask)
            lo, hi, inner_lo, inner_hi, st_lo, st_hi, buf_lo, buf_hi = _resolve_bounds(
                bm_exposure, tol, buf, lhs_h,
            )
            constraints.append(
                AttributeConstraint(
                    multiplier=multiplier,
                    attribute="AGGREGATE",
                    value="ALL",
                    asset_mask=mask,
                    lower=lo,
                    upper=hi,
                    inner_lower=inner_lo,
                    inner_upper=inner_hi,
                    state_lower=st_lo,
                    state_upper=st_hi,
                    buffer_lower=buf_lo,
                    buffer_upper=buf_hi,
                )
            )
            continue

        # --------------------------------------------------------------
        # Per-category constraint: col is a column name or compound key
        # --------------------------------------------------------------
        label_vec, attribute_key = _resolve_label_vector(bonds, col)

        if multiplier == Multiplier.WEIGHT:
            masks = make_attribute_value_mask(label_vec)
        elif multiplier == Multiplier.DTS:
            masks = make_attribute_value_mask(
                label_vec,
                values=bonds["OAS"] * bonds["EFFECTIVE_DURATION"],
            )
        elif multiplier == Multiplier.OAD:
            masks = make_attribute_value_mask(
                label_vec,
                values=bonds["EFFECTIVE_DURATION"],
            )
        else:
            raise ValueError(f"Unknown multiplier: {multiplier}")

        for label, mask in masks.items():
            bm_exposure, lhs_h = _bm_and_lhs(mask)
            lo, hi, inner_lo, inner_hi, st_lo, st_hi, buf_lo, buf_hi = _resolve_bounds(
                bm_exposure, tol, buf, lhs_h,
            )
            constraints.append(
                AttributeConstraint(
                    multiplier=multiplier,
                    attribute=attribute_key,
                    value=label,
                    asset_mask=mask,
                    lower=lo,
                    upper=hi,
                    inner_lower=inner_lo,
                    inner_upper=inner_hi,
                    state_lower=st_lo,
                    state_upper=st_hi,
                    buffer_lower=buf_lo,
                    buffer_upper=buf_hi,
                )
            )

    logger.debug(
        "Built %d attribute constraints from %d specs",
        len(constraints),
        len(constraint_specs),
    )
    return constraints


# ---------------------------------------------------------------------------
# Portfolio feasibility diagnostic
# ---------------------------------------------------------------------------


@dataclass
class ConstraintViolation:
    """A single attribute constraint violated by a portfolio weight vector.

    Attributes
    ----------
    name : str
        Human-readable label (``"budget"`` or ``"<MULT>/<ATTR>/<VALUE>"``).
    lhs : float
        Aggregate exposure evaluated at the supplied weight vector.
    lower : float
        Constraint lower bound (absolute exposure).
    upper : float
        Constraint upper bound (absolute exposure).
    violation : float
        Signed magnitude: positive = above ``upper``, negative = below ``lower``.
    """

    name: str
    lhs: float
    lower: float
    upper: float
    violation: float


def diagnose_portfolio_feasibility(
    holdings: np.ndarray,
    attribute_constraints: list[AttributeConstraint],
    atol: float = 1e-8,
    budget_target: float = 1.0,
) -> list[ConstraintViolation]:
    """Evaluate the YAML attribute constraints against a portfolio weight vector.

    For each :class:`AttributeConstraint`, computes ``lhs = mask · holdings``
    (boolean mask sums weights; float mask is the OAD or DTS coefficient
    vector) and reports any row outside ``[lower, upper]``.  Also reports a
    ``budget`` row when ``sum(holdings)`` deviates from *budget_target*.

    This probe is unaffected by per-bond ``w_min`` / ``t_min`` floors and
    integer-linking constraints, so violations correspond directly to the
    user-configured attribute tolerances in the YAML ``constraints:`` block.
    Use it on the held portfolio at the start of a rebalance to identify
    which buckets the optimiser must move out of.

    Parameters
    ----------
    holdings : np.ndarray
        Portfolio weight vector (length *n*).
    attribute_constraints : list[AttributeConstraint]
        Attribute constraints built by :func:`build_attribute_constraints`
        from the YAML ``constraints:`` specs.
    atol : float
        Absolute tolerance for violation detection.  Default ``1e-8``.
    budget_target : float
        Target sum for ``holdings``.  Default ``1.0``.  Set to ``None`` to
        skip the budget check.

    Returns
    -------
    list[ConstraintViolation]
        Empty when every attribute row is satisfied within *atol*.
    """
    h = np.asarray(holdings, dtype=float).ravel()
    violations: list[ConstraintViolation] = []

    if budget_target is not None:
        total = float(h.sum())
        if abs(total - budget_target) > atol:
            violations.append(ConstraintViolation(
                name="budget",
                lhs=total,
                lower=budget_target,
                upper=budget_target,
                violation=total - budget_target,
            ))

    for sc in attribute_constraints:
        mask = np.asarray(sc.asset_mask)
        if mask.dtype == bool:
            lhs = float(h[mask].sum())
        else:
            lhs = float((h * mask).sum())

        if lhs < sc.lower - atol:
            violations.append(ConstraintViolation(
                name=f"{sc.multiplier.value}/{sc.attribute}/{sc.value}",
                lhs=lhs,
                lower=float(sc.lower),
                upper=float(sc.upper),
                violation=lhs - float(sc.lower),
            ))
        elif lhs > sc.upper + atol:
            violations.append(ConstraintViolation(
                name=f"{sc.multiplier.value}/{sc.attribute}/{sc.value}",
                lhs=lhs,
                lower=float(sc.lower),
                upper=float(sc.upper),
                violation=lhs - float(sc.upper),
            ))

    return violations


# ---------------------------------------------------------------------------
# MILP constraint assembly
# ---------------------------------------------------------------------------


def build_milp_constraints(
    n: int,
    config: MILPOptimizerConfig,
    holdings: np.ndarray,
    ticker_map: np.ndarray | None = None,
    n_tickers: int = 0,
) -> list[LinearConstraint]:
    """Assemble scipy ``LinearConstraint`` objects for the MILP.

    Variable layout: ``x = [w (n) | ρ (n) | t⁺ (n) | t⁻ (n) | τ (n) | π (K) | c (1)]``
    where *K* = ``n_tickers`` (0 when tickers are not used) and ``c`` is the
    scalar cash weight bounded by ``[cash_min, cash_max]``.

    Constraints
    -----------
    1.  Budget:           ``1ᵀw + c = 1``
    2.  Link upper:       ``wᵢ ≤ w_max · ρᵢ``
    3.  Link lower:       ``wᵢ ≥ w_min_i · ρᵢ``  where ``w_min_i = min(w_min, hᵢ)`` if ``hᵢ > 0`` else ``w_min``
    4.  Position count:   ``ρ_L ≤ 1ᵀρ ≤ ρ_U``
    5.  Trade balance:    ``t⁺ᵢ − t⁻ᵢ = wᵢ − hᵢ``
    6.  Attribute:        ``l ≤ Mᵀw ≤ u``      (WEIGHT)
                          ``l ≤ (M·D)ᵀw ≤ u``  (OAD)
                          ``l ≤ (M·D·S)ᵀw ≤ u``  (DTS)
                          where ``l, u`` are the *absolute* exposure bounds
                          produced by :func:`build_attribute_constraints`
                          (benchmark exposure already added to the active
                          tolerance, so the row reads on raw ``w`` not on
                          ``w − b``).
    7.  Trade-link lower: ``t⁺ᵢ + t⁻ᵢ ≥ t_min_i · τᵢ``  where ``t_min_i = min(t_min, hᵢ)`` if ``hᵢ > 0`` else ``t_min``
    8.  Trade-link upper: ``t⁺ᵢ + t⁻ᵢ ≤ max(hᵢ, w_max − hᵢ) · τᵢ``
    9.  Trade count:      ``τ_L ≤ 1ᵀτ ≤ τ_U``
    10. Bond→Ticker:      ``ρᵢ ≤ π_{k(i)}``
    11. Ticker→Bond:      ``πₖ ≤ Σ_{i∈Iₖ} ρᵢ``
    12. Ticker count:     ``π_L ≤ 1ᵀπ ≤ π_U``
    13. Ticker wt upper:  ``ωₖ ≤ ω_max · πₖ``   where ``ωₖ = Σ_{i∈Mₖ} wᵢ``
    14. Ticker wt lower:  ``ωₖ ≥ ω_min · πₖ``

    Parameters
    ----------
    n : int
        Number of assets.
    config : MILPOptimizerConfig
        Optimiser configuration (position limits, bounds, attribute constraints).
    holdings : np.ndarray
        Current portfolio weights *h* (length *n*).
    ticker_map : np.ndarray, optional
        Integer array of length *n* mapping each bond to a ticker index
        ``0 … K-1``.  Required when ``n_tickers > 0``.
    n_tickers : int
        Number of unique tickers *K*.  Default 0 (no ticker constraints).

    Returns
    -------
    list[LinearConstraint]
    """
    constraints: list[LinearConstraint] = []
    Z = csc_array((n, n))  # n×n zero block

    # 1) Budget: sum(w) = 1
    #    [1…1 | 0…0 | 0…0 | 0…0 | 0…0] x = 1
    A_budget = csc_array(
        np.concatenate([np.ones(n), np.zeros(4 * n)]).reshape(1, -1)
    )
    constraints.append(LinearConstraint(A_budget, lb=1.0, ub=1.0))

    # 2) Position-linking (upper):  w_i − w_max·ρ_i ≤ 0
    #    [I | −w_max·I | 0 | 0 | 0] x ≤ 0
    A_link_upper = sphstack([
        speye(n, format="csc"),
        -config.w_max * speye(n, format="csc"),
        Z, Z, Z,
    ])
    constraints.append(LinearConstraint(A_link_upper, lb=-np.inf, ub=0.0))

    # 3) Position-linking (lower):  w_i ≥ w_min_i · ρ_i
    #    Per-bond floor grandfathers drifted holdings: w_min_i = min(w_min, h_i)
    #    when h_i > 0, else w_min. Allows held positions that have drifted
    #    below w_min to be retained without forcing a top-up or full sell.
    #    [−I | diag(w_min_i) | 0 | 0 | 0] x ≤ 0
    w_lb = np.where(holdings > 0, np.minimum(config.w_min, holdings), config.w_min)
    A_link_lower = sphstack([
        -speye(n, format="csc"),
        spdiags(w_lb, format="csc"),
        Z, Z, Z,
    ])
    constraints.append(LinearConstraint(A_link_lower, lb=-np.inf, ub=0.0))

    # 4) Position count:  ρ_L ≤ sum(ρ) ≤ ρ_U
    #    [0…0 | 1…1 | 0…0 | 0…0 | 0…0] x ∈ [ρ_L, ρ_U]
    A_count = csc_array(
        np.concatenate([np.zeros(n), np.ones(n), np.zeros(3 * n)]).reshape(1, -1)
    )
    constraints.append(
        LinearConstraint(
            A_count,
            lb=float(config.min_positions),
            ub=float(config.max_positions),
        )
    )

    # 5) Trade-balance equality:  t⁺ − t⁻ = w − h  ⟺  w − t⁺ + t⁻ = h
    #    [I | 0 | −I | I | 0] x = h
    A_trade = sphstack([
        speye(n, format="csc"),
        Z,
        -speye(n, format="csc"),
        speye(n, format="csc"),
        Z,
    ])
    constraints.append(LinearConstraint(A_trade, lb=holdings, ub=holdings))

    # 6) Attribute constraints:  l ≤ maskᵀw ≤ u
    #    [mask | 0 | 0 | 0 | 0] x ∈ [l, u]
    for sc in config.attribute_constraints:
        mask = np.asarray(sc.asset_mask, dtype=float)
        A_attr = csc_array(
            np.concatenate([mask, np.zeros(4 * n)]).reshape(1, -1)
        )
        constraints.append(LinearConstraint(A_attr, lb=sc.lower, ub=sc.upper))

    # 7) Trade-link lower:  t⁺ᵢ + t⁻ᵢ ≥ t_min_i · τᵢ
    #    Per-bond trade floor grandfathers drifted holdings:
    #    t_min_i = min(t_min, h_i) when h_i > 0, else t_min.
    #    Allows full liquidation of positions that have drifted below t_min.
    #    [0 | 0 | I | I | −diag(t_min_i)] x ≥ 0
    t_lb = np.where(holdings > 0, np.minimum(config.t_min, holdings), config.t_min)
    A_tau_lo = sphstack([
        Z, Z,
        speye(n, format="csc"),
        speye(n, format="csc"),
        csc_array(np.diag(-t_lb)),
    ])
    constraints.append(LinearConstraint(A_tau_lo, lb=0.0, ub=np.inf))

    # 8) Trade-link upper:  t⁺ᵢ + t⁻ᵢ ≤ Mᵢ · τᵢ
    #    where Mᵢ = max(hᵢ, w_max − hᵢ)
    #    [0 | 0 | I | I | −diag(M)] x ≤ 0
    M_big = np.maximum(holdings, config.w_max - holdings)
    A_tau_hi = sphstack([
        Z, Z,
        speye(n, format="csc"),
        speye(n, format="csc"),
        csc_array(np.diag(-M_big)),
    ])
    constraints.append(LinearConstraint(A_tau_hi, lb=-np.inf, ub=0.0))

    # 9) Trade count:  τ_L ≤ sum(τ) ≤ τ_U
    #    [0…0 | 0…0 | 0…0 | 0…0 | 1…1] x ∈ [τ_L, τ_U]
    A_trade_count = csc_array(
        np.concatenate([np.zeros(4 * n), np.ones(n)]).reshape(1, -1)
    )
    constraints.append(
        LinearConstraint(
            A_trade_count,
            lb=float(config.min_trades),
            ub=float(config.max_trades),
        )
    )

    # -- ticker constraints (only when ticker_map is provided) --------------
    K = n_tickers
    if K > 0 and ticker_map is not None:
        total_cols = 5 * n + K

        # Pad existing constraint matrices from 5n to 5n+K columns
        for idx in range(len(constraints)):
            lc = constraints[idx]
            A_old = lc.A
            A_padded = sphstack([A_old, csc_array((A_old.shape[0], K))])
            constraints[idx] = LinearConstraint(A_padded, lb=lc.lb, ub=lc.ub)

        # 10) Bond→Ticker:  ρ_i ≤ π_{k(i)}  →  ρ_i − π_{k(i)} ≤ 0
        bond_rows = np.arange(n)
        A_bond_tkr = csc_array(
            (np.concatenate([np.ones(n), -np.ones(n)]),
             (np.concatenate([bond_rows, bond_rows]),
              np.concatenate([n + np.arange(n), 5 * n + ticker_map]))),
            shape=(n, total_cols),
        )
        constraints.append(LinearConstraint(A_bond_tkr, lb=-np.inf, ub=0.0))

        # 11) Ticker→Bond:  π_k ≤ Σ_{i∈I_k} ρ_i  →  −Σρ_i + π_k ≤ 0
        A_tkr_bond = csc_array(
            (np.concatenate([-np.ones(n), np.ones(K)]),
             (np.concatenate([ticker_map, np.arange(K)]),
              np.concatenate([n + np.arange(n), 5 * n + np.arange(K)]))),
            shape=(K, total_cols),
        )
        constraints.append(LinearConstraint(A_tkr_bond, lb=-np.inf, ub=0.0))

        # 12) Ticker count:  π_L ≤ 1ᵀπ ≤ π_U
        A_tkr_count = csc_array(
            np.concatenate([np.zeros(5 * n), np.ones(K)]).reshape(1, -1)
        )
        constraints.append(
            LinearConstraint(
                A_tkr_count,
                lb=float(config.min_tickers),
                ub=float(config.max_tickers),
            )
        )

        # 13) Ticker weight upper:  Σ_{i∈I_k} w_i ≤ W_max · π_k
        A_tkr_wu = csc_array(
            (np.concatenate([np.ones(n), -config.ticker_w_max * np.ones(K)]),
             (np.concatenate([ticker_map, np.arange(K)]),
              np.concatenate([np.arange(n), 5 * n + np.arange(K)]))),
            shape=(K, total_cols),
        )
        constraints.append(LinearConstraint(A_tkr_wu, lb=-np.inf, ub=0.0))

        # 14) Ticker weight lower:  Σ_{i∈I_k} w_i ≥ W_min · π_k
        #     → −Σw_i + W_min · π_k ≤ 0
        A_tkr_wl = csc_array(
            (np.concatenate([-np.ones(n), config.ticker_w_min * np.ones(K)]),
             (np.concatenate([ticker_map, np.arange(K)]),
              np.concatenate([np.arange(n), 5 * n + np.arange(K)]))),
            shape=(K, total_cols),
        )
        constraints.append(LinearConstraint(A_tkr_wl, lb=-np.inf, ub=0.0))

    # -- final cash-column padding ------------------------------------------
    # Append a single scalar variable c to every constraint matrix.
    # Budget (constraints[0]) gets coefficient 1 so 1ᵀw + c = 1.
    # All other rows get 0 — cash is structurally invisible to attribute,
    # position-count, trade-count, ticker, and KRD constraints.
    for idx in range(len(constraints)):
        lc = constraints[idx]
        A_old = lc.A
        n_rows = A_old.shape[0]
        if idx == 0:
            cash_col = csc_array(np.ones((n_rows, 1)))
        else:
            cash_col = csc_array((n_rows, 1))
        A_padded = sphstack([A_old, cash_col])
        constraints[idx] = LinearConstraint(A_padded, lb=lc.lb, ub=lc.ub)

    return constraints