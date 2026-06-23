"""Treasury futures KRD hedge overlay.

Implements a physical futures hedge that neutralises the portfolio's
interest-rate duration mismatch versus the benchmark.  Hedge ratios
are derived from the relative KRD vector (portfolio − benchmark) at
each futures tenor node.

The module provides:

- ``FuturesContract`` / ``TREASURY_FUTURES``: contract specifications
- ``FuturesHedgeConfig``: YAML-driven configuration
- ``FuturesHedgeState``: per-period hedge snapshot
- Pure functions: ``compute_futures_dv01``, ``compute_relative_krd``,
  ``compute_hedge_ratios``, ``compute_hedge_return``
- ``FuturesHistorySource`` / ``CachedFuturesHistorySource``: data access
- ``FuturesHedgeEngine``: orchestration wrapper
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuturesContract:
    """Specification for a single government bond futures contract."""

    ticker: str  # e.g. "TU1"
    tenor: float  # bucket tenor in years (2, 5, 7, 10, 20, 30)
    contract_size: float  # notional in contract currency (e.g. 200_000 USD, 100_000 EUR)
    notional_coupon: float = 0.06  # CME / Eurex 6% notional (Buxl: 6% post-Sep-2021)
    currency: str = "USD"  # contract denomination currency (ISO 4217)


TREASURY_FUTURES: tuple[FuturesContract, ...] = (
    FuturesContract("TU1", 2.0, 200_000, currency="USD"),
    FuturesContract("FV1", 5.0, 100_000, currency="USD"),
    FuturesContract("TY1", 7.0, 100_000, currency="USD"),
    FuturesContract("UXY1", 10.0, 100_000, currency="USD"),
    FuturesContract("US1", 20.0, 100_000, currency="USD"),
    FuturesContract("WN1", 30.0, 100_000, currency="USD"),
)

# Eurex EUR-denominated government bond futures. Buxl notional coupon
# was redesigned from 4% to 6% effective Sep-2021; all entries here use
# 6%, valid for backtest windows fully after Sep-2021.
EUREX_FUTURES: tuple[FuturesContract, ...] = (
    FuturesContract("DU1", 2.0, 100_000, currency="EUR"),   # Schatz
    FuturesContract("OE1", 5.0, 100_000, currency="EUR"),   # Bobl
    FuturesContract("RX1", 10.0, 100_000, currency="EUR"),  # Bund
    FuturesContract("UB1", 30.0, 100_000, currency="EUR"),  # Buxl (6% post-Sep-2021)
)

DEFAULT_FUTURES_BY_CURRENCY: dict[str, tuple[FuturesContract, ...]] = {
    "USD": TREASURY_FUTURES,
    "EUR": EUREX_FUTURES,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuturesHedgeConfig:
    """Configuration for the futures hedge overlay.

    Contracts are organised per currency: each currency's bonds are
    hedged with that currency's contract set. The default preserves
    single-currency USD behaviour exactly.

    ``duration_calc`` selects how each bond's rate exposure is mapped to
    the futures nodes:

    - ``"krd"``: full key-rate-duration vector from the term-structure
      model (uses the backtest's ``krd_method`` = ``"zero"`` | ``"par"``).
    - ``"bullet"``: the bond's entire ``EFFECTIVE_DURATION`` is assigned
      to the single node nearest its maturity (degenerate one-hot KRD
      row); the term-structure model is bypassed.
    """

    enabled: bool = False
    hedge_scale: float = 1.0
    missing_data_policy: Literal["raise", "skip"] = "raise"
    duration_calc: Literal["krd", "bullet"] = "krd"
    contracts_by_currency: dict[str, tuple[FuturesContract, ...]] = field(
        default_factory=lambda: {"USD": TREASURY_FUTURES}
    )

    @property
    def currencies(self) -> tuple[str, ...]:
        return tuple(sorted(self.contracts_by_currency.keys()))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuturesHedgeState:
    """Snapshot of hedge state at a single rebalance date.

    Established at date T; consumed at T+1 to compute hedge P&L.
    Ticker keys are globally unique across currencies; the
    ``contract_currencies`` map enables per-currency FX overlay of P&L.
    """

    as_of: date
    hedge_ratios: dict[str, float]  # ticker → contracts short per $1 NAV (base ccy)
    start_prices: dict[str, float]  # ticker → PX_LAST at as_of (contract ccy)
    contract_sizes: dict[str, float]  # ticker → CS (contract ccy units)
    contract_currencies: dict[str, str] = field(default_factory=dict)  # ticker → ISO ccy
    dv01s: dict[str, float] = field(default_factory=dict)  # ticker → DV01 per contract (contract ccy)


# ---------------------------------------------------------------------------
# Pure computation functions
# ---------------------------------------------------------------------------


def compute_futures_dv01(
    duration_value: float,
    price: float,
    contract_size: float,
    *,
    notional_coupon: float = 0.06,
) -> float:
    """Compute DV01 for a single futures contract.

    FUT_EQV_DUR_NOTL is Macaulay duration of the 6% notional bond.
    Convert to modified duration under semi-annual convention, then:

        DV01 = D_mod × (P / 100) × CS × 0.0001

    Parameters
    ----------
    duration_value : float
        Macaulay duration (years) from FUT_EQV_DUR_NOTL.
    price : float
        Futures price (percentage of par).
    contract_size : float
        Contract notional in USD.
    notional_coupon : float
        Notional coupon rate (default 0.06).

    Returns
    -------
    float
        Dollar DV01 per contract.
    """
    d_mod = duration_value / (1.0 + notional_coupon / 2.0)
    return d_mod * (price / 100.0) * contract_size * 0.0001


def nearest_tenor_index(maturity_years: float, tenors: Sequence[float]) -> int:
    """Index of the contract tenor closest to a bond's maturity.

    Bullet-hedge bucketing rule: each bond is hedged with the single
    futures contract whose node tenor minimises ``|maturity - tenor|``.
    Ties (equidistant between two nodes) resolve to the shorter tenor
    via ``np.argmin`` returning the first minimum.

    Parameters
    ----------
    maturity_years : float
        Bond years-to-maturity from the rebalance date.
    tenors : Sequence[float]
        Contract node tenors for the bond's currency (need not be sorted).

    Returns
    -------
    int
        Index into ``tenors`` of the nearest node.
    """
    return int(np.argmin(np.abs(np.asarray(tenors, dtype=float) - maturity_years)))


def compute_relative_krd(
    port_weights: np.ndarray,
    port_krd_matrix: np.ndarray,
    bmk_weights: np.ndarray,
    bmk_krd_matrix: np.ndarray,
) -> np.ndarray:
    """Compute relative KRD vector (portfolio − benchmark).

    Parameters
    ----------
    port_weights : ndarray, shape (N,)
        Portfolio weights (fraction of NAV).
    port_krd_matrix : ndarray, shape (N, K)
        Per-bond KRD vectors for portfolio bonds.
    bmk_weights : ndarray, shape (M,)
        Benchmark weights (fraction of NAV).
    bmk_krd_matrix : ndarray, shape (M, K)
        Per-bond KRD vectors for benchmark bonds.

    Returns
    -------
    ndarray, shape (K,)
        Relative KRD at each tenor node.
    """
    port_krd = port_weights @ port_krd_matrix  # (K,)
    bmk_krd = bmk_weights @ bmk_krd_matrix  # (K,)
    return port_krd - bmk_krd


def compute_hedge_ratios(
    relative_krd: np.ndarray,
    futures_dv01s: np.ndarray,
    *,
    hedge_scale: float = 1.0,
) -> np.ndarray:
    """Compute number of contracts short per $1 NAV at each bucket.

    Parameters
    ----------
    relative_krd : ndarray, shape (K,)
        Relative KRD (portfolio − benchmark) per tenor node.
    futures_dv01s : ndarray, shape (K,)
        DV01 per contract at each tenor node.
    hedge_scale : float
        Scaling factor (1.0 = full hedge).

    Returns
    -------
    ndarray, shape (K,)
        Hedge ratios. Positive = short futures, Negative = long futures.
    """
    # N_k = KRD_rel(T_k) × 0.0001 / DV01_contract_k
    return hedge_scale * (relative_krd * 0.0001) / futures_dv01s


def compute_hedge_return(
    hedge_ratios: np.ndarray,
    price_start: np.ndarray,
    price_end: np.ndarray,
    contract_sizes: np.ndarray,
) -> float:
    """Compute total hedge P&L per $1 NAV.

    Short futures profit when prices decline (rates rise):
        R_hedge = sum_k N_k × (P_start_k − P_end_k) × CS_k / 100

    Parameters
    ----------
    hedge_ratios : ndarray, shape (K,)
        Contracts short per $1 NAV (positive = short).
    price_start : ndarray, shape (K,)
        Futures prices at period start (pct of par).
    price_end : ndarray, shape (K,)
        Futures prices at period end (pct of par).
    contract_sizes : ndarray, shape (K,)
        Contract notional in USD.

    Returns
    -------
    float
        Total hedge return per $1 NAV.
    """
    return float(
        np.sum(hedge_ratios * (price_start - price_end) * contract_sizes / 100.0)
    )


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

# Per-date observation: {ticker: {field: value}}
FuturesObservation = dict[str, dict[str, float]]
# Full cache: {date: FuturesObservation}
FuturesCache = dict[date, FuturesObservation]


class FuturesHistorySource:
    """Snowflake-backed source for government futures history.

    Queries ``quant.research.GOVT_FUTURES_HISTORY`` for all rebalance
    dates in a single round-trip.

    Parameters
    ----------
    sf_config : object
        Snowflake connection configuration.
    """

    def __init__(self, sf_config: object) -> None:
        from archipelago.data.connectors.snowflake_client import SnowflakeClient

        self._client = SnowflakeClient.from_config(sf_config)
        self._cache: FuturesCache = {}

    def load_all(self, dates: list[date]) -> None:
        """Bulk-load futures history for *dates* into the internal cache."""
        if not dates:
            return

        import json
        from importlib import resources

        iso_dates = sorted({d.strftime("%Y-%m-%d") for d in dates})
        dates_json = json.dumps(iso_dates)

        pkg_root = resources.files("archipelago")
        sql_path = pkg_root.joinpath("data/connectors/sql/futures_history.sql")
        sql = sql_path.read_text(encoding="utf-8")
        sql = sql.replace("%dates%", dates_json)

        df = self._client.execute_query(sql)
        df.columns = [c.upper() for c in df.columns]

        import pandas as pd_mod

        for rebal_date, grp in df.groupby("REBAL_DATE"):
            d = pd_mod.Timestamp(rebal_date).date()
            obs: FuturesObservation = {}
            for row in grp.itertuples(index=False):
                ticker = str(row.CONTRACT)
                if ticker not in obs:
                    obs[ticker] = {}
                obs[ticker]["PX_LAST"] = float(row.PX_LAST)
                obs[ticker]["FUT_EQV_DUR_NOTL"] = float(row.FUT_EQV_DUR_NOTL)
            self._cache[d] = obs

        logger.info(
            "FuturesHistorySource: loaded %d dates (%d total rows)",
            len(self._cache), len(df),
        )

    def get_observation(self, as_of: date) -> FuturesObservation:
        """Return futures data for a single date."""
        obs = self._cache.get(as_of)
        if obs is None:
            raise KeyError(
                f"FuturesHistorySource: no data cached for {as_of}; "
                f"available={sorted(self._cache.keys())}"
            )
        return obs

    def extract_cache(self) -> FuturesCache:
        """Return the internal cache for shipping to worker processes."""
        return dict(self._cache)


class CachedFuturesHistorySource:
    """Picklable cache-backed futures history source for worker processes.

    Parameters
    ----------
    cache : FuturesCache
        Pre-populated ``{date: {ticker: {field: value}}}`` dict.
    """

    def __init__(self, cache: FuturesCache) -> None:
        self._cache = cache

    def get_observation(self, as_of: date) -> FuturesObservation:
        obs = self._cache.get(as_of)
        if obs is None:
            raise KeyError(
                f"CachedFuturesHistorySource: no data cached for {as_of}; "
                f"available={sorted(self._cache.keys())}"
            )
        return obs


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FuturesHedgeEngine:
    """Orchestration wrapper for the futures hedge overlay.

    Combines data source access with hedge computation.  Designed to be
    instantiated once and called per-period from the backtest loop.

    Parameters
    ----------
    config : FuturesHedgeConfig
        Hedge configuration.
    source : FuturesHistorySource | CachedFuturesHistorySource
        Data provider for futures prices and durations.
    """

    def __init__(
        self,
        config: FuturesHedgeConfig,
        source: FuturesHistorySource | CachedFuturesHistorySource,
    ) -> None:
        self.config = config
        self.source = source
        # Per-currency vector caches (positionally aligned within each ccy).
        self._contracts_by_ccy: dict[str, tuple[FuturesContract, ...]] = {
            ccy.upper(): tuple(contracts)
            for ccy, contracts in config.contracts_by_currency.items()
        }
        self._tickers_by_ccy: dict[str, list[str]] = {
            ccy: [c.ticker for c in contracts]
            for ccy, contracts in self._contracts_by_ccy.items()
        }
        self._tenors_by_ccy: dict[str, np.ndarray] = {
            ccy: np.array([c.tenor for c in contracts])
            for ccy, contracts in self._contracts_by_ccy.items()
        }
        self._sizes_by_ccy: dict[str, np.ndarray] = {
            ccy: np.array([c.contract_size for c in contracts])
            for ccy, contracts in self._contracts_by_ccy.items()
        }
        self._coupons_by_ccy: dict[str, np.ndarray] = {
            ccy: np.array([c.notional_coupon for c in contracts])
            for ccy, contracts in self._contracts_by_ccy.items()
        }
        # Flat union of tickers/contracts (used for source observation lookup
        # and for the worker-side serialization).
        self._all_contracts: tuple[FuturesContract, ...] = tuple(
            c for cs in self._contracts_by_ccy.values() for c in cs
        )
        self._all_tickers: list[str] = [c.ticker for c in self._all_contracts]
        # Sanity: ticker uniqueness across currencies.
        if len(set(self._all_tickers)) != len(self._all_tickers):
            dupes = sorted({t for t in self._all_tickers if self._all_tickers.count(t) > 1})
            raise ValueError(
                f"FuturesHedgeEngine: duplicate tickers across currency buckets: {dupes}"
            )

    @property
    def currencies(self) -> tuple[str, ...]:
        return tuple(sorted(self._contracts_by_ccy.keys()))

    def _fetch_prices_and_durations(
        self,
        as_of: date,
        contracts: tuple[FuturesContract, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Look up (prices, durations) for *contracts* at *as_of*.

        Honours ``missing_data_policy``: missing/invalid observations
        either raise or are zeroed out with a logged warning.
        """
        obs = self.source.get_observation(as_of)
        n = len(contracts)
        prices = np.zeros(n)
        durations = np.zeros(n)
        for i, contract in enumerate(contracts):
            ticker_data = obs.get(contract.ticker)
            if ticker_data is None:
                if self.config.missing_data_policy == "raise":
                    raise ValueError(
                        f"FuturesHedgeEngine: no data for {contract.ticker} on {as_of}"
                    )
                logger.warning(
                    "FuturesHedgeEngine: no data for %s on %s — skipping",
                    contract.ticker, as_of,
                )
                continue
            px = ticker_data.get("PX_LAST", 0.0)
            dur = ticker_data.get("FUT_EQV_DUR_NOTL", 0.0)
            if px <= 0.0 or dur <= 0.0:
                if self.config.missing_data_policy == "raise":
                    raise ValueError(
                        f"FuturesHedgeEngine: invalid PX_LAST={px} or "
                        f"FUT_EQV_DUR_NOTL={dur} for {contract.ticker} on {as_of}"
                    )
                logger.warning(
                    "FuturesHedgeEngine: invalid data for %s on %s (px=%.2f, dur=%.2f)",
                    contract.ticker, as_of, px, dur,
                )
                continue
            prices[i] = px
            durations[i] = dur
        return prices, durations

    def build_state(
        self,
        as_of: date,
        port_weights: np.ndarray,
        port_krd_matrix_by_ccy: dict[str, np.ndarray],
        bmk_weights: np.ndarray,
        bmk_krd_matrix_by_ccy: dict[str, np.ndarray],
    ) -> FuturesHedgeState:
        """Build hedge state for a rebalance date (per-currency dispatch).

        Each currency's bonds contribute non-zero rows to that currency's
        KRD matrix; bonds of other currencies contribute zero rows. This
        keeps the ``port_weights @ krd_matrix`` contraction unchanged
        while isolating per-currency rate exposure.

        Parameters
        ----------
        as_of : date
            Current rebalance date.
        port_weights : ndarray, shape (N,)
            Post-optimization portfolio weights.
        port_krd_matrix_by_ccy : dict[str, ndarray]
            Per-currency KRD matrix, shape ``(N, K_ccy)`` each.
        bmk_weights : ndarray, shape (N,)
            Benchmark weights (must share row order with *port_weights*).
        bmk_krd_matrix_by_ccy : dict[str, ndarray]
            Per-currency benchmark KRD matrix, shape ``(N, K_ccy)`` each.

        Returns
        -------
        FuturesHedgeState
        """
        hedge_ratios: dict[str, float] = {}
        start_prices: dict[str, float] = {}
        contract_sizes: dict[str, float] = {}
        contract_currencies: dict[str, str] = {}
        dv01s_out: dict[str, float] = {}

        # Surface any currency with bond exposure but no contracts.
        all_ccys = set(port_krd_matrix_by_ccy) | set(bmk_krd_matrix_by_ccy)
        unhedged = sorted(c for c in all_ccys if c not in self._contracts_by_ccy)
        if unhedged:
            logger.warning(
                "FuturesHedgeEngine.build_state: no contracts configured for "
                "currencies %s on %s — rate exposure in these currencies will "
                "be left unhedged",
                unhedged, as_of,
            )

        for ccy, contracts in self._contracts_by_ccy.items():
            tickers = self._tickers_by_ccy[ccy]
            sizes = self._sizes_by_ccy[ccy]
            coupons = self._coupons_by_ccy[ccy]
            n_contracts = len(contracts)

            port_krd_mat = port_krd_matrix_by_ccy.get(ccy)
            bmk_krd_mat = bmk_krd_matrix_by_ccy.get(ccy)
            if port_krd_mat is None or bmk_krd_mat is None:
                # No KRD inputs for this currency → no positions taken; still
                # publish zero ratios so downstream state lookups don't KeyError.
                # Skip data fetch to preserve prior behavior (don't hit source
                # when there is no exposure to hedge).
                prices = np.zeros(n_contracts)
                dv01s = np.zeros(n_contracts)
                ratios = np.zeros(n_contracts)
            else:
                prices, durations = self._fetch_prices_and_durations(as_of, contracts)
                dv01s = np.array([
                    compute_futures_dv01(
                        durations[i], prices[i], sizes[i],
                        notional_coupon=coupons[i],
                    )
                    for i in range(n_contracts)
                ])
                rel_krd = compute_relative_krd(
                    port_weights, port_krd_mat, bmk_weights, bmk_krd_mat,
                )
                ratios = np.zeros(n_contracts)
                valid_mask = dv01s > 0.0
                if valid_mask.any():
                    ratios[valid_mask] = compute_hedge_ratios(
                        rel_krd[valid_mask],
                        dv01s[valid_mask],
                        hedge_scale=self.config.hedge_scale,
                    )

            for i, ticker in enumerate(tickers):
                hedge_ratios[ticker] = float(ratios[i])
                start_prices[ticker] = float(prices[i])
                contract_sizes[ticker] = float(sizes[i])
                contract_currencies[ticker] = ccy
                dv01s_out[ticker] = float(dv01s[i])

        return FuturesHedgeState(
            as_of=as_of,
            hedge_ratios=hedge_ratios,
            start_prices=start_prices,
            contract_sizes=contract_sizes,
            contract_currencies=contract_currencies,
            dv01s=dv01s_out,
        )

    def realized_return(
        self,
        prev_state: FuturesHedgeState,
        curr_date: date,
        *,
        fx_overlay: Callable[[str, date, date], float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Realized hedge P&L from previous state to current date.

        Splits state by ``contract_currencies`` and computes P&L
        per currency in that currency's units. When ``fx_overlay`` is
        provided, each currency's P&L is multiplied by the overlay
        scalar before aggregation, converting to base currency under
        the locked-forward FX hedge convention.

        Parameters
        ----------
        prev_state : FuturesHedgeState
            Hedge state established at the previous rebalance.
        curr_date : date
            Current rebalance date (hedge P&L realized here).
        fx_overlay : callable | None
            ``(currency, prev_date, curr_date) -> float`` factor that
            converts hedge P&L from contract currency to base currency.
            ``None`` (default) leaves P&L in contract currency units —
            equivalent to assuming base currency == contract currency.

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(total_in_base, by_currency)``. ``by_currency`` values are
            in **base** currency after overlay (matches the sum).
        """
        obs = self.source.get_observation(curr_date)
        prev_date = prev_state.as_of

        by_ccy: dict[str, float] = {}
        for ccy, contracts in self._contracts_by_ccy.items():
            tickers = self._tickers_by_ccy[ccy]
            n_contracts = len(contracts)
            ratios = np.array([prev_state.hedge_ratios.get(t, 0.0) for t in tickers])
            price_start = np.array([prev_state.start_prices.get(t, 0.0) for t in tickers])
            sizes = np.array([prev_state.contract_sizes.get(t, 0.0) for t in tickers])
            price_end = np.empty(n_contracts)
            for i, contract in enumerate(contracts):
                ticker_data = obs.get(contract.ticker)
                if ticker_data is None or ticker_data.get("PX_LAST", 0.0) <= 0.0:
                    if self.config.missing_data_policy == "raise":
                        raise ValueError(
                            f"FuturesHedgeEngine: no end price for "
                            f"{contract.ticker} on {curr_date}"
                        )
                    price_end[i] = price_start[i]  # no P&L if data missing
                else:
                    price_end[i] = ticker_data["PX_LAST"]

            ccy_return = compute_hedge_return(ratios, price_start, price_end, sizes)
            if fx_overlay is not None:
                ccy_return *= float(fx_overlay(ccy, prev_date, curr_date))
            by_ccy[ccy] = float(ccy_return)

        total = float(sum(by_ccy.values()))
        return total, by_ccy
