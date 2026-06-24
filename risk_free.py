"""Risk-free return computation for excess-return decomposition.

Provides benchmark modes for computing per-bond risk-free returns
from the USD swap curve, enabling the split::

    r_excess_i = r_total_i - r_rf_i

Supported modes (see :class:`RfBenchmarkMode`):

- **CASHFLOW_MATCHED**: Exact cashflow-matched RF twin (Option A).
- **KRD**: Key-rate-duration matched (Option D).
comment
"""

from __future__ import annotations

import abc
import logging
from datetime import date
from typing import TYPE_CHECKING, Callable

import numpy as np

from archipelago.instruments.discount import (
    CURVE_TIME_BASIS,
    RfBenchmarkMode,
    USD_SWAP_HEDGE_CONVENTION,
    build_krd_synth_cache,
    calc_cf_matched_realized_return,
    calc_krd_matched_realized_return,
)

if TYPE_CHECKING:
    import pandas as pd

    from archipelago.instruments.discount import KrdHedgeConvention

logger = logging.getLogger(__name__)

# Type aliases
ZeroRateFunc = Callable[[float], float]
DiscountFunc = Callable[[float], float]


# ---------------------------------------------------------------------------
# Abstract source
# ---------------------------------------------------------------------------


class RiskFreeCurveSource(abc.ABC):
    """ABC for providers of the zero-rate and discount functions at a given date."""

    @abc.abstractmethod
    def get_zero_rate_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> ZeroRateFunc:
        """Return a callable ``z(t)`` for the given valuation date.

        Parameters
        ----------
        as_of : date
            Valuation date.
        currency : str | None
            ISO currency code selecting which curve to return. ``None``
            (default) selects the source's base/default currency.
            Single-currency sources ignore this argument.

        Returns
        -------
        ZeroRateFunc
            Callable mapping tenor (years, float) to continuously
            compounded zero rate.
        """

    @abc.abstractmethod
    def get_discount_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> DiscountFunc:
        """Return a callable ``D(t)`` for the given valuation date.

        Parameters
        ----------
        as_of : date
            Valuation date.
        currency : str | None
            ISO currency code selecting which curve to return. ``None``
            (default) selects the source's base/default currency.
            Single-currency sources ignore this argument.

        Returns
        -------
        DiscountFunc
            Callable mapping tenor (years, float) to the risk-free
            discount factor D(0, t).
        """


# ---------------------------------------------------------------------------
# Concrete: SwapCurveBuilder-backed source
# ---------------------------------------------------------------------------


class SwapCurveSource(RiskFreeCurveSource):
    """Derives ``z(t)`` from :class:`~archipelago.instruments.discount.SwapCurveBuilder`.

    A per-date in-memory cache holds the factory output so that the
    ``get_zero_rate_function()`` / ``get_discount_function()`` pair share
    a single ``builder_factory(as_of)`` call per date.  Without this
    cache the factory (which builds and calibrates a swap curve from
    market data) would run twice per date.

    Parameters
    ----------
    builder_factory : Callable[[date], object]
        Factory returning a *built and calibrated* ``SwapCurveBuilder``
        instance for the requested date.  The builder must already have
        ``build_curve()`` and ``calibrate_cubic_spline()`` called.  The
        factory is responsible for sourcing market data (SOFR, futures,
        swaps) for that date.
    """

    def __init__(self, builder_factory: Callable[[date], dict]) -> None:
        self._factory = builder_factory
        self._cache: dict[date, dict] = {}

    def _get_calibration(self, as_of: date) -> dict:
        """Return the cached calibration dict for *as_of*, building if needed."""
        cached = self._cache.get(as_of)
        if cached is None:
            cached = self._factory(as_of)
            self._cache[as_of] = cached
        return cached

    def get_zero_rate_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> ZeroRateFunc:
        """Build (or fetch from cache) and return the zero-rate function for *as_of*.

        The ``currency`` argument is accepted for interface compatibility
        with :class:`MultiCurrencySwapCurveSource` and is ignored: a
        single-currency source returns its sole calibrated curve.
        """
        del currency  # single-currency source
        return self._get_calibration(as_of)["zero_rate_function"]

    def get_discount_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> DiscountFunc:
        """Build (or fetch from cache) and return the discount function D(t) for *as_of*.

        The ``currency`` argument is accepted for interface compatibility
        with :class:`MultiCurrencySwapCurveSource` and is ignored.
        """
        del currency  # single-currency source
        return self._get_calibration(as_of)["discount_function"]

    def extract_curve_cache(
        self, dates: list[date],
    ) -> dict[date, tuple[ZeroRateFunc, DiscountFunc]]:
        """Return a picklable snapshot of ``(zero_rate_function, discount_function)`` per date.

        Used by parallel simulation to ship pre-built curve objects to
        worker processes without sending the (closure-bearing) factory.

        Any date not yet cached is built on demand and added to ``_cache``.
        """
        snapshot: dict[date, tuple[ZeroRateFunc, DiscountFunc]] = {}
        for d in dates:
            calib = self._get_calibration(d)
            snapshot[d] = (
                calib["zero_rate_function"],
                calib["discount_function"],
            )
        return snapshot


# ---------------------------------------------------------------------------
# Concrete: cache-backed source for worker processes
# ---------------------------------------------------------------------------


class CachedRfCurveSource(RiskFreeCurveSource):
    """RF curve source backed by a pre-built ``(zero_rate, discount)`` cache.

    Used inside worker processes during parallel simulation: the parent
    process builds and calibrates every required date's swap curve once,
    extracts the resulting picklable callable objects (``DiscountFunction``,
    ``ZeroRateFunction`` wrapping ``PchipInterpolator``), and ships the
    cache dict to workers.  Workers reconstruct this source so that
    ``_run_period()`` can look up RF curves without invoking the
    (non-picklable, closure-bearing) factory in :class:`SwapCurveSource`.

    Parameters
    ----------
    cache : dict[date, tuple[ZeroRateFunc, DiscountFunc]]
        Pre-built per-date curve objects.
    """

    def __init__(
        self,
        cache: dict[date, tuple[ZeroRateFunc, DiscountFunc]],
    ) -> None:
        self._cache = cache

    def get_zero_rate_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> ZeroRateFunc:
        del currency  # single-currency source
        entry = self._cache.get(as_of)
        if entry is None:
            raise KeyError(
                f"CachedRfCurveSource: no curve cached for {as_of}; "
                f"available={sorted(self._cache.keys())}"
            )
        return entry[0]

    def get_discount_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> DiscountFunc:
        del currency  # single-currency source
        entry = self._cache.get(as_of)
        if entry is None:
            raise KeyError(
                f"CachedRfCurveSource: no curve cached for {as_of}; "
                f"available={sorted(self._cache.keys())}"
            )
        return entry[1]


# ---------------------------------------------------------------------------
# Concrete: multi-currency RF curve source
# ---------------------------------------------------------------------------


class MultiCurrencySwapCurveSource(RiskFreeCurveSource):
    """Wraps a per-currency dict of :class:`SwapCurveSource` instances.

    The ``base_currency`` is used when callers do not specify a currency
    (preserves single-curve behaviour for code paths that have not yet
    been made currency-aware, e.g. excess-return computation).

    Parameters
    ----------
    sources : dict[str, SwapCurveSource]
        ISO currency code → per-currency curve source.
    base_currency : str
        Default currency returned when ``get_*_function(as_of)`` is
        called without an explicit ``currency`` argument.
    """

    def __init__(
        self,
        sources: dict[str, SwapCurveSource],
        base_currency: str,
    ) -> None:
        self._sources: dict[str, SwapCurveSource] = {
            ccy.upper(): src for ccy, src in sources.items()
        }
        self._base = base_currency.upper()
        if self._base not in self._sources:
            raise ValueError(
                f"MultiCurrencySwapCurveSource: base_currency={base_currency!r} "
                f"not in sources keys={sorted(self._sources.keys())}"
            )

    @property
    def currencies(self) -> tuple[str, ...]:
        return tuple(self._sources.keys())

    @property
    def base_currency(self) -> str:
        return self._base

    def _resolve(self, currency: str | None) -> SwapCurveSource:
        key = (currency or self._base).upper()
        src = self._sources.get(key)
        if src is None:
            raise KeyError(
                f"MultiCurrencySwapCurveSource: no source for currency={key!r}; "
                f"available={sorted(self._sources.keys())}"
            )
        return src

    def get_zero_rate_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> ZeroRateFunc:
        return self._resolve(currency).get_zero_rate_function(as_of)

    def get_discount_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> DiscountFunc:
        return self._resolve(currency).get_discount_function(as_of)

    def extract_curve_cache(
        self, dates: list[date],
    ) -> dict[str, dict[date, tuple[ZeroRateFunc, DiscountFunc]]]:
        """Return a per-currency, picklable curve cache for worker shipment."""
        return {
            ccy: src.extract_curve_cache(dates) for ccy, src in self._sources.items()
        }


class CachedMultiCurrencyRfCurveSource(RiskFreeCurveSource):
    """Worker-side counterpart to :class:`MultiCurrencySwapCurveSource`.

    Parameters
    ----------
    cache : dict[str, dict[date, tuple[ZeroRateFunc, DiscountFunc]]]
        Per-currency, per-date pre-built ``(zero_rate, discount)`` callables.
    base_currency : str
        Default currency returned when no explicit ``currency`` is given.
    """

    def __init__(
        self,
        cache: dict[str, dict[date, tuple[ZeroRateFunc, DiscountFunc]]],
        base_currency: str,
    ) -> None:
        self._cache: dict[str, dict[date, tuple[ZeroRateFunc, DiscountFunc]]] = {
            ccy.upper(): bucket for ccy, bucket in cache.items()
        }
        self._base = base_currency.upper()
        if self._base not in self._cache:
            raise ValueError(
                f"CachedMultiCurrencyRfCurveSource: base_currency={base_currency!r} "
                f"not in cache keys={sorted(self._cache.keys())}"
            )

    @property
    def currencies(self) -> tuple[str, ...]:
        return tuple(self._cache.keys())

    @property
    def base_currency(self) -> str:
        return self._base

    def _entry(
        self, as_of: date, currency: str | None,
    ) -> tuple[ZeroRateFunc, DiscountFunc]:
        key = (currency or self._base).upper()
        bucket = self._cache.get(key)
        if bucket is None:
            raise KeyError(
                f"CachedMultiCurrencyRfCurveSource: no curves for currency={key!r}; "
                f"available={sorted(self._cache.keys())}"
            )
        entry = bucket.get(as_of)
        if entry is None:
            raise KeyError(
                f"CachedMultiCurrencyRfCurveSource: no curve cached for "
                f"({key}, {as_of}); available_dates={sorted(bucket.keys())}"
            )
        return entry

    def get_zero_rate_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> ZeroRateFunc:
        return self._entry(as_of, currency)[0]

    def get_discount_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> DiscountFunc:
        return self._entry(as_of, currency)[1]


# ---------------------------------------------------------------------------
# Synthetic hedge return sources (for "native" returns_calc mode)
# ---------------------------------------------------------------------------

# Per-date ISIN → RETURN_SYNTH map
SynthHedgeCache = dict[date, dict[str, float]]

# Map universe_source cadence to the PERIOD column value in Snowflake
_PERIOD_BY_SOURCE: dict[str, str] = {
    "monthly": "m",
    "weekly": "w",
}


class SynthHedgeReturnSource:
    """Snowflake-backed source for per-bond synthetic hedge returns.

    Queries ``quant.research.synth_hedge_returns`` for all rebalance
    dates in a single round-trip.  Results are cached internally so that
    ``get_hedge_returns()`` is a pure dict lookup.

    Parameters
    ----------
    sf_config : object
        Snowflake connection configuration.
    synth_type : str
        Hedge methodology type (``"exact"``, ``"krd_ice"``, ``"krd13"``).
    period : str
        Period cadence column value (``"m"`` or ``"w"``).
    """

    def __init__(self, sf_config: object, synth_type: str, period: str) -> None:
        from archipelago.data.connectors.snowflake_client import SnowflakeClient

        self._client = SnowflakeClient.from_config(sf_config)
        self._synth_type = synth_type
        self._period = period
        self._cache: SynthHedgeCache = {}

    def load_all(self, dates: list[date]) -> None:
        """Bulk-load hedge returns for *dates* into the internal cache.

        Issues a single Snowflake query using the PARSE_JSON/FLATTEN
        pattern.  Populates ``self._cache`` so that subsequent
        ``get_hedge_returns()`` calls are cache hits.
        """
        if not dates:
            return

        import json
        from importlib import resources

        iso_dates = sorted({d.strftime("%Y-%m-%d") for d in dates})
        dates_json = json.dumps(iso_dates)

        pkg_root = resources.files("archipelago")
        sql_path = pkg_root.joinpath("data/connectors/sql/synth_hedge_returns.sql")
        sql = sql_path.read_text(encoding="utf-8")
        # Replace longest placeholders first to avoid substring collisions.
        sql = sql.replace("%synth_type%", f"'{self._synth_type}'")
        sql = sql.replace("%period%", f"'{self._period}'")
        sql = sql.replace("%dates%", f"'{dates_json}'")

        df = self._client.execute_query(sql)
        df.columns = [c.upper() for c in df.columns]

        import pandas as pd_mod

        for data_date, grp in df.groupby("DATA_DATE"):
            d = pd_mod.Timestamp(data_date).date()
            self._cache[d] = dict(zip(grp["ISIN"], grp["RETURN_SYNTH"].astype(float)))

        logger.info(
            "SynthHedgeReturnSource: loaded %d dates (%d total rows) for type=%s period=%s",
            len(self._cache), len(df), self._synth_type, self._period,
        )

    def get_hedge_returns(self, as_of: date, isins: np.ndarray) -> np.ndarray:
        """Return per-bond hedge returns aligned to *isins*.

        ISINs not found in the cache for *as_of* receive 0.0 (no hedge
        return → excess return equals total return for that bond).
        """
        lookup = self._cache.get(as_of, {})
        return np.array(
            [lookup.get(isin, 0.0) for isin in isins], dtype=float,
        )

    def extract_cache(self) -> SynthHedgeCache:
        """Return the internal cache for shipping to worker processes."""
        return dict(self._cache)


class CachedSynthHedgeReturnSource:
    """Picklable cache-backed synthetic hedge return source for worker processes.

    Parameters
    ----------
    cache : SynthHedgeCache
        Pre-populated ``{date: {ISIN: RETURN_SYNTH}}`` dict extracted
        from :meth:`SynthHedgeReturnSource.extract_cache`.
    """

    def __init__(self, cache: SynthHedgeCache) -> None:
        self._cache = cache

    def get_hedge_returns(self, as_of: date, isins: np.ndarray) -> np.ndarray:
        lookup = self._cache.get(as_of, {})
        return np.array(
            [lookup.get(isin, 0.0) for isin in isins], dtype=float,
        )


# ---------------------------------------------------------------------------
# Duration snapshot
# ---------------------------------------------------------------------------

# ISIN → effective duration (years)
DurationSnapshot = dict[str, float]


def build_duration_snapshot(bonds: pd.DataFrame) -> DurationSnapshot:
    """Extract ISIN → EFFECTIVE_DURATION from Goblin data.

    Parameters
    ----------
    bonds : pd.DataFrame
        Filtered Goblin universe with ``ISIN`` and ``EFFECTIVE_DURATION``.

    Returns
    -------
    DurationSnapshot
    """
    snap: DurationSnapshot = {}
    for row in bonds.itertuples(index=False):
        dur = float(row.EFFECTIVE_DURATION)
        snap[row.ISIN] = 0.0 if np.isnan(dur) else dur
    return snap


# ---------------------------------------------------------------------------
# Bond metadata snapshot (for cashflow-matched benchmarks)
# ---------------------------------------------------------------------------

# ISIN → (coupon_rate, maturity_date, frequency, day_count_convention, currency, oas)
# ``oas`` is the constant option-adjusted spread in **decimal**
# (continuously-compounded), i.e. Goblin ``OAS`` (basis points) / 10 000.
BondMetadataSnapshot = dict[str, tuple[float, date, int, str, str, float]]


def build_bond_metadata_snapshot(bonds: pd.DataFrame) -> BondMetadataSnapshot:
    """Extract ISIN → (coupon_rate, maturity_date, frequency, day_count_convention, currency, oas) from Goblin data.

    Parameters
    ----------
    bonds : pd.DataFrame
        Filtered Goblin universe with ``ISIN``, ``COUPON``,
        ``MATURITY_DATE``, ``CPN_FREQ``, and optionally
        ``DAY_COUNT_CONVENTION``, ``CURRENCY`` and ``OAS`` columns.
        ``CPN_FREQ`` is required.
        If ``DAY_COUNT_CONVENTION`` is absent, defaults to ``"30/360"``.
        If ``CURRENCY`` is absent or NaN for a row, defaults to ``"USD"``
        (preserves single-currency backward compatibility).
        If ``OAS`` is absent or NaN for a row, defaults to ``0.0`` (the
        bond is treated as discounting on the risk-free curve only).
        ``OAS`` is read in Goblin basis points and stored as a decimal
        continuously-compounded spread (``OAS / 10 000``).

    Returns
    -------
    BondMetadataSnapshot
    """
    import pandas as pd_mod

    has_dcc = "DAY_COUNT_CONVENTION" in bonds.columns
    has_ccy = "CURRENCY" in bonds.columns
    has_oas = "OAS" in bonds.columns

    snap: BondMetadataSnapshot = {}
    for row in bonds.itertuples(index=False):
        isin: str = row.ISIN
        coupon = float(row.COUPON) / 100.0  # Goblin stores as percentage
        mat = row.MATURITY_DATE
        if isinstance(mat, pd_mod.Timestamp):
            mat = mat.date()
        freq = int(row.CPN_FREQ)
        dcc = "30/360"
        if has_dcc:
            raw_dcc = getattr(row, "DAY_COUNT_CONVENTION", None)
            if raw_dcc is not None and not pd_mod.isna(raw_dcc):
                dcc = str(raw_dcc)
        ccy = "USD"
        if has_ccy:
            raw_ccy = getattr(row, "CURRENCY", None)
            if raw_ccy is not None and not pd_mod.isna(raw_ccy):
                ccy = str(raw_ccy).upper()
        oas = 0.0
        if has_oas:
            raw_oas = getattr(row, "OAS", None)
            if raw_oas is not None and not pd_mod.isna(raw_oas):
                oas = float(raw_oas) / 10_000.0  # Goblin OAS is in basis points
        snap[isin] = (coupon, mat, freq, dcc, ccy, oas)
    return snap


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_excess_returns(
    isins: np.ndarray,
    total_returns: np.ndarray,
    prev_dur_snap: DurationSnapshot,
    curr_dur_snap: DurationSnapshot,
    prev_zr: ZeroRateFunc,
    curr_zr: ZeroRateFunc,
    *,
    mode: RfBenchmarkMode = RfBenchmarkMode.KRD,
    prev_df: DiscountFunc | None = None,
    curr_df: DiscountFunc | None = None,
    prev_date: date | None = None,
    curr_date: date | None = None,
    bond_meta: BondMetadataSnapshot | None = None,
    convention: "KrdHedgeConvention | None" = None,
) -> np.ndarray:
    """Per-bond excess returns = total return - risk-free benchmark return.

    Parameters
    ----------
    isins : np.ndarray
        ISIN array defining row order.
    total_returns : np.ndarray
        Per-bond total returns (same order as *isins*).
    prev_dur_snap : DurationSnapshot
        ISIN → effective duration at the previous period.
    curr_dur_snap : DurationSnapshot
        ISIN → effective duration at the current period.
    prev_zr : ZeroRateFunc
        Zero-rate function at the previous period.
    curr_zr : ZeroRateFunc
        Zero-rate function at the current period.
    mode : RfBenchmarkMode
        Benchmark mode (default: ZCB_SLIDE for backward compatibility).
    prev_df : DiscountFunc | None
        Discount function D(t) at previous date (required for CASHFLOW_MATCHED, FORWARD, KRD).
    curr_df : DiscountFunc | None
        Discount function D(t) at current date (required for CASHFLOW_MATCHED, KRD).
    prev_date : date | None
        Previous rebalance date (required for CASHFLOW_MATCHED, KRD).
    curr_date : date | None
        Current rebalance date (required for CASHFLOW_MATCHED, YTM, KRD).
    bond_meta : BondMetadataSnapshot | None
        ISIN → (coupon_rate, maturity_date, frequency, day_count_convention).
        Required for CASHFLOW_MATCHED and KRD.

    Returns
    -------
    np.ndarray
        Per-bond excess returns aligned to *isins*.
    """
    n = len(isins)
    excess = np.zeros(n, dtype=float)

    # Pre-build KRD synthetic cache once per rebalance date (BUG-005 fix)
    krd_cache = None
    if mode == RfBenchmarkMode.KRD and prev_df is not None and curr_df is not None:
        if prev_date is not None and curr_date is not None:
            krd_cache = build_krd_synth_cache(
                prev_df, curr_df, prev_date, curr_date,
                convention=convention or USD_SWAP_HEDGE_CONVENTION,
            )

    for i, isin in enumerate(isins):
        rf = _compute_single_rf_return(
            isin, mode,
            prev_dur_snap, curr_dur_snap,
            prev_zr, curr_zr,
            prev_df, curr_df,
            prev_date, curr_date,
            bond_meta,
            krd_cache=krd_cache,
        )
        if rf is None:
            # Missing RF data → preserve total return as excess (BUG-006 fix)
            excess[i] = total_returns[i]
        else:
            excess[i] = total_returns[i] - rf

    return excess


def _compute_single_rf_return(
    isin: str,
    mode: RfBenchmarkMode,
    prev_dur_snap: DurationSnapshot,
    curr_dur_snap: DurationSnapshot,
    prev_zr: ZeroRateFunc,
    curr_zr: ZeroRateFunc,
    prev_df: DiscountFunc | None,
    curr_df: DiscountFunc | None,
    prev_date: date | None,
    curr_date: date | None,
    bond_meta: BondMetadataSnapshot | None,
    *,
    krd_cache=None,
) -> float | None:
    """Compute single-bond RF return for the given mode. Returns None if data missing."""
    if mode == RfBenchmarkMode.CASHFLOW_MATCHED:
        if bond_meta is None or prev_df is None or curr_df is None:
            return None
        if prev_date is None or curr_date is None:
            return None
        meta = bond_meta.get(isin)
        if meta is None:
            return None
        coupon_rate, maturity_date, frequency, dcc, _ccy, _oas = meta
        return calc_cf_matched_realized_return(
            prev_df, curr_df, coupon_rate, maturity_date,
            prev_date, curr_date, frequency, dcc,
        )

    if mode == RfBenchmarkMode.KRD:
        if bond_meta is None or prev_df is None or curr_df is None:
            return None
        if prev_date is None or curr_date is None:
            return None
        if krd_cache is None:
            return None
        meta = bond_meta.get(isin)
        if meta is None:
            return None
        coupon_rate, maturity_date, frequency, dcc, _ccy, _oas = meta
        return calc_krd_matched_realized_return(
            prev_df, coupon_rate, maturity_date,
            prev_date, frequency, dcc,
            hedge_cache=krd_cache,
        )

    return None
