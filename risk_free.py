"""Risk-free return computation for excess-return decomposition.

Provides benchmark modes for computing per-bond risk-free returns
from the USD swap curve, enabling the split::

    r_excess_i = r_total_i - r_rf_i

Supported modes (see :class:`RfBenchmarkMode`):

- **CASHFLOW_MATCHED**: Exact cashflow-matched RF twin (Option A).
- **KRD**: Key-rate-duration matched (Option D).
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

ZeroRateFunc = Callable[[float], float]
DiscountFunc = Callable[[float], float]


class RiskFreeCurveSource(abc.ABC):
    """ABC for providers of the zero-rate and discount functions at a given date."""

    @abc.abstractmethod
    def get_zero_rate_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> ZeroRateFunc:
        """Return a callable ``z(t)`` for the given valuation date."""

    @abc.abstractmethod
    def get_discount_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> DiscountFunc:
        """Return a callable ``D(t)`` for the given valuation date."""


class SwapCurveSource(RiskFreeCurveSource):
    """Derives ``z(t)`` from :class:`~archipelago.instruments.discount.SwapCurveBuilder`."""

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
        del currency
        return self._get_calibration(as_of)["zero_rate_function"]

    def get_discount_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> DiscountFunc:
        del currency
        return self._get_calibration(as_of)["discount_function"]

    def extract_curve_cache(
        self, dates: list[date],
    ) -> dict[date, tuple[ZeroRateFunc, DiscountFunc]]:
        """Return a picklable snapshot of ``(zero_rate_function, discount_function)`` per date."""
        snapshot: dict[date, tuple[ZeroRateFunc, DiscountFunc]] = {}
        for d in dates:
            calib = self._get_calibration(d)
            snapshot[d] = (
                calib["zero_rate_function"],
                calib["discount_function"],
            )
        return snapshot


class CachedRfCurveSource(RiskFreeCurveSource):
    """RF curve source backed by a pre-built ``(zero_rate, discount)`` cache."""

    def __init__(
        self,
        cache: dict[date, tuple[ZeroRateFunc, DiscountFunc]],
    ) -> None:
        self._cache = cache

    def get_zero_rate_function(
        self, as_of: date, *, currency: str | None = None,
    ) -> ZeroRateFunc:
        del currency
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
        del currency
        entry = self._cache.get(as_of)
        if entry is None:
            raise KeyError(
                f"CachedRfCurveSource: no curve cached for {as_of}; "
                f"available={sorted(self._cache.keys())}"
            )
        return entry[1]


class MultiCurrencySwapCurveSource(RiskFreeCurveSource):
    """Wraps a per-currency dict of :class:`SwapCurveSource` instances."""

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
    """Worker-side counterpart to :class:`MultiCurrencySwapCurveSource`."""

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


SynthHedgeCache = dict[date, dict[str, float]]

_PERIOD_BY_SOURCE: dict[str, str] = {
    "monthly": "m",
    "weekly": "w",
}


class SynthHedgeReturnSource:
    """Snowflake-backed source for per-bond synthetic hedge returns."""

    def __init__(self, sf_config: object, synth_type: str, period: str) -> None:
        from archipelago.data.connectors.snowflake_client import SnowflakeClient

        self._client = SnowflakeClient.from_config(sf_config)
        self._synth_type = synth_type
        self._period = period
        self._cache: SynthHedgeCache = {}

    def load_all(self, dates: list[date]) -> None:
        """Bulk-load hedge returns for *dates* into the internal cache."""
        if not dates:
            return

        import json
        from importlib import resources

        iso_dates = sorted({d.strftime("%Y-%m-%d") for d in dates})
        dates_json = json.dumps(iso_dates)

        pkg_root = resources.files("archipelago")
        sql_path = pkg_root.joinpath("data/connectors/sql/synth_hedge_returns.sql")
        sql = sql_path.read_text(encoding="utf-8")
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
        """Return per-bond hedge returns aligned to *isins*."""
        lookup = self._cache.get(as_of, {})
        return np.array(
            [lookup.get(isin, 0.0) for isin in isins], dtype=float,
        )

    def extract_cache(self) -> SynthHedgeCache:
        """Return the internal cache for shipping to worker processes."""
        return dict(self._cache)


class CachedSynthHedgeReturnSource:
    """Picklable cache-backed synthetic hedge return source for worker processes."""

    def __init__(self, cache: SynthHedgeCache) -> None:
        self._cache = cache

    def get_hedge_returns(self, as_of: date, isins: np.ndarray) -> np.ndarray:
        lookup = self._cache.get(as_of, {})
        return np.array(
            [lookup.get(isin, 0.0) for isin in isins], dtype=float,
        )


DurationSnapshot = dict[str, float]


def build_duration_snapshot(bonds: pd.DataFrame) -> DurationSnapshot:
    """Extract ISIN → EFFECTIVE_DURATION from Goblin data."""
    snap: DurationSnapshot = {}
    for row in bonds.itertuples(index=False):
        dur = float(row.EFFECTIVE_DURATION)
        snap[row.ISIN] = 0.0 if np.isnan(dur) else dur
    return snap


BondMetadataSnapshot = dict[str, tuple[float, date, int, str, str, float]]


def build_bond_metadata_snapshot(bonds: pd.DataFrame) -> BondMetadataSnapshot:
    """Extract ISIN → (coupon_rate, maturity_date, frequency, day_count_convention, currency, oas) from Goblin data."""
    import pandas as pd_mod

    has_dcc = "DAY_COUNT_CONVENTION" in bonds.columns
    has_ccy = "CURRENCY" in bonds.columns
    has_oas = "OAS" in bonds.columns

    snap: BondMetadataSnapshot = {}
    for row in bonds.itertuples(index=False):
        isin: str = row.ISIN
        coupon = float(row.COUPON) / 100.0
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
                oas = float(raw_oas) / 100.0

        snap[isin] = (coupon, mat, freq, dcc, ccy, oas)

    return snap


def _period_years(prev_date: date | None, curr_date: date | None) -> float:
    """Return elapsed period in curve years."""
    if prev_date is None or curr_date is None:
        return 0.0
    return (curr_date - prev_date).days / CURVE_TIME_BASIS


def _format_return_bps(value: float) -> str:
    """Format a decimal return as basis points."""
    return f"{value * 100:.2f} bps"


def _linear_period_return(rate: float, years: float) -> float:
    """Convert a continuously quoted annual rate into a period return."""
    return rate * years


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
    """Per-bond excess returns = total return - risk-free benchmark return."""
    n = len(isins)
    excess = np.zeros(n, dtype=float)

    period_years = _period_years(prev_date, curr_date)
    if period_years > 0.0:
        logger.debug(
            "Computing RF excess returns over %s years (%s)",
            period_years,
            _format_return_bps(_linear_period_return(float(prev_zr(period_years)), period_years)),
        )

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
            excess[i] = total_returns[i]
        else:
            excess[i] = total_returns[i] + rf

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
            curr_df, coupon_rate, maturity_date,
            prev_date, frequency, dcc,
            hedge_cache=krd_cache,
        )

    return None