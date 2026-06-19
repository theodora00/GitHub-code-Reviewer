"""
USD Swap Curve Stripper

This module implements functionality to build a USD swap curve from market observables:
- SOFR (Secured Overnight Financing Rate) for short-term rates
- Futures contracts for medium-term rates
- Swap rates for long-term rates

The curve is bootstrapped to ensure consistent discount factors across all tenors.
"""

import math
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq, minimize, root
from scipy.interpolate import PchipInterpolator
from enum import Enum
from typing import Dict, List, NamedTuple, Tuple, Optional, Callable, Union


# ---------------------------------------------------------------------------
# Curve time-axis convention
# ---------------------------------------------------------------------------
# The curve's internal time parameterization uses ACT/365 Fixed (calendar
# days / 365).  All calibrated callables (discount_function, zero_rate_function,
# forward_rate_function) expect `t` on this basis.  Downstream consumers MUST
# convert dates to this same basis before calling curve functions.
CURVE_TIME_BASIS: float = 365.0


def curve_time(valuation_date: datetime, target_date: datetime) -> float:
    """Convert a date to the curve's time-axis value (ACT/365 Fixed).

    This is the authoritative date→t mapping for all curve interpolation,
    zero-rate annualization, and callable inputs.
    """
    return (target_date - valuation_date).days / CURVE_TIME_BASIS


def _raise_on_right_extrapolation(t_arr: np.ndarray, max_time: float, name: str) -> None:
    """Reject right-end extrapolation for calibrated curve functions."""
    if np.any(t_arr > max_time + 1e-12):
        requested = float(np.max(t_arr))
        raise ValueError(
            f"{name} is only defined through {max_time:.10f} years; "
            f"received {requested:.10f} years"
        )



class DayCountConvention(Enum):
    """Day count conventions used in rate calculations."""
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    THIRTY_360 = "30/360"
    ACT_ACT_ISDA = "ACT/ACT ISDA"


class Instrument(Enum):
    """Types of instruments used in curve construction."""
    CASH = "CASH"        # SOFR, Fed Funds
    FUTURES = "FUTURES"  # SOFR Futures
    SWAP = "SWAP"        # Interest Rate Swaps


# ============================================================================
# Function wrappers for cubic spline interpolation
# ============================================================================

class DiscountFunction:
    """
    Wrapper for discount function using PCHIP interpolation on u(t) = -ln(DF).
    """
    def __init__(self, spline, times):
        """
        Args:
            spline: PchipInterpolator fitted to u(t)
            times: List of time points used for extrapolation
        """
        self.spline = spline
        self.times = times
        self.max_time = float(max(times))

    def __call__(self, t):
        """Evaluate calibrated discount factor at time t (supports vector inputs)."""
        t_arr = np.asarray(t, dtype=float)
        _raise_on_right_extrapolation(t_arr, self.max_time, "discount_function")
        out = np.ones_like(t_arr)
        mask = t_arr > 0.0
        if np.any(mask):
            out[mask] = np.exp(-self.spline(t_arr[mask]))
        return float(out) if out.ndim == 0 else out


class ZeroRateFunction:
    """
    Wrapper for zero rate function using PCHIP interpolation on u(t) = -ln(DF).
    """
    def __init__(self, spline, times):
        """
        Args:
            spline: PchipInterpolator fitted to u(t)
            times: List of time points
        """
        self.spline = spline
        self.times = times
        self.max_time = float(max(times))

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        _raise_on_right_extrapolation(t_arr, self.max_time, "zero_rate_function")
        out = np.zeros_like(t_arr)
        mask = t_arr > 0.0
        if np.any(mask):
            uval = self.spline(t_arr[mask])
            out[mask] = uval / t_arr[mask]
        uprime0 = float(self.spline.derivative()(self.times[0]))
        out[~mask] = max(0.0, uprime0)
        return float(out) if out.ndim == 0 else out


class ForwardRateFunction:
    """
    Wrapper for forward rate function using PCHIP interpolation on u(t) = -ln(DF).
    """
    def __init__(self, spline, times):
        """
        Args:
            spline: PchipInterpolator fitted to u(t)
            times: List of time points
        """
        self.spline = spline
        self.times = times
        self.max_time = float(max(times))

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        _raise_on_right_extrapolation(t_arr, self.max_time, "forward_rate_function")
        out = np.zeros_like(t_arr)
        mask = t_arr > 0.0
        if np.any(mask):
            uprime = self.spline.derivative()(t_arr[mask])
            out[mask] = np.maximum(0.0, uprime)
        uprime0 = float(self.spline.derivative()(self.times[0]))
        out[~mask] = max(0.0, uprime0)
        return float(out) if out.ndim == 0 else out


# ============================================================================
# Function wrappers for log-linear interpolation on discount factors
# ============================================================================

class LogLinearDiscountFunction:
    """Piecewise log-linear interpolation on discount factors (Bloomberg standard)."""
    def __init__(self, times: np.ndarray, log_dfs: np.ndarray):
        self.times = times
        self.log_dfs = log_dfs
        self.max_time = float(times[-1])

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        _raise_on_right_extrapolation(t_arr, self.max_time, "discount_function")
        out = np.ones_like(t_arr)
        mask = t_arr > 0.0
        if np.any(mask):
            log_df_interp = np.interp(t_arr[mask], self.times, self.log_dfs)
            out[mask] = np.exp(log_df_interp)
        return float(out) if out.ndim == 0 else out


class LogLinearZeroRateFunction:
    """Zero rate derived from piecewise log-linear DF interpolation."""
    def __init__(self, times: np.ndarray, log_dfs: np.ndarray):
        self.times = times
        self.log_dfs = log_dfs
        self.max_time = float(times[-1])

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        _raise_on_right_extrapolation(t_arr, self.max_time, "zero_rate_function")
        out = np.zeros_like(t_arr)
        mask = t_arr > 0.0
        if np.any(mask):
            log_df_interp = np.interp(t_arr[mask], self.times, self.log_dfs)
            out[mask] = -log_df_interp / t_arr[mask]
        # At t=0, use instantaneous rate from first segment
        if np.any(~mask) and len(self.times) >= 2:
            out[~mask] = -self.log_dfs[1] / self.times[1] if self.times[1] > 0 else 0.0
        return float(out) if out.ndim == 0 else out


class LogLinearForwardRateFunction:
    """Piecewise-constant forward rate implied by log-linear DF interpolation."""
    def __init__(self, times: np.ndarray, log_dfs: np.ndarray):
        self.times = times
        self.log_dfs = log_dfs
        self.max_time = float(times[-1])

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        _raise_on_right_extrapolation(t_arr, self.max_time, "forward_rate_function")
        out = np.zeros_like(t_arr)
        mask = t_arr > 0.0
        if np.any(mask):
            # Forward rate is piecewise constant between nodes
            idx = np.searchsorted(self.times, t_arr[mask], side='right') - 1
            idx = np.clip(idx, 0, len(self.times) - 2)
            dt = self.times[idx + 1] - self.times[idx]
            dlog = self.log_dfs[idx + 1] - self.log_dfs[idx]
            safe_dt = np.where(dt > 0, dt, 1.0)
            out[mask] = -dlog / safe_dt
        if np.any(~mask) and len(self.times) >= 2:
            dt0 = self.times[1] - self.times[0]
            out[~mask] = -(self.log_dfs[1] - self.log_dfs[0]) / dt0 if dt0 > 0 else 0.0
        return float(out) if out.ndim == 0 else out





class SwapCurveBuilder:
    """Class for constructing USD swap curves from market data."""
    
    def __init__(
        self,
        valuation_date: datetime,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        swap_fixed_frequency_months: int = 12,
    ):
        """
        Initialize the curve builder.
        
        Args:
            valuation_date: Reference date for curve construction
            day_count: Day count convention for calculations
            swap_fixed_frequency_months: Number of months between successive
                fixed-leg coupons (12 = annual, USD SOFR OIS market standard
                per CME cleared SOFR OIS conventions and ISDA 2021 SOFR
                Definitions; 6 = semi-annual for legacy USD LIBOR IRS).
                Must be a positive integer dividing 12 (1, 2, 3, 4, 6, 12).
        """
        if not isinstance(swap_fixed_frequency_months, int) or swap_fixed_frequency_months <= 0:
            raise ValueError(
                f"swap_fixed_frequency_months must be a positive int, got {swap_fixed_frequency_months!r}"
            )
        if 12 % swap_fixed_frequency_months != 0:
            raise ValueError(
                f"swap_fixed_frequency_months must divide 12, got {swap_fixed_frequency_months}"
            )
        self.valuation_date = valuation_date
        self.day_count = day_count
        self.swap_fixed_frequency_months = swap_fixed_frequency_months
        
        # Store discount factors and zero rates
        self.discount_factors: Dict[datetime, float] = {}
        self.zero_rates: Dict[datetime, float] = {}
        
        # Store raw market data
        self.market_data: List[Dict] = []

    def _curve_time(self, target_date: datetime) -> float:
        """Convert a date to the curve's time-axis value (ACT/365 Fixed).

        This is the single authoritative date→t mapping for all internal
        interpolation, zero-rate annualization, and calibration knot placement.
        """
        return (target_date - self.valuation_date).days / CURVE_TIME_BASIS
        
    def add_market_rate(self, 
                      instrument_type: Instrument,
                      rate: float,
                      tenor: Union[str, int],
                      maturity_date: Optional[datetime] = None,
                      start_date: Optional[datetime] = None) -> None:
        """
        Add market rate data for curve construction.
        
        Args:
            instrument_type: Type of instrument (CASH, FUTURES, SWAP)
            rate: Market observed rate
            tenor: Time period as string ("1D", "3M", "10Y") or number of days
            maturity_date: Optional explicit maturity date
            start_date: Futures contract reference period start date (required for
                FUTURES instruments; ignored for CASH and SWAP)
        """
        if instrument_type == Instrument.FUTURES and start_date is None:
            raise ValueError(
                "start_date is required for FUTURES instruments: it must be the "
                "explicit reference period start date of the contract, not inferred "
                "from the curve state."
            )
        if maturity_date is None:
            maturity_date = self._calculate_maturity_date(tenor)
            
        self.market_data.append({
            "instrument_type": instrument_type,
            "rate": rate,
            "tenor": tenor,
            "maturity_date": maturity_date,
            "start_date": start_date,
        })
        
    def _calculate_maturity_date(self, tenor: Union[str, int]) -> datetime:
        """
        Calculate maturity date based on tenor string or days.
        
        Args:
            tenor: Time period as string ("1D", "3M", "10Y") or number of days
            
        Returns:
            Maturity date
        """
        if isinstance(tenor, int):
            return self.valuation_date + timedelta(days=tenor)

        if not isinstance(tenor, str):
            raise TypeError(f"tenor must be int or str, got {type(tenor).__name__}")

        # Strict whole-token regex: digits then a single D/W/M/Y suffix.
        # Rejects whitespace, floats, mixed units, garbage prefixes.
        m = re.fullmatch(r"(\d+)([DWMY])", tenor.upper())
        if m is None:
            raise ValueError(
                f"Unsupported tenor {tenor!r}: expected '<int><D|W|M|Y>' (e.g. '3M', '10Y')"
            )
        value = int(m.group(1))
        unit = m.group(2)

        if unit == 'D':
            return self.valuation_date + timedelta(days=value)
        elif unit == 'W':
            return self.valuation_date + timedelta(weeks=value)
        elif unit == 'M':
            return self.valuation_date + relativedelta(months=value)
        else:  # 'Y'
            return self.valuation_date + relativedelta(years=value)
    
    def _day_count_factor(self, start_date: datetime, end_date: datetime) -> float:
        """
        Calculate day count factor based on convention.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Day count factor
        """
        days = (end_date - start_date).days
        
        if self.day_count == DayCountConvention.ACT_360:
            return days / 360
        elif self.day_count == DayCountConvention.ACT_365:
            return days / 365
        elif self.day_count == DayCountConvention.THIRTY_360:
            # Simplified 30/360 calculation
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            
            # Adjust day values according to 30/360 convention
            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 >= 30:
                d2 = 30
                
            return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360
        
        elif self.day_count == DayCountConvention.ACT_ACT_ISDA:
            # Simplified ACT/ACT ISDA
            year = start_date.year
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            days_in_year = 366 if is_leap else 365
            return days / days_in_year
        
        else:
            raise ValueError(f"Unsupported day count convention: {self.day_count}")
    
    def _get_discount_factor(self, date: datetime) -> float:
        """
        Get discount factor for a specific date.
        
        Args:
            date: Target date
            
        Returns:
            Discount factor
        """
        if date in self.discount_factors:
            return self.discount_factors[date]

        # Interpolate if date not exactly in our curve points
        # Find closest dates before and after
        earlier_dates = [d for d in self.discount_factors.keys() if d <= date]
        later_dates = [d for d in self.discount_factors.keys() if d >= date]

        if not earlier_dates and not later_dates:
            raise ValueError(f"Cannot interpolate discount factor for {date}: no curve points available")

        # Handle extrapolation beyond curve endpoints
        if not earlier_dates:
            # Extrapolate before first point - use flat extrapolation
            earliest_date = min(self.discount_factors.keys())
            return self.discount_factors[earliest_date]

        if not later_dates:
            last_curve_date = max(self.discount_factors.keys())
            raise ValueError(
                f"Cannot interpolate discount factor for {date}: "
                f"curve ends at {last_curve_date}"
            )

        d1 = max(earlier_dates)
        d2 = min(later_dates)
        
        if d1 == d2:
            return self.discount_factors[d1]
        
        # Linear interpolation of log discount factors
        df1 = self.discount_factors[d1]
        df2 = self.discount_factors[d2]
        
        t1 = self._curve_time(d1)
        t2 = self._curve_time(d2)
        t = self._curve_time(date)
        
        # Linear interpolation in the log space
        log_df1 = np.log(df1)
        log_df2 = np.log(df2)
        
        alpha = (t - t1) / (t2 - t1) if (t2 - t1) > 0 else 0
        log_df = log_df1 + alpha * (log_df2 - log_df1)
        
        return np.exp(log_df)
    
    def _calculate_zero_rate(self, date: datetime) -> float:
        """
        Calculate continuous zero rate from discount factor.
        
        Args:
            date: Target date
            
        Returns:
            Zero rate (continuous compounding)
        """
        df = self._get_discount_factor(date)
        t = self._curve_time(date)
        
        if t <= 0:
            return 0
        
        return -np.log(df) / t
    
    def _bootstrap_cash_rate(self, rate: float, maturity_date: datetime) -> float:
        """
        Bootstrap discount factor from cash rate.
        
        Args:
            rate: Market rate
            maturity_date: Maturity date
            
        Returns:
            Discount factor
        """
        t = self._day_count_factor(self.valuation_date, maturity_date)
        
        # For cash rates: DF = 1 / (1 + r * t)
        df = 1 / (1 + rate * t)
        return df
    
    def _bootstrap_futures(self, rate: float, start_date: datetime, maturity_date: datetime) -> float:
        """
        Bootstrap discount factor from futures rates.
        
        Args:
            rate: Futures rate
            start_date: Start date of the futures contract
            maturity_date: Maturity date of the futures contract
            
        Returns:
            Discount factor at maturity date
        """
        # We need discount factor at start_date
        df_start = self._get_discount_factor(start_date)
        
        t = self._day_count_factor(start_date, maturity_date)
        
        # For futures rates: DF_maturity = DF_start / (1 + rate * t)
        df_maturity = df_start / (1 + rate * t)
        return df_maturity

    def _generate_swap_schedule(self, maturity_date: datetime) -> List[datetime]:
        """Generate the fixed-leg schedule up to maturity.

        Forward-rolled from ``valuation_date`` using
        ``self.swap_fixed_frequency_months`` between successive coupons; any
        trailing stub is trimmed back to ``maturity_date``.  For an 18M
        annual-leg swap (frequency=12) this yields ``[+12M, +18M]``.
        """
        step = relativedelta(months=self.swap_fixed_frequency_months)
        payment_dates: List[datetime] = []
        current_date = self.valuation_date
        while current_date < maturity_date:
            current_date = current_date + step
            if current_date > maturity_date:
                current_date = maturity_date
            payment_dates.append(current_date)
        return payment_dates
    
    def _bootstrap_swap(self, swap_rate: float, maturity_date: datetime) -> float:
        """
        Bootstrap discount factor from swap rate.
        
        Args:
            swap_rate: Market swap rate (already in decimal form, e.g., 0.0365 for 3.65%)
            maturity_date: Swap maturity date
            
        Returns:
            Discount factor at maturity date
        """

        payment_dates = self._generate_swap_schedule(maturity_date)
        if not payment_dates:
            raise ValueError(f"Swap maturity {maturity_date} produces an empty payment schedule")

        previous_curve_date = max(d for d in self.discount_factors.keys() if d < maturity_date)
        previous_curve_df = self.discount_factors[previous_curve_date]
        t_prev = self._curve_time(previous_curve_date)
        t_maturity = self._curve_time(maturity_date)
        if t_maturity <= t_prev:
            raise ValueError(
                f"Swap maturity {maturity_date} must extend beyond last curve pillar {previous_curve_date}"
            )

        known_pv = 0.0
        gap_terms: List[Tuple[float, float]] = []
        previous_date = self.valuation_date

        for payment_date in payment_dates[:-1]:
            dcf = self._day_count_factor(previous_date, payment_date)
            if payment_date <= previous_curve_date:
                known_pv += swap_rate * dcf * self._get_discount_factor(payment_date)
            else:
                t_coupon = self._curve_time(payment_date)
                alpha = (t_coupon - t_prev) / (t_maturity - t_prev)
                coeff = swap_rate * dcf * (previous_curve_df ** (1.0 - alpha))
                gap_terms.append((coeff, alpha))
            previous_date = payment_date

        dcf_last = self._day_count_factor(previous_date, maturity_date)
        if dcf_last <= 0.0:
            raise ValueError(
                f"Non-positive final accrual {dcf_last} for swap maturity {maturity_date}"
            )

        def objective(df_last: float) -> float:
            gap_pv = sum(coeff * (df_last ** alpha) for coeff, alpha in gap_terms)
            return known_pv + gap_pv + swap_rate * dcf_last * df_last - (1.0 - df_last)

        lower = 1e-12
        upper = max(1.0, previous_curve_df)
        f_lower = objective(lower)
        f_upper = objective(upper)
        expansion_steps = 0
        while f_lower * f_upper > 0.0 and upper < 1e12 and expansion_steps < 50:
            upper *= 2.0
            f_upper = objective(upper)
            expansion_steps += 1
        if f_lower * f_upper > 0.0:
            raise ValueError(
                f"Failed to bracket swap root for maturity {maturity_date}: "
                f"rate={swap_rate}, f({lower})={f_lower}, f({upper})={f_upper}"
            )

        try:
            return brentq(objective, lower, upper, xtol=1e-14, rtol=1e-14)
        except ValueError as exc:
            raise ValueError(
                f"Swap bootstrap calibration failed for maturity {maturity_date}: "
                f"rate={swap_rate}, f({lower})={f_lower}, f({upper})={f_upper}"
            ) from exc
    
    def build_curve(self) -> None:
        """
        Build the curve by bootstrapping all instruments.
        """
        # Sort market data by maturity
        sorted_data = sorted(self.market_data, key=lambda x: x["maturity_date"])
        
        # First add valuation date point (discount factor = 1)
        self.discount_factors[self.valuation_date] = 1.0
        self.zero_rates[self.valuation_date] = 0.0
        
        # Bootstrap curve
        for data in sorted_data:
            instrument_type = data["instrument_type"]
            rate = data["rate"]
            maturity_date = data["maturity_date"]
            
            # Skip if maturity date is already in our curve
            if maturity_date in self.discount_factors:
                raise ValueError(
                    f"build_curve: duplicate maturity date {maturity_date.strftime('%Y-%m-%d')} "
                    f"for instrument {instrument_type}"
                )
                
            if instrument_type == Instrument.CASH:
                df = self._bootstrap_cash_rate(rate, maturity_date)
            elif instrument_type == Instrument.FUTURES:
                start_date = data["start_date"]
                df = self._bootstrap_futures(rate, start_date, maturity_date)
            elif instrument_type == Instrument.SWAP:
                df = self._bootstrap_swap(rate, maturity_date)
            else:
                raise ValueError(f"Unsupported instrument type: {instrument_type}")

            # Guard: each bootstrapped DF must be strictly positive and finite.
            # A non-positive DF (e.g. pathological negative cash rate where
            # 1 + r*t <= 0) would contaminate downstream power-law alpha
            # interpolation in _bootstrap_swap via DF**alpha for fractional alpha.
            if not math.isfinite(df) or df <= 0.0:
                raise ValueError(
                    f"Bootstrap produced non-positive / non-finite discount factor "
                    f"{df!r} for instrument {instrument_type} maturity {maturity_date} "
                    f"(rate={rate}); curve is mathematically degenerate"
                )

            self.discount_factors[maturity_date] = df
            self.zero_rates[maturity_date] = self._calculate_zero_rate(maturity_date)
            
    def get_discount_factors(self) -> pd.DataFrame:
        """
        Get the discount factors as a DataFrame.
        
        Returns:
            DataFrame with dates and discount factors
        """
        if not self.discount_factors:
            raise ValueError("Curve has not been built. Call build_curve() first.")
            
        df = pd.DataFrame({
            "date": list(self.discount_factors.keys()),
            "discount_factor": list(self.discount_factors.values())
        })
        # Ensure pandas datetime dtype for .dt access
        df["date"] = pd.to_datetime(df["date"])
        val_ts = pd.Timestamp(self.valuation_date)
        df["days"] = (df["date"] - val_ts).dt.days
        df["years"] = df["days"] / CURVE_TIME_BASIS
        return df.sort_values("date")
    
    def get_zero_rates(self) -> pd.DataFrame:
        """
        Get the zero rates as a DataFrame.
        
        Returns:
            DataFrame with dates and zero rates
        """
        if not self.zero_rates:
            raise ValueError("Curve has not been built. Call build_curve() first.")
            
        df = pd.DataFrame({
            "date": list(self.zero_rates.keys()),
            "zero_rate": list(self.zero_rates.values())
        })
        df["date"] = pd.to_datetime(df["date"])
        val_ts = pd.Timestamp(self.valuation_date)
        df["days"] = (df["date"] - val_ts).dt.days
        df["years"] = df["days"] / CURVE_TIME_BASIS
        return df.sort_values("date")
    
    def get_forward_rates(self, forward_tenor: str = "3M") -> pd.DataFrame:
        """
        Calculate rolling simple forwards with fixed tenor from the curve.
        """
        if not self.discount_factors:
            raise ValueError("Curve has not been built. Call build_curve() first.")

        # helper: add tenor to an arbitrary date
        def add_tenor(dt: datetime, tenor: str) -> datetime:
            unit = tenor[-1].upper()
            value = int(tenor[:-1])
            if unit == 'D':
                return dt + timedelta(days=value)
            elif unit == 'W':
                return dt + timedelta(weeks=value)
            elif unit == 'M':
                return dt + relativedelta(months=value)
            elif unit == 'Y':
                return dt + relativedelta(years=value)
            else:
                raise ValueError(f"Unsupported tenor unit: {unit}")

        dates = sorted(list(self.discount_factors.keys()))
        forward_rates = []
        forward_dates = []
        
        for d in dates:
            if d < self.valuation_date:
                continue
            d_end = add_tenor(d, forward_tenor)
            # Skip if end date is before last known curve point and we cannot interpolate
            try:
                df_start = self._get_discount_factor(d)
                df_end = self._get_discount_factor(d_end)
            except Exception:
                continue
            year_frac = self._day_count_factor(d, d_end)
            if year_frac <= 0:
                continue
            fwd = (df_start / df_end - 1.0) / year_frac  # simple forward
            forward_rates.append(fwd)
            forward_dates.append(d)
        
        return pd.DataFrame({
            "date": forward_dates,
            "forward_rate": forward_rates
        })
    
    def calibrate_exponential_polynomial(self, degree: int = 5) -> Dict:
        """
        Calibrate an exponential polynomial model to the discount factors.
        
        The model has the form: DF(t) = exp(-sum(a_i * t^i)) for i=1 to degree
        This enforces DF(0) = 1 automatically.
        
        Args:
            degree: Degree of the polynomial (default 5)
            
        Returns:
            Dictionary with calibrated parameters and evaluation function
        """
        if not self.discount_factors:
            raise ValueError("Curve has not been built. Call build_curve() first.")
        
        # Extract times and discount factors
        dates = sorted(self.discount_factors.keys())
        # Keep only strictly positive DFs to avoid log issues
        dates = [dt for dt in dates if self.discount_factors[dt] > 0.0]
        times = [self._curve_time(d) for d in dates]
        log_dfs = [-np.log(self.discount_factors[d]) for d in dates]
        
        if len(times) <= 1:
            raise ValueError("Not enough positive discount factors to calibrate exponential-polynomial model.")
        
        # Skip valuation date (t=0) for fitting
        times = times[1:]
        log_dfs = log_dfs[1:]
        
        # Create Vandermonde matrix for polynomial regression
        X = np.vstack([np.power(times, i) for i in range(1, degree+1)]).T
        
        # Grid on which to enforce non-negative forward rates (P'(t) >= 0)
        grid = np.linspace(0.0, 50.0, 501)

        # Objective function: sum of squared errors
        def objective(coeffs):
            pred = X @ coeffs
            return np.sum((pred - log_dfs) ** 2)
        
        # Initialize with ordinary least squares solution
        initial_coeffs, _, _, _ = np.linalg.lstsq(X, log_dfs, rcond=None)
        
        # For monotonicity: ensure derivative of DF is negative at grid points
        # DF'(t) = -DF(t) * sum(i * a_i * t^(i-1))
        # For DF to be decreasing, sum(i * a_i * t^(i-1)) must be positive
        def forward_nonnegative_constraints(coeffs):
            # Enforce f(t) = P'(t) >= 0 on the grid (vector inequality)
            g = grid.copy()
            g[0] = 1e-6  # avoid t=0 exactly for powers
            deriv = np.zeros_like(g)
            for i in range(1, degree + 1):
                deriv += i * coeffs[i - 1] * np.power(g, i - 1)
            return deriv  # SLSQP treats each element >= 0

        # Constraints
        constraints = [
            # Forward non-negativity across grid
            {'type': 'ineq', 'fun': forward_nonnegative_constraints},
            # Short-end forward non-negative: a1 >= 0
            {'type': 'ineq', 'fun': lambda c: c[0]},
            # Long-end stability: leading coeff >= 0 keeps f(t) >= 0 as t -> inf
            {'type': 'ineq', 'fun': lambda c: c[-1]}
        ]
        
        # Perform constrained optimization
        result = minimize(objective, initial_coeffs, constraints=constraints, method='SLSQP')
        
        # Get optimized coefficients
        coeffs = result.x
        
        # Create evaluation function
        def discount_function(t):
            """Evaluate calibrated discount factor at time t (supports vector inputs)."""
            t_arr = np.asarray(t, dtype=float)
            # P(t) = sum a_i * t^i
            exponent = np.zeros_like(t_arr, dtype=float)
            for i in range(1, degree + 1):
                exponent += coeffs[i - 1] * np.power(t_arr, i)
            df = np.exp(-exponent)
            # Enforce DF(0) = 1 explicitly (handles t<=0 too)
            mask = t_arr <= 0.0
            if np.any(mask):
                df = df.astype(float, copy=True)
                df[mask] = 1.0
            return float(df) if df.ndim == 0 else df
         
        # Create rate function (continuously compounded), r(t) = P(t)/t for t>0
        def zero_rate_function(t):
            """Evaluate calibrated zero rate at time t (supports vector inputs)."""
            t_arr = np.asarray(t, dtype=float)
            # P(t) = sum a_i * t^i
            exponent = np.zeros_like(t_arr, dtype=float)
            for i in range(1, degree + 1):
                exponent += coeffs[i - 1] * np.power(t_arr, i)
            r = np.zeros_like(t_arr, dtype=float)
            mask_pos = t_arr > 0.0
            if np.any(mask_pos):
                r[mask_pos] = exponent[mask_pos] / t_arr[mask_pos]
            # Short-end approximation for t<=0
            r[~mask_pos] = coeffs[0] if len(coeffs) > 0 else 0.0
            return float(r) if r.ndim == 0 else r
         
        # Create forward rate function f(t) = P'(t)
        def forward_rate_function(t):
            """Evaluate instantaneous forward rate at time t (supports vector inputs)."""
            t_arr = np.asarray(t, dtype=float)
            deriv = np.zeros_like(t_arr, dtype=float)
            for i in range(1, degree + 1):
                deriv += i * coeffs[i - 1] * np.power(t_arr, i - 1)
            # Short-end value at t<=0 uses a1 (short rate approx) and clamp tiny negatives
            if len(coeffs) > 0:
                deriv = np.where(t_arr <= 0.0, coeffs[0], deriv)
            deriv = np.maximum(0.0, deriv)  # avoid tiny negative due to numeric noise
            return float(deriv) if deriv.ndim == 0 else deriv
        
        # Get model fit statistics
        predicted_dfs = [discount_function(t) for t in times]
        actual_dfs = [np.exp(-log_df) for log_df in log_dfs]
        mse = np.mean((np.array(predicted_dfs) - np.array(actual_dfs))**2)
        rmse = np.sqrt(mse)
        
        # Return results
        return {
            "coefficients": coeffs,
            "degree": degree,
            "discount_function": discount_function,
            "zero_rate_function": zero_rate_function,
            "forward_rate_function": forward_rate_function,
            "rmse": rmse,
            "r2": 1 - np.sum((np.array(actual_dfs) - np.array(predicted_dfs))**2) / 
                  np.sum((np.array(actual_dfs) - np.mean(actual_dfs))**2) if len(actual_dfs) > 1 else 1.0,
            "success": result.success,
            "message": result.message
        }
    
    def calibrate_cubic_spline(self) -> Dict:
        """
        Calibrate a PCHIP spline on u(t) = -ln(DF(t)).

        Returns callable wrappers for discount_function, zero_rate_function,
        and forward_rate_function with non-negative instantaneous forwards
        guaranteed by PCHIP shape preservation.
        """
        if not self.discount_factors:
            raise ValueError("Curve has not been built. Call build_curve() first.")

        dates = sorted(self.discount_factors.keys())
        times = [self._curve_time(d) for d in dates]
        dfs = [self.discount_factors[d] for d in dates]
        # guard: strictly positive DFs
        times, dfs = zip(*[(t, df) for t, df in zip(times, dfs) if df > 0.0])
        times = list(times); dfs = list(dfs)

        # Fit u(t) = -ln DF(t), monotone increasing => f(t) = u'(t) >= 0
        u = [-np.log(df) for df in dfs]
        cs = PchipInterpolator(times, u, extrapolate=False)

        discount_function = DiscountFunction(cs, times)
        zero_rate_function = ZeroRateFunction(cs, times)
        forward_rate_function = ForwardRateFunction(cs, times)

        return {
            "discount_function": discount_function,
            "zero_rate_function": zero_rate_function,
            "forward_rate_function": forward_rate_function,
            "spline": cs,
            "time_basis": CURVE_TIME_BASIS,
        }

    def calibrate_log_linear(self) -> Dict:
        """Piecewise log-linear interpolation on bootstrapped discount factors.

        This is the Bloomberg/ASKB standard interpolation method for OIS
        curves: linear interpolation in log(DF) space, producing piecewise-
        constant forward rates between nodes.

        Returns:
            Dictionary with callable ``discount_function``,
            ``zero_rate_function``, and ``forward_rate_function``.
        """
        if not self.discount_factors:
            raise ValueError("Curve has not been built. Call build_curve() first.")

        dates = sorted(self.discount_factors.keys())
        times = np.array([self._curve_time(d) for d in dates])
        dfs = np.array([self.discount_factors[d] for d in dates])
        # guard: strictly positive DFs
        pos_mask = dfs > 0.0
        times = times[pos_mask]
        dfs = dfs[pos_mask]
        log_dfs = np.log(dfs)

        discount_function = LogLinearDiscountFunction(times, log_dfs)
        zero_rate_function = LogLinearZeroRateFunction(times, log_dfs)
        forward_rate_function = LogLinearForwardRateFunction(times, log_dfs)

        return {
            "discount_function": discount_function,
            "zero_rate_function": zero_rate_function,
            "forward_rate_function": forward_rate_function,
            "time_basis": CURVE_TIME_BASIS,
        }
    
    def calibrate_nelson_siegel(self) -> Dict:
        """
        Calibrate a Nelson-Siegel model to the discount factors.
        
        The Nelson-Siegel model is a parsimonious yield curve parameterization:
        r(t) = β₀ + β₁[(1-e^(-t/τ))/(t/τ)] + β₂[(1-e^(-t/τ))/(t/τ) - e^(-t/τ)]
        
        Returns:
            Dictionary with calibrated parameters and evaluation function
        """
        if not self.discount_factors:
            raise ValueError("Curve has not been built. Call build_curve() first.")
        
        from scipy.optimize import minimize
        
        # Extract times and rates
        dates = sorted(self.discount_factors.keys())
        times = [self._curve_time(d) for d in dates]
        rates = [self.zero_rates[d] for d in dates]
        
        # Skip t=0 point
        times = times[1:]
        rates = rates[1:]
        
        # Nelson-Siegel formula
        def ns_formula(t, beta0, beta1, beta2, tau):
            if t == 0:
                # Limit as t approaches 0
                return beta0 + beta1
            
            exp_term = np.exp(-t / tau)
            term1 = (1 - exp_term) / (t / tau)
            term2 = term1 - exp_term

            return beta0 + beta1 * term1 + beta2 * term2
        
        # Objective function to minimize
        def objective(params):
            beta0, beta1, beta2, tau = params
            
            # Ensure tau is positive
            if tau <= 0:
                return 1e10  # Large penalty
            
            # Calculate rates using Nelson-Siegel
            pred_rates = [ns_formula(t, beta0, beta1, beta2, tau) for t in times]
            
            # Return sum of squared errors
            return np.sum((np.array(pred_rates) - np.array(rates))**2)
        
        # Initial parameter guesses
        # beta0: long-term rate (use longest tenor)
        # beta1: short-term component (short rate - long rate)
        # beta2: medium-term component (try 0 initially)
        # tau: decay parameter (try 2 years initially)
        initial_params = [
            rates[-1],              # beta0: long rate
            rates[0] - rates[-1],   # beta1: short - long
            0.0,                    # beta2: curvature
            2.0                     # tau: decay rate
        ]
        
        # Parameter bounds
        bounds = [
            (-0.1, 0.2),    # beta0: reasonable range for long rate
            (-0.2, 0.2),    # beta1: allow for positive or negative slope
            (-0.2, 0.2),    # beta2: allow for positive or negative curvature
            (0.1, 10.0)     # tau: reasonable range for decay parameter
        ]
        
        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Extract calibrated parameters
        beta0, beta1, beta2, tau = result.x
        
        # Create evaluation functions
        def zero_rate_function(t: float) -> float:
            """Evaluate zero rate at time t using Nelson-Siegel model"""
            return ns_formula(t, beta0, beta1, beta2, tau)
        
        def discount_function(t: float) -> float:
            """Evaluate discount factor at time t using Nelson-Siegel model"""
            if t <= 0:
                return 1.0
            
            r = zero_rate_function(t)
            return np.exp(-r * t)
        
        def forward_rate_function(t: float) -> float:
            """Evaluate instantaneous forward rate at time t using Nelson-Siegel model"""
            if t <= 0:
                return beta0 + beta1
            
            exp_term = np.exp(-t / tau)
            return beta0 + beta1 * exp_term + beta2 * (t / tau) * exp_term
        
        # Calculate model fit statistics
        predicted_rates = [zero_rate_function(t) for t in times]
        mse = np.mean((np.array(predicted_rates) - np.array(rates))**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((np.array(rates) - np.array(predicted_rates))**2) / \
             np.sum((np.array(rates) - np.mean(rates))**2)
        
        return {
            "parameters": {
                "beta0": beta0,
                "beta1": beta1, 
                "beta2": beta2,
                "tau": tau
            },
            "discount_function": discount_function,
            "zero_rate_function": zero_rate_function,
            "forward_rate_function": forward_rate_function,
            "rmse": rmse,
            "r2": r2,
            "success": result.success,
            "message": result.message
        }


# ============================================================================
# Zero-coupon bond return helpers (for excess-return decomposition)
# ============================================================================


def calc_zcb_slide_return(
    df_func: Callable[[float], float],
    duration: float,
    holding_period: float = 1.0,
) -> float:
    """Return on a duration-matched zero-coupon bond under slide (rolldown) pricing.

    The investor buys a ZCB with maturity equal to the bond's modified
    duration *d* at price D(0, d) and, after *Δt* years, re-prices it on the
    **same** (unchanged) spot curve at the shorter tenor *d − Δt*:

    .. math::

        R_{rf}^{slide} = \\frac{D(0,\\, d - \\Delta t)}{D(0,\\, d)} - 1

    This equals the 1-year **implied forward rate** at tenor *d*, making the
    benchmark duration-dependent and therefore useful for stripping curve
    carry from credit signals in cross-sectional relative-value ranking.

    Edge cases:
        * *d ≤ 0*  → 0.0 (expired / no duration)
        * *d ≤ Δt* → ZCB matures within the holding period; return = 1/D(0, d) − 1

    Args:
        df_func: Risk-free discount function D(t), continuously compounded.
        duration: Modified duration of the corporate bond (years).
        holding_period: Investment horizon in years (default 1.0).

    Returns:
        Slide (rolldown) total return of the duration-matched ZCB.
    """
    if duration <= 0.0:
        return 0.0

    d0 = float(df_func(duration))

    if duration <= holding_period:
        # ZCB matures within the holding period — investor receives par
        return 1.0 / d0 - 1.0

    d1 = float(df_func(duration - holding_period))
    return d1 / d0 - 1.0


def calc_zcb_forward_return(
    df_func: Callable[[float], float],
    holding_period: float = 1.0,
) -> float:
    """No-arbitrage risk-free return over the holding period.

    Under the no-arbitrage forward-pricing framework used by
    ``Bond.calc_return_survival`` (where D(t₁, T) = D(0, T) / D(0, t₁)),
    the return on **any** risk-free zero-coupon bond is:

    .. math::

        R_{rf}^{fwd} = \\frac{1}{D(0,\\, \\Delta t)} - 1

    This is duration-independent and represents the pure funding rate over
    the holding period.  Subtracting it from the corporate survival return
    yields the excess return consistent with the forward-pricing framework.

    Args:
        df_func: Risk-free discount function D(t), continuously compounded.
        holding_period: Investment horizon in years (default 1.0).

    Returns:
        Forward (no-arbitrage) risk-free return.
    """
    d = float(df_func(holding_period))
    if d <= 0.0:
        return 0.0
    return 1.0 / d - 1.0


# ============================================================================
# Cashflow-matched risk-free benchmark (Options A–D)
# ============================================================================


class RfBenchmarkMode(Enum):
    """Selectable risk-free benchmark modes for excess-return computation."""
    CASHFLOW_MATCHED = "cashflow_matched"   # Option A: exact cashflow-matched RF twin
    KRD = "krd"                             # Option D: key-rate-duration matched


def _build_30360_schedule(
    value_date: "date",
    maturity_date: "date",
    frequency: int = 2,
) -> List["date"]:
    """Generate a coupon schedule from maturity backward using 30/360 convention.

    Returns dates strictly after value_date, up to and including maturity_date.
    """
    from datetime import date as date_type

    step_months = 12 // frequency
    schedule: List[date_type] = []
    current = maturity_date

    while current > value_date:
        schedule.append(current)
        new_month = current.month - step_months
        new_year = current.year
        while new_month <= 0:
            new_month += 12
            new_year -= 1
        day = min(current.day, 28) if current.month == 2 else current.day
        try:
            current = date_type(new_year, new_month, day)
        except ValueError:
            current = date_type(new_year, new_month, 28)

    return sorted(schedule)


def _yearfrac_30360(d0: "date", d1: "date") -> float:
    """Year fraction under 30/360 day-count convention (US Bond Basis)."""
    y0, m0, day0 = d0.year, d0.month, d0.day
    y1, m1, day1 = d1.year, d1.month, d1.day
    if day0 == 31:
        day0 = 30
    if day1 == 31 and day0 >= 30:
        day1 = 30
    return (360 * (y1 - y0) + 30 * (m1 - m0) + (day1 - day0)) / 360.0


def _yearfrac_act360(d0: "date", d1: "date") -> float:
    """Year fraction under ACT/360 day-count convention."""
    return (d1 - d0).days / 360.0


def _yearfrac(d0: "date", d1: "date", day_count_convention: str = "30/360") -> float:
    """Year fraction dispatcher respecting the bond's day-count convention."""
    if day_count_convention.upper() == "ACT/360":
        return _yearfrac_act360(d0, d1)
    return _yearfrac_30360(d0, d1)


def calc_cf_matched_slide_return(
    df_func: Callable[[float], float],
    coupon_rate: float,
    maturity_date: "date",
    value_date: "date",
    horizon_date: "date",
    frequency: int = 2,
    day_count_convention: str = "30/360",
) -> float:
    """Cashflow-matched risk-free slide return (Option A, scenario).

    Computes the slide return of a synthetic riskless bond (S(t)≡1) with
    the same coupon, frequency, and maturity as the corporate bond. Both the
    pricing and reinvestment use the unchanged-curve-shape (slide) framework.

    .. math::

        R_{rf}^{cf\\text{-}slide} = \\frac{P_{rf,slide}(t_1) + c\\sum_{t_i\\in(t_0,t_1]} 1/D(0,\\,t_1-t_i)}
                                         {P_{rf}(t_0)} - 1

    where:
        P_rf(t0) = c * Σ D(0, t_i) + N * D(0, T)
        P_rf_slide(t1) = c * Σ D(0, t_i - t1) + N * D(0, T - t1)   for t_i > t1

    Args:
        df_func: Risk-free discount function D(t).
        coupon_rate: Annual coupon rate (e.g. 0.05 for 5%).
        maturity_date: Bond maturity date.
        value_date: Valuation date t0.
        horizon_date: Horizon date t1.
        frequency: Coupon frequency per year (default 2 = semi-annual).
        day_count_convention: Day-count convention ('30/360' or 'ACT/360').

    Returns:
        Cashflow-matched risk-free slide return.
    """
    coupon_payment = 100.0 * coupon_rate / frequency
    schedule = _build_30360_schedule(value_date, maturity_date, frequency)

    if not schedule:
        return 0.0

    # Time fractions from value_date
    del_t = np.array([_yearfrac(value_date, d, day_count_convention) for d in schedule])

    # P_rf(t0): price the riskless bond at t0
    df_values = np.array([float(df_func(t)) for t in del_t])
    p_rf_t0 = coupon_payment * np.sum(df_values) + 100.0 * df_values[-1]

    if p_rf_t0 <= 0.0:
        return 0.0

    # Split schedule into interim (t0, t1] and future (> t1)
    t1 = _yearfrac(value_date, horizon_date, day_count_convention)
    interim_mask = del_t <= t1 + 1e-10  # coupon dates in (t0, t1]
    future_mask = ~interim_mask

    # Maturity falls within the horizon — all cashflows are interim
    maturity_in_horizon = maturity_date <= horizon_date

    # Reinvested interim coupons at slide rates: 1/D(0, t1 - t_i)
    coupons_reinvested = 0.0
    if np.any(interim_mask):
        del_t_slide_interim = t1 - del_t[interim_mask]
        # Clamp tiny negative values from floating-point
        del_t_slide_interim = np.maximum(del_t_slide_interim, 0.0)
        df_slide_interim = np.array([
            float(df_func(max(t, 1e-10))) if t > 1e-10 else 1.0
            for t in del_t_slide_interim
        ])
        coupons_reinvested = coupon_payment * float(np.sum(1.0 / df_slide_interim))

    # Add principal reinvested if maturity falls inside the horizon
    principal_reinvested = 0.0
    if maturity_in_horizon:
        t_mat = _yearfrac(value_date, maturity_date, day_count_convention)
        del_t_mat_to_horizon = t1 - t_mat
        if del_t_mat_to_horizon > 1e-10:
            principal_reinvested = 100.0 / float(df_func(del_t_mat_to_horizon))
        else:
            principal_reinvested = 100.0

    # P_rf_slide(t1): reprice future cashflows at shortened tenors
    p_rf_slide_t1 = 0.0
    if np.any(future_mask):
        del_t_future_from_t1 = del_t[future_mask] - t1
        df_future = np.array([float(df_func(t)) for t in del_t_future_from_t1])
        p_rf_slide_t1 = coupon_payment * np.sum(df_future) + 100.0 * df_future[-1]

    return (p_rf_slide_t1 + coupons_reinvested + principal_reinvested) / p_rf_t0 - 1.0


def calc_cf_matched_realized_return(
    prev_df_func: Callable[[float], float],
    curr_df_func: Callable[[float], float],
    coupon_rate: float,
    maturity_date: "date",
    prev_date: "date",
    curr_date: "date",
    frequency: int = 2,
    day_count_convention: str = "30/360",
) -> float:
    """Cashflow-matched risk-free realized return (Option A, backtest).

    Computes the realized return of a synthetic riskless twin between two
    actual curve observations. The bond is priced on ``prev_df_func`` at
    ``prev_date`` and repriced on ``curr_df_func`` at ``curr_date``.

    .. math::

        R_{rf}^{cf\\text{-}realized} = \\frac{P_{rf}(t_1;\\,\\text{curve}_1) + \\text{coupons}_{(t_0,t_1]} + \\text{principal}_{\\mathbb{1}_{T \\le t_1}}}
                                             {P_{rf}(t_0;\\,\\text{curve}_0)} - 1

    Coupons paid in (t0, t1] are accumulated at face value (no reinvestment
    assumed for short backtest periods — typically 1 week or 1 month).

    Args:
        prev_df_func: Discount function at previous date (curve_0).
        curr_df_func: Discount function at current date (curve_1).
        coupon_rate: Annual coupon rate.
        maturity_date: Bond maturity date.
        prev_date: Previous rebalance date t0.
        curr_date: Current rebalance date t1.
        frequency: Coupon frequency per year (default 2).
        day_count_convention: Day-count convention ('30/360' or 'ACT/360').

    Returns:
        Cashflow-matched risk-free realized return.
    """
    coupon_payment = 100.0 * coupon_rate / frequency

    # Full schedule from prev_date
    schedule_full = _build_30360_schedule(prev_date, maturity_date, frequency)
    if not schedule_full:
        return 0.0

    # P_rf(t0; curve_0): price on prev curve
    del_t_prev = np.array([_yearfrac(prev_date, d, day_count_convention) for d in schedule_full])
    df_prev = np.array([float(prev_df_func(t)) for t in del_t_prev])
    p_rf_t0 = coupon_payment * np.sum(df_prev) + 100.0 * df_prev[-1]

    if p_rf_t0 <= 0.0:
        return 0.0

    # Split: coupons paid in (prev_date, curr_date], and detect principal
    coupons_paid = 0.0
    principal_paid = 0.0
    remaining_schedule = []
    for d in schedule_full:
        if d <= curr_date:
            coupons_paid += coupon_payment
            # If this is the maturity date, principal is also received
            if d == maturity_date:
                principal_paid = 100.0
        else:
            remaining_schedule.append(d)

    # P_rf(t1; curve_1): price remaining cashflows on curr curve
    p_rf_t1 = 0.0
    if remaining_schedule:
        del_t_curr = np.array([_yearfrac(curr_date, d, day_count_convention) for d in remaining_schedule])
        df_curr = np.array([float(curr_df_func(t)) for t in del_t_curr])
        p_rf_t1 = coupon_payment * np.sum(df_curr) + 100.0 * df_curr[-1]

    return (p_rf_t1 + coupons_paid + principal_paid) / p_rf_t0 - 1.0


# ============================================================================
# KRD-matched risk-free benchmark (Option D)
# ============================================================================

KRD_TENORS: List[float] = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]


class KrdHedgeConvention(NamedTuple):
    """Immutable specification of the synthetic KRD hedge basket.

    Defines the standard instruments used to construct the hedge portfolio.
    The bond being hedged uses its own frequency and day-count for its KRD
    vector; the hedge basket is independent of the bond.
    """
    tenors: List[float]
    frequency: int
    day_count: str
    bump_size: float


USD_SWAP_HEDGE_CONVENTION = KrdHedgeConvention(
    tenors=KRD_TENORS,
    frequency=2,
    day_count="30/360",
    bump_size=0.0001,
)

ICE_HEDGE_CONVENTION = KrdHedgeConvention(
    tenors=[0.5, 2.0, 5.0, 10.0, 20.0, 30.0],
    frequency=2,
    day_count="30/360",
    bump_size=0.0001,
)

USD_FUT_HEDGE_CONVENTION = KrdHedgeConvention(
    tenors=[2.0, 5.0, 7.0, 10.0, 20.0, 30.0],
    frequency=2,
    day_count="30/360",
    bump_size=0.0001,
)

EUR_FUT_HEDGE_CONVENTION = KrdHedgeConvention(
    tenors=[2.0, 5.0, 10.0, 30.0],
    frequency=2,
    day_count="30/360",
    bump_size=0.0001,
)

HEDGE_CONVENTION_BY_CURRENCY: dict[str, KrdHedgeConvention] = {
    "USD": USD_FUT_HEDGE_CONVENTION,
    "EUR": EUR_FUT_HEDGE_CONVENTION,
}


def _krd_bump_weight(t: float, bump_tenor: float, all_tenors: List[float]) -> float:
    """Ho (1992) triangular bump weight at tenor *t* for a bump at *bump_tenor*.

    The bump profiles partition unity: ``sum(w_k(t) for k in all_tenors) == 1``
    for all ``t >= 0``.
    """
    idx = all_tenors.index(bump_tenor)
    k = bump_tenor

    # First tenor: flat from 0 to k, then ramp down to next tenor
    if idx == 0:
        if t <= k:
            return 1.0
        k_right = all_tenors[idx + 1]
        if t < k_right:
            return (k_right - t) / (k_right - k)
        return 0.0

    # Last tenor: ramp up from previous tenor, then flat beyond k
    if idx == len(all_tenors) - 1:
        k_left = all_tenors[idx - 1]
        if t <= k_left:
            return 0.0
        if t <= k:
            return (t - k_left) / (k - k_left)
        return 1.0

    # Interior tenor: triangular — ramp up from k_left, ramp down to k_right
    k_left = all_tenors[idx - 1]
    k_right = all_tenors[idx + 1]
    if t <= k_left:
        return 0.0
    if t <= k:
        return (t - k_left) / (k - k_left)
    if t < k_right:
        return (k_right - t) / (k_right - k)
    return 0.0


def _krd_bump_weights_vec(
    t_arr: np.ndarray, bump_tenor: float, all_tenors: List[float],
) -> np.ndarray:
    """Vectorized Ho (1992) triangular bump weights for an array of cashflow tenors.

    Equivalent to ``np.array([_krd_bump_weight(t, bump_tenor, all_tenors) for t in t_arr])``
    but operates on the entire array without Python loops.
    """
    idx = all_tenors.index(bump_tenor)
    k = bump_tenor
    weights = np.zeros(len(t_arr))

    if idx == 0:
        k_right = all_tenors[1]
        mask_le_k = t_arr <= k
        mask_ramp = (t_arr > k) & (t_arr < k_right)
        weights[mask_le_k] = 1.0
        weights[mask_ramp] = (k_right - t_arr[mask_ramp]) / (k_right - k)
    elif idx == len(all_tenors) - 1:
        k_left = all_tenors[idx - 1]
        mask_ramp = (t_arr > k_left) & (t_arr <= k)
        mask_ge_k = t_arr > k
        weights[mask_ramp] = (t_arr[mask_ramp] - k_left) / (k - k_left)
        weights[mask_ge_k] = 1.0
    else:
        k_left = all_tenors[idx - 1]
        k_right = all_tenors[idx + 1]
        mask_up = (t_arr > k_left) & (t_arr <= k)
        mask_down = (t_arr > k) & (t_arr < k_right)
        weights[mask_up] = (t_arr[mask_up] - k_left) / (k - k_left)
        weights[mask_down] = (k_right - t_arr[mask_down]) / (k_right - k)

    return weights


def _price_riskless_bond(
    df_func: Callable[[float], float],
    coupon_rate: float,
    schedule_yearfracs: np.ndarray,
    frequency: int,
) -> float:
    """Price a riskless bond: P = sum(c * D(t_i)) + 100 * D(T).

    ``df_func`` must accept array inputs (e.g. ``DiscountFunction``).
    """
    coupon_payment = 100.0 * coupon_rate / frequency
    df_values = np.asarray(df_func(schedule_yearfracs), dtype=float).ravel()
    return float(coupon_payment * np.sum(df_values) + 100.0 * df_values[-1])


@dataclass(frozen=True)
class KrdCurveSet:
    """Per-currency/per-date bumped-curve cache shared across the bond universe.

    The bumped discount functions are bond-independent (they depend only on
    the risk-free curve, the node grid, and the bump size), so a single
    ``KrdCurveSet`` is built once per currency/date and reused to price every
    bond's KRD vector via :func:`compute_bond_krd`.

    Two construction methods share this container:

    * ``"zero"`` — analytic Ho (1992) triangular tent on the continuously
      compounded zero rate.  ``df_up``/``df_down`` are unused; the bumped
      discount factors are derived inline in :func:`compute_bond_krd` so the
      arithmetic is bit-identical to the legacy zero-bump primitive.
    * ``"par"`` — par-coupon yield bump at one node, re-bootstrapped to a
      discount curve (Bloomberg/ICE).  ``df_up[k]``/``df_down[k]`` are the
      re-bootstrapped discount functions for an up/down bump at node ``k``.

    The constant-OAS overlay ``exp(-oas·t)`` is *not* baked in here (OAS is a
    per-bond attribute); it is applied in :func:`compute_bond_krd`.
    """
    method: str
    tenors: Tuple[float, ...]
    bump_size: float
    df_base: Callable[[float], float]
    df_up: Optional[Tuple[Callable[[float], float], ...]] = None
    df_down: Optional[Tuple[Callable[[float], float], ...]] = None


class _ParCurveDiscountFunction:
    """Discount function for a re-bootstrapped par curve.

    PCHIP interpolation on ``u(t) = -ln D(t)`` over ``[0, t_max]`` (matching
    the production ``calibrate_cubic_spline`` convention), with flat
    instantaneous-forward extrapolation beyond the longest node (constant
    forward ``u'(t_max)``, floored at zero) so bonds maturing past the node
    grid remain priceable.
    """
    def __init__(self, times: np.ndarray, u: np.ndarray):
        self.times = np.asarray(times, dtype=float)
        self.spline = PchipInterpolator(
            self.times, np.asarray(u, dtype=float), extrapolate=False,
        )
        self.t_max = float(self.times[-1])
        self.u_max = float(self.spline(self.t_max))
        self.f_max = max(0.0, float(self.spline.derivative()(self.t_max)))

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        out = np.ones_like(t_arr)
        mask_in = (t_arr > 0.0) & (t_arr <= self.t_max)
        mask_ext = t_arr > self.t_max
        if np.any(mask_in):
            out[mask_in] = np.exp(-self.spline(t_arr[mask_in]))
        if np.any(mask_ext):
            u_ext = self.u_max + self.f_max * (t_arr[mask_ext] - self.t_max)
            out[mask_ext] = np.exp(-u_ext)
        return float(out) if out.ndim == 0 else out


def build_zero_krd_curve_set(
    df_func: Callable[[float], float],
    tenors: Optional[List[float]] = None,
    bump_size: float = 0.0001,
) -> KrdCurveSet:
    """Build a zero-rate-bump KRD curve set (analytic Ho tent).

    No curve construction is performed: the bumped discount factors are
    derived analytically from *df_func* inside :func:`compute_bond_krd`.
    *tenors* must be ascending (the Ho tent uses node neighbours).
    """
    if tenors is None:
        tenors = KRD_TENORS
    return KrdCurveSet("zero", tuple(tenors), float(bump_size), df_func, None, None)


def compute_bond_krd(
    curve_set: KrdCurveSet,
    coupon_rate: float,
    maturity_date: "date",
    value_date: "date",
    frequency: int = 2,
    day_count_convention: str = "30/360",
    oas: float = 0.0,
    *,
    _schedule_yearfracs: Optional[np.ndarray] = None,
) -> Dict[float, float]:
    """Per-bond key rate durations via central-difference bump-and-reprice.

    The bumped discount functions live in *curve_set* and are shared across
    the bond universe.  Cashflows are discounted at ``risk-free + oas`` with
    the spread held fixed across the up/down scenarios (constant-OAS
    convention, Bloomberg/ICE).

    .. math::

        KRD_k = -\\frac{P_{up} - P_{down}}{2h \\cdot P_{base}}

    Args:
        curve_set: Shared :class:`KrdCurveSet` (``"zero"`` or ``"par"``).
        coupon_rate: Annual coupon rate (decimal, e.g. 0.05).
        maturity_date: Bond maturity date.
        value_date: Valuation date.
        frequency: Coupon frequency per year.
        day_count_convention: Day-count convention.
        oas: Constant option-adjusted spread (decimal, continuously
            compounded). Overlays ``exp(-oas·t)`` on every discount factor
            so the spread is held fixed under the bump. ``0.0`` (the default)
            reproduces the pure risk-free KRD.
        _schedule_yearfracs: Pre-computed yearfrac array. When provided,
            skips internal schedule construction (avoids duplicate work
            when the caller has already built the schedule).

    Returns:
        Dict mapping tenor → KRD value.
    """
    tenors = list(curve_set.tenors)
    bump_size = curve_set.bump_size

    if _schedule_yearfracs is not None:
        del_t = _schedule_yearfracs
    else:
        schedule = _build_30360_schedule(value_date, maturity_date, frequency)
        if not schedule:
            return {k: 0.0 for k in tenors}
        del_t = np.array([_yearfrac(value_date, d, day_count_convention) for d in schedule])

    n_cf = len(del_t)
    n_t = len(tenors)
    coupon_payment = 100.0 * coupon_rate / frequency

    # Base discount factors. The constant-OAS overlay exp(-oas·t) is a fixed
    # factor independent of the bump, so it propagates unchanged into the
    # up/down scenarios and holds the spread constant.
    df_base = np.asarray(curve_set.df_base(del_t), dtype=float).ravel()
    if oas:
        df_base = df_base * np.exp(-oas * del_t)
    p_base = float(coupon_payment * np.sum(df_base) + 100.0 * df_base[-1])

    if p_base <= 0.0:
        return {k: 0.0 for k in tenors}

    if curve_set.method == "zero":
        # Identify active tenors: skip those where all cashflows fall
        # at or below the left neighbour — KRD is exactly 0.
        max_cf_tenor = float(del_t[-1])
        n_active = n_t
        for i in range(1, n_t):
            if tenors[i - 1] >= max_cf_tenor:
                n_active = i
                break

        # Build bump weight matrix W[i, j] for active tenors only.
        W = np.empty((n_active, n_cf))
        for ai in range(n_active):
            W[ai] = _krd_bump_weights_vec(del_t, tenors[ai], tenors)

        # Bumped DFs: D_bumped = D_base * exp(∓h·W·t)
        exp_term = bump_size * W * del_t[np.newaxis, :]       # (n_active, n_cf)
        df_up = df_base[np.newaxis, :] * np.exp(-exp_term)    # (n_active, n_cf)
        df_down = df_base[np.newaxis, :] * np.exp(exp_term)   # (n_active, n_cf)

        p_up = coupon_payment * df_up.sum(axis=1) + 100.0 * df_up[:, -1]
        p_down = coupon_payment * df_down.sum(axis=1) + 100.0 * df_down[:, -1]

        krd_active = -(p_up - p_down) / (2.0 * bump_size * p_base)

        return {
            k: (float(krd_active[i]) if i < n_active else 0.0)
            for i, k in enumerate(tenors)
        }

    # method == "par": reprice on the re-bootstrapped per-node curves.
    if curve_set.df_up is None or curve_set.df_down is None:
        raise ValueError("par KrdCurveSet missing bumped discount functions")

    oas_overlay = np.exp(-oas * del_t) if oas else None
    krd: Dict[float, float] = {}
    for i, k in enumerate(tenors):
        up = np.asarray(curve_set.df_up[i](del_t), dtype=float).ravel()
        down = np.asarray(curve_set.df_down[i](del_t), dtype=float).ravel()
        if oas_overlay is not None:
            up = up * oas_overlay
            down = down * oas_overlay
        p_up = coupon_payment * np.sum(up) + 100.0 * up[-1]
        p_down = coupon_payment * np.sum(down) + 100.0 * down[-1]
        krd[k] = float(-(p_up - p_down) / (2.0 * bump_size * p_base))
    return krd


def _compute_par_yield(
    df_func: Callable[[float], float],
    schedule_yearfracs: np.ndarray,
    frequency: int = 2,
) -> float:
    """Par coupon rate for a bond on the given schedule.

    .. math::

        c_{par} = f \\cdot \\frac{1 - D(T)}{\\sum_i D(t_i)}

    Args:
        df_func: Discount function D(t).
        schedule_yearfracs: Yearfrac array for the bond's cashflow dates.
        frequency: Coupon frequency per year.

    Returns:
        Par coupon rate (decimal).
    """
    df_values = np.asarray(df_func(schedule_yearfracs), dtype=float).ravel()
    annuity = np.sum(df_values)
    if annuity < 1e-15:
        return 0.0
    return float(frequency * (1.0 - df_values[-1]) / annuity)


def _rebootstrap_par_curve(
    par_yields: np.ndarray,
    node_schedules: List[np.ndarray],
    node_times: List[float],
    frequency: int,
) -> _ParCurveDiscountFunction:
    """Sequentially bootstrap a discount curve from par-coupon yields.

    Pillars are the node maturities; ``(0, 0)`` anchors the short end.  A
    sequential solve (each node bracketed and root-found against the pillars
    fixed so far) provides a warm start, then a global Newton refine makes the
    *final* multi-pillar PCHIP curve reprice **every** par bond to par
    simultaneously.  The global step is required because PCHIP-on-``-ln D`` is
    non-local: adding a later pillar perturbs the spline shape over earlier
    segments, so a purely sequential solve would leave earlier par bonds
    mispriced on the final curve.
    """
    times: List[float] = [0.0]
    u: List[float] = [0.0]
    for k in range(len(node_times)):
        s = node_schedules[k]
        c = float(par_yields[k])
        t_k = node_times[k]
        pillars_t = list(times)
        pillars_u = list(u)

        def price_err(u_k: float, _s=s, _c=c, _t_k=t_k,
                      _pt=pillars_t, _pu=pillars_u) -> float:
            spline = PchipInterpolator(
                _pt + [_t_k], _pu + [u_k], extrapolate=False,
            )
            d = np.exp(-spline(_s))
            return float((_c / frequency) * np.sum(d) + d[-1] - 1.0)

        # Adaptive bracket in rate space r = u_k / t_k (handles negative
        # rates and steep curves); price_err is monotone decreasing in u_k.
        r_lo, r_hi = -0.5, 1.0
        u_lo = u_hi = 0.0
        bracketed = False
        for _ in range(8):
            u_lo, u_hi = r_lo * t_k, r_hi * t_k
            f_lo, f_hi = price_err(u_lo), price_err(u_hi)
            if f_lo > 0.0 > f_hi:
                bracketed = True
                break
            if f_lo <= 0.0:
                r_lo -= 0.5
            if f_hi >= 0.0:
                r_hi += 1.0
        if not bracketed:
            raise ValueError(
                f"par bootstrap failed to bracket node {k} "
                f"(T={t_k:.3f}, c={c:.6f})"
            )
        u_k = brentq(price_err, u_lo, u_hi)
        times.append(t_k)
        u.append(u_k)

    times_arr = np.array(times)

    # Global refine: solve all pillar u-values jointly so the final PCHIP
    # curve (built on every pillar) reprices each par bond to par.  The
    # sequential solution above is the warm start.
    par_arr = np.asarray(par_yields, dtype=float)

    def _residuals(u_vec: np.ndarray) -> np.ndarray:
        pillars_u = np.concatenate(([0.0], u_vec))
        spline = PchipInterpolator(times_arr, pillars_u, extrapolate=False)
        res = np.empty(len(node_times))
        for j in range(len(node_times)):
            d = np.exp(-spline(node_schedules[j]))
            res[j] = (par_arr[j] / frequency) * np.sum(d) + d[-1] - 1.0
        return res

    sol = root(_residuals, np.array(u[1:]), method="hybr")
    if not sol.success:
        raise ValueError(
            f"par bootstrap global refine failed: {sol.message}"
        )
    u_final = np.concatenate(([0.0], np.asarray(sol.x, dtype=float)))

    return _ParCurveDiscountFunction(times_arr, u_final)


def build_par_krd_curve_set(
    df_func: Callable[[float], float],
    value_date: "date",
    tenors: Optional[List[float]] = None,
    bump_size: float = 0.0001,
    frequency: int = 2,
    day_count_convention: str = "30/360",
) -> KrdCurveSet:
    """Build a par-bump KRD curve set (Bloomberg/ICE re-bootstrap).

    Par-coupon yields are read off *df_func* at the node tenors, one node is
    bumped ``±bump_size`` at a time, and a discount curve is re-bootstrapped
    for each bump (and for the unbumped base).  The bond is then repriced on
    the re-bootstrapped curves in :func:`compute_bond_krd`.  Both the base
    price and the bumped prices use re-bootstrapped curves, so the central
    difference is a clean derivative of one curve family.
    """
    if tenors is None:
        tenors = KRD_TENORS
    tenors = sorted(float(t) for t in tenors)

    node_schedules: List[np.ndarray] = []
    node_times: List[float] = []
    base_par: List[float] = []
    for tenor in tenors:
        synth_maturity = value_date + relativedelta(months=int(round(tenor * 12)))
        schedule = _build_30360_schedule(value_date, synth_maturity, frequency)
        if not schedule:
            raise ValueError(f"empty par-node schedule for tenor {tenor}")
        del_t = np.array([
            _yearfrac(value_date, d, day_count_convention) for d in schedule
        ])
        node_schedules.append(del_t)
        node_times.append(float(del_t[-1]))
        base_par.append(_compute_par_yield(df_func, del_t, frequency))

    base_par_arr = np.array(base_par)
    df_base_curve = _rebootstrap_par_curve(
        base_par_arr, node_schedules, node_times, frequency,
    )

    df_up: List[_ParCurveDiscountFunction] = []
    df_down: List[_ParCurveDiscountFunction] = []
    for j in range(len(tenors)):
        up_par = base_par_arr.copy()
        up_par[j] += bump_size
        down_par = base_par_arr.copy()
        down_par[j] -= bump_size
        df_up.append(_rebootstrap_par_curve(up_par, node_schedules, node_times, frequency))
        df_down.append(_rebootstrap_par_curve(down_par, node_schedules, node_times, frequency))

    return KrdCurveSet(
        "par", tuple(tenors), float(bump_size), df_base_curve,
        tuple(df_up), tuple(df_down),
    )


def _compute_synthetic_krd_and_return(
    prev_df_func: Callable[[float], float],
    curr_df_func: Callable[[float], float],
    tenor: float,
    prev_date: "date",
    curr_date: "date",
    convention: KrdHedgeConvention,
) -> Tuple[Dict[float, float], float]:
    """Compute the full KRD vector and total return for a synthetic par security at *tenor*.

    The synthetic instrument is fully defined by *convention* (frequency,
    day-count, bump size), decoupled from the corporate bond being hedged.

    Returns:
        (krd_vector, total_return) where krd_vector maps tenor → KRD value.
    """
    synth_dcc = convention.day_count
    frequency = convention.frequency
    tenors = convention.tenors
    bump_size = convention.bump_size

    # Construct synthetic maturity date using proper month arithmetic
    synth_maturity = prev_date + relativedelta(months=int(round(tenor * 12)))

    # Build the actual schedule FIRST, then compute par yield from it (BUG-008)
    schedule_prev = _build_30360_schedule(prev_date, synth_maturity, frequency)
    if not schedule_prev:
        return {k: 0.0 for k in tenors}, 0.0

    del_t_prev = np.array([
        _yearfrac(prev_date, d, synth_dcc) for d in schedule_prev
    ])

    # Par coupon computed on the EXACT schedule (BUG-008 fix)
    par_coupon = _compute_par_yield(prev_df_func, del_t_prev, frequency)

    # Full KRD vector of the synthetic (BUG-012 fix)
    synth_curve_set = build_zero_krd_curve_set(prev_df_func, tenors, bump_size)
    synth_krd_vec = compute_bond_krd(
        synth_curve_set, par_coupon, synth_maturity, prev_date,
        frequency, synth_dcc,
        _schedule_yearfracs=del_t_prev,
    )

    # Total return of the synthetic over the period
    p0 = _price_riskless_bond(prev_df_func, par_coupon, del_t_prev, frequency)

    if p0 <= 0.0:
        return synth_krd_vec, 0.0

    coupon_payment = 100.0 * par_coupon / frequency

    # Split: coupons paid in (prev_date, curr_date], remaining schedule
    coupons_paid = 0.0
    principal_paid = 0.0
    remaining_schedule = []
    for d in schedule_prev:
        if d <= curr_date:
            coupons_paid += coupon_payment
            if d == synth_maturity:
                principal_paid = 100.0
        else:
            remaining_schedule.append(d)

    # Price remaining cashflows on curr_df_func
    p1 = 0.0
    if remaining_schedule:
        del_t_curr = np.array([
            _yearfrac(curr_date, d, synth_dcc) for d in remaining_schedule
        ])
        p1 = _price_riskless_bond(curr_df_func, par_coupon, del_t_curr, frequency)

    r_synth = (p1 + coupons_paid + principal_paid) / p0 - 1.0
    return synth_krd_vec, float(r_synth)


class KrdSynthCache(NamedTuple):
    """Pre-computed synthetic data for a rebalance date (IMP-004).

    Hoists the A⁻¹ matrix, synthetic return vector, and cash return out of
    the per-bond loop so that each bond only requires:
    ``w = A_inv @ b; hedge = w · r_synth + (1 - Σw) · r_cash``.
    """
    A_inv: Optional[np.ndarray]   # (n, n) inverse of KRD matrix; None if singular
    r_synth: np.ndarray           # (n,) synthetic returns
    r_cash: float                 # cash return over the holding period
    tenors: List[float]           # tenor list (for b-vector assembly)
    convention: KrdHedgeConvention  # hedge convention used to build this cache


def calc_krd_matched_realized_return(
    prev_df_func: Callable[[float], float],
    coupon_rate: float,
    maturity_date: "date",
    prev_date: "date",
    frequency: int,
    day_count_convention: str,
    hedge_cache: KrdSynthCache,
) -> float:
    """KRD-matched risk-free realized return (Option D).

    Per-bond work:
    1. Compute the bond's KRD vector b (one ``compute_bond_krd`` call)
       using the bond's own *frequency* and *day_count_convention*.
    2. ``w = A⁻¹ b`` (one matrix-vector multiply).
    3. ``g = w · r_synth + (1 − Σw) · r_cash``.

    The hedge basket (A matrix, synthetic returns, cash return) is
    defined by ``hedge_cache.convention`` and is independent of the bond.

    Args:
        prev_df_func: Discount function at previous date.
        coupon_rate: Annual coupon rate (decimal).
        maturity_date: Bond maturity date.
        prev_date: Previous rebalance date.
        frequency: Bond's coupon frequency per year.
        day_count_convention: Bond's day-count convention.
        hedge_cache: Pre-computed :class:`KrdSynthCache` from
            :func:`build_krd_synth_cache`.

    Returns:
        KRD-matched risk-free realized return.
    """
    # Singular A matrix → return 0.0
    if hedge_cache.A_inv is None:
        return 0.0

    conv = hedge_cache.convention

    # Bond KRD vector — uses the bond's own frequency and day-count
    bond_curve_set = build_zero_krd_curve_set(
        prev_df_func, conv.tenors, conv.bump_size,
    )
    bond_krd = compute_bond_krd(
        bond_curve_set, coupon_rate, maturity_date, prev_date,
        frequency, day_count_convention,
    )
    b = np.array([bond_krd[k] for k in hedge_cache.tenors])

    # Hedge weights via pre-computed A⁻¹
    w = hedge_cache.A_inv @ b

    # Basket return + cash position
    hedge_return = float(np.dot(w, hedge_cache.r_synth))
    w_cash = 1.0 - float(np.sum(w))
    hedge_return += w_cash * hedge_cache.r_cash

    return float(hedge_return)


def build_krd_synth_cache(
    prev_df_func: Callable[[float], float],
    curr_df_func: Callable[[float], float],
    prev_date: "date",
    curr_date: "date",
    convention: KrdHedgeConvention = USD_SWAP_HEDGE_CONVENTION,
) -> KrdSynthCache:
    """Pre-compute everything rebalance-date-specific for KRD hedging.

    Call once per rebalance period.  Pass the result as ``hedge_cache`` to
    :func:`calc_krd_matched_realized_return` for each bond.  Per-bond work
    reduces to one KRD vector computation + one matrix-vector multiply.

    The hedge basket is fully defined by *convention* and is independent
    of any individual bond's coupon frequency or day-count.

    Returns:
        KrdSynthCache with pre-inverted A matrix, synthetic returns, and
        cash return.
    """
    tenors = convention.tenors
    n = len(tenors)

    # Compute synthetic KRD vectors and returns
    synth_data: Dict[float, Tuple[Dict[float, float], float]] = {}
    for k in tenors:
        synth_data[k] = _compute_synthetic_krd_and_return(
            prev_df_func, curr_df_func, k, prev_date, curr_date,
            convention,
        )

    # Assemble the N×N KRD matrix A and synthetic return vector
    A = np.zeros((n, n))
    r_synth_vec = np.zeros(n)
    for k_idx, tk in enumerate(tenors):
        krd_vec_k, r_k = synth_data[tk]
        r_synth_vec[k_idx] = r_k
        for j, tj in enumerate(tenors):
            A[j, k_idx] = krd_vec_k.get(tj, 0.0)

    # Pre-invert A (shared across all bonds at this rebalance date)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = None

    # Cash return (shared across all bonds)
    holding_period_years = (curr_date - prev_date).days / CURVE_TIME_BASIS
    if holding_period_years > 1e-10:
        d_cash = float(prev_df_func(holding_period_years))
        r_cash = (1.0 / d_cash - 1.0) if d_cash > 0.0 else 0.0
    else:
        r_cash = 0.0

    return KrdSynthCache(
        A_inv=A_inv,
        r_synth=r_synth_vec,
        r_cash=r_cash,
        tenors=list(tenors),
        convention=convention,
    )