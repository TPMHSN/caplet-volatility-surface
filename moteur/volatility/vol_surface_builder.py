"""
Volatility surface builder - converts CapFloorVolData to QuantLib CapFloorTermVolSurface.

Supports two modes:
  - Bilinear: direct interpolation of market grid (default)
  - SABR Normal: strip caplet vols via OptionletStripper1, calibrate Normal
    SABR (beta=0) per caplet expiry, store SABR params for direct
    StrippedOptionlet construction at pricing time.
"""

import logging
import math
import QuantLib as ql
from typing import List, Optional, Tuple, Dict

from market_data.schemas import CapFloorVolData
from curves.helpers import parse_tenor
from .schemas import VolType
from .capfloor_sabr import CapFloorSabrCalibrator

logger = logging.getLogger(__name__)


class VolSurfaceBuilder:
    """
    Builds a QuantLib CapFloorTermVolSurface from raw market data.

    Supports both normal (Bachelier) and lognormal (Black) volatilities.
    Two modes: bilinear (direct grid) and SABR Normal (calibrated smile).
    """

    def build(
        self,
        data: CapFloorVolData,
        vol_type: VolType = VolType.NORMAL,
        smile_method: str = "bilinear",
        forward_rates: Optional[List[float]] = None,
        ibor_index: Optional[ql.IborIndex] = None,
        discount_curve: Optional[ql.YieldTermStructureHandle] = None,
    ) -> Tuple[
        ql.CapFloorTermVolSurface, VolType, Optional[List[Dict[str, float]]], Optional[Dict]
    ]:
        """
        Build a CapFloorTermVolSurface from market data.

        Args:
            data: Raw cap/floor vol data (expiries, strikes, vol matrix)
            vol_type: Volatility quoting convention
            smile_method: "bilinear" or "sabr_normal"
            forward_rates: Forward rates per expiry (required for SABR mode)
            ibor_index: IborIndex linked to forward curve (required for SABR mode)
            discount_curve: Discount curve handle (required for SABR mode)

        Returns:
            Tuple of (QuantLib surface, vol_type, sabr_params or None, caplet_metadata or None)
        """
        if smile_method == "sabr_normal":
            return self._build_sabr_surface(
                data,
                vol_type,
                forward_rates,
                ibor_index,
                discount_curve,
            )
        surface, vt, _ = self._build_bilinear_surface(data, vol_type)
        return surface, vt, None, None

    def _build_bilinear_surface(
        self,
        data: CapFloorVolData,
        vol_type: VolType,
    ) -> Tuple[ql.CapFloorTermVolSurface, VolType, None]:
        """Build surface with direct bilinear interpolation (original behavior)."""
        periods = [parse_tenor(t) for t in data.expiries]
        strikes = data.strikes
        n_expiries = len(periods)
        n_strikes = len(strikes)

        matrix = ql.Matrix(n_expiries, n_strikes)
        for i in range(n_expiries):
            for j in range(n_strikes):
                matrix[i][j] = data.volatilities[i][j]

        calendar = ql.TARGET()
        bdc = ql.ModifiedFollowing
        dc = ql.Actual365Fixed()

        # Explicit reference date (not 0 settlement days): the stored surface stays pinned
        # at the market date for display and SABR calibration. Forward-date repricing does
        # not strip it as is: CapFloorVolHelpersMixin._build_vol_handle re-anchors the quoted
        # tenors to the evaluation date first (sticky tenor).
        ref_date = ql.Settings.instance().evaluationDate
        surface = ql.CapFloorTermVolSurface(
            ref_date,
            calendar,
            bdc,
            periods,
            strikes,
            matrix,
            dc,
        )
        surface.enableExtrapolation()

        return surface, vol_type, None

    # ------------------------------------------------------------------
    # SABR Normal surface building
    # ------------------------------------------------------------------

    def _build_sabr_surface(
        self,
        data: CapFloorVolData,
        vol_type: VolType,
        forward_rates: Optional[List[float]] = None,
        ibor_index: Optional[ql.IborIndex] = None,
        discount_curve: Optional[ql.YieldTermStructureHandle] = None,
    ) -> Tuple[ql.CapFloorTermVolSurface, VolType, List[Dict[str, float]], Optional[Dict]]:
        """
        Build surface via SABR Normal calibration on stripped caplet vols.

        Flow:
          1. Build temporary bilinear CapFloorTermVolSurface
          2. Strip caplet vols via OptionletStripper1 (+ ATM correction)
          3. Calibrate SABR per caplet expiry on stripped vols
          4. Return bilinear surface + SABR params + caplet metadata
        """
        if data.atm_vols is None or not any(v > 0 for v in data.atm_vols):
            logger.warning("No ATM vols for SABR calibration, falling back to bilinear")
            s, vt, _ = self._build_bilinear_surface(data, vol_type)
            return s, vt, [], None

        # If ibor_index or discount_curve not provided, fall back to old behavior
        # (calibrate on flat cap vols — less accurate but still functional)
        if ibor_index is None or discount_curve is None:
            logger.warning(
                "No ibor_index/discount_curve for SABR stripping, "
                "falling back to flat-cap-vol calibration (less accurate)"
            )
            return self._build_sabr_surface_legacy(data, vol_type, forward_rates)

        # --- Step 1: Build temporary bilinear surface ---
        temp_surface, _, _ = self._build_bilinear_surface(data, vol_type)

        # --- Step 2: Strip caplet vols ---
        caplet_data = self._strip_caplet_vols(
            temp_surface,
            ibor_index,
            discount_curve,
            vol_type,
            data,
        )
        if not caplet_data:
            logger.warning("Caplet stripping failed, falling back to flat-cap-vol calibration")
            return self._build_sabr_surface_legacy(data, vol_type, forward_rates)

        # --- Step 3: Calibrate SABR on caplet vols ---
        calibrator = CapFloorSabrCalibrator()

        expiry_labels = [c["expiry_label"] for c in caplet_data]
        caplet_strikes = [c["strikes"] for c in caplet_data]
        caplet_vols = [c["vols"] for c in caplet_data]
        caplet_atm_vols = [c["atm_vol"] for c in caplet_data]
        caplet_fwd_rates = [c["forward_rate"] for c in caplet_data]
        caplet_times = [c["T"] for c in caplet_data]
        ref_date = ql.Settings.instance().evaluationDate
        dc = ql.Actual365Fixed()
        calendar = ql.TARGET()
        first_market_date = calendar.advance(ref_date, parse_tenor(data.expiries[0]))
        first_market_time = dc.yearFraction(ref_date, first_market_date)

        sabr_params = calibrator.calibrate_surface(
            expiries=expiry_labels,
            strikes_per_expiry=caplet_strikes,
            volatilities=caplet_vols,
            atm_vols=caplet_atm_vols,
            forward_rates=caplet_fwd_rates,
            expiry_times=caplet_times,
            first_market_time=first_market_time,
        )

        # --- Step 4: Build caplet metadata for pricer bypass ---
        # Use the common strike grid from stripping (all caplets share the same grid)
        common_strikes = caplet_data[0]["strikes"] if caplet_data else data.strikes
        caplet_dates = [c["fixing_date"] for c in caplet_data]

        caplet_metadata = {
            "caplet_dates": caplet_dates,
            "caplet_strikes": common_strikes,
            "forward_rates": caplet_fwd_rates,
            "expiry_times": caplet_times,
        }

        logger.info(
            "SABR calibration on %d caplet expiries, avg RMSE = %.2f bp",
            len(sabr_params),
            sum(p["calibration_error"] for p in sabr_params) / max(len(sabr_params), 1) * 1e4,
        )

        # Return the bilinear surface (for display/fallback), SABR params, and metadata
        return temp_surface, vol_type, sabr_params, caplet_metadata

    def _strip_caplet_vols(
        self,
        temp_surface: ql.CapFloorTermVolSurface,
        ibor_index: ql.IborIndex,
        discount_curve: ql.YieldTermStructureHandle,
        vol_type: VolType,
        data: CapFloorVolData,
    ) -> List[Dict]:
        """
        Strip caplet vols from flat cap surface via OptionletStripper1 + ATM correction.

        Returns list of dicts per caplet expiry:
            {expiry_label, fixing_date, T, forward_rate, strikes, vols, atm_vol}
        """
        is_normal = vol_type == VolType.NORMAL
        ref_date = ql.Settings.instance().evaluationDate
        dc = ql.Actual365Fixed()
        calendar = ql.TARGET()

        try:
            if is_normal:
                stripper = ql.OptionletStripper1(
                    temp_surface,
                    ibor_index,
                    ql.nullDouble(),
                    1.0e-6,
                    100,
                    discount_curve,
                    ql.Normal,
                    0.0,
                )
            else:
                stripper = ql.OptionletStripper1(
                    temp_surface,
                    ibor_index,
                    ql.nullDouble(),
                    1.0e-6,
                    100,
                    discount_curve,
                    ql.ShiftedBlackVol,
                    0.0,
                )
        except Exception as e:
            logger.error("OptionletStripper1 failed during SABR build: %s", e)
            return []

        n_opt = stripper.optionletMaturities()
        fixing_dates = stripper.optionletFixingDates()
        atm_rates = stripper.atmOptionletRates()

        # --- ATM correction (same logic as pricer) ---
        has_atm = data.atm_vols is not None and any(v > 0 for v in data.atm_vols)

        # Extract raw stripped data
        all_stripped_strikes = []
        all_stripped_vols = []
        optionlet_times = []

        for i in range(n_opt):
            all_stripped_strikes.append(list(stripper.optionletStrikes(i)))
            all_stripped_vols.append(list(stripper.optionletVolatilities(i)))
            T = dc.yearFraction(ref_date, fixing_dates[i])
            optionlet_times.append(T)

        if has_atm:
            all_stripped_strikes, all_stripped_vols = self._apply_atm_correction_to_stripped(
                stripper,
                data,
                ibor_index,
                discount_curve,
                is_normal,
                all_stripped_strikes,
                all_stripped_vols,
                optionlet_times,
            )

        # --- Build caplet data list ---
        # Keep all optionlets for SABR metadata/pricing, including sub-first-expiry.
        first_expiry_date = calendar.advance(ref_date, parse_tenor(data.expiries[0]))
        T_first = dc.yearFraction(ref_date, first_expiry_date)

        caplet_data = []
        for i in range(n_opt):
            T = optionlet_times[i]

            strikes_i = all_stripped_strikes[i]
            vols_i = all_stripped_vols[i]
            fwd_i = atm_rates[i]

            # Compute ATM vol by interpolation
            atm_vol_i = self._interpolate_vol_at_strike(strikes_i, vols_i, fwd_i)

            # Map to tenor label
            months = round(T * 12)
            if months <= 0:
                months = 1
            if months % 12 == 0:
                label = f"{months // 12}Y"
            else:
                label = f"{months}M"

            caplet_data.append(
                {
                    "expiry_label": label,
                    "fixing_date": fixing_dates[i],
                    "T": T,
                    "forward_rate": fwd_i,
                    "strikes": strikes_i,
                    "vols": vols_i,
                    "atm_vol": atm_vol_i,
                    "is_pre_first": (T < T_first - 0.01),
                }
            )

        pre_first_count = sum(1 for c in caplet_data if c["is_pre_first"])
        logger.info(
            "Stripped %d caplet expiries from %d raw optionlets (pre-first=%d, first T=%.2f)",
            len(caplet_data),
            n_opt,
            pre_first_count,
            T_first,
        )
        return caplet_data

    def _apply_atm_correction_to_stripped(
        self,
        stripper: ql.OptionletStripper1,
        data: CapFloorVolData,
        ibor_index: ql.IborIndex,
        discount_curve: ql.YieldTermStructureHandle,
        is_normal: bool,
        all_stripped_strikes: List[List[float]],
        all_stripped_vols: List[List[float]],
        optionlet_times: List[float],
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Apply ATM correction with QuantLib OptionletStripper2 semantics.

        Each ATM cap quote adds one corrected strike to every optionlet covered
        by that cap. The strike grids therefore remain local to each optionlet.
        """
        ref_date = ql.Settings.instance().evaluationDate
        dc = ql.Actual365Fixed()
        forwarding_curve = ibor_index.forwardingTermStructure().currentLink()
        atm_anchors: List[Tuple[float, float, float]] = []

        def stripped_vol(time_to_expiry: float, strike: float) -> float:
            vols_by_time = [
                self._interpolate_vol_at_strike(strikes, vols, strike)
                for strikes, vols in zip(all_stripped_strikes, all_stripped_vols)
            ]
            if time_to_expiry <= optionlet_times[0]:
                return vols_by_time[0]
            if time_to_expiry >= optionlet_times[-1]:
                return vols_by_time[-1]
            for i in range(len(optionlet_times) - 1):
                if optionlet_times[i] <= time_to_expiry <= optionlet_times[i + 1]:
                    weight = (time_to_expiry - optionlet_times[i]) / (
                        optionlet_times[i + 1] - optionlet_times[i]
                    )
                    return vols_by_time[i] + weight * (vols_by_time[i + 1] - vols_by_time[i])
            return vols_by_time[-1]

        for expiry_str, atm_vol_flat in zip(data.expiries, data.atm_vols or []):
            if atm_vol_flat <= 0:
                continue

            expiry = parse_tenor(expiry_str)
            seed_cap = ql.MakeCapFloor(ql.CapFloor.Cap, expiry, ibor_index, strike=0.0)
            atm_cap_strike = seed_cap.atmRate(forwarding_curve)
            cap = ql.MakeCapFloor(
                ql.CapFloor.Cap,
                expiry,
                ibor_index,
                strike=atm_cap_strike,
            )
            flat_quote = ql.QuoteHandle(ql.SimpleQuote(atm_vol_flat))
            if is_normal:
                flat_engine = ql.BachelierCapFloorEngine(discount_curve, flat_quote)
            else:
                flat_engine = ql.BlackCapFloorEngine(discount_curve, flat_quote, dc, 0.0)
            cap.setPricingEngine(flat_engine)

            caplet_data = []
            caplet_times = []
            for cashflow in cap.floatingLeg():
                coupon = ql.as_floating_rate_coupon(cashflow)
                if coupon is None:
                    continue
                T_i = dc.yearFraction(ref_date, coupon.fixingDate())
                caplet_times.append(T_i)
                caplet_data.append(
                    (
                        coupon.indexFixing(),
                        atm_cap_strike,
                        stripped_vol(T_i, atm_cap_strike),
                        T_i,
                        coupon.accrualPeriod(),
                        discount_curve.discount(cashflow.date()),
                    )
                )

            if not caplet_data or not caplet_times:
                continue

            spread_j = self._solve_atm_spread(
                caplet_data,
                cap.NPV(),
                is_normal,
            )
            atm_anchors.append((atm_cap_strike, spread_j, max(caplet_times)))

        corrected_strikes = []
        corrected_vols = []
        for i in range(stripper.optionletMaturities()):
            T_i = optionlet_times[i]
            local_strikes = list(all_stripped_strikes[i])
            local_vols = list(all_stripped_vols[i])
            for atm_cap_strike, spread, T_cap in atm_anchors:
                if T_i > T_cap + 0.01:
                    continue
                base_vol = self._interpolate_vol_at_strike(
                    all_stripped_strikes[i],
                    all_stripped_vols[i],
                    atm_cap_strike,
                )
                corrected_vol = max(base_vol + spread, 1.0e-8)
                insert_at = next(
                    (
                        idx
                        for idx, strike in enumerate(local_strikes)
                        if strike >= atm_cap_strike - 1.0e-12
                    ),
                    len(local_strikes),
                )
                if (
                    insert_at < len(local_strikes)
                    and abs(local_strikes[insert_at] - atm_cap_strike) <= 1.0e-12
                ):
                    local_vols[insert_at] = corrected_vol
                else:
                    local_strikes.insert(insert_at, atm_cap_strike)
                    local_vols.insert(insert_at, corrected_vol)

            corrected_strikes.append(local_strikes)
            corrected_vols.append(local_vols)

        return corrected_strikes, corrected_vols

    @staticmethod
    def _price_caplet(
        forward: float,
        strike: float,
        vol: float,
        T: float,
        yf: float,
        df: float,
        is_normal: bool,
    ) -> float:
        """Price a single caplet for ATM correction."""
        if vol < 1e-12 or T <= 0:
            return max(forward - strike, 0.0) * yf * df
        stddev = vol * math.sqrt(T)
        if is_normal:
            return ql.bachelierBlackFormula(ql.Option.Call, strike, forward, stddev) * yf * df
        return ql.blackFormula(ql.Option.Call, strike, forward, stddev) * yf * df

    @staticmethod
    def _solve_atm_spread(
        caplet_data: list,
        target_price: float,
        is_normal: bool,
    ) -> float:
        """Bisection solver for ATM spread."""

        def _reprice(spread: float) -> float:
            total = 0.0
            for fwd, K, vol_i, T_i, yf_i, df_i in caplet_data:
                v = max(vol_i + spread, 1e-12)
                stddev = v * math.sqrt(T_i) if T_i > 0 else 0.0
                if is_normal:
                    total += ql.bachelierBlackFormula(ql.Option.Call, K, fwd, stddev) * yf_i * df_i
                else:
                    total += ql.blackFormula(ql.Option.Call, K, fwd, stddev) * yf_i * df_i
            return total

        lo, hi = -0.10, 0.10
        tol = max(abs(target_price) * 1e-8, 1e-10)

        f_lo = _reprice(lo) - target_price
        f_hi = _reprice(hi) - target_price

        if f_lo * f_hi > 0:
            return lo if abs(f_lo) < abs(f_hi) else hi

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = _reprice(mid) - target_price
            if abs(f_mid) < tol or (hi - lo) < 1e-12:
                return mid
            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _interpolate_vol_at_strike(strikes: list, vols: list, target: float) -> float:
        """Linear interpolation of vol at a target strike."""
        if not strikes or not vols:
            return 0.0
        if target <= strikes[0]:
            return vols[0]
        if target >= strikes[-1]:
            return vols[-1]
        for j in range(len(strikes) - 1):
            if strikes[j] <= target <= strikes[j + 1]:
                t = (target - strikes[j]) / (strikes[j + 1] - strikes[j])
                return vols[j] + t * (vols[j + 1] - vols[j])
        return vols[-1]

    # ------------------------------------------------------------------
    # Legacy SABR (calibrate on flat cap vols — less accurate fallback)
    # ------------------------------------------------------------------

    def _build_sabr_surface_legacy(
        self,
        data: CapFloorVolData,
        vol_type: VolType,
        forward_rates: Optional[List[float]] = None,
    ) -> Tuple[ql.CapFloorTermVolSurface, VolType, List[Dict[str, float]], None]:
        """
        Legacy SABR: calibrate on flat cap vols (when curves not available).

        Less accurate but functional. Returns 4-tuple for compatibility.
        """
        ref_date = ql.Settings.instance().evaluationDate
        dc = ql.Actual365Fixed()
        calendar = ql.TARGET()
        expiry_times = []
        for exp_str in data.expiries:
            period = parse_tenor(exp_str)
            exp_date = calendar.advance(ref_date, period)
            T = dc.yearFraction(ref_date, exp_date)
            expiry_times.append(max(T, 0.01))

        if forward_rates is None:
            mid_strike = data.strikes[len(data.strikes) // 2]
            forward_rates = [mid_strike] * len(data.expiries)

        calibrator = CapFloorSabrCalibrator()
        sabr_params = calibrator.calibrate_surface(
            expiries=data.expiries,
            strikes_per_expiry=data.strikes,
            volatilities=data.volatilities,
            atm_vols=data.atm_vols,
            forward_rates=forward_rates,
            expiry_times=expiry_times,
        )

        fine_strikes, fine_vols = calibrator.generate_fine_grid_surface(
            sabr_params_list=sabr_params,
            forward_rates=forward_rates,
            expiry_times=expiry_times,
            original_strikes=data.strikes,
            num_strikes=50,
        )

        periods = [parse_tenor(t) for t in data.expiries]
        n_expiries = len(periods)
        n_strikes = len(fine_strikes)
        matrix = ql.Matrix(n_expiries, n_strikes)
        for i in range(n_expiries):
            for j in range(n_strikes):
                matrix[i][j] = fine_vols[i][j]

        # Pinned at the market date like the bilinear carrier; pricing re-anchors the
        # quoted tenors to the evaluation date before any stripping (sticky tenor).
        ref_date = ql.Settings.instance().evaluationDate
        surface = ql.CapFloorTermVolSurface(
            ref_date,
            calendar,
            ql.ModifiedFollowing,
            periods,
            fine_strikes,
            matrix,
            dc,
        )
        surface.enableExtrapolation()

        return surface, vol_type, sabr_params, None

    # ------------------------------------------------------------------
    # ATM curve builder
    # ------------------------------------------------------------------

    def build_atm_curve(
        self,
        data: CapFloorVolData,
    ) -> Optional[ql.CapFloorTermVolCurve]:
        """
        Build a CapFloorTermVolCurve from ATM vols.

        Returns None if no ATM vols available.
        """
        if data.atm_vols is None or not any(v > 0 for v in data.atm_vols):
            return None

        periods = [parse_tenor(t) for t in data.expiries]
        atm_vals = list(data.atm_vols)

        calendar = ql.TARGET()
        bdc = ql.ModifiedFollowing
        dc = ql.Actual365Fixed()

        curve = ql.CapFloorTermVolCurve(
            0,  # settlement days
            calendar,
            bdc,
            periods,
            atm_vals,
            dc,
        )
        curve.enableExtrapolation()
        return curve
