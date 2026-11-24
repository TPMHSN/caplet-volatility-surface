"""
SABR Normal (beta=0) calibration for cap/floor caplet smiles.

Key implementation choices:
  - Uses shifted SABR in normal mode to support negative strikes with QL 1.41.
  - Shift is constant per expiry: shift = max(1e-4, -min(K_min, F) + 1e-4)
  - Calibrates only (rho, nu); alpha is solved to anchor ATM exactly.

Important: this calibrator expects stripped optionlet vols, not flat cap vols.
"""

import logging
import math
import QuantLib as ql
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Union, Optional

logger = logging.getLogger(__name__)


class CapFloorSabrCalibrator:
    """
    Calibrate Normal SABR (beta=0) per expiry for cap/floor vol smile.

    Uses scipy L-BFGS-B with multiple starting points for robustness.
    Typical calibration error should be < 1 bp RMSE.
    """

    BETA = 0.0
    NUMERIC_PENALTY = 1.0e6
    SHIFT_MAX = 0.08

    @staticmethod
    def _compute_shift(forward: float, strikes: List[float]) -> float:
        """Compute a constant shift that makes F and all K strictly positive."""
        floor_level = min([forward] + list(strikes)) if strikes else forward
        return max(1.0e-4, -floor_level + 1.0e-4)

    def _sabr_normal_vol(
        self,
        strike: float,
        forward: float,
        T: float,
        alpha: float,
        rho: float,
        nu: float,
        shift: float,
    ) -> float:
        """
        Compute shifted-normal SABR vol for one strike.

        Uses QuantLib shiftedSabrVolatility(..., ql.Normal) to support K <= 0.
        """
        vol = ql.shiftedSabrVolatility(
            strike,
            forward,
            T,
            alpha,
            self.BETA,
            nu,
            rho,
            shift,
            ql.Normal,
        )
        if not math.isfinite(vol) or vol <= 0.0:
            raise ValueError("Invalid shifted-normal SABR volatility")
        return vol

    def _solve_alpha_for_atm(
        self,
        atm_vol: float,
        forward_rate: float,
        T: float,
        rho: float,
        nu: float,
        shift: float,
    ) -> Optional[float]:
        """
        Solve alpha so that sigma_model(F,F,T) == atm_vol.

        Uses robust bracketing + bisection.
        """
        if T <= 0.0 or atm_vol <= 0.0:
            return None

        def f(alpha: float) -> float:
            try:
                model_atm = self._sabr_normal_vol(
                    forward_rate,
                    forward_rate,
                    T,
                    alpha,
                    rho,
                    nu,
                    shift,
                )
                return model_atm - atm_vol
            except Exception:
                return float("nan")

        lo = 1.0e-8
        hi = max(0.1, atm_vol * 30.0)
        f_lo = f(lo)
        f_hi = f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            return None

        expand_steps = 0
        while f_lo * f_hi > 0.0 and expand_steps < 40:
            hi *= 2.0
            f_hi = f(hi)
            if not math.isfinite(f_hi):
                return None
            expand_steps += 1

        if f_lo * f_hi > 0.0:
            return None

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                return None
            if abs(f_mid) < 1.0e-12 or (hi - lo) < 1.0e-12:
                return mid
            if f_lo * f_mid <= 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------
    # Per-expiry calibration
    # ------------------------------------------------------------------

    def calibrate_expiry(
        self,
        atm_vol: float,
        strikes: List[float],
        vols: List[float],
        forward_rate: float,
        T: float,
    ) -> Dict[str, float]:
        """
        Calibrate SABR Normal for one expiry.

        Calibrates (rho, nu) via multi-start L-BFGS-B and solves alpha such that
        the model matches ATM exactly.

        Returns diagnostics:
          alpha, beta, rho, nu, shift, calibration_error (RMSE),
          atm_input, atm_model, atm_gap_bp, objective_value.
        """
        shift = self._compute_shift(forward_rate, strikes)
        default = {
            "alpha": max(atm_vol, 1.0e-8),
            "beta": self.BETA,
            "rho": 0.0,
            "nu": 0.3,
            "shift": shift,
            "calibration_error": 0.0,
            "atm_input": atm_vol,
            "atm_model": atm_vol,
            "atm_gap_bp": 0.0,
            "objective_value": 0.0,
        }
        if T <= 0 or atm_vol <= 0:
            return default

        valid_pairs = [(k, v) for k, v in zip(strikes, vols) if (v > 0.0 and math.isfinite(v))]
        if len(valid_pairs) < 2:
            return default

        valid_strikes = [k for k, _ in valid_pairs]
        valid_vols = [v for _, v in valid_pairs]
        shift = self._compute_shift(forward_rate, valid_strikes)
        n_pts = len(valid_strikes)

        # ----- objective : SSE on smile points, with ATM anchored through alpha -----
        def objective(x):
            r, n, s = x
            alpha = self._solve_alpha_for_atm(atm_vol, forward_rate, T, r, n, s)
            if alpha is None:
                return self.NUMERIC_PENALTY

            sse = 0.0
            for k, v_mkt in zip(valid_strikes, valid_vols):
                try:
                    v_model = self._sabr_normal_vol(
                        k,
                        forward_rate,
                        T,
                        alpha,
                        r,
                        n,
                        s,
                    )
                except Exception:
                    return self.NUMERIC_PENALTY
                diff = v_model - v_mkt
                sse += diff * diff
            if not math.isfinite(sse):
                return self.NUMERIC_PENALTY
            return sse

        # ----- bounds for (rho, nu, shift) -----
        shift_floor = shift
        shift_upper = max(shift_floor + 1.0e-6, self.SHIFT_MAX)
        bounds = [
            (-0.999, 0.999),
            (0.001, 5.0),
            (shift_floor, shift_upper),
        ]

        # ----- multi-start over (rho, nu, shift) -----
        best_fun = float("inf")
        best_x = [0.0, 0.3, shift_floor]
        rho_starts = [-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8]
        nu_starts = [0.10, 0.30, 0.60, 1.00, 1.50]
        shift_starts = [
            shift_floor,
            min(shift_floor + 0.005, shift_upper),
            min(shift_floor + 0.010, shift_upper),
            min(shift_floor + 0.020, shift_upper),
        ]
        # remove duplicates while preserving order
        shift_starts = list(dict.fromkeys(shift_starts))

        for r0 in rho_starts:
            for n0 in nu_starts:
                for s0 in shift_starts:
                    try:
                        res = minimize(
                            objective,
                            [r0, n0, s0],
                            method="L-BFGS-B",
                            bounds=bounds,
                            options={"maxiter": 500, "ftol": 1e-15},
                        )
                    except Exception:
                        continue
                    if math.isfinite(res.fun) and res.fun < best_fun:
                        best_fun = float(res.fun)
                        best_x = list(res.x)

        if not math.isfinite(best_fun):
            out = dict(default)
            out["shift"] = shift
            out["objective_value"] = self.NUMERIC_PENALTY
            return out

        rho, nu, shift = best_x
        alpha = self._solve_alpha_for_atm(atm_vol, forward_rate, T, rho, nu, shift)
        if alpha is None:
            out = dict(default)
            out["shift"] = shift
            out["objective_value"] = self.NUMERIC_PENALTY
            return out

        objective_value = objective([rho, nu, shift])
        if not math.isfinite(objective_value):
            objective_value = self.NUMERIC_PENALTY
        cal_error = math.sqrt(max(objective_value, 0.0) / n_pts) if n_pts > 0 else 0.0

        try:
            atm_model = self._sabr_normal_vol(
                forward_rate,
                forward_rate,
                T,
                alpha,
                rho,
                nu,
                shift,
            )
        except Exception:
            atm_model = atm_vol
        atm_gap_bp = (atm_model - atm_vol) * 1.0e4

        return {
            "alpha": alpha,
            "beta": self.BETA,
            "rho": rho,
            "nu": nu,
            "shift": shift,
            "calibration_error": cal_error,
            "atm_input": atm_vol,
            "atm_model": atm_model,
            "atm_gap_bp": atm_gap_bp,
            "objective_value": objective_value,
        }

    # ------------------------------------------------------------------
    # Full surface calibration
    # ------------------------------------------------------------------

    def calibrate_surface(
        self,
        expiries: List[str],
        strikes_per_expiry: Union[List[float], List[List[float]]],
        volatilities: List[List[float]],
        atm_vols: List[float],
        forward_rates: List[float],
        expiry_times: List[float],
        first_market_time: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """
        Calibrate SABR Normal for all expiries.

        strikes_per_expiry: either a single flat list (same strikes for all
            expiries) or a list-of-lists (one strike grid per expiry, as
            returned by OptionletStripper1).

        If first_market_time is provided, expiries strictly before this time are
        not independently calibrated; they reuse (rho, nu) from the first
        calibratable expiry and re-anchor alpha to their ATM vol.

        Returns: List of SABR param dicts per expiry.
        """
        # Normalize strikes to per-expiry lists
        if strikes_per_expiry and not isinstance(strikes_per_expiry[0], (list, tuple)):
            # Flat list — broadcast to all expiries
            strikes_list = [strikes_per_expiry] * len(expiries)
        else:
            strikes_list = strikes_per_expiry

        sabr_params: List[Dict[str, float]] = []

        calibration_start_idx = 0
        if first_market_time is not None:
            for i, t in enumerate(expiry_times):
                if t >= first_market_time - 1.0e-10:
                    calibration_start_idx = i
                    break

        first_calibrated_params = None
        for i in range(calibration_start_idx, len(expiries)):
            params = self.calibrate_expiry(
                atm_vols[i],
                strikes_list[i],
                volatilities[i],
                forward_rates[i],
                expiry_times[i],
            )
            if math.isfinite(params.get("objective_value", float("inf"))):
                first_calibrated_params = params
                break

        if first_calibrated_params is None and len(expiries) > 0:
            first_calibrated_params = self.calibrate_expiry(
                atm_vols[0],
                strikes_list[0],
                volatilities[0],
                forward_rates[0],
                expiry_times[0],
            )

        for i, expiry in enumerate(expiries):
            T = expiry_times[i]
            fwd = forward_rates[i]
            atm_v = atm_vols[i]
            row_vols = volatilities[i]

            if i < calibration_start_idx and first_calibrated_params is not None:
                row_strikes = list(strikes_list[i]) if i < len(strikes_list) else []
                shift = first_calibrated_params.get(
                    "shift",
                    self._compute_shift(fwd, row_strikes),
                )
                rho = first_calibrated_params["rho"]
                nu = first_calibrated_params["nu"]
                alpha = self._solve_alpha_for_atm(atm_v, fwd, T, rho, nu, shift)
                if alpha is None:
                    alpha = first_calibrated_params["alpha"]

                valid_pairs = [(k, v) for k, v in zip(row_strikes, row_vols) if v > 0.0]
                if valid_pairs:
                    sse = 0.0
                    for k, v_mkt in valid_pairs:
                        try:
                            v_model = self._sabr_normal_vol(
                                k,
                                fwd,
                                T,
                                alpha,
                                rho,
                                nu,
                                shift,
                            )
                        except Exception:
                            sse = self.NUMERIC_PENALTY
                            break
                        diff = v_model - v_mkt
                        sse += diff * diff
                    rmse = math.sqrt(max(sse, 0.0) / len(valid_pairs))
                else:
                    sse = 0.0
                    rmse = 0.0

                try:
                    atm_model = self._sabr_normal_vol(
                        fwd,
                        fwd,
                        T,
                        alpha,
                        rho,
                        nu,
                        shift,
                    )
                except Exception:
                    atm_model = atm_v

                params = {
                    "alpha": alpha,
                    "beta": self.BETA,
                    "rho": rho,
                    "nu": nu,
                    "shift": shift,
                    "calibration_error": rmse,
                    "atm_input": atm_v,
                    "atm_model": atm_model,
                    "atm_gap_bp": (atm_model - atm_v) * 1.0e4,
                    "objective_value": sse,
                    "frozen": True,
                    "frozen_pre_first": True,
                }
            else:
                params = self.calibrate_expiry(
                    atm_v,
                    strikes_list[i],
                    row_vols,
                    fwd,
                    T,
                )
                params["frozen"] = False
                params["frozen_pre_first"] = False

            params["expiry"] = expiry
            sabr_params.append(params)

        return sabr_params

    # ------------------------------------------------------------------
    # Smile interpolation
    # ------------------------------------------------------------------

    def interpolate_smile(
        self,
        sabr_params: Dict[str, float],
        forward_rate: float,
        T: float,
        strikes: List[float],
    ) -> List[float]:
        """Generate smooth vol smile from calibrated SABR params."""
        alpha = sabr_params["alpha"]
        rho = sabr_params["rho"]
        nu = sabr_params["nu"]
        shift = sabr_params.get("shift")
        if shift is None:
            shift = self._compute_shift(forward_rate, strikes)

        out = []
        for k in strikes:
            try:
                vol = self._sabr_normal_vol(k, forward_rate, T, alpha, rho, nu, shift)
            except Exception:
                vol = max(alpha, 1.0e-8)
            out.append(max(vol, 1.0e-8))
        return out

    # ------------------------------------------------------------------
    # Generate caplet vol surface from SABR params
    # ------------------------------------------------------------------

    def generate_sabr_caplet_surface(
        self,
        sabr_params_list: List[Dict[str, float]],
        forward_rates: List[float],
        expiry_times: List[float],
        strikes: List[float],
    ) -> List[List[float]]:
        """
        Generate caplet vols from SABR params at a given strike grid.

        Used by the pricer to build a StrippedOptionlet directly from
        calibrated SABR parameters (bypassing OptionletStripper1).

        Returns: matrix [n_expiries x n_strikes] of normal caplet vols.
        """
        result = []
        for i, params in enumerate(sabr_params_list):
            T = expiry_times[i]
            fwd = forward_rates[i]
            row = self.interpolate_smile(params, fwd, T, strikes)
            row = [max(v, 1e-6) for v in row]
            result.append(row)
        return result

    # ------------------------------------------------------------------
    # Fine grid surface for OptionletStripper1 (legacy / bilinear fallback)
    # ------------------------------------------------------------------

    def generate_fine_grid_surface(
        self,
        sabr_params_list: List[Dict[str, float]],
        forward_rates: List[float],
        expiry_times: List[float],
        original_strikes: List[float],
        num_strikes: int = 50,
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Generate a fine-grid vol surface from SABR params.

        Grid range is computed from original strikes + forward rates,
        extended ±50 % of total width, floored at -2 % (negative-rate support).

        Returns: (fine_strikes, fine_volatilities_matrix)
        """
        all_points = list(original_strikes) + list(forward_rates)
        data_min = min(all_points)
        data_max = max(all_points)
        width = data_max - data_min
        margin = max(width * 0.5, 0.005)  # at least 50 bp margin
        min_strike = max(data_min - margin, -0.02)
        max_strike = data_max + margin
        step = (max_strike - min_strike) / (num_strikes - 1)
        fine_strikes = [min_strike + i * step for i in range(num_strikes)]

        fine_vols = []
        for i, params in enumerate(sabr_params_list):
            T = expiry_times[i]
            fwd = forward_rates[i]
            row = self.interpolate_smile(params, fwd, T, fine_strikes)
            # Vol floor: prevent near-zero vols that crash OptionletStripper1
            row = [max(v, 1e-6) for v in row]
            fine_vols.append(row)

        return fine_strikes, fine_vols
