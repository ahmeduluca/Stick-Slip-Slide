#mechanics.py
import inspect
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# avoid top-level import of fitting (circular import); import lazily where needed
from .math_utils import (
    summarize_dist, area_from_flat_end_fit,
    _num, phase_to_rad, rms_to_peak, robust_fit_line,
)

def corrected_normal_load(F_raw_N: np.ndarray, z_raw_m: np.ndarray, k_sup: float, b_sup: float) -> np.ndarray:
    return F_raw_N - (k_sup * z_raw_m + b_sup)

def contact_depth_h_m(z_raw_m: np.ndarray, touch_i: int, P_N: np.ndarray, k_frame_z: Optional[float]) -> np.ndarray:
    """
    Displacement increases with indentation, so:
      z_contact = z_raw - P/k_frame_z  (optional)
      h(t) = z_contact(t) - z_contact(touch)
    """
    zc = z_raw_m.copy()
    if k_frame_z is not None:
        zc = zc - (P_N / float(k_frame_z))
    z0 = zc[touch_i]
    return zc - z0

def vertical_stiffness_frame_corrected(Sz_raw_N_per_m: np.ndarray, k_frame_z: Optional[float]) -> np.ndarray:
    """
    1/Sz = 1/Sz_raw - 1/k_frame_z
    """
    Sz = Sz_raw_N_per_m.copy()
    if k_frame_z is not None:
        Sz = 1.0 / np.maximum(1e-30, (1.0 / np.maximum(1e-30, Sz_raw_N_per_m)) - (1.0 / float(k_frame_z)))
    return Sz

def area_pi_h_R(h_m: np.ndarray, R_m) -> np.ndarray:
    """
    Area proxy A = pi * h * R.

    Backward-compatible:
      - If R_m is a scalar.
      - If R_m is an array: vectorized (broadcasts with h_m).
    """
    h = np.maximum(0.0, np.asarray(h_m, float))
    R = np.asarray(R_m, float)  # scalar or array
    return np.pi * h * R


def normal_pressure_Pa(P_N: np.ndarray, A_m2: np.ndarray) -> np.ndarray:
    return P_N / np.maximum(1e-30, A_m2)

def shear_stress_Pa(Ft_N: np.ndarray, A_m2: np.ndarray) -> np.ndarray:
    return Ft_N / np.maximum(1e-30, A_m2)

## Total sliding parameters calculated;
def total_sliding_cyc_dist_speed(
    time_s: np.ndarray,
    amp: np.ndarray,
    freq_Hz: float,
    start_i: int,
    stop_i: int,
) -> Dict:
    """
    Parameters
    ----------
    time_s : array
        Time vector [s].
    amp : array
        Lateral displacement amplitude (same length as time_s).
        If amp_is_rms=True, amp is RMS and will be converted to peak.
        Units should be meters if you want distance in meters and speeds in m/s.
    freq_Hz : float
        Oscillation frequency [Hz].
    start_i : int
        Start index for sliding interval (inclusive). stick->slide point.
    stop_i : int    
        Stop index for sliding interval (inclusive). restick point.
    Returns
    -------
    dict with:
      - totals: dict (overall totals across all hold intervals)
    """
    t = np.asarray(time_s, dtype=float)
    A = np.asarray(amp, dtype=float)

    if not (np.isfinite(freq_Hz) and freq_Hz > 0):
        raise ValueError("freq_Hz must be finite and > 0.")

    if not (np.isfinite(start_i) and np.isfinite(stop_i)):
        raise ValueError("start_i and stop_i must be finite indices.")
    
    total_time = 0.0
    N = 0.0
    D = 0.0

    sl = slice(start_i, stop_i + 1)
    tt = t[sl]
    AA = A[sl]

    # finite mask
    m = np.isfinite(tt) & np.isfinite(AA)
    if m.sum() < 2:
        return {"totals": {
            "ok": 0,
            "total_sliding_time_s": np.nan,
            "total_osc_cycles": np.nan,
            "total_slide_dist_m": np.nan,
            "max_instantaneous_speed_m_per_s": np.nan,
            "mean_instantaneous_speed_m_per_s": np.nan,
            "overall_mean_speed_m_per_s": np.nan,
        }}

    tt = tt[m]
    AA = AA[m]

    total_time = float(tt[-1] - tt[0])

    N = float(freq_Hz * total_time)          # oscillation cycles
    D = float(4.0 * freq_Hz * np.trapezoid(AA, tt))   # total slide distance over sinusoid
    v_max = float(2.0 * np.pi * freq_Hz * max(AA))  # max(|v|) over a sinusoid
    A_mean_time = float(np.trapezoid(AA, tt) / total_time)
    v_mean = float(4.0 * freq_Hz * A_mean_time) # mean(|v|) over a sinusoid

    totals = {
        "ok": 1 if total_time > 0 else 0,
        "total_sliding_time_s": float(total_time),
        "total_osc_cycles": float(N),
        "total_slide_dist_m": float(D),
        "max_instantaneous_speed_m_per_s": float(v_max),
        "mean_instantaneous_speed_m_per_s": float(v_mean),
        "overall_mean_speed_m_per_s": float(D / total_time) if total_time > 0 else np.nan,
    }

    return {"totals": totals}


def compute_lateral_corrected(
    df: pd.DataFrame,
    cfg,
    scale_to_SI: Dict[str, float],
    cal_sl: Optional[slice],
) -> pd.DataFrame:
    out = df.copy()

    # Convert RMS channels to SI
    F2_rms_SI = _num(out, cfg.F2_rms_col) * scale_to_SI[cfg.F2_rms_col]   # N
    X2_rms_SI = _num(out, cfg.X2_rms_col) * scale_to_SI[cfg.X2_rms_col]   # m
    phi = phase_to_rad(_num(out, cfg.PH2_col))

    # RMS -> peak
    F2_pk = rms_to_peak(F2_rms_SI)
    X2_pk = rms_to_peak(X2_rms_SI)

    out["F2_pk_N"] = F2_pk
    out["X2_pk_m"] = X2_pk
    out["phi2_rad"] = phi

    # Fit lateral parallel spring from calibration: F = kx*X + b
    if cal_sl is not None:
        try:
            kx_sup, bx_sup = robust_fit_line(X2_pk[cal_sl], F2_pk[cal_sl])
            if not np.isfinite(kx_sup):
                raise RuntimeError("kx_sup not finite")
        except Exception:
            kx_sup, bx_sup = (np.nan, np.nan)
    else:
        kx_sup, bx_sup = (np.nan, np.nan)

    # Fallbacks if calibration missing/failed
    if (not np.isfinite(kx_sup)) or (not np.isfinite(bx_sup)):
        if cfg.k_sup_x_fallback is not None and np.isfinite(cfg.k_sup_x_fallback):
            kx_sup = float(cfg.k_sup_x_fallback)
            bx_sup = float(cfg.b_sup_x_fallback)
        elif cfg.allow_no_cal:
            # last resort: no spring subtraction
            kx_sup = 0.0
            bx_sup = 0.0
        else:
            raise RuntimeError(
                "Calibration failed and no fallback provided. "
                "Pass --k_sup_x (N/m) or set --allow_no_cal to proceed with k_sup_x=0."
            )
    # Apply spring subtraction
    out["kx_sup_est_N_per_m"] = kx_sup
    out["bx_sup_est_N"] = bx_sup

    out["F2_pk_spring_N"] = kx_sup * out["X2_pk_m"] + bx_sup
    out["F2_pk_corr_N"] = out["F2_pk_N"] - out["F2_pk_spring_N"]

    # Optional frame correction in X for contact displacement amplitude
    if cfg.k_frame_x is not None:
        out["X2_pk_contact_m"] = out["X2_pk_m"] - (out["F2_pk_corr_N"] / float(cfg.k_frame_x))
    else:
        out["X2_pk_contact_m"] = out["X2_pk_m"]

    # Phase is displacement relative to force => K* = (F/x)*exp(-i phi)
    ratio = out["F2_pk_corr_N"].to_numpy() / np.maximum(1e-30, out["X2_pk_contact_m"].to_numpy())
    Kstar = ratio * np.exp(-1j * out["phi2_rad"].to_numpy())

    out["Stiffness_lateral"] = np.real(Kstar)
    out["Damping_lateral"] = np.imag(Kstar)
    out["E_diss_J_per_cycle"] = np.pi * np.abs(out["Damping_lateral"].to_numpy()) * (out["X2_pk_contact_m"].to_numpy() ** 2)

    return out
# ============================================================
# Inserted Contact Mechanics (Hertz/JKR) functions
# ============================================================

def effective_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """Hertz reduced modulus E* (Pa)."""
    inv = (1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2
    return 1.0 / inv if inv > 0 else np.nan


def hertz_fit_radius(h_m: np.ndarray, P_N: np.ndarray, E_star_Pa: float, hardness_Pa: float,
                     plasticity_p0_frac: float = 1.0, min_h_m: float = 5e-9,
                     max_frac_of_Pmax: float = 0.95, min_points: int = 50,
                     n_iter: int = 3) -> dict:
    """Fits Hertz sphere: P = (4/3) E* sqrt(R) h^(3/2)"""
    from .math_utils import safe_nanmax
    h = np.asarray(h_m, float)
    P = np.asarray(P_N, float)
    m = np.isfinite(h) & np.isfinite(P) & (h > min_h_m) & (P > 0) & np.isfinite(E_star_Pa) & (E_star_Pa > 0)
    if m.sum() < min_points:
        return {"ok": 0, "reason": "not enough points", "R_eff_m": np.nan, "C": np.nan, "rmse_N": np.nan}
    Pmax = safe_nanmax(P[m])
    if not np.isfinite(Pmax) or Pmax <= 0:
        return {"ok": 0, "reason": "Pmax invalid", "R_eff_m": np.nan, "C": np.nan, "rmse_N": np.nan}
    m &= (P <= max_frac_of_Pmax * Pmax)
    if m.sum() < min_points:
        return {"ok": 0, "reason": "not enough points after Pmax fraction cut", "R_eff_m": np.nan, "C": np.nan, "rmse_N": np.nan}
    m_fit = m.copy()
    C = np.nan
    R_eff = np.nan
    rmse = np.nan
    for it in range(max(1, int(n_iter))):
        idx = np.where(m_fit)[0]
        if idx.size < min_points:
            break
        x = np.power(h[idx], 1.5)
        y = P[idx]
        denom = float(np.dot(x, x))
        if denom <= 0:
            break
        C = float(np.dot(x, y) / denom)
        R_eff = float(((3.0 * C) / (4.0 * E_star_Pa)) ** 2)
        yhat = C * x
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        if not (np.isfinite(hardness_Pa) and hardness_Pa > 0):
            continue
        a = np.sqrt(np.maximum(1e-30, R_eff * h))
        p0 = (3.0 * P) / (2.0 * np.pi * np.maximum(1e-30, a**2))
        m_new = m_fit & np.isfinite(p0) & (p0 <= plasticity_p0_frac * hardness_Pa)
        if np.array_equal(m_new, m_fit):
            break
        m_fit = m_new
    ok = 1 if (np.isfinite(R_eff) and R_eff > 0 and np.isfinite(C)) else 0
    return {"ok": ok, "E_star_Pa": float(E_star_Pa), "C": float(C), "R_eff_m": float(R_eff), "rmse_N": float(rmse), "n_used": int(np.where(m_fit)[0].size), "mask_used": m_fit}

def hertz_apparent_radius_R_of_h(h_m: np.ndarray, P_N: np.ndarray, E_star_Pa: float) -> np.ndarray:
    """Pointwise apparent radius from rearranged Hertz: R = [ (3P)/(4E* h^(3/2)) ]^2"""
    h = np.asarray(h_m, float)
    P = np.asarray(P_N, float)
    denom = 4.0 * E_star_Pa * np.power(np.maximum(1e-30, h), 1.5)
    sqrtR = (3.0 * P) / denom
    R = np.power(sqrtR, 2.0)
    R[~np.isfinite(R)] = np.nan
    return R

def a_from_stiffness_Sneddon(Sz_N_per_m: np.ndarray, E_star_Pa: float) -> np.ndarray:
    Sz = np.asarray(Sz_N_per_m, float)
    a = np.full_like(Sz, np.nan, dtype=float)
    if np.isfinite(E_star_Pa) and E_star_Pa > 0:
        m = np.isfinite(Sz) & (Sz > 0)
        a[m] = Sz[m] / (2.0 * E_star_Pa)
    return a

def a_from_depth_sphere(h_m: np.ndarray, R_m: float) -> np.ndarray:
    h = np.asarray(h_m, float)
    a = np.full_like(h, np.nan, dtype=float)
    if np.isfinite(R_m) and R_m > 0:
        m = np.isfinite(h) & (h > 0)
        a[m] = np.sqrt(R_m * h[m])
    return a

def w_eff_from_roughness(w_J_per_m2: float, sigma_rms_m: float | None, model: str = "none", delta0_m: float = 0.3e-9) -> float:
    w = float(w_J_per_m2)
    if not (np.isfinite(w) and w >= 0):
        return np.nan
    if model == "none" or sigma_rms_m is None:
        return w
    sig = float(sigma_rms_m)
    if not (np.isfinite(sig) and sig >= 0 and np.isfinite(delta0_m) and delta0_m > 0):
        return w
    if model == "exp":
        return w * float(np.exp(- (sig / delta0_m) ** 2))
    return w

def _hertz_load_from_h(h_m: np.ndarray, R_m: float, E_star_Pa: float) -> np.ndarray:
    h = np.asarray(h_m, float)
    out = np.full_like(h, np.nan, dtype=float)
    m = np.isfinite(h) & (h > 0) & np.isfinite(R_m) & (R_m > 0) & np.isfinite(E_star_Pa) & (E_star_Pa > 0)
    if np.any(m):
        out[m] = (4.0/3.0) * E_star_Pa * np.sqrt(R_m) * (h[m] ** 1.5)
    return out

def _c_from_tabor(mu: float) -> float:
    if not np.isfinite(mu):
        return 0.0
    x = np.log10(max(mu, 1e-12))
    x0, x1 = np.log10(0.1), np.log10(5.0)
    t = (x - x0) / (x1 - x0)
    t = float(np.clip(t, 0.0, 1.0))
    return float(2.0 - 0.5*t)

def _tabor_mu(R_m: float, w_eff: float, E_star_Pa: float, z0_m: float) -> float:
    if not (np.isfinite(R_m) and R_m > 0 and np.isfinite(w_eff) and w_eff >= 0 and np.isfinite(E_star_Pa) and E_star_Pa > 0 and np.isfinite(z0_m) and z0_m > 0):
        return np.nan
    return float(((R_m * (w_eff**2)) / ((E_star_Pa**2) * (z0_m**3))) ** (1.0/3.0))

def _jkr_P_from_a(a_m: np.ndarray, R_m: float, E_star_Pa: float, w_J_per_m2: float) -> np.ndarray:
    a = np.asarray(a_m, dtype=float)
    R = float(R_m); E = float(E_star_Pa); w = float(w_J_per_m2)
    term_el = (4.0/3.0) * E * (a**3) / R
    term_adh = np.sqrt(np.maximum(0.0, 8.0*np.pi*w*E*(a**3)))
    return term_el - term_adh

def _jkr_h_from_a(a_m: np.ndarray, R_m: float, E_star_Pa: float, w_J_per_m2: float) -> np.ndarray:
    a = np.asarray(a_m, dtype=float)
    R = float(R_m); E = float(E_star_Pa); w = float(w_J_per_m2)
    term_geom = (a**2)/R
    term_adh = np.sqrt(np.maximum(0.0, (8.0*np.pi*w*a)/(3.0*E)))
    return term_geom - term_adh

def _jkr_P_from_h(h_m: np.ndarray, R_m: float, E_star_Pa: float, w_J_per_m2: float, n_bisect: int = 60) -> np.ndarray:
    h = np.asarray(h_m, dtype=float)
    out = np.full_like(h, np.nan, dtype=float)
    R = float(R_m); E = float(E_star_Pa); w = float(w_J_per_m2)
    if not (R > 0 and E > 0 and w > 0):
        return out
    for i in range(h.size):
        hi = h[i]
        if not (np.isfinite(hi) and hi >= 0):
            continue
        a_lo = 0.0
        a_hi = np.sqrt(max(R*hi, 0.0)) * 5.0 + 1e-12
        for _ in range(12):
            h_hi = _jkr_h_from_a(np.array([a_hi]), R, E, w)[0]
            if np.isfinite(h_hi) and (h_hi >= hi):
                break
            a_hi *= 2.0
        h_hi = _jkr_h_from_a(np.array([a_hi]), R, E, w)[0]
        if not (np.isfinite(h_hi) and h_hi >= hi):
            continue
        lo, hi_a = a_lo, a_hi
        for _ in range(n_bisect):
            mid = 0.5*(lo + hi_a)
            h_mid = _jkr_h_from_a(np.array([mid]), R, E, w)[0]
            if not np.isfinite(h_mid):
                hi_a = mid
                continue
            if h_mid >= hi:
                hi_a = mid
            else:
                lo = mid
        a_sol = 0.5*(lo + hi_a)
        out[i] = _jkr_P_from_a(np.array([a_sol]), R, E, w)[0]
    return out

def _auto_model_from_mu(mu: float, mu_dmt: float = 0.1, mu_jkr: float = 5.0) -> str:
    if not np.isfinite(mu):
        return "hertz"
    if mu <= mu_dmt:
        return "dmt"
    if mu >= mu_jkr:
        return "jkr"
    return "transition"

def a_from_Sz(Sz_N_per_m, E_star_Pa):
    Sz = np.asarray(Sz_N_per_m, float)
    if not (np.isfinite(E_star_Pa) and E_star_Pa > 0):
        return np.full_like(Sz, np.nan)
    return Sz / (2.0 * E_star_Pa)

def A_from_a(a_m):
    a = np.asarray(a_m, float)
    return np.pi * a*a

def estimate_R_from_a_and_h(a_m, h_m, *, min_h=1e-9):
    a = np.asarray(a_m, float); h = np.asarray(h_m, float)
    m = np.isfinite(a) & np.isfinite(h) & (h > min_h) & (a > 0)
    if np.sum(m) < 5:
        return np.nan
    R = (a[m]**2) / h[m]
    return float(np.nanmedian(R))

def estimate_R_from_a_and_P(a_m, P_N, E_star_Pa, *, min_P=1e-6):
    a = np.asarray(a_m, float); P = np.asarray(P_N, float)
    m = np.isfinite(a) & np.isfinite(P) & (a > 0) & (P > min_P)
    if np.sum(m) < 5:
        return np.nan
    R = ((4.0/3.0) * E_star_Pa * (a[m]**3)) / P[m]
    return float(np.nanmedian(R))

# Public aliases expected by other modules (non-underscored names)
hertz_load_from_h = _hertz_load_from_h
c_from_tabor = _c_from_tabor
tabor_mu = _tabor_mu
jkr_P_from_h = _jkr_P_from_h
auto_model_from_mu = _auto_model_from_mu

# ============================================================
# 2) Uncertainty calculations
# ============================================================

def _rss(*sigmas):
    sig = np.asarray(sigmas, float)
    sig = sig[np.isfinite(sig)]
    return float(np.sqrt(np.sum(sig**2))) if sig.size else np.nan

def _rel(x, sx):
    # relative uncertainty |sx/x| with guards
    if not (np.isfinite(x) and np.isfinite(sx) and x != 0.0):
        return np.nan
    return abs(sx / x)

def sigma_div(num, den, s_num, s_den):
    """z = num/den, independent."""
    if not (np.isfinite(num) and np.isfinite(den) and den != 0):
        return np.nan
    r1 = _rel(num, s_num)
    r2 = _rel(den, s_den)
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan
    return abs(num/den) * np.sqrt(r1*r1 + r2*r2)

def sigma_mul(a, b, s_a, s_b):
    """z = a*b, independent."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    r1 = _rel(a, s_a)
    r2 = _rel(b, s_b)
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return np.nan
    return abs(a*b) * np.sqrt(r1*r1 + r2*r2)

def sigma_area_piRh(h_m: np.ndarray, sigma_h_m: np.ndarray, R_m: float, sigma_R_m: float) -> np.ndarray:
    h = np.maximum(0.0, np.asarray(h_m, float))
    sh = np.asarray(sigma_h_m, float)
    R = float(R_m)
    sR = float(sigma_R_m)
    return np.sqrt((np.pi * h * sR)**2 + (np.pi * R * sh)**2)

def sigma_contact_depth(z_m, P_N, k_frame_z, s_z, s_P, s_k=0.0):
    # h = z - P/k
    if not (np.isfinite(z_m) and np.isfinite(P_N) and np.isfinite(k_frame_z) and k_frame_z != 0):
        return np.nan
    term_z = s_z
    term_P = s_P / abs(k_frame_z)
    term_k = abs(P_N) * (s_k / (k_frame_z**2)) if (np.isfinite(s_k) and s_k > 0) else 0.0
    return _rss(term_z, term_P, term_k)

def sigma_pressure(P_N, A_m2, s_P, s_A):
    return sigma_div(P_N, A_m2, s_P, s_A)

def sigma_shear_strength(F_N, A_m2, s_F, s_A):
    return sigma_div(F_N, A_m2, s_F, s_A)

def sigma_mu(F_N, P_N, s_F, s_P):
    return sigma_div(F_N, P_N, s_F, s_P)

def _finite(x):
    return (x is not None) and np.isfinite(x)

def sigma_from_components(*sigmas):
    sigs = [float(s) for s in sigmas if _finite(s) and float(s) >= 0]
    if not sigs:
        return np.nan
    return float(np.sqrt(np.sum(np.square(sigs))))

def prop_div(a, sa, b, sb):
    # y = a/b
    if not (_finite(a) and _finite(b)) or b == 0:
        return np.nan
    y = a / b
    rel2 = 0.0
    if _finite(sa) and a != 0:
        rel2 += (sa / a) ** 2
    if _finite(sb) and b != 0:
        rel2 += (sb / b) ** 2
    return float(abs(y) * np.sqrt(rel2))

def prop_mul(a, sa, b, sb):
    # y = a*b
    if not (_finite(a) and _finite(b)):
        return np.nan
    y = a * b
    rel2 = 0.0
    if _finite(sa) and a != 0:
        rel2 += (sa / a) ** 2
    if _finite(sb) and b != 0:
        rel2 += (sb / b) ** 2
    return float(abs(y) * np.sqrt(rel2))

def sigma_P_contact(
    F_raw_N: np.ndarray,
    z_raw_m: np.ndarray,
    k_sup: float,
    b_sup: float,
    *,
    sigma_F_N: float,
    sigma_z_m: float,
    sigma_k_sup: float,
    sigma_b_sup: float,
) -> np.ndarray:
    F = np.asarray(F_raw_N, float)
    z = np.asarray(z_raw_m, float)
    return np.sqrt(
        sigma_F_N**2 +
        (z * sigma_k_sup)**2 +
        (k_sup * sigma_z_m)**2 +
        sigma_b_sup**2
    )

def sigma_h_contact(
    z_raw_m: np.ndarray,
    touch_i: int,
    P_N: np.ndarray,
    sigma_P_N: np.ndarray,
    *,
    k_frame_z: float | None,
    sigma_z_m: float,
    sigma_k_frame_z: float = 0.0,
) -> np.ndarray:
    z = np.asarray(z_raw_m, float)
    P = np.asarray(P_N, float)
    sP = np.asarray(sigma_P_N, float)

    # z0 and P0 uncertainties: treat as same channel noise at that index
    z0 = float(z[touch_i])
    P0 = float(P[touch_i])
    sP0 = float(sP[touch_i])

    # If you want more realistic z0 uncertainty, you can estimate it from a small
    # pre-touch window. For now, sigma_z_m is the right baseline.
    sigma_z0 = float(sigma_z_m)

    if k_frame_z is None:
        return np.sqrt(sigma_z_m**2 + sigma_z0**2) * np.ones_like(z)

    kf = float(k_frame_z)
    dkf = float(sigma_k_frame_z)

    dP = P - P0
    # main terms
    s2 = (sigma_z_m**2 + sigma_z0**2) + (sP / kf)**2 + (sP0 / kf)**2

    # frame stiffness uncertainty term
    if dkf > 0:
        s2 = s2 + ((dP * dkf) / (kf**2))**2

    return np.sqrt(s2)


## subtracting for actuator dynamic response, esp. for the instability stick to slip transition.
def _savgol(y, win, poly=3):
    from scipy.signal import savgol_filter
    win = int(win)
    if win % 2 == 0: win += 1
    win = max(win, poly+2 + (poly+2)%2)  # ensure > poly and odd
    win = min(win, len(y) - (1-len(y)%2))  # keep odd <= len
    return savgol_filter(y, win, poly, mode="interp")

def differentiate_wrt_t(t, y):
    # central difference wrt time, handles nonuniform t
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    dy = np.gradient(y, t)
    return dy

def fit_actuator_mck(t, x, F, smooth_win=101):
    """
    Fit F ≈ m*xdd + c*xd + k*x on a calibration window.
    Returns dict with m,c,k and diagnostics.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    F = np.asarray(F, float)

    # smooth x for derivatives
    xs = _savgol(x, smooth_win, poly=3)
    xd = differentiate_wrt_t(t, xs)
    xdd = differentiate_wrt_t(t, xd)

    A = np.column_stack([xdd, xd, xs])  # [m,c,k]
    msk = np.isfinite(A).all(axis=1) & np.isfinite(F)
    A = A[msk]
    b = F[msk]

    if A.shape[0] < 20:
        return {"ok": 0, "reason": "too_few_points"}

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    m_eff, c_eff, k_eff = coef

    Fhat = A @ coef
    resid = b - Fhat
    rmse = float(np.sqrt(np.mean(resid**2)))

    return {
        "ok": 1,
        "m_eff": float(m_eff),
        "c_eff": float(c_eff),
        "k_eff": float(k_eff),
        "rmse_N": rmse,
        "n": int(A.shape[0]),
        "smooth_win": int(smooth_win),
        "error" : None,
    }

def correct_contact_force(t, x, F_meas, m_eff, c_eff, k_eff, smooth_win=101):
    """
    Returns F_contact = F_meas - (m*xdd + c*xd + k*x)
    """
    xs = _savgol(x, smooth_win, poly=3)
    xd = differentiate_wrt_t(t, xs)
    xdd = differentiate_wrt_t(t, xd)
    F_act = m_eff*xdd + c_eff*xd + k_eff*xs
    return F_meas - F_act, F_act

## Lock-in technique and ramped force lag (die time constant and ramp rate-frequency) error-uncertainity
def robust_slope(t, y):
    m = np.isfinite(t) & np.isfinite(y)
    if np.sum(m) < 5:
        return np.nan
    # slope only
    return float(np.polyfit(t[m], y[m], 1)[0])

def sigma_lockin_lag(t, F_env, i_event, tau_s, pre_s=0.30, guard_s=0.25):
    """
    1-sigma systematic magnitude from first-order LP lag:
      sigma_lag ~ tau * |dF/dt|
    measured on window before event:
      [t_event - guard - pre, t_event - guard]
    """
    if i_event is None:
        return np.nan, np.nan
    i = int(i_event)
    if i < 0 or i >= len(t):
        return np.nan, np.nan

    t0 = t[i] - guard_s - pre_s
    t1 = t[i] - guard_s
    m = (t >= t0) & (t <= t1) & np.isfinite(F_env)
    if np.sum(m) < 5:
        return np.nan, np.nan

    slope = robust_slope(t[m], F_env[m])  # N/s
    if not np.isfinite(slope):
        return np.nan, np.nan
    return float(tau_s * abs(slope)), slope

def estimate_lockin_lag_force_sigma(t, F, i, tau_s=0.05, dt_window_s=0.2):
    """Return sigma_F due to lock-in lag around index i."""
    t = np.asarray(t, float); F = np.asarray(F, float)
    if i is None or i < 2 or i >= len(t)-2:
        return np.nan
    # local slope using robust linear fit in a small window
    m = (t >= t[i]-dt_window_s/2) & (t <= t[i]+dt_window_s/2) & np.isfinite(F)
    if np.sum(m) < 5:
        return np.nan
    tt = t[m] - np.mean(t[m])
    FF = F[m]
    # least squares slope
    denom = np.sum(tt*tt)
    if denom <= 0:
        return np.nan
    r = np.sum(tt*FF) / denom  # N/s
    return float(abs(r) * tau_s)

def estimate_instability_dynamic_sigma(t, F, i_ss, pre_s=0.2, post_s=0.1):
    """Error magnitude due to dynamic mismatch at slip onset."""
    if i_ss is None: 
        return np.nan
    t = np.asarray(t, float); F = np.asarray(F, float)
    t0 = t[i_ss]
    pre = (t >= t0-pre_s) & (t < t0) & np.isfinite(F)
    post = (t >= t0) & (t <= t0+post_s) & np.isfinite(F)
    if np.sum(pre) < 5 or np.sum(post) < 3:
        return np.nan

    # linear fit pre-slip: F = a*t + b
    tp = t[pre]; Fp = F[pre]
    A = np.vstack([tp, np.ones_like(tp)]).T
    a, b = np.linalg.lstsq(A, Fp, rcond=None)[0]

    # evaluate prediction in post window
    tpost = t[post]
    Fpred = a*tpost + b
    Fmeas = F[post]
    return float(np.nanmax(np.abs(Fpred - Fmeas)))

def safe_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x) & (x > 0)]

def tau_from_F_and_A_samples(F_N: float, A_samples: np.ndarray) -> dict:
    # tau = F/A
    t = float(F_N) / np.asarray(A_samples, float)
    return summarize_dist(t)

def compute_area_from_choice(
    h_m: np.ndarray,
    P_N: np.ndarray,
    area_mode: str,
    *,
    cfg,
    E_star_Pa: float,
    hertz: dict | None = None,
    flat_end: dict | None = None,
) -> tuple[np.ndarray, str]:

    mode = area_mode

    if mode == "fit_hertz":
        if hertz and int(hertz.get("ok", 0)) == 1:
            R = hertz.get("R_eff_m", np.nan)
            if np.isfinite(R) and R > 0:
                return area_pi_h_R(h_m, float(R)), "fit_hertz"
        return area_pi_h_R(h_m, float(cfg.tip_radius_m)), "nominal_fallback"

    if mode == "flat_end":
        if flat_end and int(flat_end.get("ok", 0)) == 1:
            S0 = flat_end.get("S0", np.nan)
            C  = flat_end.get("C", np.nan)
            if np.isfinite(S0) and np.isfinite(C):
                A_curve = area_from_flat_end_fit(
                    P_N,
                    S0_N_per_m=float(S0),
                    C=float(C),
                    E_star_Pa=float(E_star_Pa),
                )
                return A_curve, "flat_end"
        return area_pi_h_R(h_m, float(cfg.tip_radius_m)), "nominal_fallback"

    return area_pi_h_R(h_m, float(cfg.tip_radius_m)), "nominal"