#mechanics.py
import inspect
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# avoid top-level import of fitting (circular import); import lazily where needed
from .math_utils import (
    summarize_dist, area_from_flat_end_fit,
    _num, phase_to_rad, rms_to_peak, robust_mad
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
        Units should be meters; distance in meters and speeds in m/s.
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
    version = np.__version__
    if version < "2.0":
        D = float(4.0 * freq_Hz * np.trapz(AA, tt))   # total slide distance over sinusoid
        A_mean_time = float(np.trapz(AA, tt) / total_time)
    else:
        D = float(4.0 * freq_Hz * np.trapezoid(AA, tt))   # total slide distance over sinusoid
        A_mean_time = float(np.trapezoid(AA, tt) / total_time)
    v_max = float(2.0 * np.pi * freq_Hz * max(AA))  # max(|v|) over a sinusoid
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

def estimate_support_kc_from_cal(
    F_pk: np.ndarray,
    X_pk: np.ndarray,
    phi_rad: np.ndarray,
    *,
    f_Hz: float,
    cal_sl: slice,
    min_points: int = 10,
    trim_q: tuple[float, float] = (0.05, 0.95),
) -> dict:
    """
    Estimate support impedance Z_sup = k_sup + i*omega*c_sup from a calibration slice.
      phi = displacement phase relative to force
      Z = (F/X) * exp(-i*phi)

    Units:
      Z, k_sup, (omega*c_sup) are N/m
      c_sup is N*s/m
    """
    F_pk = np.asarray(F_pk, float)
    X_pk = np.asarray(X_pk, float)
    phi = np.asarray(phi_rad, float)

    if (not np.isfinite(f_Hz)) or (f_Hz <= 0):
        return dict(ok=0, reason="bad_frequency", k_sup=np.nan, c_sup=np.nan, omega=np.nan)

    omega = 2.0 * np.pi * float(f_Hz)

    F = np.abs(F_pk[cal_sl])
    X = np.abs(X_pk[cal_sl])
    ph = phi[cal_sl]

    m = np.isfinite(F) & np.isfinite(X) & np.isfinite(ph) & (X > 0)
    F = F[m]; X = X[m]; ph = ph[m]
    if F.size < min_points:
        return dict(ok=0, reason="not_enough_points", k_sup=np.nan, c_sup=np.nan, omega=omega)

    Z = (F / X) * np.exp(-1j * ph)  # N/m

    # Trim by X amplitude to reduce edge/outlier effects
    qlo, qhi = np.quantile(X, trim_q)
    keep = (X >= qlo) & (X <= qhi)
    Zk = Z[keep] if keep.sum() >= min_points else Z

    k_sup = float(np.nanmedian(np.real(Zk)))
    k_loss_sup = float(np.nanmedian(np.imag(Zk)))  # = omega*c_sup
    c_sup = float(k_loss_sup / omega)

    ok = int(np.isfinite(k_sup) and np.isfinite(c_sup))
    return dict(
        ok=ok,
        reason="" if ok else "nonfinite_fit",
        k_sup=k_sup,
        c_sup=c_sup,
        omega=omega,
        n_used=int(Zk.size),
        k_loss_sup_N_per_m=k_loss_sup,
        imag_is_negative=bool(k_loss_sup < 0),
    )


def compute_lateral_corrected(
    df: pd.DataFrame,
    cfg,
    scale_to_SI: Dict[str, float],
    cal_sl: Optional[slice],
) -> pd.DataFrame:
    """
    Dynamic lateral correction using complex impedance subtraction.

    Outputs (unit-correct):
      - Dyn. Stiffness (K_storage_lateral_N_per_m) = Re(Z_contact)
      - Dyn. Damping (K_loss_lateral_N_per_m)    = Im(Z_contact) = omega * c_contact
      - c_lateral_Ns_per_m        = K_loss / omega
      - E_diss_J_per_cycle        = pi * |K_loss| * X_contact^2
    """
    out = df.copy()

    # ---- Convert RMS channels to SI ----
    F2_rms_SI = _num(out, cfg.F2_rms_col) * scale_to_SI[cfg.F2_rms_col]   # N
    X2_rms_SI = _num(out, cfg.X2_rms_col) * scale_to_SI[cfg.X2_rms_col]   # m
    phi = phase_to_rad(_num(out, cfg.PH2_col))  # rad

    # ---- RMS -> peak amplitudes (magnitudes) ----
    F2_pk = np.abs(rms_to_peak(np.asarray(F2_rms_SI, float)))
    X2_pk = np.abs(rms_to_peak(np.asarray(X2_rms_SI, float)))

    out["F2_pk_N"] = F2_pk
    out["X2_pk_m"] = X2_pk
    out["phi2_rad"] = np.asarray(phi, float)

    # ---- Frequency ----
    f_Hz = float(getattr(cfg, "dyn_f2_freq_Hz", np.nan))
    omega = 2.0 * np.pi * f_Hz if (np.isfinite(f_Hz) and f_Hz > 0) else np.nan
    out["dyn_f2_freq_Hz"] = f_Hz
    out["omega_drive_rad_per_s"] = omega

    # ---- Support k,c from calibration, else fallback ----
    k_sup = np.nan
    c_sup = np.nan
    cal={}
    cal_used = False
    cal_reason = ""

    if (cal_sl is not None) and np.isfinite(f_Hz) and (f_Hz > 0):
        cal = estimate_support_kc_from_cal(F2_pk, X2_pk, out["phi2_rad"].to_numpy(),
                                           f_Hz=f_Hz, cal_sl=cal_sl)
        cal_used = bool(cal.get("ok", 0) == 1)
        cal_reason = cal.get("reason", "")
        if cal_used:
            k_sup = cal["k_sup"]
            c_sup = cal["c_sup"]

    # Stiffness fallback
    if not np.isfinite(k_sup):
        k_fb = getattr(cfg, "k_sup_x_fallback", None)
        if k_fb is not None and np.isfinite(k_fb):
            k_sup = float(k_fb)
        elif getattr(cfg, "allow_no_cal", False):
            k_sup = 0.0
        else:
            raise RuntimeError("Support stiffness calibration failed and no fallback provided.")

    # Damping fallback (Ns/m)
    if not np.isfinite(c_sup):
        c_fb = getattr(cfg, "c_sup_x_fallback", None)
        if c_fb is not None and np.isfinite(c_fb):
            c_sup = float(c_fb)
        elif getattr(cfg, "allow_no_cal", False):
            c_sup = 0.0
        else:
            c_sup = 0.0
    print(cal)
    out["support_cal_used"] = cal_used
    out["support_cal_reason"] = cal_reason
    out["kx_sup_est_N_per_m"] = k_sup
    out["cx_sup_est_Ns_per_m"] = c_sup

    # ---- Optional frame correction on displacement amplitude ----
    X_contact = out["X2_pk_m"].to_numpy()
    k_frame = getattr(cfg, "k_frame_x", None)
    if k_frame is not None and np.isfinite(k_frame) and float(k_frame) > 0:
        X_contact = X_contact - (out["F2_pk_N"].to_numpy() / float(k_frame))
        out["X2_pk_contact_went_negative"] = X_contact < 0
        X_contact = np.maximum(0.0, X_contact)
    out["X2_pk_contact_m"] = X_contact

    den = np.maximum(1e-30, X_contact)

    # ---- Measured impedance and corrected/contact impedance ----
    # Z_meas = (F/X)*exp(-i phi)  [N/m]
    Z_meas = (out["F2_pk_N"].to_numpy() / den) * np.exp(-1j * out["phi2_rad"].to_numpy())

    # Support impedance Z_sup = k + i*omega*c  [N/m]
    omega_eff = float(omega) if (np.isfinite(omega) and omega > 0) else 0.0
    Z_sup = k_sup + 1j * omega_eff * c_sup

    Z_contact = Z_meas - Z_sup

    # ---- Outputs (unit-correct naming) ----
    out["Stiffness_lateral"] = np.real(Z_contact)
    out["Damping_lateral"] = np.imag(Z_contact)  # = omega*c_contact

    if omega_eff > 0:
        out["c_lateral_Ns_per_m"] = out["Damping_lateral"] / omega_eff
    else:
        out["c_lateral_Ns_per_m"] = np.nan

    # Optional diagnostic: reconstructed contact force amplitude
    # X~ = X * exp(+i phi); F~_contact = Z_contact * X~
    X_tilde = den * np.exp(1j * out["phi2_rad"].to_numpy())
    F_contact_tilde = Z_contact * X_tilde
    out["F2_pk_corr_N"] = np.abs(F_contact_tilde)

    # Dissipated energy per cycle (uses loss stiffness)
    out["E_diss_J_per_cycle"] = np.pi * np.abs(out["Damping_lateral"].to_numpy()) * (den ** 2)

    return out

# ============================================================
# Inserted Contact Mechanics (Hertz/JKR) functions
# ============================================================

def effective_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """Hertz reduced modulus E* (Pa)."""
    inv = (1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2
    return 1.0 / inv if inv > 0 else np.nan

def effective_shear_modulus_Pa(G1: float, nu1: float, G2: float, nu2: float) -> float:
    """Reduced Shear Modulus"""
    inv = (2.0-nu1)/ G1 + (2-nu2)/G2
    return 1.0/inv if inv>0 else np.nan

def effective_shear_modulus_byS_Pa(Sx: float, a: float) -> float:
    """Reduced Shear Modulus by measured lateral stiffness and calculated contact radius."""
    G = 1e9*Sx/(8.0*a) ## N/m divided by nm -> Pa.. or adjust for metric.
    return G if np.isfinite(G) else np.nan 

def poisson(Sx: float, Sz: float) -> float:
    """ Poission's ratio by measured contact stiffness-Mindlin's approach """
    ratio = Sx/Sz if Sz>0. else np.nan
    return 2*((ratio-1)/(ratio-2)) if np.isfinite(ratio) else np.nan

def tau_scaled(tau: float, G: float) -> float:
    return tau/G if G>0. else np.nan

def tau_scaled_byGrosslip(dx: float, a: float) -> float:
    """ 
    Using derived Mindlin solution by Y.Gao 2*X_critical / contact radius = scaled shear strength metric
        careful on nm/nm  units.
    """
    return 2*dx/a if a>0. else np.nan

def tau_by_inverseScaled_MPa(tau_scaled: float, G: float) -> float:
    """
    inverse calculation to check by displacement-amplitude shear strength value
    -use shear modulus by one of the effective calculations.
    conversion to MPa for direct check considering G is in GPa.
    """
    return tau_scaled*G*1e3

def radius_scaled(a: float, b: float) -> float:
    """ Scaling by burgers vector b"""
    return a/b if b>0. else np.nan

def junction_growth_metric(A: float, A0: float) -> float:
    return (A/A0)**2-1 if A0>0. else np.nan

def junction_growth_scale(Load: float, Friction: float)-> float:
    return (Friction/Load)**2 if Load>0. else np.nan

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

def _jkr_P_from_h(
    h_m: np.ndarray,
    R_m: float,
    E_star_Pa: float,
    w_J_per_m2: float,
    n_bisect: int = 60,
    *,
    # ---- fast-solver knobs (safe defaults) ----
    newton_max_iter: int = 12,
    newton_tol_rel: float = 1e-12,
    fallback_bisect: bool = True,
    bracket_grow: int = 8,
) -> np.ndarray:
    """
    Fast JKR P(h) compatible existing definitions:

      h(a) = a^2/R - sqrt( (8*pi*w*a)/(3E) )
      P(a) = (4/3) E a^3 / R - sqrt( 8*pi*w*E*a^3 )

    Uses vectorized Newton iterations for a(h) with Hertz initial guess,
    with optional per-point bisection fallback for rare non-converged points.

    Signature kept identical to original (includes n_bisect) for compatibility.
    """
    h = np.asarray(h_m, dtype=float)
    out = np.full_like(h, np.nan, dtype=float)

    R = float(R_m); E = float(E_star_Pa); w = float(w_J_per_m2)
    if not (R > 0.0 and E > 0.0 and w > 0.0):
        return out

    # Only solve for finite nonnegative h (consistent with original)
    m = np.isfinite(h) & (h >= 0.0)
    if not np.any(m):
        return out

    hv = h[m]

    # ---- Vectorized Newton on a for f(a)=a^2/R - C*sqrt(a) - h = 0 ----
    # C so term_adh = C*sqrt(a) = sqrt((8*pi*w*a)/(3E))
    C = np.sqrt(8.0 * np.pi * w / (3.0 * E))

    # Initial guess from Hertz: a0 ~ sqrt(R*h)
    a = np.sqrt(np.maximum(0.0, R * hv)) + 1e-18

    # Newton
    for _ in range(int(max(1, newton_max_iter))):
        sqrt_a = np.sqrt(np.maximum(a, 1e-30))
        f = (a * a) / R - C * sqrt_a - hv

        # df/da = 2a/R - C/(2*sqrt(a))
        df = (2.0 * a) / R - (C / (2.0 * np.maximum(sqrt_a, 1e-30)))

        step = f / np.where(np.abs(df) > 1e-30, df, 1e-30)
        a_new = np.maximum(a - step, 1e-18)

        rel = np.abs(a_new - a) / np.maximum(a_new, 1e-30)
        a = a_new

        if float(np.nanmax(rel)) < float(newton_tol_rel):
            break

    # Compute P(a) (vectorized) using exact formula
    a3 = a**3
    term_el = (4.0 / 3.0) * E * a3 / R
    term_adh = np.sqrt(np.maximum(0.0, 8.0 * np.pi * w * E * a3))
    Pv = term_el - term_adh

    # Validate solution: check h(a) is close to hv
    h_check = (a * a) / R - C * np.sqrt(np.maximum(a, 1e-30))
    # Tolerances: absolute + relative; tune if needed
    ok = (
        np.isfinite(Pv)
        & np.isfinite(h_check)
        & (np.abs(h_check - hv) <= (1e-10 + 1e-6 * np.abs(hv)))
    )

    out_m = np.full_like(hv, np.nan, dtype=float)
    out_m[ok] = Pv[ok]

    # ---- Optional fallback: robust bisection (rare) ----
    if fallback_bisect:
        bad = ~ok
        if np.any(bad):
            # Scalar helper (avoids allocating arrays in _jkr_h_from_a)
            def h_from_a_scalar(a_: float) -> float:
                return (a_ * a_) / R - np.sqrt(max(0.0, (8.0 * np.pi * w * a_) / (3.0 * E)))

            def P_from_a_scalar(a_: float) -> float:
                a3_ = a_**3
                term_el_ = (4.0 / 3.0) * E * a3_ / R
                term_adh_ = np.sqrt(max(0.0, 8.0 * np.pi * w * E * a3_))
                return term_el_ - term_adh_

            bad_idx = np.where(bad)[0]
            for idx in bad_idx:
                hi = float(hv[idx])
                if not (np.isfinite(hi) and hi >= 0.0):
                    continue

                a_lo = 0.0
                a_hi = float(np.sqrt(max(R * hi, 0.0)) * 5.0 + 1e-12)

                # grow bracket
                for _ in range(int(max(1, bracket_grow))):
                    h_hi = h_from_a_scalar(a_hi)
                    if np.isfinite(h_hi) and (h_hi >= hi):
                        break
                    a_hi *= 2.0

                h_hi = h_from_a_scalar(a_hi)
                if not (np.isfinite(h_hi) and h_hi >= hi):
                    continue

                lo, hi_a = a_lo, a_hi
                for _ in range(int(max(1, n_bisect))):
                    mid = 0.5 * (lo + hi_a)
                    h_mid = h_from_a_scalar(mid)
                    if not np.isfinite(h_mid):
                        hi_a = mid
                        continue
                    if h_mid >= hi:
                        hi_a = mid
                    else:
                        lo = mid

                a_sol = 0.5 * (lo + hi_a)
                out_m[idx] = P_from_a_scalar(a_sol)

    out[m] = out_m
    return out


def _auto_model_from_mu(mu: float, mu_dmt: float = 0.1, mu_jkr: float = 5.0) -> str:
    if not np.isfinite(mu):
        return "hertz"
    if mu <= mu_dmt:
        return "dmt"
    if mu >= mu_jkr:
        return "jkr"
    return "transition"

def a_from_Sz(Sz_N_per_m: float, E_star_Pa: float)-> float:
    return Sz_N_per_m / (2.0 * E_star_Pa) if E_star_Pa>0. else np.nan

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
    """ 
    y = a/b; sa, sb uncertaninty in a, b
    returns y*rel_error-propagated
    """
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
    # NEW (optional touch offsets, SI)
    F0_N: float | None = None,
    z0_m: float | None = None,
    sigma_F0_N: float = 0.0,
    sigma_z0_m: float = 0.0,
    relative_to_touch: bool = False,
) -> np.ndarray:
    """
    If relative_to_touch=False:
        P = F - (k z + b)               (matches corrected_normal_load)
    If relative_to_touch=True:
        P = (F - F0) - k (z - z0)       (b cancels; includes offset uncertainties)
    """
    F = np.asarray(F_raw_N, float)
    z = np.asarray(z_raw_m, float)
    k = float(k_sup)
    dk = float(sigma_k_sup)
    sF = float(sigma_F_N)
    sz = float(sigma_z_m)

    if not relative_to_touch:
        db = float(sigma_b_sup)
        return np.sqrt(
            sF**2 +
            (z * dk)**2 +
            (k * sz)**2 +
            db**2
        )

    # touch-relative mode
    if (F0_N is None) or (z0_m is None) or (not np.isfinite(F0_N)) or (not np.isfinite(z0_m)):
        # safe fallback to absolute
        db = float(sigma_b_sup)
        return np.sqrt(sF**2 + (z * dk)**2 + (k * sz)**2 + db**2)

    dz = z - float(z0_m)
    sF0 = float(sigma_F0_N)
    sz0 = float(sigma_z0_m)

    return np.sqrt(
        sF**2 +
        sF0**2 +
        (k * sz)**2 +
        (k * sz0)**2 +
        (dz * dk)**2
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
    # NEW (optional): touch offset for z
    z0_m: float | None = None,
    sigma_z0_m: float = 0.0,
) -> np.ndarray:
    z = np.asarray(z_raw_m, float)
    P = np.asarray(P_N, float)
    sP = np.asarray(sigma_P_N, float)

    # z0 and its uncertainty
    if (z0_m is None) or (not np.isfinite(z0_m)):
        z0 = float(z[touch_i])
        sz0 = float(sigma_z_m)  # conservative fallback
    else:
        z0 = float(z0_m)
        sz0 = float(sigma_z0_m) if np.isfinite(sigma_z0_m) and sigma_z0_m > 0 else float(sigma_z_m)

    # P0 and its uncertainty from sigma_P array
    P0 = float(P[touch_i])
    sP0 = float(sP[touch_i]) if np.isfinite(sP[touch_i]) else 0.0

    # If no frame correction, just z - z0
    if k_frame_z is None:
        return np.sqrt(sigma_z_m**2 + sz0**2) * np.ones_like(z)

    kf = float(k_frame_z)
    dkf = float(sigma_k_frame_z)

    dP = P - P0

    # h = (z - z0) - (P - P0)/kf
    s2 = (sigma_z_m**2 + sz0**2) + (sP / kf)**2 + (sP0 / kf)**2
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

def area_from_stiffness_Sneddon(
    Sz_N_per_m: np.ndarray,
    *,
    E_star_Pa: float,
    A_min_m2: float = 1e-18,   # (e.g., π*(5 nm)^2 ~ 7.85e-17)
) -> np.ndarray:
    a = a_from_stiffness_Sneddon(np.asarray(Sz_N_per_m, float), float(E_star_Pa))
    A = np.pi * a * a
    A = np.where(np.isfinite(A) & (A > float(A_min_m2)), A, np.nan)
    return A

def _clamp_area(
    A_m2: np.ndarray,
    *,
    A_min_m2: float = 1e-18,
    A_max_m2: float | None = None,
) -> np.ndarray:
    A = np.asarray(A_m2, float)
    A = np.where(np.isfinite(A), A, np.nan)
    A = np.where(A > float(A_min_m2), A, np.nan)
    if A_max_m2 is not None and np.isfinite(A_max_m2):
        A = np.where(A <= float(A_max_m2), A, np.nan)
    return A

def compute_area_from_choice(
    h_m: np.ndarray,
    P_N: np.ndarray,
    area_mode: str,
    *,
    cfg,
    E_star_Pa: float,
    hertz: dict | None = None,
    flat_end: dict | None = None,
    Sz_meas_N_per_m: np.ndarray | None = None,   # NEW: needed for nominal_stiffness
) -> tuple[np.ndarray, str]:
    """
    Returns (A_curve, area_mode_used).

    Key behavior:
      - nominal -> prefers stiffness-based area IF stiffness is valid near peak load.
      - otherwise nominal -> depth-based (π R h).
      - fit_hertz / flat_end behave as before, with nominal fallback.
    """

    mode = (area_mode or "nominal").strip().lower()

    # configurable floors/thresholds
    a_min_m = float(getattr(cfg, "area_min_radius_m", 5e-9))  # 5 nm default
    A_min_m2 = float(np.pi * a_min_m * a_min_m)

    frac_ok_min = float(getattr(cfg, "area_stiff_frac_ok_min", 0.20))     # min finite fraction overall
    frac_hi_ok_min = float(getattr(cfg, "area_stiff_frac_hi_ok_min", 0.30))  # min finite fraction near peak
    peak_frac = float(getattr(cfg, "area_stiff_peak_frac", 0.90))         # peak-load window: P >= peak_frac*Pmax
    min_hi_pts = int(getattr(cfg, "area_stiff_min_hi_pts", 10))           # require enough peak points
    max_cv = float(getattr(cfg, "area_stiff_max_cv", 1.0))                # Ceff. of Var. guard on stiffness in peak window

    def _nominal_depth() -> tuple[np.ndarray, str]:
        A = area_pi_h_R(h_m, float(cfg.tip_radius_m))
        return _clamp_area(A, A_min_m2=A_min_m2), "nominal_depth"

    def _nominal_stiffness_if_valid() -> tuple[np.ndarray, str] | None:
        if Sz_meas_N_per_m is None:
            return None
        E = float(E_star_Pa)
        if not (np.isfinite(E) and E > 0):
            return None

        Sz = np.asarray(Sz_meas_N_per_m, float)
        if Sz.shape != np.asarray(h_m).shape:
            return None

        A_stiff = area_from_stiffness_Sneddon(Sz, E_star_Pa=E, A_min_m2=A_min_m2)

        # overall finite fraction
        frac_ok = float(np.nanmean(np.isfinite(A_stiff))) if A_stiff.size else 0.0
        if not (np.isfinite(frac_ok) and frac_ok >= frac_ok_min):
            return None

        # peak-load window validity
        P = np.asarray(P_N, float)
        mP = np.isfinite(P)
        if not np.any(mP):
            return None
        Pmax = float(np.nanmax(P[mP]))
        if not (np.isfinite(Pmax) and Pmax > 0):
            return None

        hi = mP & (P >= peak_frac * Pmax)
        n_hi = int(np.sum(hi))
        if n_hi < min_hi_pts:
            return None

        frac_hi_ok = float(np.nanmean(np.isfinite(A_stiff[hi]))) if n_hi else 0.0
        if not (np.isfinite(frac_hi_ok) and frac_hi_ok >= frac_hi_ok_min):
            return None

        # optional stability guard: stiffness CV in peak window not insane
        Sz_hi = Sz[hi]
        Sz_hi = Sz_hi[np.isfinite(Sz_hi) & (Sz_hi > 0)]
        if Sz_hi.size >= min_hi_pts:
            med = float(np.median(Sz_hi))
            sig = float(robust_mad(Sz_hi))  # ~1σ
            cv = sig / max(med, 1e-30) if (np.isfinite(sig) and sig >= 0) else np.inf
            if np.isfinite(cv) and (cv > max_cv):
                return None

        return _clamp_area(A_stiff, A_min_m2=A_min_m2), "nominal_stiffness"

    # -----------------------
    # NOMINAL: stiffness-first (strict validation), else depth
    # -----------------------
    if mode in ("nominal", "nominal_stiff_first", "nominal_stiff", "nominal_depth", "nominal_stiffness"):
        if mode != "nominal_depth":
            out = _nominal_stiffness_if_valid()
            if out is not None:
                return out
        return _nominal_depth()

    # -----------------------
    # fit_hertz (unchanged; nominal fallback)
    # -----------------------
    if mode in ("fit_hertz", "hertz_fit", "hertz"):
        if hertz and int(hertz.get("ok", 0)) == 1:
            R = hertz.get("R_eff_m", np.nan)
            if np.isfinite(R) and R > 0:
                A = area_pi_h_R(h_m, float(R))
                return _clamp_area(A, A_min_m2=A_min_m2), "fit_hertz"
        A, _ = _nominal_depth()
        return A, "nominal_fallback"

    # -----------------------
    # flat_end (unchanged; nominal fallback)
    # -----------------------
    if mode in ("flat_end", "flat", "flatend"):
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
                return _clamp_area(A_curve, A_min_m2=A_min_m2), "flat_end"
        A, _ = _nominal_depth()
        return A, "nominal_fallback"

    return _nominal_depth()

def fit_junction_growth_simple(
    Ft,
    Pz,
    K,
    smooth_n=5,
    n_ref=5,
):
    """
    Very simple junction growth fit:

        y = (K/K0)^4 - 1
        x = (Ft/Pz)^2

    Returns:
        {"slope", "intercept", "x", "y"}
    """

    Ft = np.asarray(Ft, float)
    Pz = np.asarray(Pz, float)
    K  = np.asarray(K, float)

    # first remove obviously bad raw points
    m0 = np.isfinite(Ft) & np.isfinite(Pz) & np.isfinite(K) & (Pz != 0)

    Ft = Ft[m0]
    Pz = Pz[m0]
    K  = K[m0]
    if Ft.size < 2:
        return {"slope": np.nan, "intercept": np.nan, "x": np.array([]), "y": np.array([])}

    # --- build x and y ---
    x = (Ft / Pz)**2

    K0 = np.median(K[:n_ref]) if n_ref > 1 else K[0]
    if (not np.isfinite(K0)) or (K0 == 0):
        return {"slope": np.nan, "intercept": np.nan, "x": x, "y": np.array([])}

    y = (K / K0)**4 - 1.0

    # smoothing
    if smooth_n > 1:
        kernel = np.ones(smooth_n, dtype=float) / smooth_n
        y = np.convolve(y, kernel, mode="same")

    # final cleanup
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size < 2:
        return {"slope": np.nan, "intercept": np.nan, "x": x, "y": y}

    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": slope, "intercept": intercept, "x": x, "y": y}