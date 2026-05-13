# src/stick_slip_slide/fitting.py
from __future__ import annotations

from typing import Dict, Optional, Callable, Any
import numpy as np
import pandas as pd
import inspect
from .config import Config

from .math_utils import (
    _num, 
    rms_to_peak,
    robust_fit_line_origin,
    robust_fit_line
)

from .mechanics import (
    hertz_load_from_h,
    jkr_P_from_h,
    tabor_mu,
    c_from_tabor,
    auto_model_from_mu,
    w_eff_from_roughness,
)

from .math_utils import mindlin_model


# SciPy optional
try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except Exception:
    curve_fit = None
    SCIPY_OK = False

# ---------
# 1) Mindlin fit
# ---------
def mindlin_fit(Q: np.ndarray, K: np.ndarray, *, min_points: int = 10) -> Dict[str, float]:
    """
    Fit Mindlin-like stiffness decay:
        K(Q) = a * (1 - Q/t)^(1/3), with a>0 and t>max(Q).
    Returns dict(a,t,rmse,n,ok,scipy_fit).
    """
    Q = np.asarray(Q, float)
    K = np.asarray(K, float)

    # Allow Q==0; require positive stiffness
    m = np.isfinite(Q) & np.isfinite(K) & (Q >= 0) & (K > 0)
    Q = Q[m]; K = K[m]

    n = int(Q.size)
    if n < int(min_points):
        return {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": n, "ok": 0, "scipy_fit": int(SCIPY_OK)}

    # Sort by Q (stabilizes quantiles + any future weighting)
    srt = np.argsort(Q)
    Q = Q[srt]
    K = K[srt]

    Qmax = float(np.max(Q))
    # more robust "effective max"
    Qhi = float(np.quantile(Q, 0.95)) if n >= 20 else Qmax

    # initial guesses
    q20 = float(np.quantile(Q, 0.2)) if n >= 5 else float(np.min(Q))
    a0 = float(np.nanmedian(K[Q <= q20])) if np.any(Q <= q20) else float(np.nanmedian(K))
    if not np.isfinite(a0) or a0 <= 0:
        a0 = float(np.nanmax(K))  # fallback
    t0 = 1.2 * max(Qhi, 1e-18)

    # --- SciPy branch ---
    if SCIPY_OK:
        try:
            from scipy.optimize import curve_fit

            # bounds: a positive, t must exceed Qmax
            a_ub = max(1e-9, 10.0 * float(np.nanmax(K)))  # safer than tying to a0
            t_lb = 1.001 * Qmax
            t_ub = 100.0 * max(Qmax, 1e-18)

            popt, _ = curve_fit(
                mindlin_model, Q, K,
                p0=[a0, max(t0, t_lb)],
                bounds=([1e-12, t_lb], [a_ub, t_ub]),
                maxfev=20000,
            )
            a_hat, t_hat = float(popt[0]), float(popt[1])

            # Guard domain (just in case)
            if not (np.isfinite(a_hat) and np.isfinite(t_hat) and a_hat > 0 and t_hat > Qmax):
                raise ValueError("bad_fit")

            Khat = mindlin_model(Q, a_hat, t_hat)
            rmse = float(np.sqrt(np.mean((K - Khat) ** 2)))
            return {"a": a_hat, "t": t_hat, "rmse": rmse, "n": n, "ok": 1, "scipy_fit": 1}
        except Exception:
            pass

    # --- Pure-numpy fallback: grid over t + closed-form a for each t (CORRECT) ---
    # Use a wider, more informative grid. Log-spacing tends to work better.
    t_min = 1.01 * Qmax
    t_max = 100.0 * max(Qmax, 1e-18)

    if t_min >= t_max:
        # degenerate, but keep safe
        return {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": n, "ok": 0, "scipy_fit": 0}

    t_grid = np.geomspace(t_min, t_max, 300)

    y = K**3  # linear in a^3
    best_rmse = np.inf
    best_a = np.nan
    best_t = np.nan

    eps = 1e-30
    for t in t_grid:
        # enforce domain safely
        if not (t > Qmax):
            continue
        x = 1.0 - Q / t
        x = np.maximum(eps, x)

        # Solve y ≈ b*x (through origin), b = a^3
        denom = float(np.dot(x, x))
        if denom <= eps:
            continue
        b = float(np.dot(x, y) / denom)
        if not (np.isfinite(b) and b > 0):
            continue

        a = float(b ** (1.0 / 3.0))

        Khat = mindlin_model(Q, a, float(t))
        rmse = float(np.sqrt(np.mean((K - Khat) ** 2)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_a = a
            best_t = float(t)

    ok = int(np.isfinite(best_a) and np.isfinite(best_t) and best_a > 0 and best_t > Qmax)
    return {"a": best_a, "t": best_t, "rmse": float(best_rmse), "n": n, "ok": ok, "scipy_fit": 0}

# ---------------------------------------------------------------------
# 2) Support spring fit (pre-touch)
# ---------------------------------------------------------------------
def fit_support_spring_pre_touch(
    z_m: np.ndarray,
    F_N: np.ndarray,
    touch_i: int,
    cfg,
    *,
    min_points: int = 50,
) -> tuple[float, float, float, float, dict]:
    """
    Fit pre-touch support spring F ≈ k*z + b on a *selected* pre-touch window.

    Returns k,b,sigma_k,sigma_b,meta
    """
    z = np.asarray(z_m, float)
    F = np.asarray(F_N, float)

    # --- time->index mapping (optional) ---
    # If you have time array available here, pass it; otherwise use daq_hz approximation.
    daq_hz = float(getattr(cfg, "daq_hz", 500.0))
    ignore_s = float(getattr(cfg, "touch_ignore_first_s", 0.0) or 0.0)
    margin_s = float(getattr(cfg, "touch_fit_margin_s", 0.2) or 0.2)  # stay away from contact

    i0 = int(max(0, round(ignore_s * daq_hz)))
    i1 = int(max(i0, touch_i - round(margin_s * daq_hz)))

    # Guard
    if touch_i is None or touch_i <= 10 or i1 - i0 < min_points:
        raise RuntimeError("Not enough pre-touch points after ignore/margin for support spring fit.")

    zz = z[i0:i1]
    FF = F[i0:i1]
    m = np.isfinite(zz) & np.isfinite(FF)
    if m.sum() < min_points:
        raise RuntimeError("Not enough paired pre-touch points (finite) to fit support spring.")

    zz = zz[m]
    FF = FF[m]

    # --- conditioning guard: require real z-span ---
    z_span = float(np.nanmax(zz) - np.nanmin(zz))
    z_span_min = float(getattr(cfg, "touch_fit_min_z_span_m", 50e-9))  # 50 nm default
    if not np.isfinite(z_span) or z_span < z_span_min:
        raise RuntimeError(
            f"Pre-touch fit window z-span too small ({z_span:.3e} m < {z_span_min:.3e} m). "
            "k_sup becomes ill-conditioned."
        )

    # --- optional: null in-window to reduce intercept-driven µN bias ---
    # (recommended for low-load)
    use_origin_fit = bool(getattr(cfg, "touch_fit_force_origin", True))
    if use_origin_fit:
        # center using robust medians
        z0 = float(np.nanmedian(zz))
        F0 = float(np.nanmedian(FF))
        zc = zz - z0
        Fc = FF - F0

        Szz = float(np.dot(zc, zc))
        if (not np.isfinite(Szz)) or Szz <= 0:
            raise RuntimeError("Support fit ill-conditioned (Szz ~ 0).")

        k = float(np.dot(zc, Fc) / Szz)
        b = float(F0 - k * z0)
    else:
        # your existing robust fit
        k, b = robust_fit_line(zz, FF)

    # sanity: support stiffness should be positive
    if (not np.isfinite(k)) or (k <= 0):
        if getattr(cfg, "k_sup_z_fallback", None) is not None and np.isfinite(cfg.k_sup_z_fallback):
            k = float(cfg.k_sup_z_fallback)
            b = float(getattr(cfg, "b_sup_z_fallback", 0.0))
        elif getattr(cfg, "allow_no_cal_z", False):
            k = 0.0
            b = 0.0
        else:
            raise RuntimeError("Support spring calibration failed and no fallback provided.")
        
    _, _, sigma_k, sigma_b = ols_line_cov(zz, FF)

    meta = dict(
        i0=i0, i1=i1, n_used=int(len(zz)),
        z_span_m=z_span,
        use_origin_fit=bool(use_origin_fit),
        k_sup_N_per_m=float(k),
        b_sup_N=float(b),
    )
    return float(k), float(b), float(sigma_k), float(sigma_b), meta
def fit_vertical_dynamic_support_spring(
    df: pd.DataFrame,
    cfg: Config,
    scale_to_SI: Dict[str, float],
    cal_sl_vert: Optional[slice],
) -> tuple[float, float]:
    """
    Fit vertical dynamic support spring from pre-touch vertical dyn calibration bump.
    Returns (k_sup_z_dyn_N_per_m, b_sup_z_dyn_N).
    If cal_sl_vert is None or channels missing, returns (nan, nan).
    """
    if cal_sl_vert is None:
        return (np.nan, np.nan)
    if (cfg.Fz_dyn_rms_col is None) or (cfg.Z_dyn_rms_col is None):
        return (np.nan, np.nan)
    if (cfg.Fz_dyn_rms_col not in df.columns) or (cfg.Z_dyn_rms_col not in df.columns):
        return (np.nan, np.nan)

    Fz_rms = _num(df, cfg.Fz_dyn_rms_col) * scale_to_SI[cfg.Fz_dyn_rms_col]  # N (RMS)
    Z_rms  = _num(df, cfg.Z_dyn_rms_col)  * scale_to_SI[cfg.Z_dyn_rms_col]   # m (RMS)

    Fz_pk = rms_to_peak(Fz_rms)
    Z_pk  = rms_to_peak(Z_rms)

    k, b = robust_fit_line_origin(Z_pk[cal_sl_vert], Fz_pk[cal_sl_vert])  # F ≈ k*Z + b
    return (float(k), float(b))

def fit_vertical_dynamic_coupling(
    df: pd.DataFrame,
    cfg: Config,
    scale_to_SI: Dict[str, float],
    cal_sl_vert: Optional[slice],
) -> dict:
    if cal_sl_vert is None:
        return {"kzz": np.nan, "kzx": np.nan, "b": 0.0, "ok": 0}

    need = [cfg.Fz_dyn_rms_col, cfg.Z_dyn_rms_col, cfg.X2_rms_col]
    if any((c is None) or (c not in df.columns) for c in need):
        return {"kzz": np.nan, "kzx": np.nan, "b": 0.0, "ok": 0}

    Fz = np.abs(rms_to_peak(_num(df, cfg.Fz_dyn_rms_col) * scale_to_SI[cfg.Fz_dyn_rms_col]))
    Z  = np.abs(rms_to_peak(_num(df, cfg.Z_dyn_rms_col)  * scale_to_SI[cfg.Z_dyn_rms_col]))
    X2 = np.abs(rms_to_peak(_num(df, cfg.X2_rms_col)     * scale_to_SI[cfg.X2_rms_col]))

    sl = cal_sl_vert
    y = np.asarray(Fz[sl], float)
    A = np.column_stack([np.asarray(Z[sl], float), np.asarray(X2[sl], float)])

    m = np.isfinite(y) & np.isfinite(A).all(axis=1)
    y = y[m]; A = A[m]
    if y.size < 20:
        return {"kzz": np.nan, "kzx": np.nan, "b": 0.0, "ok": 0}

    # Solve min ||A*[kzz,kzx] - y|| (no intercept)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    kzz, kzx = beta.tolist()

    # Simple fit quality
    yhat = A @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"kzz": float(kzz), "kzx": float(kzx), "b": 0.0, "r2": float(r2), "ok": 1}

# ---------------------------------------------------------------------
# 3) Hertz radius fit with adhesion + optional stiffness regularizer
#    (optimized: early-stop + no prints + faster JKR search + leverages fast jkr_P_from_h)
# ---------------------------------------------------------------------
def hertz_fit_radius_adhesion(
    h_m: np.ndarray,
    P_N: np.ndarray,
    *,
    E_star_Pa: float,
    adhesion_model: str = "auto",     # "hertz"|"dmt"|"transition"|"jkr"|"auto"
    w_J_per_m2: float = 0.0,
    sigma_rms_m: float | None = None,
    rough_model: str = "none",
    delta0_m: float = 0.3e-9,
    z0_m: float = 0.3e-9,
    mu_dmt: float = 0.1,
    mu_jkr: float = 5.0,
    min_h_m: float = 0.0,
    max_frac_of_Pmax: float = 1.0,
    min_points: int = 8,
    n_iter: int = 6,
    R0_m: float | None = None,
    Sz_meas_N_per_m: np.ndarray | None = None,
    dh_stiff_m: float = 0.25e-9,
    stiff_wt: float = 0.0,
    # ---- new knobs (safe defaults) ----
    debug: bool = False,
    early_stop_rel: float = 1e-4,   # relative R change
    jkr_use_scipy: bool = True,     # use scipy bounded minimization if available
) -> dict:
    """
    Optimizations vs old version:
      - removes unconditional prints (debug flag instead)
      - early stop when R converges and model stabilizes
      - JKR: uses bounded 1D minimization if SciPy available, else reduced grid
      - relies on mechanics.jkr_P_from_h being optimized (replace _jkr_P_from_h with fast Newton+fallback)

    Still:
      - depends only on mechanics public kernels
      - returns mask_used (useful for diagnostics + bootstrap)
    """
    h = np.asarray(h_m, float)
    P = np.asarray(P_N, float)
    if h.shape != P.shape:
        return {"ok": 0, "reason": "shape_mismatch", "n_used": 0}

    Sz = None
    if Sz_meas_N_per_m is not None:
        Sz = np.asarray(Sz_meas_N_per_m, float)
        if Sz.shape != h.shape:
            Sz = None  # ignore if misaligned

    E = float(E_star_Pa)
    if not (np.isfinite(E) and E > 0):
        return {"ok": 0, "reason": "bad_E_star", "n_used": 0}

    mask = np.isfinite(h) & np.isfinite(P) & (h >= 0)
    if Sz is not None:
        mask &= np.isfinite(Sz)

    if np.isfinite(min_h_m) and float(min_h_m) > 0:
        mask &= (h >= float(min_h_m))

    # top-load cutoff
    Pmax0 = float(np.nanmax(P[mask])) if np.any(mask) else np.nan
    if np.isfinite(Pmax0) and Pmax0 > 0 and (0 < float(max_frac_of_Pmax) < 1.0):
        mask &= (P <= float(max_frac_of_Pmax) * Pmax0)

    n_used0 = int(np.sum(mask))
    if n_used0 < int(min_points):
        return {"ok": 0, "reason": "not_enough_points", "n_used": n_used0}

    h = h[mask]
    P = P[mask]
    if Sz is not None:
        Sz = Sz[mask]

    # effective adhesion
    w_eff = w_eff_from_roughness(w_J_per_m2, sigma_rms_m, model=rough_model, delta0_m=delta0_m)
    w_eff = float(w_eff) if (np.isfinite(w_eff) and w_eff > 0) else 0.0

    # precompute x = h^(3/2) for fast Hertz-slope fit
    x = np.power(np.maximum(0.0, h), 1.5)

    # initial R guess from Hertz slope
    if R0_m is not None and np.isfinite(R0_m) and R0_m > 0:
        R = float(R0_m)
    else:
        mm = (x > 0) & np.isfinite(x) & np.isfinite(P)
        if np.sum(mm) >= 2:
            K = float(np.nanmedian(P[mm] / x[mm]))
            sqrtR = K / ((4.0 / 3.0) * E) if (np.isfinite(K) and K > 0) else np.nan
            R = float(max(1e-12, sqrtR**2)) if np.isfinite(sqrtR) else 1e-6
        else:
            R = 1e-6

    model_req = (adhesion_model or "auto").strip().lower()

    def _predict_P(hh: np.ndarray, Rm: float, model_name: str, mu_val: float) -> tuple[np.ndarray, float, float]:
        if w_eff <= 0 or model_name == "hertz":
            return hertz_load_from_h(hh, Rm, E), 0.0, 0.0

        if model_name == "dmt":
            c_pull = 2.0
            Fadh = c_pull * np.pi * Rm * w_eff
            return hertz_load_from_h(hh, Rm, E) - Fadh, c_pull, Fadh

        if model_name == "transition":
            c_pull = c_from_tabor(mu_val)
            Fadh = c_pull * np.pi * Rm * w_eff
            return hertz_load_from_h(hh, Rm, E) - Fadh, c_pull, Fadh

        if model_name == "jkr":
            c_pull = 1.5
            Fadh = c_pull * np.pi * Rm * w_eff
            # IMPORTANT: relies on mechanics.jkr_P_from_h being the optimized implementation
            return jkr_P_from_h(hh, Rm, E, w_eff, n_bisect=60), c_pull, Fadh

        return hertz_load_from_h(hh, Rm, E), 0.0, 0.0

    def _rmse(y, yhat):
        mm = np.isfinite(y) & np.isfinite(yhat)
        if np.sum(mm) < int(min_points):
            return np.inf
        d = y[mm] - yhat[mm]
        return float(np.sqrt(np.mean(d * d)))

    def _rmse_Sz(Rm: float, model_name: str, mu_val: float) -> tuple[float, int]:
        if Sz is None or not (np.isfinite(dh_stiff_m) and dh_stiff_m > 0):
            return np.nan, 0
        dh = float(dh_stiff_m)
        hm = np.maximum(0.0, h - dh)
        hp = h + dh
        Pp, _, _ = _predict_P(hp, Rm, model_name, mu_val)
        Pm, _, _ = _predict_P(hm, Rm, model_name, mu_val)
        Sz_pred = (Pp - Pm) / (2.0 * dh)
        mm = np.isfinite(Sz) & np.isfinite(Sz_pred) & (Sz > 0) & (Sz_pred > 0)
        n_ok = int(np.sum(mm))
        if n_ok < int(min_points):
            return np.inf, n_ok
        d = Sz[mm] - Sz_pred[mm]
        return float(np.sqrt(np.mean(d * d))), n_ok

    def _fit_R_from_Peff(Peff: np.ndarray) -> float:
        mm = np.isfinite(Peff) & np.isfinite(x) & (x > 0) & (Peff > 0)
        if np.sum(mm) < 2:
            return np.nan
        K = float(np.nanmedian(Peff[mm] / x[mm]))
        if not (np.isfinite(K) and K > 0):
            return np.nan
        sqrtR = K / ((4.0 / 3.0) * E)
        return float(max(1e-12, sqrtR**2))

    chosen_model = "hertz"
    mu_val = np.nan

    prev_R = None
    prev_model = None

    for it in range(int(max(1, n_iter))):
        # model selection
        if model_req == "auto" and w_eff > 0:
            mu_val = tabor_mu(R, w_eff, E, z0_m)
            chosen_model = auto_model_from_mu(mu_val, mu_dmt=mu_dmt, mu_jkr=mu_jkr)
        else:
            chosen_model = model_req
            mu_val = tabor_mu(R, w_eff, E, z0_m) if (w_eff > 0) else np.nan

        if debug:
            print(f"[hertz_fit_radius_adhesion] it={it+1} model={chosen_model} R={R:.4e} w_eff={w_eff:.3e}")

        # early stop (R stable + model stable)
        if prev_R is not None:
            rel = abs(R - prev_R) / max(abs(R), 1e-30)
            if (rel < float(early_stop_rel)) and (chosen_model == prev_model):
                break
        prev_R = float(R)
        prev_model = str(chosen_model)

        # Hertz
        if w_eff <= 0 or chosen_model == "hertz":
            R_new = _fit_R_from_Peff(P)
            if np.isfinite(R_new):
                R = 0.5 * R + 0.5 * R_new
            continue

        # DMT/transition
        if chosen_model in ("dmt", "transition"):
            c_pull = 2.0 if chosen_model == "dmt" else c_from_tabor(mu_val)
            Fadh = c_pull * np.pi * R * w_eff
            Peff = P + Fadh
            R_new = _fit_R_from_Peff(Peff)
            if np.isfinite(R_new):
                R = 0.5 * R + 0.5 * R_new
            continue

        # JKR
        if chosen_model == "jkr":
            use_stiff = (Sz is not None) and (np.isfinite(stiff_wt) and float(stiff_wt) > 0)

            def obj_R(Rc: float) -> float:
                Rc = float(max(Rc, 1e-12))
                P_pred, _, _ = _predict_P(h, Rc, "jkr", mu_val)
                obj = _rmse(P, P_pred)
                if use_stiff:
                    rS, _ = _rmse_Sz(Rc, "jkr", mu_val)
                    obj = obj + float(stiff_wt) * rS
                return float(obj)

            R_center = float(max(R, 1e-12))
            lo = R_center / 10.0
            hi = R_center * 10.0

            best_R = R_center
            best_obj = np.inf

            used_scipy = False
            if jkr_use_scipy:
                try:
                    from scipy.optimize import minimize_scalar
                    res = minimize_scalar(obj_R, bounds=(lo, hi), method="bounded")
                    if res.success and np.isfinite(res.x):
                        best_R = float(res.x)
                        best_obj = float(res.fun) if np.isfinite(res.fun) else obj_R(best_R)
                        used_scipy = True
                except Exception:
                    used_scipy = False

            if not used_scipy:
                # Reduced grid vs old (17 + 13 instead of 31 + 25)
                grid = R_center * np.logspace(-0.8, 0.8, 17)
                for Rc in grid:
                    v = obj_R(Rc)
                    if v < best_obj:
                        best_obj = v
                        best_R = float(Rc)

                grid2 = best_R * np.logspace(-0.25, 0.25, 13)
                for Rc in grid2:
                    v = obj_R(Rc)
                    if v < best_obj:
                        best_obj = v
                        best_R = float(Rc)

            R = 0.5 * R + 0.5 * best_R
            continue

        # fallback
        R_new = _fit_R_from_Peff(P)
        if np.isfinite(R_new):
            R = 0.5 * R + 0.5 * R_new

    # final model
    if model_req == "auto" and w_eff > 0:
        mu_val = tabor_mu(R, w_eff, E, z0_m)
        chosen_model = auto_model_from_mu(mu_val, mu_dmt=mu_dmt, mu_jkr=mu_jkr)
    else:
        chosen_model = model_req

    P_pred, c_pull, Fadh_N = _predict_P(h, R, chosen_model, mu_val)
    rmse_P = _rmse(P, P_pred)

    rmse_Sz_val = np.nan
    n_Sz_used = 0
    if Sz is not None and float(stiff_wt) > 0:
        rmse_Sz_val, n_Sz_used = _rmse_Sz(R, chosen_model, mu_val)

    rmse_combined = rmse_P + (
        float(stiff_wt) * rmse_Sz_val
        if (np.isfinite(rmse_Sz_val) and float(stiff_wt) > 0)
        else 0.0
    )

    return {
        "ok": 1 if (np.isfinite(R) and R > 0 and np.isfinite(rmse_P)) else 0,
        "reason": "",
        "mask_used": mask,

        "E_star_Pa": float(E),
        "R_eff_m": float(R),

        "rmse_N": float(rmse_P),
        "rmse_mN": float(rmse_P * 1e3) if np.isfinite(rmse_P) else np.nan,
        "n_used": int(h.size),

        "rmse_Sz_N_per_m": float(rmse_Sz_val) if np.isfinite(rmse_Sz_val) else np.nan,
        "n_Sz_used": int(n_Sz_used),
        "rmse_combined": float(rmse_combined) if np.isfinite(rmse_combined) else np.nan,

        "adhesion_model_requested": str(model_req),
        "adhesion_model_used": str(chosen_model),

        "rough_model": str(rough_model),
        "w0_J_per_m2": float(w_J_per_m2),
        "w_eff_J_per_m2": float(w_eff),

        "tabor_mu": float(mu_val) if np.isfinite(mu_val) else np.nan,
        "c_pull": float(c_pull),
        "Fadh_N": float(Fadh_N),

        "z0_m": float(z0_m),
        "mu_dmt": float(mu_dmt),
        "mu_jkr": float(mu_jkr),
        "stiff_wt": float(stiff_wt),
        "dh_stiff_m": float(dh_stiff_m),
    }

def ols_line_cov(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """
    Ordinary least squares y = k*x + b
    Returns: (k, b, sigma_k, sigma_b)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan

    X = np.column_stack([x, np.ones_like(x)])
    # beta = (X^T X)^-1 X^T y
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    k, b = float(beta[0]), float(beta[1])

    resid = y - (k*x + b)
    dof = max(1, n - 2)
    s2 = float(np.sum(resid**2) / dof)

    XtX_inv = np.linalg.inv(X.T @ X)
    var_k = s2 * XtX_inv[0, 0]
    var_b = s2 * XtX_inv[1, 1]
    return k, b, float(np.sqrt(max(var_k, 0.0))), float(np.sqrt(max(var_b, 0.0)))

def fit_flat_end_stiffness(
    P_N: np.ndarray,
    S_N_per_m: np.ndarray,
    *,
    E_star_Pa: float,
    P_min_N: float | None = None,
    P_max_N: float | None = None,
    robust: bool = True,
    n_iter: int = 50,
    clip_sigma: float = 3.0,
    min_points: int = 30,
) -> dict:
    """
    Fit the baseline+Hertz-stiffness scaling model for normal CSM stiffness vs load:

        S(P) = S0 + C * P^(1/3)

    Contact-mechanics meaning (diagnostic, not "unique truth"):
    ----------------------------------------------------------
    - In elastic Hertzian sphere contact, contact radius a scales as P^(1/3), and stiffness
      scales as S = 2 E* a, so S ∝ P^(1/3).
    - The additive term S0 captures an *apparent baseline stiffness* that behaves like a
      flat punch of effective radius a_flat = S0/(2E*), or residual parallel stiffness.

      a_flat is an *effective* descriptor: it may include real flatness + instrumental leakage.

    Derived parameters:
    -------------------
    - a_flat_m = S0 / (2 E*)
    - R_eff_m  = C^3 / (6 E*^2)     (equivalent Hertz radius that would yield the same C)

    Robustness:
    -----------
    If robust=True, iteratively reweighted least squares (Huber) + optional sigma-clip.
    This helps reject a few points corrupted by drift or onset of plasticity.

    Returns
    -------
    dict:
      ok (int 0/1), reason, S0, C, a_flat_m, a_flat_um, R_eff_m, rmse, R2, n_used,
      and diagnostics: weights, window_mask, cov, stderr.
    """
    # ---- Basic checks ----
    P = np.asarray(P_N, float)
    S = np.asarray(S_N_per_m, float)
    if P.shape != S.shape:
        return {"ok": 0, "reason": "shape_mismatch", "n_used": 0,
                "S0": np.nan, "C": np.nan, "a_flat_m": np.nan, "R_eff_m": np.nan}

    E = float(E_star_Pa)
    if not (np.isfinite(E) and E > 0):
        return {"ok": 0, "reason": "bad_E_star", "n_used": 0,
                "S0": np.nan, "C": np.nan, "a_flat_m": np.nan, "R_eff_m": np.nan}

    # ---- Windowing mask ----
    win = np.isfinite(P) & np.isfinite(S) & (P > 0) & (S > 0)
    if P_min_N is not None:
        win &= (P >= float(P_min_N))
    if P_max_N is not None:
        win &= (P <= float(P_max_N))

    idx = np.where(win)[0]
    if idx.size < int(min_points):
        return {
            "ok": 0, "reason": "not_enough_points",
            "n_used": int(idx.size),
            "S0": np.nan, "C": np.nan,
            "a_flat_m": np.nan, "a_flat_um": np.nan,
            "R_eff_m": np.nan,
            "rmse": np.nan, "R2": np.nan,
            "window_mask": win,
        }

    # ---- Linear regression in transformed variable x = P^(1/3) ----
    x = np.cbrt(P[idx])
    y = S[idx]
    X = np.column_stack([np.ones_like(x), x])

    # Weighted least squares solver
    def _wls_nonneg(beta_w: np.ndarray):
        """
        Pure-NumPy weighted least squares with nonnegativity constraints:
            beta = [S0, C] with S0 >= 0, C >= 0
        Minimizes: sum_i w_i * (y_i - (S0 + C*x_i))^2

        Drop-in replacement for _wls(): returns (beta, yhat, resid, XtWX)

        Safety tweaks:
        - Handles all-zero weights
        - Treats ill-conditioned XtWX as singular (cond > 1e12)
        - Uses tiny eps thresholds to avoid 0-division
        """
        w = np.asarray(beta_w, float)
        w = np.maximum(w, 0.0)

        # If all weights are zero, avoid crashes
        if not np.any(w > 0):
            beta = np.array([np.nan, np.nan], float)
            yhat = X @ beta
            resid = y - yhat
            XtWX = np.full((2, 2), np.nan, float)
            return beta, yhat, resid, XtWX

        # Precompute normal-equation pieces
        Wcol = w[:, None]
        XtWX = X.T @ (Wcol * X)          # 2x2
        XtWy = X.T @ (w * y)             # 2,

        # Helper: weighted SSE
        def sse(beta):
            r = y - (X @ beta)
            return float(np.sum(w * r * r))

        # ---- Candidate 1: unconstrained WLS ----
        beta_uc = None
        try:
            # treat very ill-conditioned as singular
            if not np.all(np.isfinite(XtWX)) or np.linalg.cond(XtWX) > 1e12:
                raise np.linalg.LinAlgError("ill_conditioned")
            beta_uc = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta_uc = None

        # If feasible, it's optimal (convex QP)
        if beta_uc is not None and np.all(np.isfinite(beta_uc)):
            if beta_uc[0] >= 0.0 and beta_uc[1] >= 0.0:
                beta = beta_uc
                yhat = X @ beta
                resid = y - yhat
                return beta, yhat, resid, XtWX

        candidates = []

        # ---- Candidate 2: boundary S0 = 0, fit C >= 0 ----
        # minimize sum w*(y - C*x)^2 => C = (x^T W y)/(x^T W x)
        x = X[:, 1]
        denom = float(np.sum(w * x * x))
        if denom > 1e-30:
            C_hat = float(np.sum(w * x * y) / denom)
        else:
            C_hat = 0.0
        C_hat = max(0.0, C_hat)
        candidates.append(np.array([0.0, C_hat], float))

        # ---- Candidate 3: boundary C = 0, fit S0 >= 0 ----
        # minimize sum w*(y - S0)^2 => S0 = weighted mean
        wsum = float(np.sum(w))
        if wsum > 1e-30:
            S0_hat = float(np.sum(w * y) / wsum)
        else:
            S0_hat = 0.0
        S0_hat = max(0.0, S0_hat)
        candidates.append(np.array([S0_hat, 0.0], float))

        # ---- Candidate 4: corner (0,0) ----
        candidates.append(np.array([0.0, 0.0], float))

        # Pick best feasible candidate by weighted SSE
        best_beta = None
        best_val = np.inf
        for b in candidates:
            if np.all(np.isfinite(b)) and (b[0] >= 0.0) and (b[1] >= 0.0):
                val = sse(b)
                if val < best_val:
                    best_val = val
                    best_beta = b

        beta = best_beta if best_beta is not None else np.array([np.nan, np.nan], float)
        yhat = X @ beta
        resid = y - yhat
        return beta, yhat, resid, XtWX


    # ---- Robust loop (Huber + optional sigma-clip) ----
    w = np.ones_like(y)
    beta = None
    XtWX = None
    for _ in range(int(max(1, n_iter if robust else 1))):
        beta, yhat, resid, XtWX = _wls_nonneg(w)

        if not robust:
            break

        # robust scale via MAD
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        sigma = 1.4826 * mad if mad > 0 else (float(np.std(resid)) if resid.size > 1 else np.nan)
        if not (np.isfinite(sigma) and sigma > 0):
            break

        r = resid / sigma

        # (a) hard sigma clip -> set weight 0 beyond clip_sigma
        keep = np.abs(r) <= float(clip_sigma)

        # don't nuke too many points
        if keep.sum() < max(int(min_points), int(0.5 * y.size)):
            keep = np.ones_like(keep, dtype=bool)

        # (b) Huber weights
        k = 1.345
        w_new = np.ones_like(y)
        big = np.abs(r) > k
        w_new[big] = k / np.abs(r[big])

        # apply hard clipping too
        w_new[~keep] = 0.0

        # stop if converged
        if np.allclose(w, w_new, atol=1e-6, rtol=0):
            w = w_new
            break

        w = w_new
        if np.sum(w > 0) < int(min_points):
            break

    if beta is None or not np.all(np.isfinite(beta)):
        return {"ok": 0, "reason": "solve_failed", "n_used": int(idx.size),
                "S0": np.nan, "C": np.nan, "a_flat_m": np.nan, "R_eff_m": np.nan,
                "window_mask": win}

    S0 = float(beta[0])
    C  = float(beta[1])

    # ---- Fit quality on used points (w>0) ----
    use = w > 0
    n_used = int(np.sum(use))
    if n_used >= 3:
        y_use = y[use]
        X_use = X[use, :]
        yhat_use = X_use @ beta
        resid_use = y_use - yhat_use

        ss_res = float(np.sum(resid_use**2))
        ss_tot = float(np.sum((y_use - float(np.mean(y_use)))**2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(ss_res / max(1, n_used - 2)))
    else:
        R2, rmse = np.nan, np.nan

    # ---- Derived contact-mechanics quantities ----
    a_flat_m = S0 / (2.0 * E) if np.isfinite(S0) else np.nan
    R_eff_m  = (C**3) / (6.0 * E**2) if (np.isfinite(C) and C > 0) else np.nan

    # ---- Covariance / stderr estimate (simple, on used points) ----
    cov = None
    stderr = None
    try:
        # unweighted covariance on used points (treat used points as final subset)
        Xw = X[use, :]
        yw = y[use]
        # ordinary LS on used points
        XtX = Xw.T @ Xw
        beta_u = np.linalg.solve(XtX, Xw.T @ yw)
        resid_u = yw - (Xw @ beta_u)
        dof = max(1, yw.size - 2)
        s2 = float(np.sum(resid_u**2) / dof)
        cov = np.linalg.inv(XtX) * s2

        se_S0 = float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.nan
        se_C  = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else np.nan
        se_a  = se_S0 / (2.0 * E) if np.isfinite(se_S0) else np.nan
        stderr = {"S0": se_S0, "C": se_C, "a_flat_m": se_a}
    except Exception:
        cov = None
        stderr = None

    return {
        "ok": 1,
        "reason": "",
        "S0": S0,
        "C": C,
        "a_flat_m": float(a_flat_m) if np.isfinite(a_flat_m) else np.nan,
        "a_flat_um": float(a_flat_m * 1e6) if np.isfinite(a_flat_m) else np.nan,
        "R_eff_m": float(R_eff_m) if np.isfinite(R_eff_m) else np.nan,
        "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
        "R2": float(R2) if np.isfinite(R2) else np.nan,
        "n_used": int(n_used),
        "window_mask": win,     # mask applied before robust reweighting
        "weights": w,           # weights on windowed data (length = idx.size)
        "cov": cov,
        "stderr": stderr,
        "notes": "Fit S(P)=S0 + C*P^(1/3) on positive-load/stiffness window.",
    }

def bootstrap_flat_end_stiffness_uncertainty(
    P_N: np.ndarray,
    S_N_per_m: np.ndarray,
    *,
    fit_fn=None,
    fit_kwargs: dict | None = None,
    n_boot: int = 400,
    seed: int | None = 0,
    keep_frac: float = 1.0,
    min_success: int = 50,
    block_size: int | None = 10,
) -> dict:
    """
    Bootstrap uncertainty for the flat-end stiffness fit S(P)=S0 + C P^(1/3).

    Why bootstrap?
    --------------
    Stiffness vs load curves often have correlated noise (sweeps), drift, and occasional
    outliers. Bootstrap provides robust uncertainty on S0, C, a_flat, and derived R_eff.

    Resampling strategy:
    --------------------
    - IID bootstrap: resample points with replacement (block_size=None or <=1)
    - Block bootstrap: resample contiguous blocks to preserve correlation structure (recommended)

    Returns:
    --------
    dict with:
      ok, reason, n_used, n_boot_ok,
      summaries for S0, C, a_flat_m, R_eff_m (+ CI95),
      samples for propagation.
    """
    fit_kwargs = dict(fit_kwargs or {})

    # ---- Filter kwargs to avoid "unexpected kwarg" errors ----
    try:
        sig = inspect.signature(fit_fn)
        accepted = set(sig.parameters.keys())
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in accepted}
    except Exception:
        pass

    P = np.asarray(P_N, float)
    S = np.asarray(S_N_per_m, float)
    if P.shape != S.shape:
        return {"ok": 0, "reason": "shape_mismatch", "n_boot_ok": 0, "n_used": 0}

    # base validity; the fitter also windows, but we remove NaNs and non-physical here
    base = np.isfinite(P) & np.isfinite(S) & (P > 0) & (S > 0)
    idx0 = np.where(base)[0]
    n = int(idx0.size)

    min_points = int(fit_kwargs.get("min_points", 30))
    if n < max(10, min_points):
        return {"ok": 0, "reason": "not_enough_valid_points", "n_used": n, "n_boot_ok": 0}

    # sample size per draw
    msize = int(max(min_points, round(float(keep_frac) * n)))
    msize = min(msize, n)

    rng = np.random.default_rng(seed)

    # ---- block bootstrap indices on idx0 ----
    def _draw_indices():
        if (block_size is None) or (int(block_size) <= 1):
            # IID within valid indices
            return rng.choice(idx0, size=msize, replace=True)

        B = int(max(1, block_size))
        nblocks = int(np.ceil(msize / B))

        # pick block starts in the *index array* coordinates
        # (this preserves the ordering of the original valid points)
        starts = rng.integers(0, max(1, n - B + 1), size=nblocks)
        chunks = [idx0[s:s + B] for s in starts]
        draw = np.concatenate(chunks)
        return draw[:msize]

    # ---- collect samples ----
    S0_s, C_s, a_s, R_s = [], [], [], []
    rmse_s, R2_s, n_used_s = [], [], []

    for _ in range(int(max(10, n_boot))):
        draw = _draw_indices()
        res = None
        try:
            res = fit_fn(P[draw], S[draw], **fit_kwargs)
        except Exception:
            continue

        if int(res.get("ok", 0)) != 1:
            continue

        S0 = res.get("S0", np.nan)
        C  = res.get("C", np.nan)
        a  = res.get("a_flat_m", np.nan)
        Rm = res.get("R_eff_m", np.nan)

        if not (np.isfinite(S0) and np.isfinite(C) and np.isfinite(a) and np.isfinite(Rm)):
            continue
        if not (a >= 0 and Rm > 0):
            continue

        S0_s.append(float(S0))
        C_s.append(float(C))
        a_s.append(float(a))
        R_s.append(float(Rm))
        rmse_s.append(float(res.get("rmse", np.nan)))
        R2_s.append(float(res.get("R2", np.nan)))
        n_used_s.append(int(res.get("n_used", np.nan)) if np.isfinite(res.get("n_used", np.nan)) else msize)

    n_ok = int(len(R_s))
    if n_ok < int(min_success):
        return {
            "ok": 0,
            "reason": "too_few_successful_bootstraps",
            "n_used": n,
            "n_boot_ok": n_ok,
            "min_success": int(min_success),
        }

    def _summ(arr: list[float]):
        x = np.asarray(arr, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"median": np.nan, "std": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan}
        med = float(np.median(x))
        std = float(np.std(x, ddof=1)) if x.size >= 2 else np.nan
        lo, hi = np.percentile(x, [2.5, 97.5])
        return {"median": med, "std": std, "ci95_lo": float(lo), "ci95_hi": float(hi)}

    S0_s = np.asarray(S0_s, float)
    C_s  = np.asarray(C_s, float)
    a_s  = np.asarray(a_s, float)
    R_s  = np.asarray(R_s, float)

    return {
        "ok": 1,
        "reason": "",
        "n_used": n,
        "n_boot_ok": n_ok,
        "n_boot": int(n_boot),
        "keep_frac": float(keep_frac),
        "block_size": (int(block_size) if block_size is not None else None),

        # summaries (SI)
        "S0_N_per_m": _summ(S0_s.tolist()),
        "C": _summ(C_s.tolist()),
        "a_flat_m": _summ(a_s.tolist()),
        "R_eff_m": _summ(R_s.tolist()),

        # convenience summaries in um
        "a_flat_um": _summ((a_s * 1e6).tolist()),
        "R_eff_um": _summ((R_s * 1e6).tolist()),

        # diagnostics
        "rmse_N_per_m": _summ(np.asarray(rmse_s, float).tolist()),
        "R2": _summ(np.asarray(R2_s, float).tolist()),

        # raw samples for propagation
        "samples": {
            "S0": S0_s,
            "C": C_s,
            "a_flat_m": a_s,
            "R_eff_m": R_s,
        }
    }

def bootstrap_hertz_radius_uncertainty(
    h_m: np.ndarray,
    P_N: np.ndarray,
    *,
    fit_fn,
    fit_args: tuple = (),
    fit_kwargs: dict | None = None,
    Sz_meas_N_per_m: np.ndarray | None = None,
    n_boot: int = 300,
    seed: int | None = 0,
    keep_frac: float = 1.0,
    min_success: int = 30,
    block_size: int | None = 10,
    collect_fields: tuple[str, ...] = ("w_eff_J_per_m2", "Fadh_N", "z0_m"),
    # ---- anti-stall controls ----
    max_attempts_factor: float = 8.0,     # max attempts = factor * n_boot
    early_stop: bool = True,              # stop once we have n_boot successes
    # ---- optional: improve success rate ----
    ensure_high_load_frac: float = 0.10,  # force at least this fraction from top-load tail (0 disables)
    high_load_quantile: float = 0.85,     # define "high load" tail
) -> dict:
    fit_kwargs = dict(fit_kwargs or {})

    # Filter kwargs to fit_fn signature
    try:
        sig = inspect.signature(fit_fn)
        accepted = set(sig.parameters.keys())
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in accepted}
    except Exception:
        pass

    h = np.asarray(h_m, float)
    P = np.asarray(P_N, float)
    if h.shape != P.shape:
        return {"ok": 0, "reason": "shape_mismatch", "n_used": 0, "n_boot_ok": 0,
                "R_eff_std_m": np.nan, "R_eff_ci95_lo_m": np.nan, "R_eff_ci95_hi_m": np.nan}

    Sz = None
    if Sz_meas_N_per_m is not None:
        Sz = np.asarray(Sz_meas_N_per_m, float)
        if Sz.shape != h.shape:
            Sz = None

    m = np.isfinite(h) & np.isfinite(P) & (h >= 0)
    if Sz is not None:
        m &= np.isfinite(Sz)
    h = h[m]; P = P[m]
    if Sz is not None:
        Sz = Sz[m]

    n = int(h.size)
    min_points = int(fit_kwargs.get("min_points", 8))
    if n < max(8, min_points):
        return {"ok": 0, "reason": "not_enough_points_for_bootstrap",
                "n_used": n, "n_boot_ok": 0,
                "R_eff_std_m": np.nan, "R_eff_ci95_lo_m": np.nan, "R_eff_ci95_hi_m": np.nan}

    # clamp keep_frac
    kf = float(keep_frac)
    if not np.isfinite(kf):
        kf = 1.0
    kf = max(1e-6, min(1.0, kf))

    # sample size per draw
    msize = int(round(kf * n))
    msize = int(max(min_points, min(msize, n)))

    rng = np.random.default_rng(seed)

    # precompute high-load indices (optional)
    hi_idx = np.array([], dtype=int)
    if np.isfinite(ensure_high_load_frac) and float(ensure_high_load_frac) > 0:
        q = float(high_load_quantile)
        q = min(0.99, max(0.5, q))
        thr = np.quantile(P, q)
        hi_idx = np.where(P >= thr)[0]
        # if tail is tiny, still ok; we’ll just sample whatever exists

    def _draw_indices() -> np.ndarray:
        # base draw: iid or block
        if (block_size is None) or (int(block_size) <= 1):
            idx = rng.integers(0, n, size=msize)
        else:
            B = int(max(1, block_size))
            nblocks = int((msize + B - 1) // B)
            starts = rng.integers(0, max(1, n - B + 1), size=nblocks)
            offs = np.arange(B, dtype=int)
            idx = (starts[:, None] + offs[None, :]).ravel()[:msize]

        # force some high-load points to prevent low-range pathological subsets
        if hi_idx.size > 0:
            frac = float(ensure_high_load_frac)
            frac = min(1.0, max(0.0, frac))
            k_hi = int(round(frac * msize))
            if k_hi >= 1 and hi_idx.size >= 1:
                replace = (hi_idx.size < k_hi)
                idx_hi = rng.choice(hi_idx, size=k_hi, replace=replace)
                # overwrite first k_hi entries (simple + fast)
                idx[:k_hi] = idx_hi

        return idx

    def _call_fit(hb: np.ndarray, Pb: np.ndarray, Szb: np.ndarray | None):
        if Szb is not None:
            return fit_fn(hb, Pb, *fit_args, Sz_meas_N_per_m=Szb, **fit_kwargs)
        return fit_fn(hb, Pb, *fit_args, **fit_kwargs)

    n_boot = int(max(1, n_boot))
    min_success = int(max(1, min_success))

    # cap attempts to avoid “stuck forever”
    max_attempts = int(max(50, np.ceil(float(max_attempts_factor) * n_boot)))

    R_list: list[float] = []
    model_used: list[str] = []
    extra_lists: dict[str, list[float]] = {k: [] for k in collect_fields}

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        idx = _draw_indices()
        hb = h[idx]; Pb = P[idx]
        Szb = Sz[idx] if Sz is not None else None

        try:
            res = _call_fit(hb, Pb, Szb)
        except Exception:
            continue

        if int(res.get("ok", 0)) != 1:
            continue

        Rb = res.get("R_eff_m", np.nan)
        if not (np.isfinite(Rb) and Rb > 0):
            continue

        R_list.append(float(Rb))
        model_used.append(str(res.get("adhesion_model_used", res.get("adhesion_model", ""))))

        for k in collect_fields:
            vb = res.get(k, np.nan)
            extra_lists[k].append(float(vb) if np.isfinite(vb) else np.nan)

        if early_stop and (len(R_list) >= n_boot):
            break

    R = np.asarray(R_list, float)
    n_ok = int(R.size)

    if n_ok < min_success:
        return {"ok": 0, "reason": "too_few_successful_bootstraps",
                "n_used": n, "n_boot_ok": n_ok, "n_boot": n_boot, "n_attempts": attempts,
                "R_eff_std_m": np.nan, "R_eff_ci95_lo_m": np.nan, "R_eff_ci95_hi_m": np.nan}

    R_std = float(np.std(R, ddof=1)) if n_ok >= 2 else np.nan
    lo, hi = np.percentile(R, [2.5, 97.5])

    out = {
        "ok": 1, "reason": "",
        "n_used": n,
        "n_boot_ok": n_ok,
        "n_boot": int(n_boot),
        "n_attempts": int(attempts),
        "keep_frac": float(kf),
        "block_size": (int(block_size) if (block_size is not None) else None),

        "R_eff_std_m": float(R_std),
        "R_eff_ci95_lo_m": float(lo),
        "R_eff_ci95_hi_m": float(hi),

        "samples": {
            "R_eff_m": R,
            "adhesion_model_used": np.asarray(model_used, dtype=str),
            **{k: np.asarray(v, float) for k, v in extra_lists.items()},
        }
    }

    if model_used:
        vals, counts = np.unique(np.asarray(model_used, dtype=str), return_counts=True)
        j = int(np.argmax(counts))
        out["adhesion_model_used_mode"] = str(vals[j])
        out["adhesion_model_used_frac"] = float(counts[j] / np.sum(counts))

    return out

def filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """Drop kwargs that `fn` does not accept (prevents TypeError on unexpected kwargs)."""
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        # if signature inspection fails, just return as-is
        return dict(kwargs)