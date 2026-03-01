# signal.py
from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from scipy.signal import savgol_filter


from .plotting import pick_indices_from_plot
from .cycle_types import CycleBounds
from .math_utils import (
    _num, median_dt, safe_nanmax, contiguous_regions,
)

def _touch_by_boolean_run(above: np.ndarray, nmin: int) -> int:
    """Return first index where `above` is True for at least nmin consecutive samples."""
    idxs = np.where(above)[0]
    for i in idxs:
        j = i + nmin
        if j <= len(above) and np.all(above[i:j]):
            return int(i)
    raise RuntimeError("Touch not found by sustained-threshold criterion.")


def _rolling_slope_and_sigma(
    F: np.ndarray,
    z: np.ndarray,
    *,
    win: int,
    sigma_F: float,
    sigma_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling least-squares slope k = dF/dz and its 1-sigma estimate per window.
    (σk is optional; we keep it because it can still be useful for debugging.)
    """
    n = len(F)
    k = np.full(n, np.nan, dtype=float)
    sigk = np.full(n, np.nan, dtype=float)

    if win < 5 or win % 2 == 0 or n < win:
        return k, sigk

    h = win // 2
    for i in range(h, n - h):
        zz = z[i - h : i + h + 1]
        FF = F[i - h : i + h + 1]
        m = np.isfinite(zz) & np.isfinite(FF)
        if m.sum() < max(5, win // 2):
            continue

        zz = zz[m]
        FF = FF[m]

        z0 = zz - np.mean(zz)
        F0 = FF - np.mean(FF)

        Szz = float(np.dot(z0, z0))
        if not np.isfinite(Szz) or Szz < 1e-24:
            continue

        ki = float(np.dot(z0, F0) / Szz)
        k[i] = ki

        sigma_F_eff = float(np.sqrt(max(sigma_F, 0.0) ** 2 + (ki * max(sigma_z, 0.0)) ** 2))
        sigk[i] = sigma_F_eff / np.sqrt(Szz)

    return k, sigk


def _robust_sigma_mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)  # ~1-sigma for Gaussian


def estimate_offset_pre_touch(
    t: np.ndarray,
    x: np.ndarray,
    touch_i: int,
    *,
    daq_hz: float = 500.0,
    seconds: float = 2.0,
    margin_s: float = 0.5,
    min_points: int = 300,
    max_points: int = 5000,
    detrend: bool = True,
) -> Tuple[float, float, float, int]:
    """
    Estimate pre-touch offset x0 and uncertainties using ONLY pre-touch samples.

    Uses last `seconds` before (touch_i - margin).
    Returns:
      x0            : median in window
      sigma_x_point : MAD-based 1-sigma of (detrended) residuals (point noise)
      sigma_x0      : offset uncertainty ~ sigma_x_point / sqrt(n_used)
      n_used        : # finite samples used
    """
    try:
        t = np.asarray(t, float)
        x = np.asarray(x, float)
        if touch_i is None or touch_i <= 10:
            return (np.nan, np.nan, np.nan, 0)

        margin = int(max(0, round(margin_s * daq_hz)))
        end = int(max(0, touch_i - margin))
        if end < min_points:
            return (np.nan, np.nan, np.nan, 0)

        nb = int(round(seconds * daq_hz))
        nb = int(np.clip(nb, min_points, min(max_points, end)))
        start = int(end - nb)

        tt = t[start:end]
        xx = x[start:end]
        m = np.isfinite(tt) & np.isfinite(xx)
        if m.sum() < min_points:
            return (np.nan, np.nan, np.nan, int(m.sum()))

        tt = tt[m]
        xx = xx[m]
        n_used = int(xx.size)

        x0 = float(np.median(xx))

        if detrend and n_used >= 50:
            tc = tt - np.mean(tt)
            A = np.vstack([tc, np.ones_like(tc)]).T
            slope, intercept = np.linalg.lstsq(A, xx, rcond=None)[0]
            resid = xx - (slope * tc + intercept)
        else:
            resid = xx - x0

        sigma_x_point = _robust_sigma_mad(resid)
        sigma_x0 = float(sigma_x_point / np.sqrt(max(n_used, 1))) if np.isfinite(sigma_x_point) else np.nan
        return (x0, float(sigma_x_point), sigma_x0, n_used)
    except Exception:
        return (np.nan, np.nan, np.nan, 0)

def _k_from_time_derivatives(
    t: np.ndarray,
    F: np.ndarray,
    z: np.ndarray,
    *,
    win_s: float,
    poly: int = 3,
    dzdt_min_frac: float = 0.15,
    require_dzdt_positive: bool = False,
    eps: float = 1e-24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    k_inst(t) ≈ (dF/dt)/(dz/dt) with SG smoothing/derivatives.
    Returns: k_inst, dzdt, dFdt, meta
    """
    t = np.asarray(t, float)
    F = np.asarray(F, float)
    z = np.asarray(z, float)

    dt = np.nanmedian(np.diff(t))
    dt = float(dt) if (np.isfinite(dt) and dt > 0) else 1.0

    win = int(max(7, round(float(win_s) / dt)))
    if win % 2 == 0:
        win += 1
    if win <= poly + 2:
        win = poly + 3
        if win % 2 == 0:
            win += 1

    # Smooth first (mostly for debugging/plotting if needed)
    F_s = savgol_filter(F, win, poly, mode="interp")
    z_s = savgol_filter(z, win, poly, mode="interp")
    dFdt = savgol_filter(F, win, poly, deriv=1, delta=dt, mode="interp")
    dzdt = savgol_filter(z, win, poly, deriv=1, delta=dt, mode="interp")

    # Robust scale of |dzdt|
    a = np.abs(dzdt[np.isfinite(dzdt)])
    if a.size >= 20:
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med))
        dz_scale = 1.4826 * mad
    else:
        dz_scale = np.nan

    dz_thr = (float(dzdt_min_frac) * dz_scale) if (np.isfinite(dz_scale) and dz_scale > 0) else 0.0
    thr = max(dz_thr, eps)

    good = np.isfinite(dFdt) & np.isfinite(dzdt) & (np.abs(dzdt) > thr)
    if require_dzdt_positive:
        good &= (dzdt > 0)

    k_inst = np.full_like(t, np.nan, dtype=float)
    k_inst[good] = dFdt[good] / dzdt[good]

    meta = dict(
        dt_s=dt,
        win=int(win),
        poly=int(poly),
        dz_scale=float(dz_scale) if np.isfinite(dz_scale) else np.nan,
        dz_thr=float(dz_thr),
        good_frac=float(np.mean(good)),
        require_dzdt_positive=bool(require_dzdt_positive),
    )
    return k_inst, dzdt, dFdt, meta

def _find_first_motion_window(
    z: np.ndarray,
    *,
    i0: int,
    win: int,
    step: int,
    min_z_span: float,
    min_finite: int,
) -> Tuple[int, int, Dict[str, Any]] | None:
    """
    Scan forward from i0 to find first window [s:e) where z-span >= min_z_span.
    Returns (s, e, diag) or None.
    """
    z = np.asarray(z, float)
    n = len(z)
    best = None

    for s in range(int(i0), max(int(i0), n - win), int(step)):
        e = s + win
        zz = z[s:e]
        m = np.isfinite(zz)
        nfin = int(m.sum())
        if nfin < min_finite:
            continue
        zmin = float(np.nanmin(zz[m]))
        zmax = float(np.nanmax(zz[m]))
        span = zmax - zmin
        zstd = float(np.nanstd(zz[m]))
        diag = {"s": int(s), "e": int(e), "nfin": nfin, "z_span_m": span, "z_std_m": zstd}
        if np.isfinite(span) and span >= float(min_z_span):
            return int(s), int(e), diag
        # keep best candidate for debugging
        if (best is None) or (span > best[2]["z_span_m"]):
            best = (int(s), int(e), diag)

    # if no pass, return None but caller can still log best
    return None


def detect_touch_index(
    df: pd.DataFrame,
    cfg,
    scale: Dict[str, float],
    markers: Dict[str, int],
) -> tuple[int, Dict[str, Any]]:
    """
    Robust touch detection:
      Primary: k(t) ~ (dF/dt)/(dz/dt) after SG smoothing/derivatives
      Fallback1: pre-touch residual vs fitted support spring (F - (k_sup z + b))
      Fallback2: dynamic stiffness channel threshold

    Also estimates pre-touch offsets:
      F0, z0 and their uncertainties (point noise and offset uncertainty)

    Returns: (touch_index, meta)

    Key NEW behavior:
      - Motion-gates the analysis: finds a window where z actually moves (avoids k_sup ~ 0 from flat z)
      - Adds detailed diagnostics into meta
    """
    meta: Dict[str, Any] = {}

    # ---- marker shortcut ----
    try:
        if getattr(cfg, "marker_surface", None) in markers:
            idx = int(markers[cfg.marker_surface])
            return idx, {"method": "marker"}
    except Exception:
        pass

    # ---- load channels (SI) ----
    try:
        t = _num(df, cfg.time_col)                           # s
        F_raw = _num(df, cfg.Fz_raw_col)
        z_raw = _num(df, cfg.z_raw_col)
        F0_raw = np.nanquantile(F_raw, 0.01)  # 1% quantile
        z0_raw = np.nanquantile(z_raw, 0.01)
        F = (F_raw - F0_raw) * scale[cfg.Fz_raw_col]    
        z = (z_raw - z0_raw) * scale[cfg.z_raw_col]
    except Exception as e:
        raise RuntimeError(f"Touch not found: failed to load required columns ({e}).") from e

    n = int(len(t))
    if n < 50:
        raise RuntimeError("Touch not found: not enough samples.")

    # ---- dt and sustained duration ----
    dt = np.nanmedian(np.diff(t))
    dt = float(dt) if (np.isfinite(dt) and dt > 0) else 1.0
    nmin = max(1, int(float(getattr(cfg, "k_touch_min_duration_s", 0.1)) / max(dt, 1e-12)))

    # ignore initial dynamic free-hang
    ignore_s = float(getattr(cfg, "touch_ignore_first_s", 0.0) or 0.0)
    i_ignore = int(np.clip(round(ignore_s / dt), 0, n - 1))

    # ---- baseline subtract force (optional) ----
    baseline_frac = float(getattr(cfg, "touch_baseline_frac", 0.0) or 0.0)
    if baseline_frac > 0:
        try:
            nb = int(np.clip(np.floor(baseline_frac * n), 5, n))
            b0 = np.nanmedian(F[:nb])
            if np.isfinite(b0):
                F = F - b0
        except Exception:
            pass

    # ---- monotonic-z mask (OPTIONAL, but risky with oscillatory protocols) ----
    require_mono = bool(getattr(cfg, "touch_require_monotonic_z", False))
    if require_mono:
        mono = np.ones(n, dtype=bool)
        try:
            dz = np.diff(z)
            mono[1:] = np.isfinite(dz) & (dz > 0)
        except Exception:
            mono[:] = True
    else:
        mono = np.ones(n, dtype=bool)

    # ---- noise inputs (used later for sigma and MC) ----
    daq_hz = float(getattr(cfg, "daq_hz", 500.0))
    sigma_F_known = float(getattr(cfg, "sigma_Fz_N", 10e-9))
    if (not np.isfinite(sigma_F_known)) or (sigma_F_known <= 0):
        sigma_F_known = 10e-9
    sigma_z_cfg = float(getattr(cfg, "sigma_z_m", np.nan))
    sigma_z_known = sigma_z_cfg if (np.isfinite(sigma_z_cfg) and sigma_z_cfg > 0) else np.nan

    meta.update({
        "dt_s": dt,
        "nmin": int(nmin),
        "i_start_ignore": int(i_ignore),
        "sigma_F_point_known_N": float(sigma_F_known),
        "sigma_z_point_cfg_m": float(sigma_z_known) if np.isfinite(sigma_z_known) else np.nan,
    })

    # ------------------------------------------------------------------
    # 0) Motion gate: find where z is actually moving (avoids flat-z fits)
    # ------------------------------------------------------------------
    # Use a short window to detect motion. Default: 0.5 s.
    motion_win_s = float(getattr(cfg, "touch_motion_win_s", 0.5))
    motion_win = max(50, int(round(motion_win_s / dt)))
    if motion_win % 2 == 0:
        motion_win += 1

    # Default: need at least 20 nm span in that window (tuneable)
    min_z_span_m = float(getattr(cfg, "touch_motion_min_z_span_m", 20e-9))
    motion_step = max(20, int(round(0.1 / dt)))  # step 0.1 s

    motion_found = _find_first_motion_window(
        z, i0=i_ignore, win=motion_win, step=motion_step,
        min_z_span=min_z_span_m, min_finite=max(30, motion_win // 2),
    )
    if motion_found is None:
        # Still proceed, but log that motion gate failed; your protocol might be weird.
        meta["motion_gate_ok"] = 0
        meta["motion_gate_reason"] = "No window found with sufficient z-span."
        i_start = i_ignore
    else:
        s_m, e_m, d_m = motion_found
        meta["motion_gate_ok"] = 1
        meta["motion_gate_s"] = int(s_m)
        meta["motion_gate_e"] = int(e_m)
        meta["motion_gate_z_span_nm"] = float(d_m["z_span_m"] * 1e9)
        meta["motion_gate_z_std_nm"] = float(d_m["z_std_m"] * 1e9)
        i_start = int(s_m)

    meta["i_start_after_motion"] = int(i_start)

    # ------------------------------------------------------------------
    # PRIMARY: derivative-ratio stiffness touch: k(t) ~ (dF/dt)/(dz/dt)
    # ------------------------------------------------------------------
    idx = None
    try:
        k_touch_min = float(getattr(cfg, "k_touch_min", 500.0))

        win_s = float(getattr(cfg, "touch_k_ratio_win_s",
                              getattr(cfg, "touch_slope_window_s", 0.05)))
        poly = int(getattr(cfg, "touch_k_ratio_poly", 3))
        dzdt_min_frac = float(getattr(cfg, "touch_k_ratio_dzdt_min_frac", 0.15))
        require_dzdt_positive = bool(getattr(cfg, "touch_k_ratio_require_dzdt_positive", True))

        k_inst, dzdt, dFdt, km = _k_from_time_derivatives(
            t, F, z,
            win_s=win_s,
            poly=poly,
            dzdt_min_frac=dzdt_min_frac,
            require_dzdt_positive=require_dzdt_positive,
        )

        # --- baseline-relative threshold option (recommended) ---
        use_rel = bool(getattr(cfg, "touch_use_relative_k_threshold", True))
        if use_rel:
            # baseline window: use motion window if available, otherwise first 1 s after i_start
            base_win_s = float(getattr(cfg, "touch_k_baseline_win_s", 0.8))
            base_win = max(50, int(round(base_win_s / dt)))
            b0 = int(i_start)
            b1 = int(min(n, b0 + base_win))
            kb = k_inst[b0:b1]
            kb = kb[np.isfinite(kb)]
            if kb.size >= 30:
                k0 = float(np.nanmedian(kb))
                sk = float(_robust_sigma_mad(kb - k0))
                ns = float(getattr(cfg, "touch_k_nsigma", 6.0))
                k_thr_rel = k0 + ns * sk if (np.isfinite(sk) and sk > 0) else k_touch_min
                k_thr = max(k_touch_min, k_thr_rel)
                meta["k_baseline_med_N_per_m"] = k0
                meta["k_baseline_sigma_N_per_m"] = sk
                meta["k_thr_rel_N_per_m"] = k_thr_rel
            else:
                k_thr = k_touch_min
                meta["k_baseline_reason"] = f"Not enough finite baseline k samples ({kb.size})."
        else:
            k_thr = k_touch_min

        above = mono & np.isfinite(k_inst) & (k_inst > float(k_thr))
        above[:i_start] = False

        idx = _touch_by_boolean_run(above, nmin)

        meta.update({
            "method": "k_ratio_dFdt_over_dzdt",
            "k_touch_min_N_per_m": float(k_touch_min),
            "k_thr_used_N_per_m": float(k_thr),
            "k_inst_at_touch_N_per_m": float(k_inst[idx]) if np.isfinite(k_inst[idx]) else np.nan,
            "k_ratio_meta_win": int(km["win"]),
            "k_ratio_meta_poly": int(km["poly"]),
            "k_ratio_meta_dz_scale": float(km.get("dz_scale", np.nan)),
            "k_ratio_meta_dz_thr": float(km["dz_thr"]),
            "k_ratio_meta_good_frac": float(km["good_frac"]),
            "k_ratio_require_dzdt_positive": bool(km["require_dzdt_positive"]),
        })

    except Exception as e_primary:
        meta["primary_fail"] = str(e_primary)

        # --------------------------------------------------------------
        # FALLBACK 1: pre-touch residual method with MOTION-BASED fit window
        # --------------------------------------------------------------
        try:
            fit_s = float(getattr(cfg, "touch_fit_pretouch_s", 2.0))
            fit_win = max(200, int(round(fit_s / dt)))
            if fit_win % 2 == 0:
                fit_win += 1

            # Need some z-span to fit; default 50 nm for this longer window
            fit_min_span = float(getattr(cfg, "touch_fit_min_z_span_m", 50e-9))
            fit_step = max(20, int(round(0.2 / dt)))

            fit_found = _find_first_motion_window(
                z, i0=i_start, win=fit_win, step=fit_step,
                min_z_span=fit_min_span, min_finite=max(50, fit_win // 2),
            )
            if fit_found is None:
                raise RuntimeError("No suitable pre-touch fit window with z motion found.")
            s_fit, e_fit, d_fit = fit_found

            zz = z[s_fit:e_fit]
            FF = F[s_fit:e_fit]
            m = np.isfinite(zz) & np.isfinite(FF)
            if int(m.sum()) < max(80, fit_win // 2):
                raise RuntimeError("Not enough finite pre-touch samples to fit support spring.")

            zz = zz[m]; FF = FF[m]
            z0 = zz - float(np.mean(zz))
            Szz = float(np.dot(z0, z0))
            if (not np.isfinite(Szz)) or Szz < float(getattr(cfg, "touch_fit_min_Szz", 1e-22)):
                raise RuntimeError(f"Pre-touch fit ill-conditioned: Szz={Szz:g}")

            A = np.vstack([zz, np.ones_like(zz)]).T
            k_sup, b_sup = np.linalg.lstsq(A, FF, rcond=None)[0]
            k_sup = float(k_sup); b_sup = float(b_sup)

            # residual contact load
            P = F - (k_sup * z + b_sup)

            Pb = P[s_fit:e_fit]
            Pb = Pb[np.isfinite(Pb)]
            P0 = float(np.nanmedian(Pb))
            sigma_P0 = float(_robust_sigma_mad(Pb - P0))

            nsig = float(getattr(cfg, "touch_P_nsigma", 6.0))
            P_thr = (P0 + nsig * sigma_P0) if (np.isfinite(sigma_P0) and sigma_P0 > 0) else float(getattr(cfg, "touch_P_abs_N", 2e-7))

            above = np.isfinite(P) & (P > P_thr)
            above[:i_start] = False
            idx = _touch_by_boolean_run(above, nmin)

            meta.update({
                "method": "pretouch_residual",
                "pretouch_fit_s": int(s_fit),
                "pretouch_fit_e": int(e_fit),
                "pretouch_fit_z_span_nm": float(d_fit["z_span_m"] * 1e9),
                "pretouch_fit_z_std_nm": float(d_fit["z_std_m"] * 1e9),
                "pretouch_fit_Szz": float(Szz),

                "k_sup_touchfit_N_per_m": float(k_sup),
                "b_sup_touchfit_N": float(b_sup),

                "P0_baseline_N": float(P0),
                "sigma_P0_baseline_N": float(sigma_P0) if np.isfinite(sigma_P0) else np.nan,
                "P_thr_N": float(P_thr),
                "P_touch_N": float(P[idx]) if np.isfinite(P[idx]) else np.nan,
            })

        except Exception as e_resid:
            meta["residual_fail"] = str(e_resid)

            # ----------------------------------------------------------
            # FALLBACK 2: dynamic stiffness channel
            # ----------------------------------------------------------
            try:
                k_col = getattr(cfg, "k_touch_col", None)
                if (k_col is None) or (k_col not in df.columns):
                    raise RuntimeError("k_touch_col fallback is unavailable.") from e_resid

                k = _num(df, k_col)
                k_touch_min = float(getattr(cfg, "k_touch_min", 500.0))
                above = np.isfinite(k) & (k > k_touch_min)
                above[:i_start] = False
                idx = _touch_by_boolean_run(above, nmin)

                meta.update({
                    "method": "dyn_k_channel",
                    "k_touch_col": str(k_col),
                    "k_touch_min_N_per_m": float(k_touch_min),
                })
            except Exception as e_fb:
                raise RuntimeError("Touch not found: all methods failed.") from e_fb

    assert idx is not None

    # ---------------- offsets + uncertainties (FORCE & DISPLACEMENT) ----------------
    try:
        off_sec = float(getattr(cfg, "touch_offset_seconds", 2.0))
        off_margin = float(getattr(cfg, "touch_offset_margin_s", 0.5))

        F0, sigma_F_point_est, sigma_F0_est, nF = estimate_offset_pre_touch(
            t, F, int(idx),
            daq_hz=daq_hz,
            seconds=off_sec,
            margin_s=off_margin,
            min_points=max(300, int(0.5 * daq_hz)),
        )

        # choose whether to use estimated force sigma or known constant
        use_est_F = bool(getattr(cfg, "touch_sigmaF_use_estimate", False))
        sigma_F_point_used = float(sigma_F_point_est) if (use_est_F and np.isfinite(sigma_F_point_est)) else float(sigma_F_known)

        sigma_F0_used = (
            float(sigma_F_point_used / np.sqrt(max(int(nF), 1)))
            if (np.isfinite(sigma_F_point_used) and int(nF) > 0)
            else float(sigma_F0_est)
        )

        z0, sigma_z_point_est, sigma_z0_est, nz = estimate_offset_pre_touch(
            t, z, int(idx),
            daq_hz=daq_hz,
            seconds=off_sec,
            margin_s=off_margin,
            min_points=max(300, int(0.5 * daq_hz)),
        )

        sigma_z_point_used = float(sigma_z_known) if np.isfinite(sigma_z_known) else float(sigma_z_point_est)
        sigma_z0_used = (
            float(sigma_z_point_used / np.sqrt(max(int(nz), 1)))
            if (np.isfinite(sigma_z_point_used) and int(nz) > 0)
            else float(sigma_z0_est)
        )

        meta.update({
            "daq_hz": float(daq_hz),

            "F0_mN": float(F0) * 1e3 if np.isfinite(F0) else np.nan,
            "F0_n_used": int(nF),
            "sigma_F_point_est_mN": float(sigma_F_point_est) * 1e3 if np.isfinite(sigma_F_point_est) else np.nan,
            "sigma_F_point_used_mN": float(sigma_F_point_used) * 1e3 if np.isfinite(sigma_F_point_used) else np.nan,
            "sigma_F0_mN": float(sigma_F0_used) * 1e3 if np.isfinite(sigma_F0_used) else np.nan,

            "z0_nm": float(z0) * 1e9 if np.isfinite(z0) else np.nan,
            "z0_n_used": int(nz),
            "sigma_z_point_est_nm": float(sigma_z_point_est) * 1e9 if np.isfinite(sigma_z_point_est) else np.nan,
            "sigma_z_point_used_nm": float(sigma_z_point_used) * 1e9 if np.isfinite(sigma_z_point_used) else np.nan,
            "sigma_z0_nm": float(sigma_z0_used) * 1e9 if np.isfinite(sigma_z0_used) else np.nan,

            # keep SI for propagation
            "sigma_F_point_used_N": float(sigma_F_point_used) if np.isfinite(sigma_F_point_used) else np.nan,
            "sigma_F0_N": float(sigma_F0_used) if np.isfinite(sigma_F0_used) else np.nan,
            "sigma_z_point_used_m": float(sigma_z_point_used) if np.isfinite(sigma_z_point_used) else np.nan,
            "sigma_z0_m": float(sigma_z0_used) if np.isfinite(sigma_z0_used) else np.nan,
        })

    except Exception as e_off:
        meta["offset_fail"] = str(e_off)
        meta.update({
            "F0_mN": np.nan, "sigma_F0_mN": np.nan,
            "z0_nm": np.nan, "sigma_z0_nm": np.nan,
        })

    # ---------------- optional Monte-Carlo CI on touch index ----------------
    # uses SI sigmas (N, m)
    try:
        mc_n = int(getattr(cfg, "touch_mc_n", 0) or 0)
        if mc_n > 0 and meta.get("method") == "k_ratio_dFdt_over_dzdt":
            sigma_F_mc = float(meta.get("sigma_F_point_used_N", sigma_F_known))
            sigma_z_mc = float(meta.get("sigma_z_point_used_m", np.nan))
            if np.isfinite(sigma_F_mc) and np.isfinite(sigma_z_mc) and sigma_F_mc > 0 and sigma_z_mc > 0:
                seed = int(getattr(cfg, "touch_mc_seed", 0) or 0)
                rng = np.random.default_rng(seed)
                idxs = []

                win_s = float(getattr(cfg, "touch_k_ratio_win_s",
                                      getattr(cfg, "touch_slope_window_s", 0.05)))
                poly = int(getattr(cfg, "touch_k_ratio_poly", 3))
                dzdt_min_frac = float(getattr(cfg, "touch_k_ratio_dzdt_min_frac", 0.15))
                require_dzdt_positive = bool(getattr(cfg, "touch_k_ratio_require_dzdt_positive", True))

                k_touch_min = float(getattr(cfg, "k_touch_min", 500.0))

                for _ in range(mc_n):
                    Fp = F + rng.normal(0.0, sigma_F_mc, size=n)
                    zp = z + rng.normal(0.0, sigma_z_mc, size=n)
                    try:
                        k_inst_p, _, _, _ = _k_from_time_derivatives(
                            t, Fp, zp,
                            win_s=win_s, poly=poly,
                            dzdt_min_frac=dzdt_min_frac,
                            require_dzdt_positive=require_dzdt_positive,
                        )
                        above_p = mono & np.isfinite(k_inst_p) & (k_inst_p > k_touch_min)
                        above_p[:i_start] = False
                        idxs.append(_touch_by_boolean_run(above_p, nmin))
                    except Exception:
                        pass

                if len(idxs) >= max(10, mc_n // 5):
                    idxs = np.asarray(idxs, dtype=int)
                    meta["touch_idx_mc_ok"] = int(len(idxs))
                    meta["touch_idx_ci95_lo"] = int(np.quantile(idxs, 0.025))
                    meta["touch_idx_ci95_hi"] = int(np.quantile(idxs, 0.975))
                    meta["touch_idx_sigma"] = float(np.std(idxs, ddof=1)) if len(idxs) > 1 else 0.0
                else:
                    meta["touch_idx_mc_ok"] = int(len(idxs))
                    meta["touch_idx_ci95_lo"] = None
                    meta["touch_idx_ci95_hi"] = None
                    meta["touch_idx_sigma"] = None
    except Exception as e_mc:
        meta["mc_fail"] = str(e_mc)

    return int(idx), meta

def manual_pick_touch(df: pd.DataFrame, cfg: Config, initial: Optional[int] = None) -> int:
    t = _num(df, cfg.time_col)
    z = _num(df, cfg.z_raw_col)
    fz = _num(df, cfg.Fz_raw_col)
    sz = _num(df, cfg.k_touch_col)
    plt.plot(z, fz)
    picks = pick_indices_from_plot(
        t,
        series=[("Stiffness (N/m)", sz),],
        prompts=["Click TOUCH point (first contact)."],
        n_clicks=1,
        predefined_picks=[initial] if initial is not None else [],
        title="Pick touch index"
    )
    return picks[0]

def detect_cycles(
    df: pd.DataFrame,
    cfg,
    start_i: int = 0,
    end_i: Optional[int] = None
) -> List[CycleBounds]:
    t = _num(df, cfg.time_col)
    a = np.nan_to_num(_num(df, cfg.F2_rms_col), nan=0.0)
    cycles: List[CycleBounds] = []

    n = len(a)
    if n < 10:
        raise RuntimeError("Too few samples for cycle detection.")

    if end_i is None:
        end_i = n - 1

    start_i = int(np.clip(start_i, 0, n - 2))
    end_i = int(np.clip(end_i, start_i + 1, n - 1))

    dt = median_dt(t)
    ramp_min_pts = max(3, int(cfg.min_ramp_s / max(dt, 1e-12)))
    hold_min_pts = max(3, int(cfg.min_hold_s / max(dt, 1e-12)))

    # Smooth envelope for amplitude
    a_s = a #rolling_median(a, cfg.smooth_n)

    # Extra smoothing for derivative stability
    a_sd = a_s #rolling_median(a, cfg.dfdt_smooth_n)
    da = np.gradient(a_sd, t)

    # zero out outside shear window
    a_s[:start_i] = 0.0
    a_s[end_i + 1:] = 0.0
    a_sd[:start_i] = 0.0
    a_sd[end_i + 1:] = 0.0
    da[:start_i] = 0.0
    da[end_i + 1:] = 0.0

    # Baseline and amplitude thresholds (secondary checks)
    base = float(np.quantile(a_s[start_i:end_i + 1], cfg.dynF2_baseline_q))
    thr_active = base + cfg.dynF2_active_delta
    thr_nz = base + cfg.dynF2_nearzero_delta

    # Derivative thresholds
    scale = float(safe_nanmax(np.abs(da[start_i:end_i + 1])))
    if not np.isfinite(scale) or scale <= 0:
        return cycles #raise RuntimeError("Cycle detection: derivative scale is zero; check signal/window.")

    dthr = cfg.dfdt_thr_frac * scale
    dhold = cfg.dfdt_hold_frac * dthr

    ramp_up = da > dthr
    ramp_dn = da < -dthr
    hold_like = np.abs(da) <= dhold

    up_regs = [(s, e) for (s, e) in contiguous_regions(ramp_up) if (e - s + 1) >= ramp_min_pts]

    cursor = start_i
    cyc = 0

    if not up_regs:
         return cycles  # No ramp-up regions found; return empty list
    
    for (us, ue) in up_regs:
        if us < cursor:
            continue

        # backtrack to near-zero to define start
        i_start = us
        while i_start > start_i and a_s[i_start] > thr_nz:
            i_start -= 1

        # find first sustained ramp-down after this ramp-up
        dn_after = [
            (s, e) for (s, e) in contiguous_regions(ramp_dn)
            if (e - s + 1) >= ramp_min_pts and s > ue
        ]

        search_end = end_i
        if dn_after:
            first_dn_s, _ = dn_after[0]
            search_end = min(search_end, first_dn_s)

        if search_end <= ue:
            continue

        seg_pk = slice(us, search_end + 1)
        i_peak = int(us + np.argmax(a_s[seg_pk]))
        amax = float(a_s[i_peak])

        # reject tiny bumps
        if not (amax > thr_active and (amax - base) > 2.0 * cfg.dynF2_active_delta):
            continue

        # hold plateau near the top
        near_top = (a_s >= cfg.hold_top_frac * amax) & hold_like
        near_top[:us] = False
        near_top[end_i + 1:] = False
        if dn_after:
            near_top[dn_after[0][0]:] = False

        top_regs = [(s, e) for (s, e) in contiguous_regions(near_top) if (e - s + 1) >= hold_min_pts]
        if top_regs:
            rs, re = min(top_regs, key=lambda r: abs((r[0] + r[1]) / 2 - i_peak))
            i_hold0, i_hold1 = int(rs), int(re)
        else:
            i_hold0, i_hold1 = i_peak, i_peak

        # end: after ramp-down, go until near-zero
        if dn_after:
            i_end = int(dn_after[0][1])
        else:
            i_end = int(i_peak)

        while i_end < end_i and a_s[i_end] > thr_nz:
            i_end += 1

        if not (i_start < i_peak < i_end):
            continue

        cyc += 1
        cycles.append(CycleBounds(cyc, i_start, i_peak, i_hold0, i_hold1, i_end))
        cursor = i_end + 1

    return cycles

def recompute_touch_meta_from_index( *,
    t_s: np.ndarray,
    F_mN: np.ndarray,
    z_nm: np.ndarray,
    touch_i: int,
    daq_hz: float = 500.0,
    # window for offsets/noise
    offset_seconds: float = 2.0,
    offset_margin_s: float = 0.5,
    min_points: int | None = None,
    # user-provided point noises (optional)
    sigma_F_point_mN: float = 0.01,   # 10 nN = 0.01 mN
    sigma_z_point_nm: float | None = None,
) -> Dict[str, Any]:
    """
    Recompute touch offsets + uncertainties *given a fixed touch_i*.
    All units are mN and nm.

    Returns keys:
      F0_mN, sigma_F0_mN, sigma_F_point_used_mN
      z0_nm, sigma_z0_nm, sigma_z_point_used_nm
      nF_used, nz_used
    """
    t_s = np.asarray(t_s, float)
    F_mN = np.asarray(F_mN, float)
    z_nm = np.asarray(z_nm, float)

    n = len(t_s)
    touch_i = int(touch_i)
    out: Dict[str, Any] = {"method": "recompute_from_index", "touch_i": touch_i}

    if touch_i <= 5 or touch_i >= n:
        out.update({
            "F0_mN": np.nan, "sigma_F0_mN": np.nan, "sigma_F_point_used_mN": float(sigma_F_point_mN),
            "z0_nm": np.nan, "sigma_z0_nm": np.nan, "sigma_z_point_used_nm": np.nan,
            "nF_used": 0, "nz_used": 0,
        })
        return out

    if min_points is None:
        min_points = max(300, int(0.5 * daq_hz))  # >=0.5s

    # pre-touch end index with margin
    margin = int(max(0, round(offset_margin_s * daq_hz)))
    end = max(0, touch_i - margin)
    if end < min_points:
        # not enough pre-touch
        out.update({
            "F0_mN": np.nan, "sigma_F0_mN": np.nan, "sigma_F_point_used_mN": float(sigma_F_point_mN),
            "z0_nm": np.nan, "sigma_z0_nm": np.nan, "sigma_z_point_used_nm": np.nan,
            "nF_used": 0, "nz_used": 0,
        })
        return out

    nb = int(round(offset_seconds * daq_hz))
    nb = int(np.clip(nb, min_points, min(end, 5000)))
    start = end - nb

    sl = slice(start, end)
    tt = t_s[sl]

    # ----- F0 -----
    FF = F_mN[sl]
    mF = np.isfinite(tt) & np.isfinite(FF)
    if mF.sum() >= min_points:
        ttF = tt[mF]
        FF = FF[mF]
        F0 = float(np.median(FF))
        # Offset uncertainty from known sigma_F (preferred)
        nF = int(FF.size)
        sF_point_used = float(sigma_F_point_mN) if np.isfinite(sigma_F_point_mN) and sigma_F_point_mN > 0 else np.nan
        sF0 = float(sF_point_used / np.sqrt(nF)) if np.isfinite(sF_point_used) else np.nan
    else:
        F0, sF0, sF_point_used, nF = np.nan, np.nan, float(sigma_F_point_mN), 0

    # ----- z0 + sigma_z_point estimate if not provided -----
    ZZ = z_nm[sl]
    mZ = np.isfinite(tt) & np.isfinite(ZZ)
    if mZ.sum() >= min_points:
        ttZ = tt[mZ]
        ZZ = ZZ[mZ]
        z0 = float(np.median(ZZ))
        nZ = int(ZZ.size)

        if sigma_z_point_nm is None or (not np.isfinite(sigma_z_point_nm)) or sigma_z_point_nm <= 0:
            # detrend linear drift vs time then MAD
            tc = ttZ - np.mean(ttZ)
            A = np.vstack([tc, np.ones_like(tc)]).T
            slope, intercept = np.linalg.lstsq(A, ZZ, rcond=None)[0]
            resid = ZZ - (slope * tc + intercept)
            sz_point_used = _robust_sigma_mad(resid)
        else:
            sz_point_used = float(sigma_z_point_nm)

        sz0 = float(sz_point_used / np.sqrt(nZ)) if np.isfinite(sz_point_used) else np.nan
    else:
        z0, sz0, sz_point_used, nZ = np.nan, np.nan, np.nan, 0

    out.update({
        "F0_mN": F0,
        "sigma_F0_mN": sF0,
        "sigma_F_point_used_mN": sF_point_used,
        "nF_used": nF,

        "z0_nm": z0,
        "sigma_z0_nm": sz0,
        "sigma_z_point_used_nm": sz_point_used,
        "nz_used": nZ,
    })
    return out

def manual_pick_shear_window(t: np.ndarray, P_contact_N: np.ndarray, F2_rms: np.ndarray, initial: Optional[List[int]] = None) -> Tuple[int, int]:
    picks = pick_indices_from_plot(
        t,
        series=[("P_contact (N)", P_contact_N), ("Dyn Force 2 RMS", F2_rms)],
        prompts=["Click shear window START.", "Click shear window END."],
        n_clicks=2,
        predefined_picks=initial,
        title="Pick shear window bounds"
    )
    i0, i1 = sorted(picks)
    return i0, i1

def manual_pick_one_cycle(t: np.ndarray, F2_env: np.ndarray, initial: CycleBounds, title: str = "") -> "CycleBounds":
    picks = pick_indices_from_plot(
        t,
        series=[("Dyn Force 2 envelope", F2_env)],
        prompts=[
            "Click cycle START (near-zero before ramp-up).",
            "Click cycle PEAK (max amplitude).",
            "Click HOLD END (plateau end).",
            "Click cycle END (back to near-zero).",
            "If no cycle to select, click f.",
        ],
        predefined_picks=[initial.i_start if initial.i_start is not None else 0
                          , initial.i_peak if initial.i_peak is not None else 0
                          , initial.i_hold1 if initial.i_hold1 is not None else 0
                          , initial.i_end if initial.i_end is not None else 0],
        n_clicks=4,

        title=title or "Pick cycle bounds"
    )
    i_start, i_peak, i_hold1, i_end = picks
    i_hold0=i_peak+1
    cycle = 1  # dummy cycle number; will be replaced by caller
    # basic ordering guard
    if not (i_start < i_peak < i_end):
        cycle, i_start, i_end, i_peak, i_hold0, i_hold1 = 1,0,0,0,0,0
    if i_peak < i_hold0:
        i_hold0, i_peak = i_peak, i_hold0
    if i_hold1 < i_hold0:
        i_hold0, i_hold1 = i_hold1, i_hold0
    return CycleBounds(cycle, i_start=i_start, i_peak=i_peak, i_hold0=i_hold0, i_hold1=i_hold1, i_end=i_end)

def manual_pick_cycles(df2: pd.DataFrame, cfg: Config, i0: int, i1: int, initial: List[CycleBounds], n_cycles: int = 3) -> List["CycleBounds"]:
    t = _num(df2, cfg.time_col)
    F2 = np.nan_to_num(_num(df2, cfg.F2_rms_col), nan=0.0)
    F2s = pd.Series(F2).rolling(cfg.smooth_n, center=True, min_periods=1).median().to_numpy()
    F2s[:i0] = 0.0
    F2s[i1+1:] = 0.0

    cycles = []
    for c in range(1, n_cycles + 1):
        cb = manual_pick_one_cycle(t, F2s, initial[c-1] if initial is not None and c-1 < len(initial) else CycleBounds(cycle=c, i_start=i0, i_peak=0, i_hold0=0, i_hold1=0, i_end=i1), title=f"Pick cycle {c}")
        if cb.cycle == 0:
            continue
        if cb.i_start == 0:
            break
        cb = CycleBounds(cycle=c, i_start=cb.i_start, i_peak=cb.i_peak, i_hold0=cb.i_hold0, i_hold1=cb.i_hold1, i_end=cb.i_end)
        cycles.append(cb)
    return cycles

# ============================================================
# Transition detection: stick->slide and re-stick
# ============================================================
def detect_stick_slide_transitions(
    df: pd.DataFrame,
    b: CycleBounds,
    sliding_lateral_stiffness_thresh: float,    # user threshold for stick->slide (Sx drops below)
    resticking_lateral_stiffness_thresh: float, # user threshold for re-stick (Sx rises above)
    frac_up: float,                             # fallback fraction for stick->slide: S_slide = frac_up * Sx_stuck
    frac_low: float,                            # fallback fraction for re-stick:  S_re   = frac_low * Sx_stuck
    low_frac_band: tuple[float, float],         # band of Ft/Ftmax to estimate Sx_stuck
    smooth_n: int,
) -> dict:
    """
    stick->slide (ramp-up): first index where Sx falls below S_slide
    re-stick    (ramp-down): after hold ends, first index where Sx rises above S_re,
                             but ONLY after it has gone below S_slide at least once.

    Sx_stuck estimated as median Sx in early ramp-up band: Ft in [low*Ftmax, high*Ftmax].
    """
    Ft = df["F2_pk_corr_N"].to_numpy()
    Xc = df["X2_pk_contact_m"].to_numpy()
    Sx = df["Stiffness_lateral"].to_numpy()

    # ---- ramp-up slice ----
    ru0 = int(b.i_start)
    ru1 = int(b.i_peak)
    ru = slice(ru0, ru1 + 1)

    # ---- ramp-down slice: START AFTER HOLD ----
    # This is the key fix: do not allow re-stick detection during peak/hold region.
    rd0 = int(max(b.i_hold1, b.i_peak))
    rd1 = int(b.i_end)
    rd = slice(rd0, rd1 + 1)

    # Smooth Sx within each slice
    Sx_ru_s = pd.Series(Sx[ru]).rolling(smooth_n, center=True, min_periods=1).median().to_numpy()
    Sx_rd_s = pd.Series(Sx[rd]).rolling(smooth_n, center=True, min_periods=1).median().to_numpy()

    Ft_ru = Ft[ru]

    Ftmax = safe_nanmax(Ft_ru) if np.isfinite(Ft_ru).any() else np.nan
    if not (np.isfinite(Ftmax) and Ftmax > 0):
        return {
            "i_ss": None, "i_rs": None,
            "Sx_stuck": None,
            "Sx_slide_used": None,
            "Sx_restick_used": None,
            "Ft_ss_N": None, "X_ss_m": None,
            "Ft_rs_N": None, "X_rs_m": None,
            "rd0": rd0,
            "went_low_first": 0,
        }

    # ---- estimate Sx_stuck from early ramp-up ----
    lo = low_frac_band[0] * Ftmax
    hi = low_frac_band[1] * Ftmax
    m_stuck = np.isfinite(Ft_ru) & np.isfinite(Sx_ru_s) & (Ft_ru >= lo) & (Ft_ru <= hi)

    if m_stuck.sum() < 10:
        idxs = np.where(np.isfinite(Sx_ru_s))[0][:max(10, min(30, len(Sx_ru_s)))]
        Sx_stuck = float(np.nanmedian(Sx_ru_s[idxs])) if idxs.size else None
    else:
        Sx_stuck = float(np.nanmedian(Sx_ru_s[m_stuck]))

    if not (np.isfinite(Sx_stuck) and Sx_stuck > 0):
        return {
            "i_ss": None, "i_rs": None,
            "Sx_stuck": Sx_stuck,
            "Sx_slide_used": None,
            "Sx_restick_used": None,
            "Ft_ss_N": None, "X_ss_m": None,
            "Ft_rs_N": None, "X_rs_m": None,
            "rd0": rd0,
            "went_low_first": 0,
        }

    # ---- decide thresholds: user-provided takes priority if finite ----
    # stick->slide threshold (must be BELOW stuck)
    Sx_slide = float(sliding_lateral_stiffness_thresh)
    if not (np.isfinite(Sx_slide) and (Sx_slide < Sx_stuck)):
        Sx_slide = float(frac_up) * Sx_stuck
        slide_source = "frac"
    else:
        slide_source = "user"

    # ---- stick->slide: first crossing below Sx_slide on ramp-up ----
    i_ss_rel = None
    for i, val in enumerate(Sx_ru_s):
        if np.isfinite(val) and (val < Sx_slide):
            i_ss_rel = i
            slide_source = "user"
            break
    if i_ss_rel is None:
        # fallback: fraction of Sx_stuck
        Sx_slide = float(frac_up) * Sx_stuck
        for i, val in enumerate(Sx_ru_s):
            if np.isfinite(val) and (val < Sx_slide):
                i_ss_rel = i
                slide_source = "frac"
                break

    i_ss = (ru0 + i_ss_rel) if i_ss_rel is not None else None

    # ---- re-stick: AFTER HOLD, require it went low first, then crosses above Sx_re ----
    went_low = False
    i_rs_rel = None
    re_source = "none"
    Sx_re = float(resticking_lateral_stiffness_thresh)
    for j, val in enumerate(Sx_rd_s):
        if not np.isfinite(val):
            continue
        # Step 1: detect "went low" (i.e., definitely in sliding regime)
        if not went_low:
            if val < Sx_slide:
                went_low = True
            continue
        # Step 2: first time it rises above re-stick threshold
        if val > Sx_re:
            i_rs_rel = j
            re_source = "user"
            break
    
    if i_rs_rel is None:
        Sx_re = float(frac_low) * Sx_stuck
        went_low = False
        # fallback: fraction of Sx_stuck
        for j, val in enumerate(Sx_rd_s):
            if not np.isfinite(val):
                continue
        # Step 1: detect "went low" (i.e., definitely in sliding regime)
            if not went_low:
                if val < Sx_slide:
                    went_low = True
                continue
        # Step 2: first time it rises above re-stick threshold
            if val > Sx_re:
                i_rs_rel = j
                re_source = "frac"
                break
    i_rs = (rd0 + i_rs_rel) if i_rs_rel is not None else None

    out = {
        "i_ss": i_ss,
        "i_rs": i_rs,
        "Sx_stuck": float(Sx_stuck),
        "Sx_slide_used": float(Sx_slide),
        "Sx_restick_used": float(Sx_re),
        "Sx_slide_source": slide_source,
        "Sx_restick_source": re_source,
        "rd0": int(rd0),
        "went_low_first": int(went_low),
    }

    if i_ss is not None:
        out["Ft_ss_N"] = float(Ft[i_ss])
        out["X_ss_m"] = float(Xc[i_ss])
    else:
        out["Ft_ss_N"] = np.nan
        out["X_ss_m"] = np.nan

    if i_rs is not None:
        out["Ft_rs_N"] = float(Ft[i_rs])
        out["X_rs_m"] = float(Xc[i_rs])
    else:
        out["Ft_rs_N"] = np.nan
        out["X_rs_m"] = np.nan

    return out

def want_manual(cfg: Config, mode: str, failed: bool) -> bool:
    if mode == "never": return False
    if mode == "always": return True
    return bool(failed)  # "on_fail"

def approve_or_repick_gate(figures, fig_title: str = "") -> str:
    """
    Returns: "accept" | "repick_touch" | "repick_window" | "repick_cycles"
    Raises on pass/abort.
    """
    decision = {"val": None}

    def on_key(event):
        k = (event.key or "").lower()
        if k == "a":
            decision["val"] = "accept"
        elif k == "t":
            decision["val"] = "repick_touch"
        elif k == "w":
            decision["val"] = "repick_window"
        elif k == "c":
            decision["val"] = "repick_cycles"
        elif k == "p":
            decision["val"] = "pass"
        elif k == "escape":
            decision["val"] = "abort"

        if decision["val"] is not None:
            plt.close("all")
    for fig in figures:
        try:
            fig.canvas.mpl_connect("key_press_event", on_key)
            if fig_title:
                fig.canvas.manager.set_window_title(fig_title)
        except Exception:
            pass
    plt.show(block=True)

    if decision["val"] == "accept":
        return "accept"
    if decision["val"] in {"repick_touch","repick_window","repick_cycles"}:
        return decision["val"]
    if decision["val"] == "pass":
        raise RuntimeError("User passed file.")
    raise RuntimeError("User aborted.")
# ============================================================
# ============================================================
# Calibration slice, cycle detection
# ============================================================
def find_calibration_slices_pre_touch(
    df: pd.DataFrame,
    cfg: Config,
    markers: Dict[str, int],
    touch_i: int,
) -> tuple[Optional[slice], Optional[slice]]:
    """
    Returns (cal_sl_lat, cal_sl_vert).
    cal_sl_lat: slice for lateral calibration (used for spring subtraction fit)
    cal_sl_vert: slice for vertical dynamic calibration bump (sanity check only)
    """
    # Markers (if present) win
    if cfg.marker_cal_up in markers and cfg.marker_cal_dn in markers:
        i0 = int(markers[cfg.marker_cal_up])
        i1 = int(markers[cfg.marker_cal_dn])
        if i1 <= i0:
            raise RuntimeError("Calibration markers out of order.")
        return slice(i0, i1 + 1), None

    # Pre-touch search window
    pre_end = max(0, int(touch_i))
    if pre_end < 50:
        # too short to detect bumps robustly
        return None, None

    idx = np.arange(pre_end)

    # Lateral bump (primary)
    y_lat = _num(df, cfg.F2_rms_col)
    cal_lat = find_first_bump_region(
        y=y_lat, idx=idx,
        smooth_n=101, thr_frac=0.08, pad=20, min_points=40
    )

    # Vertical bump (optional sanity)
    cal_vert = None
    if getattr(cfg, "Fz_dyn_rms_col", None) and cfg.Fz_dyn_rms_col in df.columns:
        y_v = _num(df, cfg.Fz_dyn_rms_col)
        cal_vert = find_first_bump_region(
            y=y_v, idx=idx,
            smooth_n=101, thr_frac=0.08, pad=20, min_points=40
        )

    return cal_lat, cal_vert
# ============================================================
def find_shear_window_from_normal_load_v2(
    t: np.ndarray,
    P_contact_N: np.ndarray,
    touch_i: int,
    cfg: Config
) -> tuple[int, int]:
    """
    Robust window:
      i0 = after loading is finished (just before/around max load plateau)
      i1 = before unloading begins
    Anchors everything around the max of smoothed P after touch.
    """
    n = len(t)
    dt = median_dt(t)
    if n < 10:
        return (0, n - 1)
    
    smooth_n = cfg.normal_load_smooth
    dpdt_thr_frac = cfg.loading_rate_threshold
    sustain_s = cfg.normal_load_sustain

    w = max(11, int(smooth_n))
    P_sm = pd.Series(P_contact_N).rolling(w, center=True, min_periods=1).median().to_numpy()
    dPdt = np.gradient(P_sm, t)
    dPdt = pd.Series(dPdt).rolling(w, center=True, min_periods=1).median().to_numpy()

    start = max(0, int(touch_i))
    if start >= n - 2:
        return (start, n - 1)

    # threshold based on post-touch derivative scale
    post = np.arange(start, n)
    scale = float(safe_nanmax(np.abs(dPdt[post]))) if post.size else float(safe_nanmax(np.abs(dPdt)))
    if not np.isfinite(scale) or scale <= 0:
        return (start, n - 1)

    thr = dpdt_thr_frac * scale
    sustain_pts = max(3, int(sustain_s / max(dt, 1e-12)))

    # locate max load AFTER touch (on smoothed)
    iPmax = int(start + np.nanargmax(P_sm[start:]))

    # sustained +slope and -slope masks
    pos = dPdt > thr
    neg = dPdt < -thr
    stable = np.abs(dPdt) <= thr

    # ---- find i0: first stable point AFTER the last sustained positive region before max ----
    pos_regs = [(s, e) for (s, e) in contiguous_regions(pos) if (e - s + 1) >= sustain_pts]
    # pick the last pos-reg that ends before iPmax
    pos_before = [r for r in pos_regs if r[1] < iPmax]
    if pos_before:
        last_pos_end = pos_before[-1][1]
        # from there, find the next stable region
        stable_regs = [(s, e) for (s, e) in contiguous_regions(stable) if (e - s + 1) >= sustain_pts and s > last_pos_end]
        i0 = stable_regs[0][0] if stable_regs else max(start, last_pos_end + 1)
    else:
        # fallback: just use a bit before max plateau
        i0 = max(start, iPmax - 5 * sustain_pts)

    # ---- find i1: first sustained negative region AFTER max ----
    neg_regs = [(s, e) for (s, e) in contiguous_regions(neg) if (e - s + 1) >= sustain_pts and s > iPmax]
    i1 = neg_regs[0][0] if neg_regs else (n - 1)

    # guards
    i0 = int(np.clip(i0, start, n - 2))
    i1 = int(np.clip(i1, i0 + 1, n - 1))
    return i0, i1

def find_first_bump_region(y: np.ndarray, idx: np.ndarray, smooth_n: int = 101, thr_frac: float = 0.08, 
                           pad: int = 20, min_points: int = 40) -> Optional[slice]:
    """
    Finds the FIRST contiguous region where smoothed y rises above baseline+thr_frac*span.
    Returns a slice in ORIGINAL indices (not local).
    """
    if idx.size < 10:
        return None

    y0 = np.nan_to_num(y[idx], nan=0.0)
    ys = pd.Series(y0).rolling(int(smooth_n), center=True, min_periods=1).median().to_numpy()

    base = float(np.quantile(ys, 0.10))
    span = float(np.quantile(ys, 0.95) - base)
    if not np.isfinite(span) or span <= 0:
        return None

    thr = base + float(thr_frac) * span
    regs = contiguous_regions(ys > thr)
    if not regs:
        return None

    s, e = regs[0]
    s = max(0, s - int(pad))
    e = min(len(ys) - 1, e + int(pad))
    if (e - s) < int(min_points):
        return None

    i0 = int(idx[s])
    i1 = int(idx[e])
    return slice(i0, i1 + 1)

def window_idx(t: np.ndarray, center_i: int, halfwidth_s: float) -> np.ndarray:
    t0 = t[center_i]
    return np.where((t >= t0 - halfwidth_s) & (t <= t0 + halfwidth_s))[0]

def window_idx_fw(t: np.ndarray, start_i: int, width_s: float) -> np.ndarray:
    t0 = t[start_i]
    return np.where((t >= t0) & (t <= t0 + width_s))[0]

def harmonics_energy_regime_dict(
    *,
    t: np.ndarray,
    x_raw: np.ndarray,
    Kpp: np.ndarray,                 # Damping_lateral (Im(K*)), N/m
    X_lockin_pk: np.ndarray | None,  # X2_pk_contact_m (peak), m  -> for direct lock-in energy
    Ft_pk: np.ndarray | None,        # F2_pk_corr_N (peak), N     -> for hysteresis width
    sl: slice,
    f1_guess_hz: float,
    prefix: str = "",

    # harmonic/FFT controls
    max_harmonic: int = 3,
    estimate_f0: bool = True,
    f0_span_hz: float = 5.0,
    nyquist_guard: float = 0.49,
    target_cycles: float = 12.0,
    target_samples: int = 120,
    min_points_floor: int = 40,
    min_points_frac: float = 0.8,
    noise_span_hz: float = 8.0,
    noise_exclude_bins: int = 3,

    # integration controls
    use_f0_for_totals: bool = False,  # if True, use measured f0(t) for FFT totals; otherwise constant f1_guess
) -> dict:
    """
    Regime-local harmonic + energy analysis.
    Returns a dictionary (optionally prefixed) ready to be merged into summarize_cycle via **dict.

    Provides TWO consistent energy channels over the slice:
      (1) Direct lock-in energy: E_LI_cyc(t) = pi*|K''(t)|*X_lockin_pk(t)^2
      (2) FFT augmented energy: E_FFT_cyc(t) = pi*|K''(t)|*(X1^2+X2^2+X3^2) with sigma

    Totals are computed as: sum(E_cyc * f * dt) over the slice.
    Mean power is total energy / slice duration.
    """

    def _p(k: str) -> str:
        return f"{prefix}_{k}" if prefix else k

    # -------------------- guards --------------------
    if sl is None:
        return {}

    i0, i1 = int(sl.start), int(sl.stop)
    if (i1 - i0) < 10:
        return {}

    t = np.asarray(t, float)
    x_raw = np.asarray(x_raw, float)
    Kpp = np.asarray(Kpp, float)
    X_lockin_pk = None if X_lockin_pk is None else np.asarray(X_lockin_pk, float)
    Ft_pk = None if Ft_pk is None else np.asarray(Ft_pk, float)

    dt = np.diff(t)
    dt_pos = dt[np.isfinite(dt) & (dt > 0)]
    if dt_pos.size < 5:
        return {}

    fs = float(1.0 / np.median(dt_pos))
    if not np.isfinite(fs) or fs <= 0:
        return {}

    f_guess = float(f1_guess_hz)

    # Precompute dt per sample (for totals); robust first element
    dt_row = np.zeros_like(t)
    dt_row[1:] = np.diff(t)

    if i1 - i0 >= 2:
        med_dt = float(np.nanmedian(dt_pos))
        if not np.isfinite(dt_row[i0]) or dt_row[i0] <= 0:
            dt_row[i0] = med_dt
    dt_row = np.where(np.isfinite(dt_row) & (dt_row > 0), dt_row, 0.0)

    # Slice duration
    T_slice = float(t[i1 - 1] - t[i0]) if (i1 - i0) >= 2 else 0.0

    # -------------------- window sizing for FFT --------------------
    Tw = max(target_cycles / f_guess, target_samples / fs)
    halfW = 0.5 * Tw
    Nwin_exp = int(max(16, round(Tw * fs)))
    min_points = int(max(min_points_floor, round(min_points_frac * Nwin_exp)))

    # -------------------- helpers --------------------
    def quadratic_peak_bin(mag, k):
        if k <= 0 or k >= len(mag) - 1:
            return float(k)
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = (a - 2*b + c)
        if denom == 0:
            return float(k)
        return float(k + 0.5*(a - c)/denom)

    def fft_amp_phase_sigma(xw, f_target):
        """
        Extract peak amplitude and phase near f_target using
        robust bin search + quadratic interpolation.
        Fully protected against OOB errors.
        """
        n = len(xw)
        if n < 16:
            return np.nan, np.nan, np.nan

        w = np.hanning(n)
        x0 = (xw - np.mean(xw)) * w
        X = np.fft.rfft(x0)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        mag = np.abs(X)

        # Find nearest bin
        k0 = int(np.argmin(np.abs(freqs - f_target)))

        # Local search window ±2 bins
        last = len(X) - 1
        k_lo = max(1, k0 - 2)
        k_hi = min(last - 1, k0 + 2)

        kk = k_lo + int(np.argmax(mag[k_lo:k_hi+1]))

        # Quadratic interpolation only if safe
        if 1 <= kk <= last - 1:
            kf = float(quadratic_peak_bin(mag, kk))
        else:
            kf = float(kk)

        # ---- HARD CLAMP ----
        if kf < 0.0:
            kf = 0.0
        elif kf > float(last):
            kf = float(last)

        k1 = int(np.floor(kf))
        if k1 < 0:
            k1 = 0
        elif k1 > last:
            k1 = last

        k2 = min(k1 + 1, last)
        frac = kf - k1

        # Interpolated complex amplitude
        Xf = (1.0 - frac) * X[k1] + frac * X[k2]

        # Hann coherent gain
        cg = 0.5
        A_pk = 2.0 * np.abs(Xf) / (n * cg)
        phi = np.angle(Xf)

        # ---- Noise floor estimate ----
        band = np.abs(freqs - f_target) <= noise_span_hz
        idx = np.arange(len(X))
        excl = np.abs(idx - kk) <= noise_exclude_bins
        mask = band & (~excl)

        if np.any(mask):
            sigma_mag = np.std(np.abs(X[mask]))
            vals = np.abs(X[mask])
            if vals.size > 5:
                med = np.median(vals)
                mad = np.median(np.abs(vals - med))
                sigma_mag = 1.4826 * mad
            else:
                sigma_mag = np.std(np.abs(X[mask]))
            sigma_A = 2.0 * sigma_mag / (n * cg)
        else:
            sigma_A = np.nan

        return A_pk, phi, sigma_A

    def estimate_f0_window(xw):
        """
        Robust fundamental frequency estimation near f_guess.
        - Searches within ±f0_span_hz
        - Refines peak locally (±2 bins)
        - Uses quadratic bin interpolation
        - Converts safely to Hz without df_hz
        - Never goes out of bounds
        """
        n = len(xw)
        if n < 32:
            return float(f_guess)

        # Window + FFT
        w = np.hanning(n)
        Xtmp = np.fft.rfft((xw - np.mean(xw)) * w)
        freqs_tmp = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(Xtmp)

        # Restrict search around expected frequency
        m = (freqs_tmp >= (f_guess - f0_span_hz)) & (freqs_tmp <= (f_guess + f0_span_hz))
        if not np.any(m):
            return float(f_guess)

        # Initial peak in masked region
        kk = np.where(m)[0][int(np.argmax(mag[m]))]

        # ---- LOCAL REFINEMENT (±2 bins) ----
        # prevents mask-edge interpolation artifacts
        last = len(mag) - 1
        k_lo = max(1, kk - 2)
        k_hi = min(last - 1, kk + 2)

        kk_refined = k_lo + int(np.argmax(mag[k_lo:k_hi+1]))

        # Quadratic bin interpolation (safe at edges)
        if 1 <= kk_refined <= last - 1:
            kf = float(quadratic_peak_bin(mag, kk_refined))
        else:
            kf = float(kk_refined)

        # Clamp to valid range
        if kf < 0.0:
            kf = 0.0
        elif kf > float(last):
            kf = float(last)

        k1 = int(np.floor(kf))
        if k1 < 0:
            k1 = 0
        elif k1 > last:
            k1 = last

        k2 = min(k1 + 1, last)
        frac = kf - k1

        # Convert bin index -> Hz via interpolation in frequency array
        f0 = (1.0 - frac) * freqs_tmp[k1] + frac * freqs_tmp[k2]

        return float(f0)
    # -------------------- per-sample direct lock-in energy (no FFT) --------------------
    E_li_cyc = None
    if X_lockin_pk is not None:
        Kabs = np.abs(Kpp)
        E_li_cyc = np.pi * Kabs * (X_lockin_pk ** 2)  # J/cycle at each sample

    # Totals for lock-in energy
    E_li_tot = np.nan
    P_li_mean = np.nan
    if E_li_cyc is not None and T_slice > 0:
        f_use = f_guess  # lock-in drive frequency
        dE = np.where(np.isfinite(E_li_cyc), E_li_cyc, 0.0) * f_use * dt_row
        E_li_tot = float(np.sum(dE[i0:i1]))
        P_li_mean = float(E_li_tot / T_slice)

    # -------------------- FFT-based harmonic + augmented energy (per sample i) --------------------
    X1_list, X2_list, X3_list = [], [], []
    s1_list, s2_list, s3_list = [], [], []
    E_fft_list, sE_fft_list = [], []
    dx_list = []
    f0_list = []
    f0_est_list = []
    idx_used = []  # indices where FFT succeeded (for totals)

    for i in range(i0, i1):
        tc = t[i]
        j0 = max(i0, np.searchsorted(t, tc - halfW, side="left"))
        j1 = min(i1, np.searchsorted(t, tc + halfW, side="right"))
        if (j1 - j0) < min_points:
            continue

        xw_nm = x_raw[j0:j1]
        if not np.all(np.isfinite(xw_nm)):
            continue
        xw = xw_nm * 1e-9  # nm -> m
        if not np.all(np.isfinite(xw)):
            continue

        f0_est = estimate_f0_window(xw) if estimate_f0 else f_guess
        f0_est_list.append(f0_est)
        f0 = f_guess #so we just pass the drive frequency as f0.
        f0_list.append(f0)

        harmonics = []
        sigmas = []
        for nh in range(1, max_harmonic + 1):
            fh = nh * f0
            if fh >= nyquist_guard * fs:
                harmonics.append(np.nan)
                sigmas.append(np.nan)
                continue
            A, ph, sA = fft_amp_phase_sigma(xw, fh)
            harmonics.append(A)
            sigmas.append(sA)

        h1, h2, h3 = (harmonics + [np.nan]*3)[:3]
        sh1, sh2, sh3 = (sigmas + [np.nan]*3)[:3]

        # Augmented energy per cycle using the same K'' at this time index
        Kval = float(np.abs(Kpp[i]))
        if not np.isfinite(Kval):
            continue

        # treat missing harmonics as "no contribution" (NaN -> 0) for energy
        h = np.asarray([h1, h2, h3], float)
        sh = np.asarray([sh1, sh2, sh3], float)

        E_fft = np.pi * Kval * float(np.nansum(h * h))

        # sigma propagation: terms with NaN just drop out
        terms = h * sh
        terms2 = terms * terms
        if np.any(np.isfinite(terms2)):
            sE_fft = 2.0 * np.pi * Kval * float(np.sqrt(np.nansum(terms2)))
        else:
            sE_fft = np.nan

        # Hysteresis width (requires tangential force amplitude)
        Fmin =  1e-12 # 10× noise or 1 pN
        if Ft_pk is not None and np.isfinite(Ft_pk[i]) and Ft_pk[i] > Fmin:
            dx_h = E_fft / (4.0 * Ft_pk[i])
        else:
            dx_h = np.nan

        X1_list.append(h1); X2_list.append(h2); X3_list.append(h3)
        s1_list.append(sh1); s2_list.append(sh2); s3_list.append(sh3)
        E_fft_list.append(E_fft); sE_fft_list.append(sE_fft)
        dx_list.append(dx_h)
        idx_used.append(i)
    # -------------------- Build base output dict (always returned) --------------------
    out = {
        _p("fs_est_Hz"): float(fs),             # f sampling _estimated
        _p("Tw_s"): float(Tw),
        _p("min_points"): int(min_points),
        _p("T_slice_s"): float(T_slice),
        _p("E_li_tot_J"): float(E_li_tot),
        _p("P_li_mean_W"): float(P_li_mean),
    }

    # If FFT failed entirely in this slice, return lock-in totals only
    if not idx_used:
        return out

    # Align "used" arrays
    idx_used = np.asarray(idx_used, int)

    X1_fft = np.asarray(X1_list, float)
    X2_fft = np.asarray(X2_list, float)
    X3_fft = np.asarray(X3_list, float)
    dxa    = np.asarray(dx_list, float)
    f0a    = np.asarray(f0_list, float)

    E_fft_arr  = np.asarray(E_fft_list, float)      # per-cycle (uncalibrated)
    sE_fft_arr = np.asarray(sE_fft_list, float)

    # -------------------- Lock-in vs FFT calibration (fundamental amplitude) --------------------
    cX = 1.0
    if X_lockin_pk is not None:
        X1_li = np.asarray(X_lockin_pk, float)[idx_used]    # lock-in peak displacement at used indices
        rr = X1_li / np.maximum(1e-30, X1_fft)
        rr = rr[np.isfinite(rr) & (rr > 0)]
        if rr.size >= 5:
            cX = float(np.nanmedian(rr))

    # Clamp to avoid crazy ratios
    if not np.isfinite(cX):
        cX = 1.0
    cX = float(np.clip(cX, 0.2, 5.0))
    out[_p("cX_li_over_fft_med")] = float(cX)

    # Calibrated harmonic amplitudes
    X1_cal = cX * X1_fft
    X2_cal = cX * X2_fft
    X3_cal = cX * X3_fft

    # -------------------- Medians (raw FFT + calibrated FFT) --------------------
    def _safe_nanmedian(arr):
        a = np.asarray(arr, float)
        return float(np.nanmedian(a)) if np.any(np.isfinite(a)) else np.nan

    out.update({
        # raw FFT medians
        _p("f0_est_med_Hz"):  _safe_nanmedian(f0_est_list), # f0 averaged by fft list
        _p("X1st_fft_med_nm"): _safe_nanmedian(X1_fft)*1e9,
        _p("X2nd_fft_med_nm"): _safe_nanmedian(X2_fft)*1e9,
        _p("X3rd_fft_med_nm"): _safe_nanmedian(X3_fft)*1e9,

        # calibrated FFT medians
        _p("X1st_fft_cal_med_nm"): _safe_nanmedian(X1_cal)*1e9,
        _p("X2nd_fft_cal_med_nm"): _safe_nanmedian(X2_cal)*1e9,
        _p("X3rd_fft_cal_med_nm"): _safe_nanmedian(X3_cal)*1e9,

        _p("THD_fft_med"): _safe_nanmedian(
            np.sqrt(np.nan_to_num(X2_fft)**2 + np.nan_to_num(X3_fft)**2) / np.maximum(1e-30, X1_fft)
        ),
        _p("E_aug_fft_med_J_per_cycle"): _safe_nanmedian(E_fft_arr),
        _p("f0_fft_med_Hz"): _safe_nanmedian(f0a),
        _p("n_fft_points"): int(idx_used.size),

        _p("dx_hyst_med_m"): _safe_nanmedian(dxa),
        _p("dx_hyst_rel_med"): _safe_nanmedian(dxa / np.maximum(1e-30, X1_fft)),
    })

    # -------------------- Lock-in anchored per-cycle baseline --------------------
    E1_li_per_cycle = np.nan
    if np.isfinite(P_li_mean) and (f_guess > 0):
        E1_li_per_cycle = float(P_li_mean / f_guess)  # since P = E_cycle * f
    out[_p("E1_li_mean_J_per_cycle")] = float(E1_li_per_cycle) if np.isfinite(E1_li_per_cycle) else np.nan

    # -------------------- Calibrated harmonic energy per cycle at used indices --------------------
    Kabs_used = np.abs(np.asarray(Kpp, float)[idx_used])  # uses same indices

    # Per-cycle harmonic energy, calibrated
    E_harm_cal = np.pi * Kabs_used * (X1_cal*X1_cal + X2_cal*X2_cal + X3_cal*X3_cal)
    out[_p("E_harm_fft_cal_med_J_per_cycle")] = _safe_nanmedian(E_harm_cal)

    # Optional: harmonic-only nonlinearity fraction (dimensionless), lock-in anchored scale
    eta_23_over_1 = (X2_cal*X2_cal + X3_cal*X3_cal) / np.maximum(1e-30, X1_cal*X1_cal)
    out[_p("eta_23_over_1_med")] = _safe_nanmedian(eta_23_over_1)

    # Lock-in anchored nonlinear energy estimate (positive-definite if harmonics exist)
    if np.isfinite(E1_li_per_cycle):
        out[_p("E_nl_star_med_J_per_cycle")] = _safe_nanmedian(E1_li_per_cycle * eta_23_over_1)

    # -------------------- Totals over slice --------------------
    if T_slice > 0:
        if use_f0_for_totals:
            f_use = np.where(np.isfinite(f0a) & (f0a > 0), f0a, f_guess)
        else:
            f_use = np.full_like(E_fft_arr, f_guess, dtype=float)

        # Uncalibrated FFT total (kept for debugging)
        dE_fft = np.where(np.isfinite(E_fft_arr), E_fft_arr, 0.0) * f_use * dt_row[idx_used]
        E_fft_tot = float(np.sum(dE_fft))
        out[_p("E_aug_fft_tot_J")] = E_fft_tot
        out[_p("P_aug_fft_mean_W")] = float(E_fft_tot / T_slice)

        # Calibrated harmonic total (recommended)
        dE_harm_cal = np.where(np.isfinite(E_harm_cal), E_harm_cal, 0.0) * f_use * dt_row[idx_used]
        E_harm_cal_tot = float(np.sum(dE_harm_cal))
        out[_p("E_harm_fft_cal_tot_J")] = E_harm_cal_tot
        out[_p("P_harm_fft_cal_mean_W")] = float(E_harm_cal_tot / T_slice)

        # “Consistency diagnostic” only (NOT physical nonlinearity): calibrated harmonic total vs lock-in total
        if np.isfinite(E_li_tot):
            out[_p("E_consistency_cal_minus_li_tot_J")] = float(E_harm_cal_tot - E_li_tot)
            out[_p("E_consistency_cal_minus_li_frac")] = float((E_harm_cal_tot - E_li_tot) / max(1e-30, E_li_tot))

    return out