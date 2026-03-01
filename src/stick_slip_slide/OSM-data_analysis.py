from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import copy, traceback

# Optional SciPy for Mindlin fit (preferred)
try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# 1) Units parsing (2nd row in CSV files)
# ============================================================

PREFIX = {
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}

def pick_folder_gui() -> str:
    """
    Opens a native folder picker (Windows/macOS/Linux) using tkinter.
    Returns the selected folder path, or raises if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(
            "tkinter is not available in this Python environment. "
            "Install/enable tkinter or pass --batch <folder>."
        ) from e

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select folder containing CSV files")
    root.destroy()

    if not folder:
        raise RuntimeError("No folder selected (cancelled).")
    return folder

def clean_unit_str(u: str) -> str:
    if u is None:
        return ""
    u = str(u).strip()
    u = u.replace("Â", "")      # fix ÂµN artifacts
    u = u.replace("μ", "µ")     # normalize mu variants
    return u

def parse_simple_unit(token: str) -> tuple[float, str]:
    token = clean_unit_str(token)
    if token == "" or token.lower() in {"none", "nan"}:
        return (1.0, "")
    if token in {"C", "°C"}:
        return (1.0, "C")

    m = re.fullmatch(r"([nµumkMG]?)([A-Za-z]+)", token)
    if not m:
        return (1.0, token)
    pref, base = m.group(1), m.group(2)
    if pref == "u":
        pref = "µ"
    return (PREFIX.get(pref, 1.0), base)

def parse_compound_unit(u: str) -> tuple[float, str]:
    u = clean_unit_str(u)
    if u == "" or u.lower() in {"none", "nan"}:
        return (1.0, "")

    if "/" in u:
        num, den = u.split("/", 1)
        s_num, base_num = parse_simple_unit(num)
        s_den, base_den = parse_simple_unit(den)
        unit_str = f"{base_num}/{base_den}".strip("/")
        return (s_num / s_den, unit_str)

    s, base = parse_simple_unit(u)
    return (s, base)

def read_csv_with_units(filepath: Path) -> tuple[pd.DataFrame, dict, dict]:
    """
    Assumes:
      Row 0: headers
      Row 1: units row
      Row 2+: numeric data
    Returns:
      df_data (units row removed)
      units_map[col] = unit string (cleaned)
      scale_to_SI[col] = multiplier to SI base units
    """
    raw = pd.read_csv(filepath, header=0, low_memory=False)
    if len(raw) < 2:
        raise RuntimeError("CSV too short: missing units row / data.")

    units_row = raw.iloc[0].to_dict()
    units_map = {c: clean_unit_str(units_row.get(c, "")) for c in raw.columns}
    scale_to_SI = {c: parse_compound_unit(units_map[c])[0] for c in raw.columns}

    df = raw.iloc[1:].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.reset_index(drop=True, inplace=True)
    return df, units_map, scale_to_SI


# ============================================================
# 2) Helpers
# ============================================================

def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy()

def safe_nanmax(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float(np.nanmax(x))

def safe_nanmin(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float(np.nanmin(x))

def robust_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan

def rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    return pd.Series(x).rolling(n, center=True, min_periods=1).median().to_numpy()

def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size - 1]
    return list(zip(starts, ends))

def median_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    return float(np.median(dt)) if dt.size else 1.0

def window_idx(t: np.ndarray, center_i: int, halfwidth_s: float) -> np.ndarray:
    t0 = t[center_i]
    return np.where((t >= t0 - halfwidth_s) & (t <= t0 + halfwidth_s))[0]

def window_idx_fw(t: np.ndarray, start_i: int, width_s: float) -> np.ndarray:
    t0 = t[start_i]
    return np.where((t >= t0) & (t <= t0 + width_s))[0]

def rms_to_peak(x_rms: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0) * x_rms

def phase_to_rad(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    if safe_nanmax(np.abs(phi)) > 7.0:
        return np.deg2rad(phi)
    return phi

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

def robust_fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 10:
        return (np.nan, np.nan)
    qlo, qhi = np.quantile(x, [0.05, 0.95])
    mm = (x >= qlo) & (x <= qhi)
    if mm.sum() >= 10:
        x2, y2 = x[mm], y[mm]
    else:
        x2, y2 = x, y
    a, b = np.polyfit(x2, y2, 1)
    return float(a), float(b)

def find_shear_window_from_normal_load(
    t: np.ndarray,
    P_contact_N: np.ndarray,
    touch_i: int,
    smooth_n: int = 501,
    dpdt_thr_frac: float = 0.02,
    min_stable_s: float = 0.5,
) -> tuple[int, int]:
    """
    Returns (i0, i1) indices for the window where:
      - loading is finished and normal load is roughly stable
      - unloading has not started yet
    Uses derivative thresholds based on max |dP/dt| after touch.

    dpdt_thr_frac: fraction of max |dP/dt| used as near-zero threshold.
    """
    n = len(t)
    dt = median_dt(t)
    w = max(11, smooth_n)

    P = pd.Series(P_contact_N).rolling(w, center=True, min_periods=1).median().to_numpy()
    dPdt = np.gradient(P, t)

    # consider region after touch
    post = np.arange(max(0, touch_i), n)

    # scale threshold from typical derivative magnitude after touch
    scale = safe_nanmax(np.abs(dPdt[post])) if post.size else safe_nanmax(np.abs(dPdt))
    if not np.isfinite(scale) or scale <= 0:
        # fallback: use a simple window after touch
        return (min(n-1, touch_i + int(1.0/dt)), n-1)

    thr = dpdt_thr_frac * scale

    # Stable region = |dP/dt| <= thr
    stable = np.abs(dPdt) <= thr

    # Find first stable segment after a loading phase (dP/dt positive earlier)
    min_stable_pts = max(5, int(min_stable_s / max(dt, 1e-12)))

    # Start search after touch
    stable_regs = [(s, e) for (s, e) in contiguous_regions(stable) if (e - s + 1) >= min_stable_pts and e > touch_i]
    if not stable_regs:
        return (min(n-1, touch_i + int(1.0/dt)), n-1)

    # choose the first stable region that occurs after some positive loading
    i0 = None
    for s, e in stable_regs:
        # check that shortly before s we had positive slope (loading)
        pre_s = max(touch_i, s - 5*min_stable_pts)
        if np.nanmedian(dPdt[pre_s:s]) > thr:
            i0 = s
            break
    if i0 is None:
        i0 = stable_regs[0][0]

    # Unloading start: first long segment where dP/dt is negative beyond threshold
    neg = dPdt < -thr
    neg_regs = [(s, e) for (s, e) in contiguous_regions(neg) if (e - s + 1) >= min_stable_pts and s > i0]
    if neg_regs:
        i1 = neg_regs[0][0]
    else:
        i1 = n - 1

    # guard
    if i1 <= i0:
        i0 = min(n-2, i0)
        i1 = n - 1

    return int(i0), int(i1)

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

def effective_modulus(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """Hertz reduced modulus E* (Pa)."""
    inv = (1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2
    return 1.0 / inv if inv > 0 else np.nan

def hertz_fit_radius(
    h_m: np.ndarray,
    P_N: np.ndarray,
    E_star_Pa: float,
    hardness_Pa: float,
    plasticity_p0_frac: float = 1.0,
    min_h_m: float = 5e-9,
    max_frac_of_Pmax: float = 0.95,
    min_points: int = 50,
    n_iter: int = 3,
) -> dict:
    """
    Fits Hertz sphere: P = (4/3) E* sqrt(R) h^(3/2)
    Fit is linear in x = h^(3/2): P = C x, with C = (4/3) E* sqrt(R).
    Returns dict with R_eff, C, rmse, masks, etc.
    """
    h = np.asarray(h_m, float)
    P = np.asarray(P_N, float)

    m = np.isfinite(h) & np.isfinite(P) & (h > min_h_m) & (P > 0) & np.isfinite(E_star_Pa) & (E_star_Pa > 0)
    if m.sum() < min_points:
        return {"ok": 0, "reason": "not enough points", "R_eff_m": np.nan, "C": np.nan, "rmse_N": np.nan}

    # restrict to <= max_frac_of_Pmax
    Pmax = safe_nanmax(P[m])
    if not np.isfinite(Pmax) or Pmax <= 0:
        return {"ok": 0, "reason": "Pmax invalid", "R_eff_m": np.nan, "C": np.nan, "rmse_N": np.nan}
    m &= (P <= max_frac_of_Pmax * Pmax)

    if m.sum() < min_points:
        return {"ok": 0, "reason": "not enough points after Pmax fraction cut", "R_eff_m": np.nan, "C": np.nan, "rmse_N": np.nan}

    # iterative fit -> compute p0 -> filter (optional)
    m_fit = m.copy()
    C = np.nan
    R_eff = np.nan
    rmse = np.nan

    for it in range(max(1, int(n_iter))):
        idx = np.where(m_fit)[0]
        if idx.size < min_points:
            break

        x = np.power(h[idx], 1.5)  # h^(3/2)
        y = P[idx]

        # least squares with zero intercept (Hertz predicts zero at h=0)
        # C = (x·y) / (x·x)
        denom = float(np.dot(x, x))
        if denom <= 0:
            break
        C = float(np.dot(x, y) / denom)

        # radius from C
        # C = (4/3) E* sqrt(R)  => sqrt(R) = (3C)/(4E*)
        R_eff = float(((3.0 * C) / (4.0 * E_star_Pa)) ** 2)

        # prediction + rmse
        yhat = C * x
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

        # plasticity filter (optional)
        if not (np.isfinite(hardness_Pa) and hardness_Pa > 0):
            # no filtering
            continue

        # Hertz contact radius a = sqrt(R h)
        a = np.sqrt(np.maximum(1e-30, R_eff * h))
        # max contact pressure p0 = 3P/(2πa^2)
        p0 = (3.0 * P) / (2.0 * np.pi * np.maximum(1e-30, a**2))

        # keep only points that remain elastic
        m_new = m_fit & np.isfinite(p0) & (p0 <= plasticity_p0_frac * hardness_Pa)
        # if no change -> stop
        if np.array_equal(m_new, m_fit):
            break
        m_fit = m_new

    ok = 1 if (np.isfinite(R_eff) and R_eff > 0 and np.isfinite(C)) else 0
    return {
        "ok": ok,
        "E_star_Pa": float(E_star_Pa),
        "C": float(C),
        "R_eff_m": float(R_eff),
        "rmse_N": float(rmse),
        "n_used": int(np.where(m_fit)[0].size),
        "mask_used": m_fit,
    }

def hertz_apparent_radius_R_of_h(h_m: np.ndarray, P_N: np.ndarray, E_star_Pa: float) -> np.ndarray:
    """
    Pointwise apparent radius from rearranged Hertz:
      P = (4/3)E* sqrt(R) h^(3/2)
      => R = [ (3P)/(4E* h^(3/2)) ]^2
    """
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

## adhesion calculations with an effective roughness in contact
def w_eff_from_roughness(
    w_J_per_m2: float,
    sigma_rms_m: float | None,
    model: str = "none",
    delta0_m: float = 0.3e-9,
) -> float:
    """
    Returns effective work of adhesion w_eff >= 0.
    model:
      - "none": w_eff = w
      - "exp":  w_eff = w * exp(-(sigma/delta0)^2)   (simple, monotonic, stable)
      - "user": caller already precomputed w_eff, just pass it as w_J_per_m2 and set model="none"
    """
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

    # fallback
    return w

### Hertz load to depth
def _hertz_load_from_h(h_m: np.ndarray, R_m: float, E_star_Pa: float) -> np.ndarray:
    h = np.asarray(h_m, float)
    out = np.full_like(h, np.nan, dtype=float)
    m = np.isfinite(h) & (h > 0) & np.isfinite(R_m) & (R_m > 0) & np.isfinite(E_star_Pa) & (E_star_Pa > 0)
    if np.any(m):
        out[m] = (4.0/3.0) * E_star_Pa * np.sqrt(R_m) * (h[m] ** 1.5)
    return out

## Tabor parameter: decide DMT or JKR
def _c_from_tabor(mu: float) -> float:
    """
    Pull-off coefficient interpolation for the *transition* (Maugis-like) regime:
      DMT: c=2
      JKR: c=1.5
    This is for Fadh = c*pi*R*w_eff (pull-off coefficient).
    For 'transition' we use it as a constant offset model.
    """ 
    if not np.isfinite(mu):
        return 0.0
    # smooth monotone map mu: 0.1 -> ~2, 5 -> ~1.5
    # simple logistic-ish in log(mu)
    x = np.log10(max(mu, 1e-12))
    # clamp between [0.1,5] in log-space roughly [-1,0.699]
    x0, x1 = np.log10(0.1), np.log10(5.0)
    t = (x - x0) / (x1 - x0)
    t = float(np.clip(t, 0.0, 1.0))
    return float(2.0 - 0.5*t)  # 2 -> 1.5

def _tabor_mu(R_m: float, w_eff: float, E_star_Pa: float, z0_m: float) -> float:
    if not (np.isfinite(R_m) and R_m > 0 and np.isfinite(w_eff) and w_eff >= 0 and
            np.isfinite(E_star_Pa) and E_star_Pa > 0 and np.isfinite(z0_m) and z0_m > 0):
        return np.nan
    return float(((R_m * (w_eff**2)) / ((E_star_Pa**2) * (z0_m**3))) ** (1.0/3.0))

# ---------------------------------------------------------
# JKR forward model P(h; R, w, E*) via solve-for-a approach
# ---------------------------------------------------------

def _jkr_P_from_a(a_m: np.ndarray, R_m: float, E_star_Pa: float, w_J_per_m2: float) -> np.ndarray:
    """
    JKR load vs contact radius a:
      P(a) = (4/3) E* a^3 / R - sqrt(8*pi*w*E* a^3)
    """
    a = np.asarray(a_m, dtype=float)
    R = float(R_m); E = float(E_star_Pa); w = float(w_J_per_m2)
    term_el = (4.0/3.0) * E * (a**3) / R
    term_adh = np.sqrt(np.maximum(0.0, 8.0*np.pi*w*E*(a**3)))
    return term_el - term_adh

def _jkr_h_from_a(a_m: np.ndarray, R_m: float, E_star_Pa: float, w_J_per_m2: float) -> np.ndarray:
    """
    JKR indentation depth (approach) vs a:
      h(a) = a^2/R - sqrt(8*pi*w*a/(3*E*))
    """
    a = np.asarray(a_m, dtype=float)
    R = float(R_m); E = float(E_star_Pa); w = float(w_J_per_m2)
    term_geom = (a**2)/R
    term_adh = np.sqrt(np.maximum(0.0, (8.0*np.pi*w*a)/(3.0*E)))
    return term_geom - term_adh
def _jkr_P_from_h(h_m: np.ndarray, R_m: float, E_star_Pa: float, w_J_per_m2: float,
                  n_bisect: int = 60) -> np.ndarray:
    """
    Compute P(h) for JKR by bisection solve of h(a)=h, then P(a).
    Robust for the *loading* branch (h >= 0). If h<0, returns NaN.
    """
    h = np.asarray(h_m, dtype=float)
    out = np.full_like(h, np.nan, dtype=float)

    R = float(R_m); E = float(E_star_Pa); w = float(w_J_per_m2)
    if not (R > 0 and E > 0 and w > 0):
        return out

    for i in range(h.size):
        hi = h[i]
        if not (np.isfinite(hi) and hi >= 0):
            continue

        # bracket on a
        a_lo = 0.0
        a_hi = np.sqrt(max(R*hi, 0.0)) * 5.0 + 1e-12  # Hertz-ish guess with margin

        # expand until h(a_hi) >= target
        for _ in range(12):
            h_hi = _jkr_h_from_a(np.array([a_hi]), R, E, w)[0]
            if np.isfinite(h_hi) and (h_hi >= hi):
                break
            a_hi *= 2.0

        h_hi = _jkr_h_from_a(np.array([a_hi]), R, E, w)[0]
        if not (np.isfinite(h_hi) and h_hi >= hi):
            continue  # failed to bracket

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

# ----------------------------------------
# AUTO decision tree (DMT / JKR / inbetween)
# ----------------------------------------

def _auto_model_from_mu(mu: float,
                        mu_dmt: float = 0.1,
                        mu_jkr: float = 5.0) -> str:
    """
    Heuristic thresholds:
      mu <= 0.1 -> DMT
      mu >= 5   -> JKR
      else      -> "transition"
    """
    if not np.isfinite(mu):
        return "hertz"
    if mu <= mu_dmt:
        return "dmt"
    if mu >= mu_jkr:
        return "jkr"
    return "transition"

## Hertz fit with adhesion model included
def hertz_fit_radius_adhesion(
    h_m: np.ndarray,
    P_N: np.ndarray,
    *,
    E_star_Pa: float,
    # --- Adhesion / interaction model ---
    adhesion_model: str = "auto",          # "hertz"|"dmt"|"transition"|"jkr"|"auto"
    w_J_per_m2: float = 0.0,               # work of adhesion (or "bare" adhesion before roughness correction)
    sigma_rms_m: float | None = None,      # rms roughness for effective adhesion reduction
    rough_model: str = "none",             # "none"|"exp"|... (w_eff_from_roughness model names)
    delta0_m: float = 0.3e-9,              # roughness/interaction length scale
    z0_m: float = 0.3e-9,                  # interaction range for Tabor parameter
    mu_dmt: float = 0.1,                   # auto threshold: mu < mu_dmt => DMT-like
    mu_jkr: float = 5.0,                   # auto threshold: mu > mu_jkr => JKR-like

    # --- Data window / robustness ---
    min_h_m: float = 0.0,
    max_frac_of_Pmax: float = 1.0,
    min_points: int = 8,
    n_iter: int = 6,                       # self-consistent iterations because mu depends on R
    R0_m: float | None = None,             # optional initial guess

    # --- Optional stiffness consistency check ---
    Sz_meas_N_per_m: np.ndarray | None = None,  # measured normal stiffness aligned with h,P
    dh_stiff_m: float = 0.25e-9,                # finite-difference step for dP/dh
    stiff_wt: float = 0.0,                      # weight of stiffness RMSE in objective (0 -> ignore stiffness)
) -> dict:
    """
    Fit an effective sphere radius R from a normal force–indentation curve P(h), optionally
    enforcing consistency with measured normal contact stiffness Sz(h) = dP/dh.

    Physics basis
    -------------
    For an elastic sphere-on-flat (effective modulus E* and radius R):

        a(h) ~ sqrt(R h)                      (geometric contact radius)
        P(h) = (4/3) E* sqrt(R) h^(3/2)       (Hertz load–depth relation)
        Sz(h) = dP/dh = 2 E* sqrt(R h)        (Hertz tangent stiffness)

    Adhesion modifies the load–depth relation. In the simplest DMT-like view, adhesion
    adds an (approximately) constant negative force offset proportional to R and w:

        P(h) ≈ P_Hertz(h;R) - 2π R w_eff      (DMT)

    In the JKR view, adhesion modifies the contact mechanics more strongly and P(h)
    is not a simple offset; we evaluate JKR numerically via _jkr_P_from_h().

    We keep "transition" model, interpolating between DMT and JKR pull-off factors,
    and "auto" selection using the Tabor parameter mu(R, w_eff, E*, z0).

    Stiffness check
    ---------------
    When Sz is provided, we compute Sz_pred(h) from the *same* P(h) model via finite
    difference:

        Sz_pred(h) ≈ [P(h+dh) - P(max(0, h-dh))] / (2 dh)

    The objective is:
        rmse_combined = rmse_P + stiff_wt * rmse_Sz

    Choose stiff_wt ~ 0.0 for "check only"
    """

    # ---------------------------------------------------------------------
    # 1) Sanitize and filter data
    # ---------------------------------------------------------------------
    h = np.asarray(h_m, float)
    P = np.asarray(P_N, float)

    Sz = None
    if Sz_meas_N_per_m is not None:
        Sz = np.asarray(Sz_meas_N_per_m, float)

    # Finite and physical (h>=0). If Sz provided, require it finite too.
    m = np.isfinite(h) & np.isfinite(P) & (h >= 0)
    if Sz is not None:
        m &= np.isfinite(Sz)
    h = h[m]
    P = P[m]
    if Sz is not None:
        Sz = Sz[m]

    if h.size < int(min_points):
        return {"ok": 0, "reason": "not_enough_points", "n_used": int(h.size)}

    # Optional minimum indentation cutoff (avoid pre-contact creep / noise)
    if np.isfinite(min_h_m) and float(min_h_m) > 0:
        mm = h >= float(min_h_m)
        h = h[mm]
        P = P[mm]
        if Sz is not None:
            Sz = Sz[mm]

    if h.size < int(min_points):
        return {"ok": 0, "reason": "not_enough_points_after_min_h", "n_used": int(h.size)}

    # Optional top-load cutoff (avoid plasticity at high load)
    Pmax = float(np.nanmax(P)) if P.size else np.nan
    if np.isfinite(Pmax) and Pmax > 0 and np.isfinite(max_frac_of_Pmax) and (0 < float(max_frac_of_Pmax) < 1.0):
        mm = P <= float(max_frac_of_Pmax) * Pmax
        h = h[mm]
        P = P[mm]
        if Sz is not None:
            Sz = Sz[mm]

    if h.size < int(min_points):
        return {"ok": 0, "reason": "not_enough_points_after_range", "n_used": int(h.size)}

    # Validate modulus
    E = float(E_star_Pa)
    if not (np.isfinite(E) and E > 0):
        return {"ok": 0, "reason": "bad_E_star", "n_used": int(h.size)}

    # ---------------------------------------------------------------------
    # 2) Effective adhesion with roughness correction
    # ---------------------------------------------------------------------
    # project convention: reduce adhesion via a roughness model -> w_eff
    w_eff = w_eff_from_roughness(w_J_per_m2, sigma_rms_m, model=rough_model, delta0_m=delta0_m)
    if not np.isfinite(w_eff) or w_eff < 0:
        w_eff = 0.0

    # ---------------------------------------------------------------------
    # 3) Initial guess for R from Hertz scaling P ~ h^(3/2)
    # ---------------------------------------------------------------------
    # For Hertz: P = (4/3) E sqrt(R) h^(3/2)  => sqrt(R) = K / ((4/3)E)
    x = np.power(h, 1.5)
    if R0_m is not None and np.isfinite(R0_m) and R0_m > 0:
        R = float(R0_m)
    else:
        mm = np.isfinite(x) & (x > 0) & np.isfinite(P)
        if np.sum(mm) >= 2:
            K = float(np.nanmedian(P[mm] / x[mm]))
            if np.isfinite(K) and K > 0:
                sqrtR = K / ((4.0 / 3.0) * E)
                R = float(max(1e-12, sqrtR**2))
            else:
                R = 1e-6
        else:
            R = 1e-6

    # Requested model choice
    model_req = (adhesion_model or "auto").strip().lower()

    # ---------------------------------------------------------------------
    # 4) Model prediction helpers: P(h;R) and stiffness Sz(h;R)
    # ---------------------------------------------------------------------
    def predict_P(hh: np.ndarray, Rm: float, model_name: str, mu_val_local: float) -> tuple[np.ndarray, float, float]:
        """
        Return (P_pred, c_pull, Fadh_mag) for the chosen adhesion model.
        Fadh_mag is returned as positive magnitude for reporting.
        """
        # If no effective adhesion or Hertz selected: standard Hertz
        if w_eff <= 0 or model_name == "hertz":
            return _hertz_load_from_h(hh, Rm, E), 0.0, 0.0

        if model_name == "dmt":
            c_pull = 2.0
            Fadh = c_pull * np.pi * Rm * w_eff
            return _hertz_load_from_h(hh, Rm, E) - Fadh, c_pull, Fadh

        if model_name == "transition":
            c_pull = _c_from_tabor(mu_val_local)
            Fadh = c_pull * np.pi * Rm * w_eff
            return _hertz_load_from_h(hh, Rm, E) - Fadh, c_pull, Fadh

        if model_name == "jkr":
            # Full JKR: evaluate P(h) numerically
            c_pull = 1.5
            Fadh = c_pull * np.pi * Rm * w_eff
            return _jkr_P_from_h(hh, Rm, E, w_eff, n_bisect=60), c_pull, Fadh

        # Fallback: Hertz
        return _hertz_load_from_h(hh, Rm, E), 0.0, 0.0

    def rmse(y: np.ndarray, yhat: np.ndarray, nmin: int) -> float:
        mm = np.isfinite(y) & np.isfinite(yhat)
        if np.sum(mm) < int(nmin):
            return np.inf
        return float(np.sqrt(np.mean((y[mm] - yhat[mm])**2)))

    def rmse_Sz_for_R(Rm: float, model_name: str, mu_val_local: float) -> tuple[float, int]:
        """
        Compute RMSE between measured Sz and predicted Sz = dP/dh by finite difference.
        """
        if Sz is None:
            return np.nan, 0
        dh = float(dh_stiff_m)
        if not (np.isfinite(dh) and dh > 0):
            return np.nan, 0

        # central difference with clamp at 0 for h-dh
        hm = np.maximum(0.0, h - dh)
        hp = h + dh
        Pp, _, _ = predict_P(hp, Rm, model_name, mu_val_local)
        Pm, _, _ = predict_P(hm, Rm, model_name, mu_val_local)
        Sz_pred = (Pp - Pm) / (2.0 * dh)

        mm = np.isfinite(Sz) & np.isfinite(Sz_pred) & (Sz > 0) & (Sz_pred > 0)
        n_ok = int(np.sum(mm))
        if n_ok < int(min_points):
            return np.inf, n_ok
        return float(np.sqrt(np.mean((Sz[mm] - Sz_pred[mm])**2))), n_ok

    # ---------------------------------------------------------------------
    # 5) Radius update step for Hertz-like models
    # ---------------------------------------------------------------------
    def fit_R_from_Peff(Peff: np.ndarray) -> float:
        """
        For Hertz-type scaling Peff ~ h^(3/2), update R from median slope.
        Works for Hertz and for DMT/transition after converting to Peff = P + Fadh.
        """
        mm = np.isfinite(Peff) & np.isfinite(x) & (x > 0) & (Peff > 0)
        if np.sum(mm) < 2:
            return np.nan
        K = float(np.nanmedian(Peff[mm] / x[mm]))
        if not (np.isfinite(K) and K > 0):
            return np.nan
        sqrtR = K / ((4.0 / 3.0) * E)
        return float(max(1e-12, sqrtR**2))

    # ---------------------------------------------------------------------
    # 6) Self-consistent loop (mu depends on R)
    # ---------------------------------------------------------------------
    chosen_model = "hertz"
    mu_val = np.nan

    for _ in range(int(max(1, n_iter))):
        # model selection
        if model_req == "auto":
            mu_val = _tabor_mu(R, w_eff, E, z0_m) if (w_eff > 0) else np.nan
            chosen_model = _auto_model_from_mu(mu_val, mu_dmt=mu_dmt, mu_jkr=mu_jkr) if (w_eff > 0) else "hertz"
        else:
            chosen_model = model_req
            mu_val = _tabor_mu(R, w_eff, E, z0_m) if (w_eff > 0) else np.nan

        # If effectively Hertz: one-step slope update
        if w_eff <= 0 or chosen_model == "hertz":
            R_new = fit_R_from_Peff(P)
            if np.isfinite(R_new):
                R = 0.5 * R + 0.5 * R_new
            continue

        # DMT/transition: convert to "effective Hertz load" Peff = P + Fadh(R)
        if chosen_model in ("dmt", "transition"):
            if chosen_model == "dmt":
                c_pull = 2.0
            else:
                c_pull = _c_from_tabor(mu_val)
            Fadh = c_pull * np.pi * R * w_eff
            Peff = P + Fadh
            R_new = fit_R_from_Peff(Peff)
            if np.isfinite(R_new):
                R = 0.5 * R + 0.5 * R_new
            continue

        # JKR: no linearization; do a lightweight grid search on R.
        if chosen_model == "jkr":
            R_center = float(max(R, 1e-12))
            grid = R_center * np.logspace(-1.0, 1.0, 31)

            best_R = R_center
            best_obj = np.inf

            for Rc in grid:
                P_pred, _, _ = predict_P(h, Rc, "jkr", mu_val)
                rP = rmse(P, P_pred, min_points)

                obj = rP
                if (Sz is not None) and (float(stiff_wt) > 0):
                    rS, _ = rmse_Sz_for_R(Rc, "jkr", mu_val)
                    obj = rP + float(stiff_wt) * rS

                if obj < best_obj:
                    best_obj = obj
                    best_R = float(Rc)

            # refine around the best radius
            grid2 = best_R * np.logspace(-0.3, 0.3, 25)
            for Rc in grid2:
                P_pred, _, _ = predict_P(h, Rc, "jkr", mu_val)
                rP = rmse(P, P_pred, min_points)

                obj = rP
                if (Sz is not None) and (float(stiff_wt) > 0):
                    rS, _ = rmse_Sz_for_R(Rc, "jkr", mu_val)
                    obj = rP + float(stiff_wt) * rS

                if obj < best_obj:
                    best_obj = obj
                    best_R = float(Rc)

            R = 0.5 * R + 0.5 * best_R
            continue

        # Unknown model -> fallback to Hertz slope
        R_new = fit_R_from_Peff(P)
        if np.isfinite(R_new):
            R = 0.5 * R + 0.5 * R_new

    # ---------------------------------------------------------------------
    # 7) Final evaluation and metrics (P RMSE + optional Sz RMSE)
    # ---------------------------------------------------------------------
    if model_req == "auto":
        mu_val = _tabor_mu(R, w_eff, E, z0_m) if (w_eff > 0) else np.nan
        chosen_model = _auto_model_from_mu(mu_val, mu_dmt=mu_dmt, mu_jkr=mu_jkr) if (w_eff > 0) else "hertz"
    else:
        chosen_model = model_req

    P_pred, c_pull, Fadh_N = predict_P(h, R, chosen_model, mu_val)
    rmse_P = rmse(P, P_pred, min_points)

    rmse_Sz_val = np.nan
    n_Sz_used = 0
    if (Sz is not None) and (float(stiff_wt) > 0):
        rmse_Sz_val, n_Sz_used = rmse_Sz_for_R(R, chosen_model, mu_val)

    rmse_combined = rmse_P
    if np.isfinite(rmse_Sz_val) and (float(stiff_wt) > 0):
        rmse_combined = rmse_P + float(stiff_wt) * rmse_Sz_val

    return {
        "ok": 1 if (np.isfinite(R) and R > 0 and np.isfinite(rmse_P)) else 0,
        "reason": "",
        "E_star_Pa": float(E),
        "R_eff_m": float(R),

        # fit quality on load curve
        "rmse_N": float(rmse_P),
        "rmse_mN": float(rmse_P * 1e3) if np.isfinite(rmse_P) else np.nan,
        "n_used": int(h.size),

        # optional stiffness diagnostics
        "rmse_Sz_N_per_m": float(rmse_Sz_val) if np.isfinite(rmse_Sz_val) else np.nan,
        "n_Sz_used": int(n_Sz_used),
        "rmse_combined": float(rmse_combined) if np.isfinite(rmse_combined) else np.nan,
        "stiff_wt": float(stiff_wt),
        "dh_stiff_m": float(dh_stiff_m),

        # model + adhesion bookkeeping
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
    }

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


def show_and_wait(fig_title: str = "", figures: List[plt.Figure] = []):
    # show the active figure and block until closed
    try:
        if fig_title and plt.get_fignums():
            plt.gcf().canvas.manager.set_window_title(fig_title)
    except Exception:
        pass
    plt.draw()
    plt.pause(0.01)
    def on_key(event):
        if event.key == "enter":
            plt.close("all")
        else:
            pass
    try:
        for fig in figures:
            fig.canvas.mpl_connect("key_press_event", on_key)
    except Exception:
        pass
    plt.show(block=True)  # blocks until the window is closed
    return []

def want_manual(cfg: Config, mode: str, failed: bool) -> bool:
    if mode == "never": return False
    if mode == "always": return True
    return bool(failed)  # "on_fail"

def approve_or_repick_gate(figures, fig_title: str = "") -> str:
    """
    Returns: "accept" | "repick_touch" | "repick_window" | "repick_cycles"
    Raises on skip/abort.
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
        elif k == "s":
            decision["val"] = "skip"
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
    if decision["val"] == "skip":
        raise RuntimeError("User skipped file.")
    raise RuntimeError("User aborted.")
# ============================================================

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

def stiffness_from_model(h, P_model_func, eps_rel=1e-4, eps_abs=1e-12):
    h = np.asarray(h, float)
    eps = np.maximum(eps_abs, eps_rel*np.maximum(h, 0.0))
    hp = h + eps
    hm = np.maximum(0.0, h - eps)
    Pp = P_model_func(hp)
    Pm = P_model_func(hm)
    return (Pp - Pm) / (hp - hm)

def area_from_flat_end_fit(
    P_N: np.ndarray,
    *,
    S0_N_per_m: float,
    C: float,
    E_star_Pa: float,
) -> np.ndarray:
    P = np.asarray(P_N, float)
    E = float(E_star_Pa)

    # S(P) defined only for P>=0; clamp negatives
    Pp = np.maximum(0.0, P)
    S_pred = float(S0_N_per_m) + float(C) * np.cbrt(Pp)

    a = S_pred / (2.0 * E)
    a = np.maximum(0.0, a)
    return np.pi * a * a

def fit_flat_end_stiffness(
    P_N: np.ndarray,
    S_N_per_m: np.ndarray,
    *,
    E_star_Pa: float,
    P_min_N: float | None = None,
    P_max_N: float | None = None,
    robust: bool = True,
    n_iter: int = 6,
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
    def _wls(beta_w: np.ndarray):
        W = beta_w[:, None]  # (n,1)
        XtWX = X.T @ (W * X)
        XtWy = X.T @ (beta_w * y)
        beta = np.linalg.solve(XtWX, XtWy)
        yhat = X @ beta
        resid = y - yhat
        return beta, yhat, resid, XtWX

    # ---- Robust loop (Huber + optional sigma-clip) ----
    w = np.ones_like(y)
    beta = None
    XtWX = None
    for _ in range(int(max(1, n_iter if robust else 1))):
        beta, yhat, resid, XtWX = _wls(w)

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

def build_Aref_samples(
    *,
    area_mode_used: str,
    A_ref: float,
    h_ref: float,
    P_ref: float,
    E_star_Pa: float,
    cfg,
    hertz: dict | None,
    boot_flat: dict | None,
    sigma_A_ref: float | None,
    n_fallback: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """
    Return samples of A_ref for uncertainty propagation.

    - nominal: Gaussian on A_ref using analytic sigma_A_ref (from sigma_h and sigma_R).
    - fit_hertz: sample R_eff from Hertz bootstrap CI/std, propagate A ~ pi*h*R for small indentation
               (Hertzian area_pi_h_R does this properly already).
    - flat_end: sample directly from flat bootstrap samples of a_flat_m (preferred),
                or from R_eff_m samples

    Important: We treat h_ref and P_ref as fixed scalars here.
    """
    rng = np.random.default_rng(int(seed))

    mode = (area_mode_used or "").lower()

    # helper: safe nominal Gaussian
    def _gauss_positive(mu, sig, n):
        if not (np.isfinite(mu) and mu > 0 and np.isfinite(sig) and sig > 0):
            return np.full(int(n), float(mu))
        x = rng.normal(float(mu), float(sig), size=int(n))
        # clamp to small positive to avoid division blowups
        return np.maximum(x, 1e-30)

    # --- nominal path ---
    if mode.startswith("nominal"):
        if sigma_A_ref is None:
            return np.full(int(n_fallback), float(A_ref))
        return _gauss_positive(A_ref, sigma_A_ref, n_fallback)

    # --- Hertz-fit path ---
    if mode == "fit_hertz":
        # Prefer bootstrap CI/std if present in `hertz`
        Rm = float(hertz.get("R_eff_m", np.nan)) if hertz else np.nan
        Rstd = float(hertz.get("R_eff_std_m", np.nan)) if hertz else np.nan

        if not (np.isfinite(Rm) and Rm > 0):
            return np.full(int(n_fallback), float(A_ref))

        # sample radius; if no std, keep deterministic
        if np.isfinite(Rstd) and Rstd > 0:
            R_s = rng.normal(Rm, Rstd, size=int(n_fallback))
            R_s = np.maximum(R_s, 1e-12)
        else:
            R_s = np.full(int(n_fallback), Rm)

        # area from geometric model
        # A = area_pi_h_R(h_ref, R)
        # but vectorize:
        return area_pi_h_R(np.full_like(R_s, float(h_ref)), R_s)

    # --- flat-end path ---
    if mode == "flat_end":
        if boot_flat and int(boot_flat.get("ok", 0)) == 1:
            samples = boot_flat.get("samples", {})
            a_s = samples.get("a_flat_m", None)
            if a_s is not None and np.asarray(a_s).size > 10:
                a_s = np.asarray(a_s, float)
                a_s = a_s[np.isfinite(a_s) & (a_s > 0)]
                if a_s.size > 10:
                    # resample to n_fallback size
                    pick = rng.choice(a_s, size=int(n_fallback), replace=True)
                    return np.pi * pick * pick

            # fallback: sample R_eff_m if present and propagate via nominal h_ref geometry
            R_s = samples.get("R_eff_m", None)
            if R_s is not None and np.asarray(R_s).size > 10:
                R_s = np.asarray(R_s, float)
                R_s = R_s[np.isfinite(R_s) & (R_s > 0)]
                if R_s.size > 10:
                    pick = rng.choice(R_s, size=int(n_fallback), replace=True)
                    return area_pi_h_R(np.full_like(pick, float(h_ref)), pick)

        return np.full(int(n_fallback), float(A_ref))

    # unknown mode -> deterministic
    return np.full(int(n_fallback), float(A_ref))

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
) -> dict:
    """
    Bootstrap uncertainty for fitted Hertz/adhesion effective radius R_eff_m.

    Why bootstrap here?
    -------------------
    The fitted radius R is extracted from a *subset* of a load–indentation curve. Noise,
    drift, and (often) correlation between neighboring samples make analytic covariance
    unreliable. Bootstrap gives a practical uncertainty estimate for R that can be propagated
    into area/pressure/shear strength.

    Resampling strategy
    -------------------
    - If data are sequential (common in loading sweeps), neighboring points are
      correlated. Using IID resampling can underestimate uncertainty.
    - A block bootstrap resamples contiguous blocks of length `block_size` (5–15 typical)
      to preserve local correlation structure.

    Parameters
    ----------
    h_m, P_N:
        Indentation (>=0) and load arrays, same length.
    Sz_meas_N_per_m:
        Optional measured normal stiffness aligned with h,P. If provided and
        `fit_fn` supports it (as in the updated hertz_fit_radius_adhesion), it is passed
        through to keep the bootstrap consistent with the fit.
    fit_fn:
        Typically hertz_fit_radius_adhesion.
    fit_kwargs:
        kwargs for fit_fn. Extra kwargs are *silently dropped* if fit_fn doesn't accept them.
    keep_frac:
        Fraction of the available points used per bootstrap draw (m out of n). Use <1 to
        reduce influence of tails/outliers and plasticity onset points.
    block_size:
        If None or <=1: IID bootstrap. Else: block bootstrap with blocks of this length.

    Returns
    -------
    dict with:
      ok, reason, n_used, n_boot_ok,
      R_eff_std_m, R_eff_ci95_lo_m, R_eff_ci95_hi_m,
      samples_R_eff_m (raw samples for propagation),
      adhesion_model_used_mode / frac (if model varies under auto)
    """
    fit_kwargs = dict(fit_kwargs or {})

    # ---- Filter kwargs to fit_fn signature to avoid any "unexpected kwarg" errors ----
    try:
        sig = inspect.signature(fit_fn)
        accepted = set(sig.parameters.keys())
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in accepted}
    except Exception:
        # If inspection fails, we proceed without filtering; fit errors are caught per draw.
        pass

    # ---- sanitize & align arrays ----
    h = np.asarray(h_m, float)
    P = np.asarray(P_N, float)

    if h.shape != P.shape:
        return {"ok": 0, "reason": "shape_mismatch", "n_used": 0, "n_boot_ok": 0,
                "R_eff_std_m": np.nan, "R_eff_ci95_lo_m": np.nan, "R_eff_ci95_hi_m": np.nan}

    Sz = None
    if Sz_meas_N_per_m is not None:
        Sz = np.asarray(Sz_meas_N_per_m, float)
        if Sz.shape != h.shape:
            # keep it simple: if misaligned, ignore Sz in bootstrap
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

    # sample size per draw
    msize = int(max(min_points, round(float(keep_frac) * n)))
    msize = min(msize, n)  # never exceed available points

    rng = np.random.default_rng(seed)

    # ---- block bootstrap sampler ----
    def _draw_indices() -> np.ndarray:
        if (block_size is None) or (int(block_size) <= 1):
            # IID resampling with replacement
            return rng.integers(0, n, size=msize)

        B = int(max(1, block_size))
        nblocks = int(np.ceil(msize / B))
        # starting indices for blocks
        starts = rng.integers(0, max(1, n - B + 1), size=nblocks)
        idx = np.concatenate([np.arange(s, s + B) for s in starts])
        return idx[:msize]

    # ---- call fit consistently with the adhesion fitter ----
    def _call_fit(hb: np.ndarray, Pb: np.ndarray, Szb: np.ndarray | None):
        if Szb is not None:
            return fit_fn(hb, Pb, *fit_args, Sz_meas_N_per_m=Szb, **fit_kwargs)
        return fit_fn(hb, Pb, *fit_args, **fit_kwargs)

    R_list: list[float] = []
    model_used: list[str] = []

    for _ in range(int(max(10, n_boot))):
        idx = _draw_indices()
        hb = h[idx]
        Pb = P[idx]
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

    R = np.asarray(R_list, float)
    if R.size < int(min_success):
        return {"ok": 0, "reason": "too_few_successful_bootstraps",
                "n_used": n, "n_boot_ok": int(R.size),
                "R_eff_std_m": np.nan, "R_eff_ci95_lo_m": np.nan, "R_eff_ci95_hi_m": np.nan}

    # Standard bootstrap summaries (distribution may be skewed; CI from percentiles is robust)
    R_std = float(np.std(R, ddof=1)) if R.size >= 2 else np.nan
    lo, hi = np.percentile(R, [2.5, 97.5])

    out = {
        "ok": 1, "reason": "",
        "n_used": n,
        "n_boot_ok": int(R.size),
        "keep_frac": float(keep_frac),
        "block_size": (int(block_size) if (block_size is not None) else None),

        "R_eff_std_m": float(R_std),
        "R_eff_ci95_lo_m": float(lo),
        "R_eff_ci95_hi_m": float(hi),

        # raw samples for propagation (area/pressure/shear)
        "samples_R_eff_m": R,
    }

    # If auto model switches across resamples, report most frequent
    if model_used:
        vals, counts = np.unique(np.asarray(model_used, dtype=str), return_counts=True)
        j = int(np.argmax(counts))
        out["adhesion_model_used_mode"] = str(vals[j])
        out["adhesion_model_used_frac"] = float(counts[j] / np.sum(counts))

    return out

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
    if fit_fn is None:
        fit_fn = fit_flat_end_stiffness
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

def summarize_dist(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": np.nan, "std": np.nan, "ci68": (np.nan, np.nan), "ci95": (np.nan, np.nan), "n": 0}
    med = float(np.median(x))
    std = float(np.std(x, ddof=1)) if x.size >= 2 else np.nan
    lo68, hi68 = np.percentile(x, [16, 84])
    lo95, hi95 = np.percentile(x, [2.5, 97.5])
    return {"median": med, "std": std, "ci68": (float(lo68), float(hi68)), "ci95": (float(lo95), float(hi95)), "n": int(x.size)}

def safe_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x) & (x > 0)]

def tau_from_F_and_A_samples(F_N: float, A_samples: np.ndarray) -> dict:
    # tau = F/A
    t = float(F_N) / np.asarray(A_samples, float)
    return summarize_dist(t)


# ============================================================
# 3) Markers + Config
# ============================================================

def extract_markers(df: pd.DataFrame, markers_col: str) -> Dict[str, int]:
    if markers_col not in df.columns:
        return {}
    out: Dict[str, int] = {}
    m = df[markers_col]
    mask = m.notna()
    # store first occurrence of each marker string
    for i in np.where(mask)[0]:
        name = str(m.iloc[i])
        out.setdefault(name, int(i))
    return out

@dataclass(frozen=True)
class Config:
    # core
    time_col: str = "Time"
    markers_col: str = "Markers"

    # raw normal channels
    Fz_raw_col: str = "Force"
    z_raw_col: str = "Displacement"

    # vertical stiffness
    Sz_col: str = "Dyn. Stiffness"
    # optional vertical lock-in (RMS) if available for calibration sanity check
    Fz_dyn_rms_col: Optional[str] = "Dyn. Force" 
    Z_dyn_rms_col: Optional[str] = "Dyn. Disp."    

    # touch detection
    k_touch_col: str = "Dyn. Stiffness"
    k_touch_min: float = 500.0
    k_touch_min_duration_s: float = 0.1
    marker_surface: str = "Surface Index"

    # lateral lock-in channels (RMS)
    F2_rms_col: str = "Dyn. Force 2"
    X2_rms_col: str = "Dyn. Disp. 2"
    PH2_col: str = "Dyn. Phase 2"  # displacement relative to force
    dyn_f2_freq_Hz: float = 80.0    # lateral CSM frequency
    lockin_slope_hw: float = 0.05  # Hz, half-width for slope calc

    # detect shear window parameters
    ## !not absolute loading rate but the max rate's percentage after the touch pt as the stopping pt of loading.
    loading_rate_threshold: float = 0.001 #threshold for finding loading/unloading and shear window by normal load gradient max' percentage

    normal_load_sustain: float = 0.5 #the minimum time (s) for indication of hold has started-to avoid measurement and gradient instability
    normal_load_smooth: int = 201 ##smoothing on the normal load-rate channel.

    # cycle detection from Dyn Force 2 RMS envelope
    smooth_n: int = 501
    dynF2_baseline_q: float = 0.05
    dynF2_active_delta: float = 0.03
    dynF2_nearzero_delta: float = 0.01
    hold_top_frac: float = 0.98
    hold_min_s: float = 0.2
    min_cycle_points: int = 200
    # derivative-based cycle detection
    dfdt_smooth_n: int = 301          # extra smoothing for derivative stability
    dfdt_thr_frac: float = 0.001       # derivative threshold as fraction of max |dF/dt| in shear window
    dfdt_hold_frac: float = 0.9      # hold condition: |dF/dt| <= hold_frac * dfdt_thr
    min_ramp_s: float = 2.0           # minimum duration of ramp-up/down (seconds)
    min_hold_s: float = 0.5           # minimum duration of hold plateau (seconds)

    # --- Uncertainty inputs (1-sigma) ---
    # normal channels
    sigma_Fz_N: float = 5e-9          # example: 5 nN
    sigma_z_m: float = 1e-9           # example: 1 nm

    # lateral amplitude / friction force channel
    sigma_Ft_N: float = 10e-9         # ~10 nN typical noise per lock-in τ

    # vertical stiffness channel
    sigma_Sz_N_per_m: float = 50.0    # example; set from hold scatter

    # model parameters
    sigma_tip_radius_m: float = 0.5e-6   # e.g. ±0.5 µm
    sigma_Estar_Pa: float = 2e9          # e.g. ±2 GPa
    sigma_k_frame_z: float = 0.0       

    # lock-in lag model (for ramped envelopes)
    lockin_tau_s: float = 0.050
    lockin_force_noise_N: float = 10e-9

    # reporting windows around cycles
    pre_window_s: float = 1.0
    post_window_s: float = 3.0
    ref_window_s: float = 1.0

    # lateral calibration markers (preferred)
    marker_cal_up: str = "dynLRampUp"
    marker_cal_dn: str = "dynLRampDown"
    k_sup_x_fallback: Optional[float] = None  # N/m
    b_sup_x_fallback: float = 0.0             # N
    allow_no_cal: bool = False

    # fallback lateral calibration heuristic if markers missing
    cal_force_thr_rms: float = 0.01
    cal_min_points: int = 400

    # frame stiffness corrections (optional)
    k_frame_z: Optional[float] = 1000000   # N/m
    k_frame_x: Optional[float] = 500000   # N/m

    # tip radius for A = pi*h*R
    tip_radius_m: float = 50e-6

    # transition detection (stick->slide and re-stick)
    trans_frac_up: float = 0.1              # S_thresh = frac * Sx_stuck (stick to slide)
    trans_frac_down: float = 0.15            # S_thresh = frac * Sx_stuck (restick)
    sliding_lateral_stiffness_thresh: float = 500.0    # N/m minimum S_thresh for stick->slide
    resticking_lateral_stiffness_thresh: float = 1000.0 # N/m minimum S_thresh for slide->stick
    trans_low_band: tuple[float, float] = (0.05, 0.20)  # early ramp-up force band to estimate S_stuck
    trans_smooth_n: int = 21

    # Mindlin fit K(Q)=a*(1-Q/t)^(1/3) on ramp-up
    mindlin_min_frac_of_maxF: float = 0.1
    mindlin_max_frac_of_maxF: float = 0.99
    mindlin_min_points: int = 30

# ------------------------------
    # Hertz diagnostics (normal F vs h)
    # ------------------------------
    hertz_enable: bool = True

    # material constants (Pa)
    E1_Pa: float = 70e9          # fused silica ~ 70 GPa
    nu1: float = 0.18            # fused silica ~ 0.18
    E2_Pa: float = 1140e9        # diamond ~ 1140 GPa
    nu2: float = 0.07            # diamond ~ 0.07

    hardness_Pa: float = 10.0e9   # optional; set None/NaN to disable plasticity filtering
    plasticity_p0_frac: float = 1.0  # require max Hertz pressure p0 <= frac * hardness -silica yields close to hardness C~1-1.50 GPa

    # data selection
    hertz_min_h_m: float = 1e-9      # ignore ultra-small depths (noise/offset), e.g. 5 nm
    hertz_max_frac_of_Pmax: float = 1  # fit only up to this fraction of peak load in loading
    hertz_min_points: int = 50

    # robust / iteration
    hertz_iter: int = 3              # iterate fit-filter-fit using p0 criterion
    hertz_plot: bool = True         # show diagnostic plot per file when live_plots is True
    area_mode: str = "nominal"   # "nominal" | "fit_hertz" "flat_end"
    
    # manual selection policy
    manual_mode: str = "always"   # "never" | "on_fail" | "always"
    manual_cycle_mode: str = "always"  # same idea, but for per-cycle indices
    expected_cycles: Optional[int] = None  # for manual cycle picking prompt and validation later..
    plot_mindlin: bool = True
    plot_cycles: bool = True

    ##Lock-in amplifier EG7280-used parameters-
    lockin_tau_s: float = 0.050
    lockin_force_noise_N: float = 10e-9
    lockin_pre_s: float = 0.30
    lockin_guard_s: float = 0.25

    adhesion_model: str = "auto"   # "hertz"|"dmt"|"jkr or "auto"
    w_J_per_m2: float = 0.5       # user-set (e.g. silica/diamond range)->via atomistic, 
    sigma_rms_m: float | None = 0.5  # RMS Roughness 
    rough_model: str = "exp"       # "none"|"exp"|user set effective adhesion by roughness
    delta0_m: float = 0.3e-9 #cut off for exponential Persson
    z0_m: float = 0.3e-9    # L-J minimum, stable pt.
    mu_dmt: float = 1 #Tabor parameter for DMT-upper limit
    mu_jkr: float = 5.0 #Tabor parameter for JKR--lower limit
    min_h_m: float = 1e-10#minimum contact depth for hertz fit,
    max_frac_of_Pmax: float = 1  # fit only up to this fraction of peak load in loading.
    min_points: int = 10
    n_iter: int = 100
# ============================================================
# ------------------------------
from dataclasses import dataclass

@dataclass(frozen=True)
class UncConfig:
    # relative uncertainties (fraction)
    E_star_rel: float = 0.05       # 5%
    R_rel: float = 0.05            # 5% TIP RADIUS
    k_frame_z_rel: float = 0.10
    k_frame_x_rel: float = 0.10
    Sx_stuck_rel: float = 0.10
    Sz_rel: float = 0.05
    Ft_N_rel: float = 0.05

    # absolute uncertainties
    force_N_abs: float = 0.0       #force calibration error
    disp_m_abs: float = 0.0

    # index uncertainties (in samples)
    # related to DAQ timebase accuracy + filtering + human pick
    touch_i_pm: int = 5 
    iPmax_pm: int = 5
    shear_i0_pm: int = 5
    ss_i_pm: int = 5
    rs_i_pm: int = 5
    window_i_pm: int = 10

@dataclass
class ManualOverrides:
    touch_i: Optional[int] = None
    iPmax: Optional[int] = None
    shear_i0: Optional[int] = None
    shear_i1: Optional[int] = None
    cycles: Optional[List["CycleBounds"]] = None  # reuse CycleBounds

# ============================================================
# 4) Touch, normal-load correction, depth/area
# ============================================================

def detect_touch_index(df: pd.DataFrame, cfg: Config, markers: Dict[str, int]) -> int:
    if cfg.marker_surface in markers:
        return int(markers[cfg.marker_surface])

    t = _num(df, cfg.time_col)
    k = _num(df, cfg.k_touch_col)
    dt = np.nanmedian(np.diff(t))
    nmin = max(1, int(cfg.k_touch_min_duration_s / max(dt, 1e-12)))

    above = np.isfinite(k) & (k > cfg.k_touch_min)
    idxs = np.where(above)[0]
    for i in idxs:
        if i + nmin < len(above) and np.all(above[i:i+nmin]):
            return int(i)
    raise RuntimeError("Touch not found (no Surface Index marker and stiffness criterion failed).")

def manual_pick_touch(df: pd.DataFrame, cfg: Config, initial: Optional[int] = None) -> int:
    t = _num(df, cfg.time_col)
    k = _num(df, cfg.k_touch_col)
    picks = pick_indices_from_plot(
        t,
        series=[("Dyn. Stiffness", k)],
        prompts=["Click TOUCH point (first contact)."],
        n_clicks=1,
        predefined_picks=initial if initial is not None else [],
        title="Pick touch index"
    )
    return picks[0]

def fit_support_spring_pre_touch(z_m: np.ndarray, F_N: np.ndarray, touch_i: int) -> tuple[float, float]:
    z = z_m[:touch_i]
    F = F_N[:touch_i]
    if np.isfinite(z).sum() < 50 or np.isfinite(F).sum() < 50:
        raise RuntimeError("Not enough pre-touch points to fit support spring.")
    k, b = robust_fit_line(z, F)  # F ≈ k*z + b
    if not np.isfinite(k):
        raise RuntimeError("Support spring fit failed.")
    # uncertainties from OLS (same slice)
    _, _, sigma_k, sigma_b = ols_line_cov(z, F)
    return float(k), float(b), float(sigma_k), float(sigma_b)

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

def area_pi_h_R(h_m: np.ndarray, R_m: float) -> np.ndarray:
    h = np.maximum(0.0, h_m)
    return np.pi * h * float(R_m)

def normal_pressure_Pa(P_N: np.ndarray, A_m2: np.ndarray) -> np.ndarray:
    return P_N / np.maximum(1e-30, A_m2)

def shear_stress_Pa(Ft_N: np.ndarray, A_m2: np.ndarray) -> np.ndarray:
    return Ft_N / np.maximum(1e-30, A_m2)

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

def choose_area_mode_gate(figures : List[plt.Figure] = [], default_mode: str = "nominal") -> str:
    """
    Keys:
      1 = nominal
      2 = fit_hertz
      3 = flat_end
      a/enter = accept default
      esc = abort
    """
    decision = {"val": None}
    fig_title = "Press 1=nominal, 2=fit_hertz, 3=flat_end; a/Enter=accept default, Esc=abort"
    def on_key(event):
        k = (event.key or "").lower()
        if k == "1":
            decision["val"] = "nominal"
            plt.close("all")
        elif k == "2":
            decision["val"] = "fit_hertz"
            plt.close("all")
        elif k == "3":
            decision["val"] = "flat_end"
            plt.close("all")
        elif k in ("a", "enter"):
            decision["val"] = default_mode
            plt.close("all")
        elif k == "escape":
            decision["val"] = "abort"
            plt.close("all")
    
    for fig in figures:
        try:
            fig.canvas.mpl_connect("key_press_event", on_key)
            if fig_title:
                fig.canvas.manager.set_window_title(fig_title)
        except Exception:
            pass
    plt.show(block=True)

    return decision["val"]

# ============================================================
# 5) Calibration slice, cycle detection
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


@dataclass(frozen=True)
class CycleBounds:
    cycle: int
    i_start: int
    i_peak: int
    i_hold0: int
    i_hold1: int
    i_end: int

def detect_cycles(df: pd.DataFrame, cfg: Config, start_i: int = 0, end_i: Optional[int] = None) -> List[CycleBounds]:
    t = _num(df, cfg.time_col)
    a = np.nan_to_num(_num(df, cfg.F2_rms_col), nan=0.0)

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
    a_s = rolling_median(a, cfg.smooth_n)

    # Extra smoothing for derivative stability
    a_sd = rolling_median(a, cfg.dfdt_smooth_n)
    da = np.gradient(a_sd, t)

    # zero out outside shear window
    a_s[:start_i] = 0.0
    a_s[end_i+1:] = 0.0
    a_sd[:start_i] = 0.0
    a_sd[end_i+1:] = 0.0
    da[:start_i] = 0.0
    da[end_i+1:] = 0.0

    # Baseline and amplitude thresholds (still used, but not the primary detector)
    base = float(np.quantile(a_s[start_i:end_i+1], cfg.dynF2_baseline_q))
    thr_active = base + cfg.dynF2_active_delta
    thr_nz = base + cfg.dynF2_nearzero_delta

    # Derivative thresholds (relative to max slope in window)
    scale = float(safe_nanmax(np.abs(da[start_i:end_i+1])))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Cycle detection: derivative scale is zero; check signal/window.")

    dthr = cfg.dfdt_thr_frac * scale
    dhold = cfg.dfdt_hold_frac * dthr

    # Masks
    ramp_up = da > dthr
    ramp_dn = da < -dthr
    hold_like = np.abs(da) <= dhold

    # Find ramp-up regions (candidate starts)
    up_regs = [(s, e) for (s, e) in contiguous_regions(ramp_up) if (e - s + 1) >= ramp_min_pts]
    if not up_regs:
        raise RuntimeError("No ramp-up regions found; tune dfdt_thr_frac/min_ramp_s.")

    cycles: List[CycleBounds] = []
    cursor = start_i
    cyc = 0

    for (us, ue) in up_regs:
        if us < cursor:
            continue

        # Candidate cycle start: walk back to near-zero (baseline neighborhood)
        i_start = us
        while i_start > start_i and a_s[i_start] > thr_nz:
            i_start -= 1

        # Peak: after ramp-up ends, find local maximum before ramp-down begins
        # Search forward from ue to find peak as argmax until we encounter sustained ramp-down
        search_end = end_i

        # Find first sustained ramp-down after this ramp-up
        dn_after = [(s, e) for (s, e) in contiguous_regions(ramp_dn) if (e - s + 1) >= ramp_min_pts and s > ue]
        if dn_after:
            first_dn_s, first_dn_e = dn_after[0]
            search_end = min(search_end, first_dn_s)  # peak should be before ramp-down starts

        if search_end <= ue:
            continue

        seg_pk = slice(us, search_end + 1)
        i_peak = int(us + np.argmax(a_s[seg_pk]))
        amax = float(a_s[i_peak])

        # Reject tiny “preload” bumps: require peak significantly above baseline
        # (Use both absolute delta and relative)
        if not (amax > thr_active and (amax - base) > 2.0 * cfg.dynF2_active_delta):
            continue

        # Hold plateau near top: contiguous region near top where derivative ~0 and amplitude near amax
        near_top = (a_s >= cfg.hold_top_frac * amax) & hold_like
        # only consider between ramp-up and ramp-down (or to end)
        near_top[:us] = False
        near_top[end_i+1:] = False
        if dn_after:
            near_top[first_dn_s:] = False

        top_regs = [(s, e) for (s, e) in contiguous_regions(near_top) if (e - s + 1) >= hold_min_pts]
        if top_regs:
            # choose one closest to i_peak
            rs, re = min(top_regs, key=lambda r: abs((r[0] + r[1]) / 2 - i_peak))
            i_hold0, i_hold1 = int(rs), int(re)
        else:
            # fallback: small window around peak
            i_hold0, i_hold1 = i_peak, i_peak

        # End: after ramp-down, walk forward until near-zero
        if dn_after:
            i_end = dn_after[0][1]
        else:
            # if no clear ramp-down detected, end when amplitude returns near baseline
            i_end = i_peak

        while i_end < end_i and a_s[i_end] > thr_nz:
            i_end += 1

        # Sanity constraints: order and minimum size
        if not (i_start < i_peak < i_end):
            continue

        cyc += 1
        cycles.append(CycleBounds(cyc, i_start, i_peak, i_hold0, i_hold1, i_end))
        cursor = i_end + 1

    if not cycles:
        raise RuntimeError("No cycles accepted after derivative-based filtering.")

    return cycles

"""
Pick indices from plot, if automatic detection fails or not wanted.
"""
def pick_indices_from_plot(
    t: np.ndarray,
    series: List[Tuple[str, np.ndarray]],
    prompts: List[str],
    n_clicks: int,
    predefined_picks: Optional[List[int]] = None,
    title: str = "",
) -> List[int]:
    """
    Shows a plot and lets user click n_clicks times on the x-axis to choose indices.
    Returns list of chosen indices (nearest in time).
    Controls:
      - left click: pick a point
      - backspace: remove last pick
      - enter: finish (only allowed after n_clicks picks)
      - escape: abort (raises)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, y in series:
        ax.plot(t, y, label=label)
    ax.legend(loc="best")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    for i in predefined_picks: 
        draw_vline(ax, t[i])  # just to set up the axis
    if n_clicks == 2 and predefined_picks is not None and len(predefined_picks) == 2:
        draw_span(ax, t[predefined_picks[0]], t[predefined_picks[1] ])

    picks: List[int] = list(predefined_picks) if predefined_picks is not None else []
    vlines = []

    msg = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top")
    def update_msg():
        left = n_clicks - len(picks)
        prompt = prompts[len(picks)] if len(picks) < len(prompts) else ""
        msg.set_text(f"{prompt}\nPicks: {len(picks)}/{n_clicks} (remaining {left})\n"
                     f"Enter=confirm, -=undo, Esc=abort")
        fig.canvas.draw_idle()

    def add_pick(x):
        i = int(np.argmin(np.abs(t - x)))
        picks.append(i)
        vl = ax.axvline(t[i], linestyle="--", linewidth=1)
        vlines.append(vl)
        update_msg()

    def undo():
        if picks:
            picks.pop()
            vl = vlines.pop()
            vl.remove()
            update_msg()

    def on_click(event):
        if event.inaxes != ax: return
        if event.button != 1: return
        if len(picks) >= n_clicks: return
        add_pick(event.xdata)

    done = {"ok": False}
    cycles_done = {"val": False}

    def on_key(event):
        if event.key == "-":
            undo()
        elif event.key == "f":
            done["ok"] = True
            cycles_done["val"] = True
            plt.close(fig)
        elif event.key == "enter":
            if len(picks) == n_clicks:
                done["ok"] = True
                plt.close(fig)
        elif event.key == "escape":
            done["ok"] = False
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_msg()
    plt.show(block=True)

    if not done["ok"]:
        raise RuntimeError("Manual picking aborted by user.")
    if cycles_done["val"]:
        picks = [0,0,0,0]  # signal for no more cycles
    return picks

def draw_vline(ax, x, label="", **kw):
    if x is None or not np.isfinite(x):
        return
    ax.axvline(x, linestyle="--", color="red", **kw)
    ax.text(x, 0.98, label, transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=9)

def draw_span(ax, x0, x1, label="", **kw):
    if x0 is None or x1 is None:
        return
    ax.axvspan(x0, x1, color="red", alpha=0.2, **kw)
    ax.text((x0+x1)/2, 0.98, label, transform=ax.get_xaxis_transform(),
            va="top", ha="center", fontsize=9)

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

    k, b = robust_fit_line(Z_pk[cal_sl_vert], Fz_pk[cal_sl_vert])  # F ≈ k*Z + b
    return (float(k), float(b))

def fit_vertical_dynamic_coupling(
    df: pd.DataFrame,
    cfg: Config,
    scale_to_SI: Dict[str, float],
    cal_sl_vert: Optional[slice],
) -> dict:
    """
    Fit a 2D linear model on the vertical dyn calibration bump:
      Fz_pk ≈ kzz*Z_pk + kzx*X2_pk + b
    Returns dict with kzz, kzx, b, and R2-like metric.
    """
    if cal_sl_vert is None:
        return {"kzz": np.nan, "kzx": np.nan, "b": np.nan, "ok": 0}

    # need vertical dyn force+disp and lateral dyn disp
    need = [cfg.Fz_dyn_rms_col, cfg.Z_dyn_rms_col, cfg.X2_rms_col]
    if any((c is None) or (c not in df.columns) for c in need):
        return {"kzz": np.nan, "kzx": np.nan, "b": np.nan, "ok": 0}

    Fz = rms_to_peak(_num(df, cfg.Fz_dyn_rms_col) * scale_to_SI[cfg.Fz_dyn_rms_col])
    Z  = rms_to_peak(_num(df, cfg.Z_dyn_rms_col)  * scale_to_SI[cfg.Z_dyn_rms_col])
    X2 = rms_to_peak(_num(df, cfg.X2_rms_col)     * scale_to_SI[cfg.X2_rms_col])

    sl = cal_sl_vert
    y = Fz[sl]
    X = np.column_stack([Z[sl], X2[sl], np.ones_like(y)])

    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() < 20:
        return {"kzz": np.nan, "kzx": np.nan, "b": np.nan, "ok": 0}

    y = y[m]
    X = X[m]

    # least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    kzz, kzx, b = beta.tolist()

    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"kzz": float(kzz), "kzx": float(kzx), "b": float(b), "r2": float(r2), "ok": 1}

# ============================================================
# 6) Lateral correction + complex stiffness + dissipation
# ============================================================

def compute_lateral_corrected(
    df: pd.DataFrame,
    cfg: Config,
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
# 7) Transition detection: stick->slide and re-stick
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
            "i_ss": np.nan, "i_rs": np.nan,
            "Sx_stuck": np.nan,
            "Sx_slide_used": np.nan,
            "Sx_restick_used": np.nan,
            "Ft_ss_N": np.nan, "X_ss_m": np.nan,
            "Ft_rs_N": np.nan, "X_rs_m": np.nan,
            "rd0": rd0,
            "went_low_first": 0,
        }

    # ---- estimate Sx_stuck from early ramp-up ----
    lo = low_frac_band[0] * Ftmax
    hi = low_frac_band[1] * Ftmax
    m_stuck = np.isfinite(Ft_ru) & np.isfinite(Sx_ru_s) & (Ft_ru >= lo) & (Ft_ru <= hi)

    if m_stuck.sum() < 10:
        idxs = np.where(np.isfinite(Sx_ru_s))[0][:max(10, min(30, len(Sx_ru_s)))]
        Sx_stuck = float(np.nanmedian(Sx_ru_s[idxs])) if idxs.size else np.nan
    else:
        Sx_stuck = float(np.nanmedian(Sx_ru_s[m_stuck]))

    if not (np.isfinite(Sx_stuck) and Sx_stuck > 0):
        return {
            "i_ss": np.nan, "i_rs": np.nan,
            "Sx_stuck": Sx_stuck,
            "Sx_slide_used": np.nan,
            "Sx_restick_used": np.nan,
            "Ft_ss_N": np.nan, "X_ss_m": np.nan,
            "Ft_rs_N": np.nan, "X_rs_m": np.nan,
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


# ============================================================
# 8) Mindlin fit
# ============================================================

def mindlin_model(Q: np.ndarray, a: float, t: float) -> np.ndarray:
    return a * np.power(np.maximum(1e-30, 1.0 - (Q / t)), 1.0 / 3.0)

def mindlin_fit(Q: np.ndarray, K: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(Q) & np.isfinite(K) & (Q > 0) & (K > 0)
    Q = Q[m]; K = K[m]
    if Q.size < 10:
        return {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": int(Q.size), "ok": 0}

    Qmax = float(np.max(Q))
    a0 = float(np.median(K[Q <= np.quantile(Q, 0.2)])) if np.any(Q <= np.quantile(Q, 0.2)) else float(np.median(K))
    t0 = 1.2 * Qmax

    if SCIPY_OK:
        try:
            popt, _ = curve_fit(
                mindlin_model, Q, K,
                p0=[a0, t0],
                bounds=([10., 0.], [100 * a0, 10.0 * t0]),
                maxfev=10000
            )
            a_hat, t_hat = float(popt[0]), float(popt[1])
            Khat = mindlin_model(Q, a_hat, t_hat) ## Lateral stiffness prediction
            rmse = float(np.sqrt(np.mean((K - Khat) ** 2))) ## Root mean square error-by comparing predicted and actual lateral stiffness
            return {"a": a_hat, "t": t_hat, "rmse": rmse, "n": int(Q.size), "ok": 1}
        except Exception:
            return {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": int(Q.size), "ok": 0}

    # fallback grid if SciPy missing
    t_grid = np.linspace(1.01 * Qmax, 3.0 * Qmax, 200)
    best = {"rmse": np.inf, "a": np.nan, "t": np.nan}

    invK3 = 1.0 / np.power(K, 3.0)
    for t in t_grid:
        x = (1.0 - Q / t)
        denom = np.dot(x, x)
        if denom <= 0:
            continue
        c = float(np.dot(x, invK3) / denom)  # c = 1/a^3
        if not np.isfinite(c) or c <= 0:
            continue
        a = float(np.power(1.0 / c, 1.0 / 3.0))
        Khat = mindlin_model(Q, a, t)
        rmse = float(np.sqrt(np.mean((K - Khat) ** 2)))
        if rmse < best["rmse"]:
            best = {"rmse": rmse, "a": a, "t": float(t)}

    ok = 1 if np.isfinite(best["a"]) and np.isfinite(best["t"]) else 0
    return {"a": best["a"], "t": best["t"], "rmse": best["rmse"], "n": int(Q.size), "ok": ok}


# ============================================================
# 9) Live plots + folder summary plots
# ============================================================

def maybe_live_plots(df: pd.DataFrame, cfg: Config, live_plots: bool, title_prefix: str = "") -> None:
    if not live_plots:
        return

    t = _num(df, cfg.time_col)

    plt.figure("Sanity: Lateral")
    plt.clf()
    plt.plot(t, _num(df, cfg.F2_rms_col), label="Dyn. Force 2 (RMS) [native]")
    if "F2_pk_corr_N" in df.columns:
        plt.plot(t, df["F2_pk_corr_N"].to_numpy()*1e3, label="F2_pk_corr (mN)")
    plt.title(f"{title_prefix} — lateral")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()

    if "P_contact_N" in df.columns:
        plt.figure("Sanity: Normal load")
        plt.clf()
        plt.plot(t, df["P_contact_N"].to_numpy() * 1e3, label="P_contact (mN)")
        plt.axhline(0, linestyle="--")
        plt.title(f"{title_prefix} — corrected normal load")
        plt.xlabel("Time (s)")
        plt.ylabel("mN")
        plt.legend()
        plt.tight_layout()

def sanity_plot_window_cycles(
    df2: pd.DataFrame,
    cfg: Config,
    t: np.ndarray,
    P_contact_N: np.ndarray,
    i0: int,
    i1: int,
    cycles: List[CycleBounds],
    cal_sl: Optional[slice],
    title: str = "",
) -> List[plt.Figure]:
    """
    Live sanity plots (shows, does not save):
      (A) P_contact and dP/dt with shear window highlighted
      (B) Dyn. Force 2 (RMS) and smoothed envelope with cycle markers + calibration slice
    """
    # ----- Plot A: Normal load and derivative -----
    figures=[]
    w = max(101, cfg.smooth_n // 2)
    P_sm = pd.Series(P_contact_N).rolling(w, center=True, min_periods=1).median().to_numpy()
    dPdt = np.gradient(P_sm, t)

    figures.append(plt.figure("Sanity A: Normal load + shear window", figsize=(10, 5)))
    plt.clf()
    ax1 = plt.gca()
    ax1.plot(t, P_contact_N * 1e3, label="P_contact (mN)")
    ax1.plot(t, P_sm * 1e3, label="P_contact smoothed (mN)")
    ax1.axvspan(t[i0], t[i1], alpha=0.15, color="red", label="shear window")

    # overlay cycle start/end vertical lines
    for cb in cycles:
        ax1.axvline(t[cb.i_start], linestyle="--", linewidth=1)
        ax1.axvline(t[cb.i_end], linestyle="--", linewidth=1)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normal load (mN)")
    ax1.set_title(title or "Normal load and detected shear window")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, dPdt * 1e3, label="dP/dt (mN/s)", color="black", linestyle=":")
    ax2.set_ylabel("dP/dt (mN/s)")
    ax2.legend(loc="upper right")
    plt.tight_layout()

    # ----- Plot B: Lateral dyn force envelope + cycles + calibration -----
    lateral_dForce = _num(df2, cfg.F2_rms_col)
    F2s = pd.Series(np.nan_to_num(lateral_dForce, nan=0.0)).rolling(cfg.smooth_n, center=True, min_periods=1).median().to_numpy()

    figures.append(plt.figure("Sanity B: Dyn. Force 2 + cycles", figsize=(10, 5)))
    plt.clf()
    ax = plt.gca()
    ax.plot(t, lateral_dForce, label="Dyn. Force 2 (RMS) raw")
    ax.plot(t, F2s, label="Dyn. Force 2 (RMS) smoothed")
    da = np.gradient(pd.Series(F2s).rolling(cfg.dfdt_smooth_n, center=True, min_periods=1).median().to_numpy(), t)
    ax2 = ax.twinx()
    ax2.plot(t, da, label="dF/dt (smoothed)")
    ax2.set_ylabel("dF/dt")

    # calibration slice shading
    if cal_sl is not None:
        ax.axvspan(t[cal_sl.start], t[cal_sl.stop - 1], alpha=0.15, label="calibration slice", color="green")

    # shear window shading
    ax.axvspan(t[i0], t[i1], alpha=0.10, label="shear window", color="red")

    # cycles
    for cb in cycles:
        ax.axvline(t[cb.i_start], linestyle="--", linewidth=1)
        ax.axvline(t[cb.i_peak], linestyle=":", linewidth=1)
        ax.axvline(t[cb.i_hold1], linestyle=":", linewidth=1)
        ax.axvline(t[cb.i_end], linestyle="--", linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dyn. Force 2 (RMS, native units)")
    ax.set_title(title or "Lateral envelope and detected cycles")
    ax.legend(loc="upper right")
    plt.tight_layout()

# ----- Plot C: Vertical Stiffness + cycles + calibration -----
    vertical_stiffness = _num(df2, cfg.k_touch_col)

    figures.append(plt.figure("Sanity C: Vertical Stiffness + cycles", figsize=(10, 5)))
    plt.clf()
    ax = plt.gca()
    ax.plot(t, vertical_stiffness, label="Vertical Stiffness raw")

    # calibration slice shading
    if cal_sl is not None:
        ax.axvspan(t[cal_sl.start], t[cal_sl.stop - 1], alpha=0.15, label="calibration slice", color="green")

    # shear window shading
    ax.axvspan(t[i0], t[i1], alpha=0.10, label="shear window", color="red")

    # cycles
    for cb in cycles:
        ax.axvline(t[cb.i_start], linestyle="--", linewidth=1)
        ax.axvline(t[cb.i_peak], linestyle=":", linewidth=1)
        ax.axvline(t[cb.i_end], linestyle="--", linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Stiffness (N/m)")
    ax.set_title(title or "Lateral envelope and detected cycles")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return figures

def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def make_folder_summary_plots(
    all_cycles_df: pd.DataFrame,
    outdir: Path,
    *,
    max_cycle_to_plot: Optional[int] = None,
    min_points_per_cycle: int = 3,
    save_png: bool = True,
) -> List[plt.Figure]:
    """
    Folder-level QC plots for friction experiments.

    Uses long-form cycles df (one row per cycle per file).
    Saves a small set of physics-first plots:
      1) mu_ss vs file (per cycle)
      2) tau_ss vs pressure_ref (per cycle)
      3) tau_ss vs A_ref (per cycle)
      4) junction growth A_ratio_to_ref vs cycle index (aggregated)
      (+ optional Mindlin and hold metrics if present)

    Notes:
      - does not assume fixed # cycles
      - avoids explicit color settings (matplotlib defaults)
      - uses robust filtering for finite values
    """
    outdir.mkdir(parents=True, exist_ok=True)
    figs: List[plt.Figure] = []

    if all_cycles_df is None or all_cycles_df.empty:
        return figs

    df = all_cycles_df.copy()
    if "file" not in df.columns or "cycle" not in df.columns:
        return figs

    df["cycle"] = _safe_numeric(df["cycle"]).astype("Int64")
    df = df.dropna(subset=["cycle"])
    df["cycle"] = df["cycle"].astype(int)

    if max_cycle_to_plot is None:
        max_cycle_to_plot = int(df["cycle"].max())
    else:
        max_cycle_to_plot = int(max(1, max_cycle_to_plot))

    # Helpful base columns (may not exist in all runs)
    for c in ["mu_ss", "tau_ss_MPa", "pressure_ref_GPa", "A_ref_um2", "A_ratio_to_ref", "mu_hold", "mindlin_t_N", "mindlin_a_N_per_m"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c])

    # Build a stable file ordering by first appearance
    files = list(pd.unique(df["file"].astype(str)))
    file_to_x = {f: i for i, f in enumerate(files)}
    df["file_x"] = df["file"].map(file_to_x)

    # ------------------------------------------------------------
    # 1) mu_ss vs file index, per cycle
    # ------------------------------------------------------------
    if "mu_ss" in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for cyc in range(1, max_cycle_to_plot + 1):
            g = df[df["cycle"] == cyc]
            g = g[np.isfinite(g["mu_ss"]) & np.isfinite(g["file_x"])]
            if len(g) < min_points_per_cycle:
                continue
            ax.plot(g["file_x"].to_numpy(), g["mu_ss"].to_numpy(), marker="o", linestyle="None", label=f"C{cyc:02d}")
        ax.set_xlabel("Experiment index (file order)")
        ax.set_ylabel("μ at stick→slide (mu_ss)")
        ax.set_title("Friction coefficient at stick→slide across experiments (by cycle)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        figs.append(fig)
        if save_png:
            fig.savefig(outdir / "FIG01_mu_ss_by_cycle_across_experiments.png", dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------
    # 2) tau_ss vs pressure_ref, per cycle
    # ------------------------------------------------------------
    if ("tau_ss_MPa" in df.columns) and ("pressure_ref_GPa" in df.columns):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for cyc in range(1, max_cycle_to_plot + 1):
            g = df[df["cycle"] == cyc]
            g = g[np.isfinite(g["tau_ss_MPa"]) & np.isfinite(g["pressure_ref_GPa"])]
            if len(g) < min_points_per_cycle:
                continue
            ax.plot(g["pressure_ref_GPa"].to_numpy(), g["tau_ss_MPa"].to_numpy(), marker="o", linestyle="None", label=f"C{cyc:02d}")
        ax.set_xlabel("Reference pressure (GPa)")
        ax.set_ylabel("τ at stick→slide (MPa)")
        ax.set_title("Shear strength vs pressure (by cycle)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        figs.append(fig)
        if save_png:
            fig.savefig(outdir / "FIG02_tau_ss_vs_pressure_ref_by_cycle.png", dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------
    # 3) tau_ss vs A_ref (or area), per cycle
    # ------------------------------------------------------------
    if ("tau_ss_MPa" in df.columns) and ("A_ref_um2" in df.columns):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for cyc in range(1, max_cycle_to_plot + 1):
            g = df[df["cycle"] == cyc]
            g = g[np.isfinite(g["tau_ss_MPa"]) & np.isfinite(g["A_ref_um2"]) & (g["A_ref_um2"] > 0)]
            if len(g) < min_points_per_cycle:
                continue
            ax.plot(g["A_ref_um2"].to_numpy(), g["tau_ss_MPa"].to_numpy(), marker="o", linestyle="None", label=f"C{cyc:02d}")
        ax.set_xlabel("Reference area A_ref (µm²)")
        ax.set_ylabel("τ at stick→slide (MPa)")
        ax.set_title("Shear strength vs contact area (by cycle)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        figs.append(fig)
        if save_png:
            fig.savefig(outdir / "FIG03_tau_ss_vs_Aref_by_cycle.png", dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------
    # 4) junction growth proxy A_ratio_to_ref vs cycle number (aggregated)
    # ------------------------------------------------------------
    if "A_ratio_to_ref" in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        g = df[np.isfinite(df["A_ratio_to_ref"]) & (df["A_ratio_to_ref"] > 0)]
        if not g.empty:
            ax.plot(g["cycle"].to_numpy(), g["A_ratio_to_ref"].to_numpy(), marker="o", linestyle="None")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("A/A0 (junction growth proxy)")
        ax.set_title("Junction growth proxy across all experiments")
        ax.grid(True, alpha=0.3)
        figs.append(fig)
        if save_png:
            fig.savefig(outdir / "FIG04_junction_growth_A_ratio_vs_cycle.png", dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------
    # Optional 5) mu_hold vs cycle (aggregated)
    # ------------------------------------------------------------
    if "mu_hold" in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        g = df[np.isfinite(df["mu_hold"])]
        if not g.empty:
            ax.plot(g["cycle"].to_numpy(), g["mu_hold"].to_numpy(), marker="o", linestyle="None")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("μ (hold)")
        ax.set_title("Hold friction coefficient across all experiments")
        ax.grid(True, alpha=0.3)
        figs.append(fig)
        if save_png:
            fig.savefig(outdir / "FIG05_mu_hold_vs_cycle.png", dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------
    # Optional 6) Mindlin parameters vs cycle (aggregated)
    # ------------------------------------------------------------
    if "mindlin_t_N" in df.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        g = df[np.isfinite(df["mindlin_t_N"])]
        if not g.empty:
            ax.plot(g["cycle"].to_numpy(), g["mindlin_t_N"].to_numpy(), marker="o", linestyle="None")
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Mindlin t (N)")
        ax.set_title("Mindlin parameter t across all experiments")
        ax.grid(True, alpha=0.3)
        figs.append(fig)
        if save_png:
            fig.savefig(outdir / "FIG06_mindlin_t_vs_cycle.png", dpi=200, bbox_inches="tight")

    return figs


def plot_check_friction_and_transitions(
    df: pd.DataFrame,
    cfg: Config,
    b: CycleBounds,
    P_contact_N: np.ndarray,
    A_m2: np.ndarray,
    tr: dict,
    title: str = "",
) -> dict:

    t = _num(df, cfg.time_col)
    Ft = df["F2_pk_corr_N"].to_numpy()
    Kp = df["Stiffness_lateral"].to_numpy()
    phase = df["phi2_rad"].to_numpy()
    Ft_raw = df["F2_pk_N"].to_numpy()
    Dt = df["X2_pk_contact_m"].to_numpy()
    En  = df["E_diss_J_per_cycle"].to_numpy()

    sl = slice(b.i_start, b.i_end + 1)
    tt = t[sl]
    Ftt = Ft[sl]
    Ktt = Kp[sl]
    phitt = phase[sl]
    Ft_rawt = Ft_raw[sl]
    Dtt = Dt[sl]
    Ent = En[sl]

    i_ss = tr.get("i_ss", None)
    i_rs = tr.get("i_rs", None)

    fig1=plt.figure(figsize=(10, 6))
    ax0 = plt.gca()
    ax0.plot(tt, Ft_rawt * 1e3, label="Lateral force amp (mN)")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Lateral force amplitude (mN)")

    # key boundaries
    ax0.axvline(t[b.i_start], linestyle="--", linewidth=1, label="cycle start", color="black")
    ax0.axvline(t[b.i_peak], linestyle=":", linewidth=1, label="cycle peak", color="olive")
    ax0.axvline(t[b.i_end], linestyle="--", linewidth=1, label="cycle end", color="black")
    ax0.axvspan(t[b.i_hold0], t[b.i_hold1], alpha=0.15, label="hold window")
    line1, line2 = None, None
    if i_ss is not None:
        line1 = ax0.axvline(t[i_ss], linewidth=2, label="stick→slide", color="red")
    if i_rs is not None:
        line2 = ax0.axvline(t[i_rs], linewidth=2, label="re-stick", color="green")

    ax00 = ax0.twinx()
    ax00.plot(tt, phitt, label=r"$ \phi_{x} (rad)$", color="orange")
    ax00.set_ylabel("Lateral Phase")

    fig2=plt.figure(figsize=(10, 6))
    axlfd = plt.gca()
    axlfd.plot(Dtt * 1e9, Ftt * 1e3, label="Lateral force amp (mN)")
    axlfd.set_xlabel("Lateral Displacement Amplitude (nm)")
    axlfd.set_ylabel("Lateral Force amplitude (mN)")

    axlfd1 = axlfd.twinx()
    axlfd1.plot(Dtt * 1e9, Ent * 6.242e12, label=r"Energy Loss", color="orange")
    axlfd1.set_ylabel("Energy Loss per cycle (MeV)")


    picks=[]
    if i_ss is not None:
        picks.append(i_ss)
    if i_rs is not None:
        picks.append(i_rs)

    fig = plt.figure(figsize=(10, 6))
    plt.clf()
    ax1 = plt.gca()
    ax1.plot(tt, Ftt * 1e3, label="Friction force amp (mN)  [F2_pk_corr]")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Friction force amplitude (mN)")

    # key boundaries
    ax1.axvline(t[b.i_start], linestyle="--", linewidth=1, label="cycle start", color="black")
    ax1.axvline(t[b.i_peak], linestyle=":", linewidth=1, label="cycle peak", color="olive")
    ax1.axvline(t[b.i_end], linestyle="--", linewidth=1, label="cycle end", color="black")
    ax1.axvspan(t[b.i_hold0], t[b.i_hold1], alpha=0.15, label="hold window")
    line1, line2 = None, None
    if i_ss is not None:
        line1 = ax1.axvline(t[i_ss], linewidth=2, label="stick→slide", color="red")
    if i_rs is not None:
        line2 = ax1.axvline(t[i_rs], linewidth=2, label="re-stick", color="green")

    ax2 = ax1.twinx()
    ax2.plot(tt, Ktt, label=r"$ S_{x} (N/m)$", color="orange")
    ax2.set_ylabel("Lateral stiffness (N/m)")

    # show Sx_stuck and threshold as horizontals
    Sx_stuck = tr.get("Sx_stuck", np.nan)
    Sx_thresh = tr.get("Sx_slide_used", np.nan)
    if np.isfinite(Sx_stuck):
        ax2.axhline(Sx_stuck, linestyle="--", linewidth=1, label="Sx_stuck")
    if np.isfinite(Sx_thresh):
        ax2.axhline(Sx_thresh, linestyle=":", linewidth=1, label="Sx_thresh", color="red")

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.title(title or f"Cycle {b.cycle}: friction force + transitions + stiffness")
    plt.tight_layout()

#### Check for stick->slip and restick transitions -interactive
    vlines = []
    msg = ax1.text(0.01, 0.99, "", transform=ax1.transAxes, va="top")
    prompts = ["Select the stick->slip", "Select the restick", "All set!"]
    def update_msg():
        left = 2 - len(picks)
        prompt = prompts[len(picks)]
        msg.set_text(f"{prompt}\nPicks: {len(picks)}/{2} (remaining {left})\n"
                     f"Enter=confirm, -=undo, Esc=abort")
        fig.canvas.draw_idle()

    def add_pick(x):
        i = int(np.argmin(np.abs(t - x)))
        picks.append(i)
        if len(picks) == 1:
            vl = ax1.axvline(t[i], linewidth=2, label="stick→slide", color="red")
            vlines.append(vl)
        elif len(picks) == 2:
            vl = ax1.axvline(t[i], linewidth=2, label="re-stick", color="green")
            vlines.append(vl)
        update_msg()

    def undo():
        if picks:
            picks.pop()
            vl = vlines.pop()
            vl.remove()
            update_msg()

    def on_click(event):
        if event.inaxes != ax1:
            if event.inaxes != ax2:
                return
        if event.button != 1: return
        if len(picks) >= 2: return
        add_pick(event.xdata)

    done = {"ok": False}

    def on_key(event):
        if event.key == "-":
            undo()
        elif event.key == "enter":
            if len(picks) == 2:
                done["ok"] = True
                plt.close('all')

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_msg()

    plt.show(block=True)
    if done["ok"]:
        tr.update(i_ss = picks[0])
        tr.update(i_rs = picks[1])
        if tr["i_ss"] is not None:
            tr.update(Ft_ss_N = float(Ft[tr["i_ss"]]))
            tr.update(X_ss_m = float(Dt[tr["i_ss"]]))
            tr.update(Sx_slide_used = float(Kp[tr["i_ss"]]))
            tr.update(Sx_slide_source = "manual")
        else:
            tr.update(Ft_ss_N = np.nan)
            tr.update(X_ss_m = np.nan)

        if tr["i_rs"] is not None:
            tr.update(Ft_rs_N = float(Ft[tr["i_rs"]]))
            tr.update(X_rs_m = float(Dt[tr["i_rs"]]))
            tr.update(Sx_restick_used = float(Kp[tr["i_rs"]]))
            tr.update(Sx_restick_source = "manual")
        else:
            tr.update(Ft_rs_N = np.nan)
            tr.update(X_rs_m = np.nan)

    return tr

def plot_mindlin_fit(
    df: pd.DataFrame,
    cfg: Config,
    b: CycleBounds, dir:bool,
    mind: dict,
    title: str = "",
) -> plt.Figure:

    Ft = df["F2_pk_corr_N"].to_numpy()
    Kp = df["Stiffness_lateral"].to_numpy()
    label= ""
    if dir:
        ru = slice(b.i_start, b.i_peak + 1)
        label = "Ramp-up data (S vs Q)"
    else:
        ru = slice(b.i_hold1 + 1, b.i_end + 1)
        label = "Ramp-down data (S vs Q)"
    Q = Ft[ru]
    K = Kp[ru]
    m = np.isfinite(Q) & np.isfinite(K) & (Q > 0) & (K > 0)
    Q = Q[m]; K = K[m]

    figure=plt.figure(figsize=(8, 6))
    plt.clf()
    plt.plot(Q * 1e3, K, marker="o", linestyle="", label=label)

    # highlight fit range used in summarize_cycle
    if Q.size > 0:
        Qmax = float(np.max(Q))
        lo = cfg.mindlin_min_frac_of_maxF * Qmax
        hi = cfg.mindlin_max_frac_of_maxF * Qmax
        in_fit = (Q >= lo) & (Q <= hi)
        if np.any(in_fit):
            plt.plot(Q[in_fit] * 1e3, K[in_fit], marker="o", linestyle="", label="Fit subset")

    if int(mind.get("ok", 0)) == 1:
        a = float(mind["a"])
        tpar = float(mind["t"])
        # curve (avoid hitting singularity)
        qmin = float(np.min(Q)) if Q.size else 0.0
        qmax = float(np.max(Q)) if Q.size else 0.0
        qgrid = np.linspace(qmin, min(qmax, 0.98 * tpar), 250)
        kgrid = mindlin_model(qgrid, a, tpar)
        plt.plot(qgrid * 1e3, kgrid, label=f"Mindlin fit: a={a:.3g} N/m, t={tpar*1e3:.3g} mN")

    plt.xlabel("Q = friction force amplitude (mN)")
    plt.ylabel(r"$S_{x} (N/m)$")
    plt.title(title or f"Cycle {b.cycle}: Mindlin fit on {'ramp-up' if dir else 'ramp-down'}")
    plt.legend(loc="best")
    plt.tight_layout()
    return figure

def plot_hertz_diagnostic(h_m, P_N, fit: dict, 
                          title: str = "", hardness_Pa: float = np.nan, 
                          plasticity_p0_frac: float = 1.0) -> List[plt.Figure]:

    mask = fit.get("mask_used", None)
    if mask is None:
        mask = np.isfinite(h_m) & np.isfinite(P_N)

    E_star = fit.get("E_star_Pa", np.nan)
    R_eff = fit.get("R_eff_m", np.nan)
    C = fit.get("C", np.nan)

    figures=[]
    # 1) P vs h^(3/2) with fit
    figures.append(plt.figure(figsize=(9, 6)))
    plt.clf()
    x = np.power(np.maximum(0.0, h_m), 1.5)
    plt.plot(x[mask], P_N[mask]*1e3, "o", label="used (mN)")
    if np.isfinite(C):
        xx = np.linspace(safe_nanmin(x[mask]), safe_nanmax(x[mask]), 300)
        yy = C * xx
        plt.plot(xx, yy*1e3, "-", label="Hertz fit")
    plt.xlabel(r"$h^{3/2}$ (m$^{3/2}$)")
    plt.ylabel("P (mN)")
    if np.isfinite(R_eff):
        plt.title(f"{title} — Hertz fit: R_eff={R_eff*1e6:.2f} µm, rmse={fit.get('rmse_N',np.nan)*1e3:.3g} mN")
    else:
        plt.title(f"{title} — Hertz fit")
    plt.legend()
    plt.tight_layout()

    # 2) R_app(h)
    figures.append(plt.figure(figsize=(9, 6)))
    plt.clf()
    R_app = hertz_apparent_radius_R_of_h(h_m, P_N, E_star) if np.isfinite(E_star) else np.full_like(h_m, np.nan)
    plt.plot(h_m[mask]*1e9, R_app[mask]*1e6, "o")
    if np.isfinite(R_eff):
        plt.axhline(R_eff*1e6, linestyle="--", label="R_eff")
        plt.legend()
    plt.xlabel("h (nm)")
    plt.ylabel("R_app (µm)")
    plt.title(f"{title} — apparent radius vs depth")
    plt.tight_layout()

    # 3) plasticity check: p0(h)
    if np.isfinite(hardness_Pa) and hardness_Pa > 0 and np.isfinite(R_eff) and R_eff > 0:
        figures.append(plt.figure(figsize=(9, 6)))
        plt.clf()
        a = np.sqrt(np.maximum(1e-30, R_eff * h_m))
        p0 = (3.0 * P_N) / (2.0 * np.pi * np.maximum(1e-30, a**2))
        plt.plot(h_m[mask]*1e9, p0[mask]/1e9, "o", label="p0 (GPa)")
        plt.axhline((plasticity_p0_frac*hardness_Pa)/1e9, linestyle="--", label="H criterion")
        plt.xlabel("h (nm)")
        plt.ylabel("p0 (GPa)")
        plt.title(f"{title} — plasticity check")
        plt.legend()
        plt.tight_layout()
    return figures

def plot_touch_with_pick(df: pd.DataFrame, cfg: Config, touch_i: int, title: str = "") -> None:
    t = _num(df, cfg.time_col)
    k = _num(df, cfg.k_touch_col)

    plt.figure(figsize=(10, 5))
    plt.clf()
    plt.plot(t, k, label=cfg.k_touch_col)
    if touch_i is not None and 0 <= touch_i < len(t):
        plt.axvline(t[touch_i], linestyle="--", linewidth=2, label=f"touch @ {touch_i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Dyn. Stiffness (native)")
    plt.title(title or "Touch detection (A=accept, R=repick, S=skip)")
    plt.legend(loc="best")
    plt.tight_layout()

def plot_shear_window_with_pick(t: np.ndarray, P_contact_N: np.ndarray, F2_rms: np.ndarray, i0: int, i1: int, title: str="") -> None:
    plt.figure(figsize=(10, 5))
    plt.clf()
    ax = plt.gca()
    ax.plot(t, P_contact_N * 1e3, label="P_contact (mN)")
    ax.axhline(0, linestyle="--", linewidth=1)
    if i0 is not None and i1 is not None and i0 < i1:
        ax.axvspan(t[i0], t[i1], alpha=0.15, label=f"shear window [{i0},{i1}]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normal load (mN)")
    ax2 = ax.twinx()
    ax2.plot(t, F2_rms, label="Dyn Force 2 RMS")
    ax2.set_ylabel("Dyn Force 2 (native)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")
    plt.title(title or "Shear window (A=accept, R=repick, S=skip)")
    plt.tight_layout()


def set_plot_defaults():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (12, 7),
        "figure.dpi": 200,
        "savefig.dpi": 200,

        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,

        "lines.linewidth": 2,
        "axes.linewidth": 1.5,
        "grid.linewidth": 1.0,

        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": True,
        "legend.framealpha": 0.9,

        # make labels less cramped
        "axes.titlepad": 10,
        "axes.labelpad": 8,
    })

def plot_normal_loading_depth_stiffness(
    t: np.ndarray,
    P_contact_N: np.ndarray,
    h_m: np.ndarray,
    Sz_N_per_m: np.ndarray | None,
    touch_i: int,
    i0: int,
    ref_i: np.ndarray | None = None,
    title: str = "",
) -> plt.Figure:
    """
    X: P_contact (mN)
    Y1: h (nm)
    Y2: Sz (N/m), optional
    """
    P_mN = np.asarray(P_contact_N, float) * 1e3
    h_nm = np.asarray(h_m, float) * 1e9

    sl = slice(max(0, int(touch_i)), min(len(P_mN), int(i0) + 1))
    if sl.stop - sl.start < 5:
        sl = slice(0, len(P_mN))
    fig,ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(P_mN[sl], h_nm[sl], label="h (nm)")
    ax1.set_xlabel("P_contact (mN)")
    ax1.set_ylabel("h (nm)")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(title or "Normal loading sanity: depth + stiffness vs load")

    # mark key points
    ax1.axvline(P_mN[touch_i], linestyle="--", linewidth=1, label="touch")
    ax1.axvline(P_mN[i0], linestyle="--", linewidth=1, label="end of loading (i0)")

    if ref_i is not None and len(ref_i) > 0:
        p0 = float(np.nanmedian(P_mN[ref_i]))
        ax1.axvspan(float(np.nanmin(P_mN[ref_i])), float(np.nanmax(P_mN[ref_i])),
                    alpha=0.15, label="ref window")

        # optional: mark median ref point
        ax1.axvline(p0, linestyle=":", linewidth=1)

    # stiffness on twin axis
    if Sz_N_per_m is not None:
        Sz = np.asarray(Sz_N_per_m, float)
        ax2 = ax1.twinx()
        ax2.plot(P_mN[sl], Sz[sl], label="Sz (N/m)", color="orange")
        ax2.set_ylabel("Sz (N/m)")

        # merged legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax1.legend(loc="best")
    plt.tight_layout()

    return fig

def plot_contact_radius_sanity(
    P_contact_N: np.ndarray,
    a_csm_m: np.ndarray,
    a_geo_m: np.ndarray,
    touch_i: int,
    i0: int,
    title: str = "",
) -> plt.Figure:
    P_mN = np.asarray(P_contact_N, float) * 1e3
    sl = slice(max(0, int(touch_i)), min(len(P_mN), int(i0) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(P_mN[sl], a_csm_m[sl] * 1e6, label="a_CSM (µm)")
    ax.plot(P_mN[sl], a_geo_m[sl] * 1e6, label="a_geo from h (µm)")
    ax.set_xlabel("P_contact (mN)")
    ax.set_ylabel("Contact radius a (µm)")
    ax.grid(True, alpha=0.25)
    ax.set_title(title or "Contact radius sanity (CSM vs geometry)")
    ax.legend(loc="best")
    plt.tight_layout()
    return fig


def finalize_figure(fig, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)  # leave space for suptitle
    return fig

def overlay_index(ax, t, idx, label, ypos=0.95, **kw):
    if idx is None:
        return
    x = float(t[idx])
    ax.axvline(x, **kw)
    ax.text(x, ypos, label, transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.2))

def plot_flat_end_fit(
    P_N: np.ndarray,
    S_N_per_m: np.ndarray,
    fit: dict,
    *,
    E_star_Pa: float | None = None,
)-> plt.Figure:
    """
    Diagnostic plot for the flat-end stiffness fit:
        S(P) = S0 + C * P^(1/3)

    Produces:
      - Scatter of measured S vs P
      - Fit curve across the plotted load range
      - Optional: also show inferred a(P)=S/(2E*) if E_star_Pa provided

    Parameters
    ----------
    P_N, S_N_per_m : arrays
        Load (N) and corrected stiffness (N/m).
    fit : dict
        Output from fit_flat_end_stiffness().
    E_star_Pa : float | None
        If provided, overlays contact radius a = S/(2E*) on a secondary y-axis.

    Returns
    -------
    figure
    """
    P = np.asarray(P_N, dtype=float)
    S = np.asarray(S_N_per_m, dtype=float)

    S0 = float(fit["S0"])
    C = float(fit["C"])
    title = "Flat-end stiffness fit diagnostic"
    # Window mask (positive load/stiffness + any user range)
    base_mask = np.isfinite(P) & np.isfinite(S) & (P > 0) & (S > 0)
    win_mask = fit.get("mask", base_mask)
    m = base_mask & win_mask

    fig, ax = plt.subplots()

    # Scatter: all valid points faint, fit-window points emphasized
    ax.plot(P[base_mask], S[base_mask], ".", markersize=3, alpha=0.25, label="data (valid)")
    ax.plot(P[m], S[m], ".", markersize=4, alpha=0.85, label="data (fit window)")

    # Fit curve over range
    Pmin = np.nanmin(P[m]) if np.any(m) else np.nanmin(P[base_mask])
    Pmax = np.nanmax(P[m]) if np.any(m) else np.nanmax(P[base_mask])
    Pgrid = np.logspace(np.log10(Pmin), np.log10(Pmax), 300)
    Sfit = S0 + C * np.cbrt(Pgrid)
    ax.plot(Pgrid, Sfit, "-", linewidth=2, label="fit: S0 + C·P^(1/3)")

    ax.set_xscale("log")
    ax.set_xlabel("Normal load P (N)")
    ax.set_ylabel("Corrected normal stiffness S (N/m)")
    ax.grid(True, which="both", alpha=0.25)

    # Annotation box
    txt = (
        f"S0 = {S0:.3g} N/m\n"
        f"C  = {C:.3g} N/m·N^(-1/3)\n"
        f"R_eff = {fit.get('R_eff_m', np.nan)*1e6:.3g} µm\n"
        f"R² = {fit.get('R2', np.nan):.3f}\n"
        f"RMSE = {fit.get('rmse', np.nan):.3g} N/m\n"
        f"n = {fit.get('n', 0)}"
    )
    if E_star_Pa is not None:
        a_flat = S0 / (2.0 * float(E_star_Pa))
        txt += f"\na_flat = {a_flat*1e6:.3g} µm"

    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    # Optional secondary axis for a(P)
    if E_star_Pa is not None and np.isfinite(E_star_Pa) and E_star_Pa > 0:
        ax2 = ax.twinx()
        # show radii in µm
        ax2.set_ylabel("Apparent contact radius a = S/(2E*) (µm)")
        # map y-lims from S to a
        y0, y1 = ax.get_ylim()
        ax2.set_ylim(y0 / (2.0 * E_star_Pa) * 1e6, y1 / (2.0 * E_star_Pa) * 1e6)
        # light guide line for baseline radius
        ax2.axhline((S0 / (2.0 * E_star_Pa)) * 1e6, linestyle="--", linewidth=1, alpha=0.5)

    if title is None:
        title = "Flat-end stiffness fit diagnostic"
    ax.set_title(title)
    ax.legend(loc="best")

    return fig

# ============================================================
# 10) Origin-friendly CSV exporters
# ============================================================

def _csv_escape(x) -> str:
    s = "" if x is None else str(x)
    if any(ch in s for ch in [",", "\"", "\n"]):
        s = "\"" + s.replace("\"", "\"\"") + "\""
    return s

def export_origin_csv(df: pd.DataFrame, outpath: Path, long_names: dict, units: dict) -> None:
    """
    Origin-friendly CSV:
      Row 1 = column names
      Row 2 = Long Name
      Row 3 = Units
      Row 4+ = data
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    ln_row = [long_names.get(c, c) for c in cols]
    un_row = [units.get(c, "") for c in cols]

    with outpath.open("w", newline="", encoding="utf-8") as f:
        f.write(",".join(map(_csv_escape, cols)) + "\n")
        f.write(",".join(map(_csv_escape, ln_row)) + "\n")
        f.write(",".join(map(_csv_escape, un_row)) + "\n")
        df.to_csv(f, index=False, header=False)

def origin_long_names_and_units_cycles() -> tuple[dict, dict]:
    long_names = {
        "file": "File",
        "cycle": "Cycle #",
        "P_hold_mN": "Normal load (hold, median)",
        "p_hold_GPa": "Contact pressure (hold, median)",
        "Sz_sliding": "Vertical stiffness (hold, median)",
        "Sliding_Sx_hold": "Sliding lateral stiffness (hold)",
        "Ft_hold_mN": "Lateral force amp (hold, corr)",
        "mu_hold": "Friction coefficient (hold)",
        "tau_hold_MPa": "Shear strength (hold)",
        "Ft_ss_mN": "Stick to slide force (corr)",
        "Ft_rs_mN": "Re-stick force (corr)",
        "X_ss_nm": "Slip distance at stick to slide",
        "X_rs_nm": "Slip distance at re-stick",
        "mu_ss": "Friction coefficient at stick to slide",
        "mu_rs": "Friction coefficient at re-stick",
        "tau_ss_MPa": "Shear strength at stick to slide",
        "tau_rs_MPa": "Shear strength at re-stick",
        "A_ratio_to_ref": "Junction growth proxy A/A0",
        "mindlin_a_N_per_m": "Mindlin a (full-stick stiffness)",
        "mindlin_t_N": "Mindlin t (static friction force amp)",
        "mindlin_rmse": "Mindlin fit RMSE",
    }
    units = {
        "P_hold_mN": "mN",
        "p_hold_GPa": "GPa",
        "Sz_sliding": "N/m",
        "Sliding_Sx_hold": "N/m",
        "Ft_hold_mN": "mN",
        "mu_hold": "",
        "tau_hold_MPa": "MPa",
        "Ft_ss_mN": "mN",
        "Ft_rs_mN": "mN",
        "X_ss_nm": "nm",
        "X_rs_nm": "nm",
        "mu_ss": "",
        "mu_rs": "",
        "tau_ss_MPa": "MPa",
        "tau_rs_MPa": "MPa",
        "A_ratio_to_ref": "",
        "mindlin_a_N_per_m": "N/m",
        "mindlin_t_N": "N",
        "mindlin_rmse": "",
    }
    return long_names, units


# ============================================================
# 11) SummaryNanoRo-like wide template (exact columns + units row first)
# ============================================================

TEMPLATE_COLS = [
  "Test","Load",
  "Pristine Friction Force","2nd Cycle Friction Force","3rd Cycle Friction Force",
  "1st Cycle Re-stick Friction Force","2nd Cycle Re-stick Friction Force","3rd Cycle Re-stick Friction Force",
  "Initial Vertical Stiffness","1st Cycle Vertical Stiffness","2nd Cycle Vertical Stiffness","3rd Cycle Vertical Stiffness",
  "1st Cycle Lateral Stiffness","2nd Cycle Lateral Stiffness","3rd Cycle Lateral Stiffness",
  "Static Friction Coefficient Pristine","Static Friction Coefficient 2nd Cycle","Static Friction Coefficient 3rd Cycle",
  "Static Friction Coefficient Re-Stick 1st","Static Friction Coefficient Re-Stick 2nd","Static Friction Coefficient Re-Stick 3rd",
  "Slip Distance Pristine","Slip Distance 2nd Cycle","Slip Distance 3rd Cycle",
  "Slip Distance Re-stick Pristine","Slip Distance Re-stick 2nd Cycle","Slip Distance Re-Stick 3rd Cycle",
  "Contact Depth","Initial Contact Area", "Contact Pressure",
  "Junction Growth 1st Cycle","Junction Growth 2nd Cycle","Junction Growth 3rd Cycle",
  "Shear Strength Pristine","Shear Strength 2nd Cycle","Shear Strength 3rd Cycle",
  "Shear Strength Re-Stick 1st","Shear Strength Re-Stick 2nd","Shear Strength Re-Stick 3rd"
]

TEMPLATE_UNITS_ROW = {
  "Test":"",
  "Load":"mN",
  "Pristine Friction Force":"mN",
  "2nd Cycle Friction Force":"mN",
  "3rd Cycle Friction Force":"mN",
  "1st Cycle Re-stick Friction Force":"mN",
  "2nd Cycle Re-stick Friction Force":"mN",
  "3rd Cycle Re-stick Friction Force":"mN",
  "Initial Vertical Stiffness":"N/m",
  "1st Cycle Vertical Stiffness":"N/m",
  "2nd Cycle Vertical Stiffness":"N/m",
  "3rd Cycle Vertical Stiffness":"N/m",
  "1st Cycle Lateral Stiffness":"N/m",
  "2nd Cycle Lateral Stiffness":"N/m",
  "3rd Cycle Lateral Stiffness":"N/m",
  "Static Friction Coefficient Pristine":"",
  "Static Friction Coefficient 2nd Cycle":"",
  "Static Friction Coefficient 3rd Cycle":"",
  "Static Friction Coefficient Re-Stick 1st":"",
  "Static Friction Coefficient Re-Stick 2nd":"",
  "Static Friction Coefficient Re-Stick 3rd":"",
  "Slip Distance Pristine":"nm",
  "Slip Distance 2nd Cycle":"nm",
  "Slip Distance 3rd Cycle":"nm",
  "Slip Distance Re-stick Pristine":"nm",
  "Slip Distance Re-stick 2nd Cycle":"nm",
  "Slip Distance Re-Stick 3rd Cycle":"nm",
  "Contact Depth":"nm",
  "Initial Contact Area":"µm^2",
  "Contact Pressure": "GPa",
  "Junction Growth 1st Cycle":"",
  "Junction Growth 2nd Cycle":"",
  "Junction Growth 3rd Cycle":"",
  "Shear Strength Pristine":"MPa",
  "Shear Strength 2nd Cycle":"MPa",
  "Shear Strength 3rd Cycle":"MPa",
  "Shear Strength Re-Stick 1st":"MPa",
  "Shear Strength Re-Stick 2nd":"MPa",
  "Shear Strength Re-Stick 3rd":"MPa",
}

def export_like_summarynanoro(df: pd.DataFrame, out_csv: Path) -> None:
    """
    Exports:
      header row
      units row (first data row)
      data rows
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    units_row = [TEMPLATE_UNITS_ROW.get(c, "") for c in df.columns]
    units_df = pd.DataFrame([units_row], columns=df.columns)
    out = pd.concat([units_df, df], ignore_index=True)
    out.to_csv(out_csv, index=False)

def build_wide_summary_like_template(all_cycles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses:
      - stick->slide transition values: Ft_ss_mN, mu_ss, tau_ss_MPa, X_ss_nm
      - re-stick transition values: Ft_rs_mN, mu_rs, tau_rs_MPa, X_rs_nm
      - lateral stiffness: S_x_hold_N_per_m-sliding; stuck stiffnesses per cycle-> Sx_stuck; 
      - vertical stiffness: S_z and cycles _for later ratio calcs. 
      - load/depth/->initial area from cycle 1 hold
      - junction growth: A_ratio_to_ref-initial area
    """
    if all_cycles_df.empty:
        return pd.DataFrame(columns=TEMPLATE_COLS)

    df = all_cycles_df.sort_values(["file", "cycle"]).copy()

    rows = []
    for fname, g in df.groupby("file"):
        g = g.sort_values("cycle")

        r = {c: np.nan for c in TEMPLATE_COLS}
        r["Test"] = Path(fname).stem

        # Load, depth, area from cycle 1 hold
        c1 = g[g["cycle"] == 1]
        if not c1.empty:
            r["Load"] = float(c1["P_hold_mN"].iloc[0])
            r["Contact Depth"] = float(c1["h_hold_nm"].iloc[0])
            r["Initial Contact Area"] = float(c1["A_hold_um2"].iloc[0])
            r["Contact Pressure"] = float(c1["p_hold_GPa"].iloc[0])

        # Vertical stiffness: initial + per cycle
        # initial vertical stiffness = S_z_N_per_m from file summary (copied into each cycle row)
        if "Sz_initial_N_per_m" in g.columns and np.isfinite(g["Sz_initial_N_per_m"].iloc[0]):
            r["Initial Vertical Stiffness"] = float(g["Sz_initial_N_per_m"].iloc[0])

        for cyc, col in [(1,"1st Cycle Sliding Vertical Stiffness"), (2,"2nd Cycle Sliding Vertical Stiffness"), (3,"3rd Cycle Sliding Vertical Stiffness")]:
            gg = g[g["cycle"] == cyc]
            if not gg.empty and "Sz_sliding" in gg.columns:
                r[col] = float(gg["Sz_sliding"].iloc[0])
        
                # Junction growth (A_ratio_to_ref)
        for cyc, col in [(1,"Junction Growth 1st Cycle"), (2,"Junction Growth 2nd Cycle"), (3,"Junction Growth 3rd Cycle")]:
            gg = g[g["cycle"] == cyc]
            if not gg.empty:
                r[col] = float(gg["A_ratio_to_ref"].iloc[0])

        # Lateral stiffness from S_x stuck
        for cyc, col in [(1,"1st Cycle Lateral Stiffness"), (2,"2nd Cycle Lateral Stiffness"), (3,"3rd Cycle Lateral Stiffness")]:
            gg = g[g["cycle"] == cyc]
            if not gg.empty:
                r[col] = float(gg["Sx_stuck_N_per_m"].iloc[0])

        # Stick->slide mapping
        map_ff = {1:"Pristine Friction Force", 2:"2nd Cycle Friction Force", 3:"3rd Cycle Friction Force"}
        map_mu = {1:"Static Friction Coefficient Pristine", 2:"Static Friction Coefficient 2nd Cycle", 3:"Static Friction Coefficient 3rd Cycle"}
        map_sd = {1:"Slip Distance Pristine", 2:"Slip Distance 2nd Cycle", 3:"Slip Distance 3rd Cycle"}
        map_tau= {1:"Shear Strength Pristine", 2:"Shear Strength 2nd Cycle", 3:"Shear Strength 3rd Cycle"}

        for cyc in [1,2,3]:
            gg = g[g["cycle"] == cyc]
            if not gg.empty:
                r[map_ff[cyc]] = float(gg["Ft_ss_mN"].iloc[0])
                r[map_mu[cyc]] = float(gg["mu_ss"].iloc[0])
                r[map_sd[cyc]] = float(gg["X_ss_nm"].iloc[0])
                r[map_tau[cyc]] = float(gg["tau_ss_MPa"].iloc[0])

        # Re-stick mapping
        map_rff = {1:"1st Cycle Re-stick Friction Force", 2:"2nd Cycle Re-stick Friction Force", 3:"3rd Cycle Re-stick Friction Force"}
        map_rmu = {1:"Static Friction Coefficient Re-Stick 1st", 2:"Static Friction Coefficient Re-Stick 2nd", 3:"Static Friction Coefficient Re-Stick 3rd"}
        map_rsd = {1:"Slip Distance Re-stick Pristine", 2:"Slip Distance Re-stick 2nd Cycle", 3:"Slip Distance Re-Stick 3rd Cycle"}
        map_rta = {1:"Shear Strength Re-Stick 1st", 2:"Shear Strength Re-Stick 2nd", 3:"Shear Strength Re-Stick 3rd"}

        for cyc in [1,2,3]:
            gg = g[g["cycle"] == cyc]
            if not gg.empty:
                r[map_rff[cyc]] = float(gg["Ft_rs_mN"].iloc[0])
                r[map_rmu[cyc]] = float(gg["mu_rs"].iloc[0])
                r[map_rsd[cyc]] = float(gg["X_rs_nm"].iloc[0])
                r[map_rta[cyc]] = float(gg["tau_rs_MPa"].iloc[0])

        rows.append([r[c] for c in TEMPLATE_COLS])

    return pd.DataFrame(rows, columns=TEMPLATE_COLS)

def build_wide_summary_dynamic(all_cycles_df: pd.DataFrame, summaries_df: pd.DataFrame, max_cycles: int | None = None) -> pd.DataFrame:
    if all_cycles_df.empty:
        return pd.DataFrame()

    df = all_cycles_df.copy()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df = df.dropna(subset=["cycle"])
    df["cycle"] = df["cycle"].astype(int)

    if max_cycles is None:
        max_cycles = int(df["cycle"].max())

    # which cycle-level columns to spread
    cycle_cols = [c for c in [
        "Ft_ss_mN","mu_ss","tau_ss_MPa","tau_ss_ci95_lo_MPa","tau_ss_ci95_hi_MPa","X_ss_nm",
        "Ft_rs_mN","mu_rs","tau_rs_MPa","tau_rs_ci95_lo_MPa","tau_rs_ci95_hi_MPa","X_rs_nm",
        "sigma_tau_ss_MPa","sigma_Ft_ss_uN","sigma_tau_rs_MPa","sigma_Ft_rs_uN",
        "Sx_stuck_N_per_m","Sx_thresh_N_per_m",
        "Sz_sliding","A_ratio_to_ref","K_ratio_to_ref",
        "mindlin_a_N_per_m","mindlin_t_N","mindlin_rmse","mindlin_ok",
        "mindlin_a_rd_N_per_m","mindlin_t_rd_N","mindlin_rmse_rd","mindlin_ok_rd",
        "total_sliding_time_s","total_osc_cycles","total_slide_dist_m",
        "overall_mean_speed_m_per_s",
    ] if c in df.columns]


    # per-file base from summaries_df
    base_cols = [c for c in [
        # identity / status
        "file", "error",

        # reference geometry + load
        "area_mode_used",
        "initial_h_nm",
        "load_max_mN",
        "A_ref_um2",
        "pressure_ref_GPa",
        "pressure_ref_ci95_lo_GPa",
        "pressure_ref_ci95_hi_GPa",
        "Sz_initial_N_per_m",

        # Hertz / adhesion fit (per-file)
        "hertz_ok",
        "hertz_reason",
        "E_star_GPa",
        "R_eff_um",
        "hertz_rmse_mN",
        "hertz_n_used",
        "adhesion_model",
        "w_eff_J_per_m2",
        "tabor_mu",
        "Fadh_N",
        "R_eff_std_um",
        "R_eff_ci95_lo_um",
        "R_eff_ci95_hi_um",
        "hertz_boot_ok",
        "hertz_boot_n_ok",

        # CSM sanity (optional but very useful)
        "R_from_CSM_a_h_um",
        "R_from_CSM_a_P_um",
        "A_ref_csm_um2",
        "a_ref_csm_um",

        # Flat-end stiffness model (only summaries; keep it compact)
        "flat_end_boot_ok",
        "flat_end_boot_n_success",
        "flat_end_boot_keep_frac",
        "flat_end_a_flat_med_um",
        "flat_end_a_flat_ci95_lo_um",
        "flat_end_a_flat_ci95_hi_um",
        "flat_end_R_eff_med_um",
        "flat_end_R_eff_ci95_lo_um",
        "flat_end_R_eff_ci95_hi_um",
        "flat_end_S0_med_N_per_m",
        "flat_end_C_med",

        # actuator transfer function fits (optional; can be removed if too noisy)
        "mass_kg", "damp_act_N_s_per_m", "k_act_N_per_m",
        "mass_lat_kg", "damp_lat_N_s_per_m", "k_lat_N_per_m",
    ] if c in summaries_df.columns]


    base = summaries_df[base_cols].copy() if base_cols else summaries_df[["file"]].copy()
    base = base.drop_duplicates("file")

    out = base.set_index("file")

    for cyc in range(1, max_cycles + 1):
        g = df[df["cycle"] == cyc].set_index("file")
        for c in cycle_cols:
            out[f"C{cyc:02d}_{c}"] = g[c]

    out = out.reset_index()
    out.insert(1, "Test", out["file"].map(lambda x: Path(x).stem))
    return out

# ============================================================
# 12) Per-cycle summarizer
# ============================================================

def summarize_cycle(
    df: pd.DataFrame,
    cfg: Config,
    b: CycleBounds,
    tr: dict,
    h_ref: float,
    A_ref: float,
    Sz_ref: float,
) -> Dict[str, float]:
    t = _num(df, cfg.time_col)

    hold = slice(b.i_hold0, b.i_hold1 + 1)  ##hold slice
    ru = slice(b.i_start, b.i_peak + 1) ##ramp-up slice
    rd = slice(b.i_hold1 + 1, b.i_end + 1) ##ramp-down slice
    between = window_idx_fw(t, b.i_end, cfg.post_window_s) ##post shear window must be set prior.

    P_contact_N = df["P_contact_N"].to_numpy()
    h_m = df["h_m"].to_numpy()
    A_m2 = df["A_m2"].to_numpy()
    Sz = df["Sz_corrected"].to_numpy()
    Ft = df["F2_pk_corr_N"].to_numpy()
    Xc = df["X2_pk_contact_m"].to_numpy()
    phi = df["phi2_rad"].to_numpy()
    Sx = df["Stiffness_lateral"].to_numpy()
    Dx = df["Damping_lateral"].to_numpy()
    Ed = df["E_diss_J_per_cycle"].to_numpy()

    A_ref_m2 = A_ref
    Sx_ref = 0.0 ##lateral stiffness at reference point-for stuck lateral stiffness before-after comparison
    
    ###Calculations over sliding period averages of each cycle. -normal direction values are just for reference info
    ##not gonna be used for any direct calculations as they are prone to noise and drift.
    P_hold = robust_median(P_contact_N[hold])
    h_hold = robust_median(h_m[hold])
    A_hold = robust_median(A_m2[hold])
    Sz_hold = robust_median(Sz[hold])

    ## values to be calculated by median over hold period
    Ft_hold = robust_median(Ft[hold])
    X_hold = robust_median(Xc[hold])
    phi_hold = robust_median(phi[hold])
    Sx_hold = robust_median(Sx[hold])
    Dx_hold = robust_median(Dx[hold])
    Ed_hold = robust_median(Ed[hold])

    # Derived hold quantities
    mu_hold = Ft_hold / P_hold if (np.isfinite(Ft_hold) and np.isfinite(P_hold) and P_hold > 0) else np.nan
    p_hold_GPa = (normal_pressure_Pa(np.array([P_hold]), np.array([A_hold]))[0] / 1e9) if (np.isfinite(A_hold) and A_hold > 0 and np.isfinite(P_hold)) else np.nan
    tau_hold_MPa = (shear_stress_Pa(np.array([Ft_hold]), np.array([A_hold]))[0] / 1e6) if (np.isfinite(A_hold) and A_hold > 0 and np.isfinite(Ft_hold)) else np.nan

    ##Sz between cycles for reference junction growth calcs:
    ##Sz_cycle calculations to obtain A_hold; (Sz_cycle/Sz_ref)^2 = A_hold/A_ref
    Sz_end_of_cycle = robust_median(Sz[between]) if (Sz is not None and between.size > 0) else np.nan

    # Junction growth proxies
    K_ratio = (Sz_end_of_cycle / Sz_ref) if (np.isfinite(Sz_end_of_cycle) and np.isfinite(Sz_ref) and Sz_ref != 0) else np.nan
    A_ratio = (K_ratio**2) if np.isfinite(K_ratio) else np.nan
    A_cycle = (A_ratio * A_ref_m2) if (np.isfinite(A_ratio) and np.isfinite(A_ref_m2) and A_ref_m2 > 0) else np.nan


    # --- transitions: may be missing for bad/noisy cycles
    i_ss = tr.get("i_ss", None)
    i_rs = tr.get("i_rs", None)

    Ft_ss_N = tr.get("Ft_ss_N", np.nan)
    X_ss_m  = tr.get("X_ss_m", np.nan)
    Ft_rs_N = tr.get("Ft_rs_N", np.nan)
    X_rs_m  = tr.get("X_rs_m", np.nan)

    # ensure indices are ints if finite
    if i_ss is not None:
        try:
            i_ss = int(i_ss)
        except Exception:
            i_ss = None
    if i_rs is not None:
        try:
            i_rs = int(i_rs)
        except Exception:
            i_rs = None

    # Normal load and area at transitions
    P_ss = float(P_contact_N[i_ss]) if i_ss is not None else np.nan
    A_ss = float(A_m2[i_ss]) if i_ss is not None else np.nan
    mu_ss = (Ft_ss_N / P_ss) if (np.isfinite(Ft_ss_N) and np.isfinite(P_ss) and P_ss > 0) else np.nan

    P_rs = float(P_contact_N[i_rs]) if i_rs is not None else np.nan
    A_rs = float(A_m2[i_rs]) if i_rs is not None else np.nan
    mu_rs = (Ft_rs_N / P_rs) if (np.isfinite(Ft_rs_N) and np.isfinite(P_rs) and P_rs > 0) else np.nan

    # Per-point uncertainties for tau at transitions (uses df-attached sigmas)
    sigma_A = df["sigma_A_m2"].to_numpy() if "sigma_A_m2" in df.columns else None

    # pick a Ft sigma model:
    # base noise from cfg + lock-in ramp + TF residual
    Ft_sig_base = cfg.sigma_Ft_N  # baseline amplitude uncertainty
    tau_li = getattr(cfg, "lockin_tau_s", 0.05)

    sigma_Ft_ss = Ft_sig_base
    if (i_ss is not None) and np.isfinite(i_ss):
        s_li = estimate_lockin_lag_force_sigma(t, Ft, int(i_ss), tau_s=tau_li, dt_window_s=cfg.lockin_slope_hw)
        if np.isfinite(s_li):
            sigma_Ft_ss = np.sqrt(sigma_Ft_ss**2 + s_li**2)

    # if TF correction residual estimated, add it:
    if "sigma_Ft_ss_TF_N" in tr and np.isfinite(tr["sigma_Ft_ss_TF_N"]):
        sigma_Ft_ss = np.sqrt(sigma_Ft_ss**2 + float(tr["sigma_Ft_ss_TF_N"])**2)

    # tau uncertainty
    if sigma_A is not None and i_ss is not None and (0 <= i_ss < len(A_m2)):
        A_ss = float(A_m2[i_ss]) if np.isfinite(i_ss) else np.nan
        sA_ss = float(sigma_A[i_ss]) if np.isfinite(i_ss) else np.nan
        s_tau_ss = sigma_shear_strength(Ft_ss_N, A_ss, sigma_Ft_ss, sA_ss) / 1e6  # -> MPa
    else:
        s_tau_ss = np.nan

    sigma_Ft_rs = Ft_sig_base
    if (i_rs is not None) and np.isfinite(i_rs):
        s_li = estimate_lockin_lag_force_sigma(t, Ft, int(i_rs), tau_s=tau_li, dt_window_s=cfg.lockin_slope_hw)
        if np.isfinite(s_li):
            sigma_Ft_rs = np.sqrt(sigma_Ft_rs**2 + s_li**2)

    # if TF correction residual estimated, add it:
    if "sigma_Ft_rs_TF_N" in tr and np.isfinite(tr["sigma_Ft_rs_TF_N"]):
        sigma_Ft_rs = np.sqrt(sigma_Ft_rs**2 + float(tr["sigma_Ft_rs_TF_N"])**2)
    # tau uncertainty
    if sigma_A is not None and i_rs is not None and (0 <= i_rs < len(A_m2)):
        A_rs = float(A_m2[i_rs]) if np.isfinite(i_rs) else np.nan
        sA_rs = float(sigma_A[i_rs]) if np.isfinite(i_rs) else np.nan
        s_tau_rs = sigma_shear_strength(Ft_rs_N, A_rs, sigma_Ft_rs, sA_rs) / 1e6  # -> MPa
    else:
        s_tau_rs = np.nan
    # Mindlin fit on ramp-up: K(Q)
    Q = Ft[ru]
    K = Sx[ru]
    m = np.isfinite(Q) & np.isfinite(K) & (Q > 0) & (K > 0)
    Q = Q[m]; K = K[m]
    if Q.size >= cfg.mindlin_min_points:
        Qmax = float(np.max(Q))
        lo = cfg.mindlin_min_frac_of_maxF * Qmax
        hi = cfg.mindlin_max_frac_of_maxF * Qmax
        mm = (Q >= lo) & (Q <= hi)
        Qf, Kf = Q[mm], K[mm]
        mind = mindlin_fit(Qf, Kf) if Qf.size >= cfg.mindlin_min_points else {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": int(Qf.size), "ok": 0}
    else:
        mind = {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": int(Q.size), "ok": 0}
    ## Mindlin fit on ramp-down:
    Q = Ft[rd]
    K = Sx[rd]
    m = np.isfinite(Q) & np.isfinite(K) & (Q > 0) & (K > 0)
    Q = Q[m]; K = K[m]
    if Q.size >= cfg.mindlin_min_points:
        Qmax = float(np.max(Q))
        lo = cfg.mindlin_min_frac_of_maxF * Qmax
        hi = cfg.mindlin_max_frac_of_maxF * Qmax
        mm = (Q >= lo) & (Q <= hi)
        Qf, Kf = Q[mm], K[mm]
        mind_rd = mindlin_fit(Qf, Kf) if Qf.size >= cfg.mindlin_min_points else {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": int(Qf.size), "ok": 0}
    else:
        mind_rd = {"a": np.nan, "t": np.nan, "rmse": np.nan, "n": int(Q.size), "ok": 0}

    return {
        "cycle": b.cycle,
        "t_start_s": float(t[b.i_start]),
        "t_hold0_s": float(t[b.i_hold0]),
        "t_hold1_s": float(t[b.i_hold1]),
        "t_end_s": float(t[b.i_end]),

        # normal + geometry (hold)
        "P_hold_mN": float(P_hold * 1e3) if np.isfinite(P_hold) else np.nan,
        "h_hold_nm": float(h_hold * 1e9) if np.isfinite(h_hold) else np.nan,
        "A_hold_um2": float(A_hold * 1e12) if np.isfinite(A_hold) else np.nan,
        "p_hold_GPa": float(p_hold_GPa),

        # vertical stiffness (hold)
        "Sz_sliding": float(Sz_hold) if np.isfinite(Sz_hold) else np.nan,

        # lateral (hold)
        "Ft_hold_mN": float(Ft_hold * 1e3) if np.isfinite(Ft_hold) else np.nan,
        "X_hold_nm": float(X_hold * 1e9) if np.isfinite(X_hold) else np.nan,
        "phi_hold_rad": float(phi_hold) if np.isfinite(phi_hold) else np.nan,
        "Sliding_lateral_stiffness": float(Sx_hold),
        "Damping_lateral": float(Dx_hold),
        "E_diss_J_per_cycle_hold": float(Ed_hold),

        # friction + shear (hold)
        "mu_hold": float(mu_hold) if np.isfinite(mu_hold) else np.nan,
        "tau_hold_MPa": float(tau_hold_MPa) if np.isfinite(tau_hold_MPa) else np.nan,

        # transitions
        "Ft_ss_mN": float(Ft_ss_N * 1e3) if np.isfinite(Ft_ss_N) else np.nan,
        "Ft_rs_mN": float(Ft_rs_N * 1e3) if np.isfinite(Ft_rs_N) else np.nan,
        "X_ss_nm": float(X_ss_m * 1e9) if np.isfinite(X_ss_m) else np.nan,
        "X_rs_nm": float(X_rs_m * 1e9) if np.isfinite(X_rs_m) else np.nan,
        "mu_ss": float(mu_ss) if np.isfinite(mu_ss) else np.nan,
        "mu_rs": float(mu_rs) if np.isfinite(mu_rs) else np.nan,
        ##"tau_ss_MPa": float(tau_ss_MPa) if np.isfinite(tau_ss_MPa) else np.nan,
        ##"tau_rs_MPa": float(tau_rs_MPa) if np.isfinite(tau_rs_MPa) else np.nan,
        "Sx_stuck_N_per_m": float(tr.get("Sx_stuck", np.nan)),
        "Sx_thresh_N_per_m": float(tr.get("Sx_slide_used", np.nan)),

        "sigma_tau_ss_MPa" : float(s_tau_ss) if np.isfinite(s_tau_ss) else np.nan,
        "sigma_Ft_ss_uN" : float(sigma_Ft_ss * 1e6) if np.isfinite(sigma_Ft_ss) else np.nan,
        "sigma_tau_rs_MPa" : float(s_tau_rs) if np.isfinite(s_tau_rs) else np.nan,
        "sigma_Ft_rs_uN" : float(sigma_Ft_rs * 1e6) if np.isfinite(sigma_Ft_rs) else np.nan,

        # junction growth proxies
        "A_ratio_to_ref": float((K_ratio**2)) if np.isfinite(K_ratio) else np.nan,
        "K_ratio_to_ref": float(K_ratio) if np.isfinite(K_ratio) else np.nan,
        #"tau_ratio_to_ref": float(tau_ratio) if np.isfinite(tau_ratio) else np.nan,

        # mindlin ramp-up
        "mindlin_a_N_per_m": float(mind.get("a", np.nan)),
        "mindlin_t_N": float(mind.get("t", np.nan)),
        "mindlin_rmse": float(mind.get("rmse", np.nan)),
        "mindlin_n": int(mind.get("n", 0)),
        "mindlin_ok": int(mind.get("ok", 0)),
        "scipy_fit": int(SCIPY_OK),
        ## mindlin ramp-down
        "mindlin_a_rd_N_per_m": float(mind_rd.get("a", np.nan)),
        "mindlin_t_rd_N": float(mind_rd.get("t", np.nan)),
        "mindlin_rmse_rd": float(mind_rd.get("rmse", np.nan)),
        "mindlin_n_rd": int(mind_rd.get("n", 0)),
        "mindlin_ok_rd": int(mind_rd.get("ok", 0)),
    }


# ============================================================
# 13) Analyze one file
# ============================================================

def analyze_one_file(fp: Path, cfg: Config, live_plots: bool, outdir: Optional[Path]) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean, single-pass analyze_one_file with:
      - correct ordering (no undefined vars)
      - optional approval + manual repick gate
      - no overwriting of manually-picked indices
      - consistent reference window for Sz_initial, h_ref, A_ref
      - safe handling of missing transitions / sliding metrics
      - fixed tau unit conversion
      - Hertz diagnostic on the loading segment (touch -> end-of-loading)

    Assumptions:
      - cfg.manual_mode in {"never","on_fail","always"}
      - manual_pick_touch(df, cfg) -> int
      - manual_pick_shear_window(t, P_contact_N, F2_rms) -> (i0:int, i1:int)
      - manual_pick_cycles(df2, cfg, i0, i1) -> List[CycleBounds]
      - approve_or_repick_gate(fig_title) -> "accept"|"repick" (raises on skip/abort)
      - sanity_plot_window_cycles(...) can accept i0/i1 possibly None; cycles possibly None/[]
      - window_idx_fw(t, center_i, halfwidth_s) exists (forward window around i0)
      - vertical_stiffness_frame_corrected(Sz_arr, cfg.k_frame_z) exists
      - effective_modulus / hertz_fit_radius_adhesion / plot_hertz_diagnostic exist if cfg.hertz_enable
      - total_sliding_cyc_dist_speed exists
    """
    # -----------------------------
    # 0) Read + basic validation
    # -----------------------------
    df, units_map, scale = read_csv_with_units(fp)

    required = [cfg.time_col, cfg.Fz_raw_col, cfg.z_raw_col, cfg.F2_rms_col, cfg.X2_rms_col, cfg.PH2_col]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    markers = extract_markers(df, cfg.markers_col)
    t = _num(df, cfg.time_col)

    # Raw normal channels to SI
    Fz_raw_N = _num(df, cfg.Fz_raw_col) * scale[cfg.Fz_raw_col]
    z_raw_m  = _num(df, cfg.z_raw_col)  * scale[cfg.z_raw_col]

    # Vertical stiffness (optional)
    Sz_arr = None
    if getattr(cfg, "Sz_col", None) and (cfg.Sz_col in df.columns):
        Sz_arr = _num(df, cfg.Sz_col) * scale[cfg.Sz_col]  # usually N/m already
        Sz_arr = vertical_stiffness_frame_corrected(Sz_arr, cfg.k_frame_z)

    # -----------------------------
    # 1) Touch detection (auto first)
    # -----------------------------
    touch_i = None
    auto_ok_touch = True
    err_touch = None
    try:
        touch_i = detect_touch_index(df, cfg, markers)
    except Exception as e:
        auto_ok_touch = False
        err_touch = str(e)

    # If manual always: we still want a first guess for plotting, but we can repick later.
    # If auto failed and manual disabled -> error later.

    # -----------------------------
    # Helper: compute normal signals given touch_i
    # -----------------------------
    def _compute_normal_signals(_touch_i: int):
        k_sup_z, b_sup_z, _, _ = fit_support_spring_pre_touch(z_raw_m, Fz_raw_N, _touch_i)
        P_contact_N = corrected_normal_load(Fz_raw_N, z_raw_m, k_sup_z, b_sup_z)
        return k_sup_z, b_sup_z, P_contact_N

    # We cannot proceed without a touch index (unless user repicks)
    k_sup_z = b_sup_z = None
    P_contact_N = None
    if touch_i is not None:
        k_sup_z, b_sup_z, P_contact_N = _compute_normal_signals(touch_i)

    # -----------------------------
    # 2) Lateral calibration + df2 (needs touch_i for cal slice search window)
    # -----------------------------
    def _compute_df2(df_in: pd.DataFrame, cfg, markers, scale, touch_i,) -> pd.DataFrame:
        if touch_i is not None:
            cal_sl_lat, cal_sl_vert = find_calibration_slices_pre_touch(df_in, cfg, markers, touch_i)
            # optional vertical dyn diagnostics
            k_sup_z_dyn, b_sup_z_dyn = fit_vertical_dynamic_support_spring(df_in, cfg, scale, cal_sl_vert)
            cpl = fit_vertical_dynamic_coupling(df_in, cfg, scale, cal_sl_vert)
            # compute lateral corrected channels
            df2 = compute_lateral_corrected(df_in, cfg, scale, cal_sl_lat)
            return df2, cal_sl_lat, cal_sl_vert, k_sup_z_dyn, b_sup_z_dyn, cpl
        else:
            return None, None, None, np.nan, np.nan, {"kzz": np.nan, "kzx": np.nan, "r2": np.nan}

    cal_sl_lat = None
    cal_sl_vert = None
    k_sup_z_dyn = np.nan
    b_sup_z_dyn = np.nan
    cpl = {"kzz": np.nan, "kzx": np.nan, "r2": np.nan}

    df2 = None

    df2, cal_sl_lat, cal_sl_vert, k_sup_z_dyn, b_sup_z_dyn, cpl = _compute_df2(df, cfg, markers, scale, touch_i)
    # -----------------------------
    # 3) Shear window + cycles (auto)
    # -----------------------------
    win = None  # (i0, i1)
    cycles = None
    auto_ok = True
    errors = {}

    if touch_i is None:
        auto_ok = False
        errors["touch"] = err_touch or "touch_i is None"
    if P_contact_N is None:
        auto_ok = False
        errors["normal"] = "P_contact_N not computed (touch missing?)"
    if df2 is None:
        auto_ok = False
        errors["df2"] = "df2 not computed (touch missing?)"

    if auto_ok:
        try:
            i0, i1 = find_shear_window_from_normal_load_v2(t, P_contact_N, touch_i, cfg)
            win = (int(i0), int(i1))
        except Exception as e:
            auto_ok = False
            errors["window"] = str(e)

    if auto_ok and win is not None:
        try:
            cycles = detect_cycles(df2, cfg, start_i=win[0], end_i=win[1])
            if not cycles:
                raise RuntimeError("No cycles detected")
        except Exception as e:
            auto_ok = False
            errors["cycles"] = str(e)

    # -----------------------------
    # 4) Approval / manual repick gate
    # -----------------------------
    need_approval = (
        (cfg.manual_mode == "always") or
        (cfg.manual_mode == "on_fail" and not auto_ok)
    )
    figs:List[plt.Figure] = []
    if need_approval:
        # Plot current best guess (may be partial)
        try:
            figs=sanity_plot_window_cycles(
                df2=df2 if df2 is not None else df,   # df2 may be None, plot what we have
                cfg=cfg,
                t=t,
                P_contact_N=P_contact_N if P_contact_N is not None else np.zeros_like(t),
                i0=(win[0] if win else None),
                i1=(win[1] if win else None),
                cycles=(cycles if cycles is not None else []),
                cal_sl=cal_sl_lat,
                title=fp.stem,
            )
        except TypeError:
            # sanity_plot_window_cycles may require i0/i1 ints; skip plot if incomplete
            if (df2 is not None) and (P_contact_N is not None) and (win is not None) and (cycles is not None):
                figs = sanity_plot_window_cycles(
                    df2=df2, cfg=cfg, t=t, P_contact_N=P_contact_N,
                    i0=win[0], i1=win[1], cycles=cycles, cal_sl=cal_sl_lat, title=fp.stem
                )
        decision = approve_or_repick_gate(figs, "approve (a)/ repick touch (t) / repick window (w) / repick cycles (c) / skip (s) / abort (esc)")
        figs = []
    else:
        decision = "approve"  # no manual mode
        if live_plots:
            figs = sanity_plot_window_cycles( df2=df2, cfg=cfg, t=t, P_contact_N=P_contact_N, 
                                      i0=win[0], i1=win[1], cycles=cycles, cal_sl=cal_sl_lat, title=fp.stem)
            figs = show_and_wait(f"{fp.stem} — auto analysis complete", figures=figs)

    if decision == "repick_touch":
        touch_i = manual_pick_touch(df2, cfg, initial=touch_i)
        # recompute spring fit + P_contact_N + h/A etc as needed
        k_sup_z, b_sup_z, P_contact_N = _compute_normal_signals(touch_i)
        # recompute cal dynamics + df2
        df2, cal_sl_lat, cal_sl_vert, k_sup_z_dyn, b_sup_z_dyn, cpl = _compute_df2(df, cfg, markers, scale, touch_i)
        # recompute shear window + cycles
        i0, i1 = find_shear_window_from_normal_load_v2(t, P_contact_N, touch_i,cfg)
        win = (int(i0), int(i1))
        i0, i1 = manual_pick_shear_window(t, P_contact_N, F2_rms=df2[cfg.F2_rms_col], initial=[i0,i1])
        win = (int(i0), int(i1))
        cycles = detect_cycles(df2, cfg, start_i=i0, end_i=i1)  # or manual_pick_cycles too
        cycles = manual_pick_cycles(df2, cfg, i0, i1, initial=cycles, n_cycles=cfg.expected_cycles)

    elif decision == "repick_window":
        i0, i1 = manual_pick_shear_window(t, P_contact_N, F2_rms=df2[cfg.F2_rms_col], initial=[i0,i1])
        win = (int(i0), int(i1))
        cycles = detect_cycles(df2, cfg, start_i=i0, end_i=i1)  # or manual_pick_cycles too
        cycles = manual_pick_cycles(df2, cfg, i0, i1, initial=cycles, n_cycles=cfg.expected_cycles)

    elif decision == "repick_cycles":
        cycles = manual_pick_cycles(df2, cfg, i0, i1, initial=cycles, n_cycles=cfg.expected_cycles)

    # accept => do nothing


    # At this point, we REQUIRE final indices
    if touch_i is None or P_contact_N is None or df2 is None or win is None or cycles is None or len(cycles) == 0:
        raise RuntimeError("Final indices incomplete after approval/repick (touch/window/cycles).")

    i0, i1 = win

    # -----------------------------
    # 5) Depth + area (needs final touch_i)
    # -----------------------------
    h_m = contact_depth_h_m(z_raw_m, touch_i, P_contact_N, cfg.k_frame_z)
    A_nominal = area_pi_h_R(h_m, cfg.tip_radius_m)
    try:
        linear_slice = slice(cal_sl_lat.stop, touch_i)
        actuator_mck_vertical = fit_actuator_mck(t[linear_slice], z_raw_m[linear_slice], Fz_raw_N[linear_slice])
        P_correct_mck =correct_contact_force(t, z_raw_m, Fz_raw_N, float(actuator_mck_vertical.get("m_eff", np.nan)),
                                            actuator_mck_vertical.get("c_eff", np.nan), actuator_mck_vertical.get("k_eff", np.nan))
        actuator_mck_lateral = fit_actuator_mck(t[cal_sl_lat], df2["X2_pk_m"][cal_sl_lat], df2["F2_pk_N"][cal_sl_lat])
        F2_pk_cor_mck = correct_contact_force(t, df2["X2_pk_m"], df2["F2_pk_N"], float(actuator_mck_lateral.get("m_eff", np.nan)),
                                            actuator_mck_lateral.get("c_eff", np.nan), actuator_mck_lateral.get("k_eff", np.nan))
    except Exception as e:
        actuator_mck_vertical = {"m_eff": np.nan, "c_eff": np.nan, "k_eff": np.nan, "error": str(e)}
        actuator_mck_lateral = {"m_eff": np.nan, "c_eff": np.nan, "k_eff": np.nan, "error": str(e)}

    # Attach normal/depth/area + metadata to df2 for downstream functions/plots
    df2 = df2.copy()
    df2["P_contact_N"] = P_contact_N
    df2["h_m"] = h_m
    df2["A_m2"] = A_nominal
    df2["Pressure_nom_GPa"] = normal_pressure_Pa(P_contact_N, A_nominal) / 1e9  # GPa
    df2["touch_index"] = int(touch_i)

    if Sz_arr is not None:
        df2["Sz_corrected"] = Sz_arr

    df2["k_sup_z_N_per_m"] = float(k_sup_z)
    df2["b_sup_z_N"] = float(b_sup_z)
    df2["k_sup_z_dyn_N_per_m"] = float(k_sup_z_dyn) if np.isfinite(k_sup_z_dyn) else np.nan
    df2["b_sup_z_dyn_N"] = float(b_sup_z_dyn) if np.isfinite(b_sup_z_dyn) else np.nan
    df2["kzz_fit_N_per_m"] = float(cpl.get("kzz", np.nan))
    df2["kzx_fit_N_per_m"] = float(cpl.get("kzx", np.nan))
    df2["Fz_coupling_r2"] = float(cpl.get("r2", np.nan))

    # Optional: show final sanity plot
    if live_plots and getattr(cfg, "final_approve_plot", False):
        figs = sanity_plot_window_cycles(
            df2=df2, cfg=cfg, t=t, P_contact_N=P_contact_N,
            i0=i0, i1=i1, cycles=cycles, cal_sl=cal_sl_lat, title=fp.stem
        )
        figs = show_and_wait(f"{fp.stem} — final indices", figs)

    # -----------------------------
    # 6) Reference window (end of normal loading plateau)
    # -----------------------------
    ref_i = window_idx_fw(t, i0, cfg.ref_window_s)  # forward/centered window
    if ref_i.size == 0:
        ref_i = np.array([i0], dtype=int)

    Sz_initial = np.nan
    if Sz_arr is not None and ref_i.size:
        Sz_initial = robust_median(Sz_arr[ref_i])

    h_ref = robust_median(h_m[ref_i]) if ref_i.size else np.nan
    A_ref_nom = robust_median(A_nominal[ref_i]) if ref_i.size else np.nan

    if (not np.isfinite(A_ref_nom)) or (A_ref_nom <= 0):
        A_ref_nom = float(A_nominal[i0]) if np.isfinite(A_nominal[i0]) else np.nan

    # load_max_mN on same ref window (useful for Hertz batch)
    load_max_mN = float(robust_median(P_contact_N[ref_i]) * 1e3) if ref_i.size else float(P_contact_N[i0] * 1e3)

    # -----------------------------
    # 7) Hertz diagnostics on loading segment (touch -> i0)
    # -----------------------------
    
    # Hertz demonstration on loading segment before hertzian fits: -only depth-load-stiffness for initial
    # checks of load on contact and effective radius relations.
    E_star = effective_modulus(cfg.E1_Pa, cfg.nu1, cfg.E2_Pa, cfg.nu2)
    if live_plots:
        figs.append(plot_normal_loading_depth_stiffness(
            t=t,
            P_contact_N=P_contact_N,
            h_m=h_m,
            Sz_N_per_m=(Sz_arr if Sz_arr is not None else None),
            touch_i=touch_i,
            i0=i0,
            ref_i=ref_i,
            title=f"{fp.stem} — loading sanity"
        ))
        figs = show_and_wait("Normal loading sanity", figs)
        a_csm = a_from_stiffness_Sneddon(Sz_arr, E_star_Pa=E_star)
        a_geo = a_from_depth_sphere(h_m, R_m=cfg.tip_radius_m)
        figs.append(plot_contact_radius_sanity(P_contact_N, a_csm, a_geo, touch_i, i0, title=f"{fp.stem} — contact radius sanity"))
        figs = show_and_wait(f"{fp.stem} — contact radius sanity", figs)

    a_csm = None
    A_csm = None
    R_csm_a_h = np.nan
    R_csm_a_P = np.nan

    flat_fit = {"ok": 0}
    boot_flat = {"ok": 0}

    if (Sz_arr is not None) and _finite(E_star):
        a_csm = a_from_Sz(Sz_arr, float(E_star))
        A_csm = A_from_a(a_csm)

        load_sl = slice(int(touch_i), int(i0) + 1)
        R_csm_a_h = estimate_R_from_a_and_h(a_csm[load_sl], h_m[load_sl], min_h=getattr(cfg,"hertz_min_h_m",1e-9))
        R_csm_a_P = estimate_R_from_a_and_P(a_csm[load_sl], P_contact_N[load_sl], float(E_star))

        # optional: store arrays for export
        df2["a_csm_m"] = a_csm
        df2["A_csm_m2"] = A_csm

        flat_fit = fit_flat_end_stiffness(
            P_contact_N[load_sl], Sz_arr[load_sl],
            E_star_Pa=E_star,
            P_min_N=getattr(cfg, "flat_Pmin_N", None),
            P_max_N=getattr(cfg, "flat_Pmax_N", None),
            robust=True,
            n_iter=int(getattr(cfg, "flat_iter", 6)),
            clip_sigma=float(getattr(cfg, "flat_clip_sigma", 3.0)),
            min_points=int(getattr(cfg, "flat_min_points", 30)),
        )

        boot_flat = bootstrap_flat_end_stiffness_uncertainty(
            P_contact_N[load_sl], Sz_arr[load_sl],
            fit_fn=fit_flat_end_stiffness,
            fit_kwargs=dict(
                E_star_Pa=E_star,
                P_min_N=getattr(cfg, "flat_Pmin_N", None),
                P_max_N=getattr(cfg, "flat_Pmax_N", None),
                robust=True,
                n_iter=int(getattr(cfg, "flat_iter", 6)),
                clip_sigma=float(getattr(cfg, "flat_clip_sigma", 3.0)),
                min_points=int(getattr(cfg, "flat_min_points", 30)),
            ),
            n_boot=int(getattr(cfg, "flat_boot_n", 400)),
            seed=int(getattr(cfg, "flat_boot_seed", 0)),
            keep_frac=float(getattr(cfg, "flat_boot_keep_frac", 1.0)),
            min_success=int(getattr(cfg, "flat_boot_min_success", 50)),
            block_size=int(getattr(cfg, "flat_boot_block", 10)),
        )
    boot = {"ok": 0}
    if getattr(cfg, "hertz_enable", False):
        if (touch_i is not None) and (i0 is not None) and (int(i0) > int(touch_i) + 5):
            load_sl = slice(int(touch_i), int(i0) + 1)
            h_load = h_m[load_sl]
            P_load = P_contact_N[load_sl]

            hertz_kwargs = dict(
                E_star_Pa=E_star,
                adhesion_model=getattr(cfg, "adhesion_model", "auto"),
                w_J_per_m2=getattr(cfg, "w_J_per_m2", 0.0),
                sigma_rms_m=getattr(cfg, "sigma_rms_m", None),
                rough_model=getattr(cfg, "rough_model", "none"),
                delta0_m=getattr(cfg, "delta0_m", 0.3e-9),
                z0_m=getattr(cfg, "z0_m", 0.3e-9),
                mu_dmt=getattr(cfg, "mu_dmt", 0.1),
                mu_jkr=getattr(cfg, "mu_jkr", 5.0),
                min_h_m=getattr(cfg, "hertz_min_h_m", 0.0),
                max_frac_of_Pmax=getattr(cfg, "hertz_max_frac_of_Pmax", 1.0),
                min_points=int(getattr(cfg, "hertz_min_points", 8)),
                n_iter=int(getattr(cfg, "hertz_iter", 6)),
                R0_m=getattr(cfg, "tip_radius_m", None),
            )

            # --- main fit ---
            hertz = hertz_fit_radius_adhesion(h_load, P_load, **filter_kwargs_for_callable(hertz_fit_radius_adhesion, hertz_kwargs))

            # --- bootstrap fit ---
            boot = bootstrap_hertz_radius_uncertainty(
                h_load, P_load,
                fit_fn=hertz_fit_radius_adhesion,
                fit_kwargs=filter_kwargs_for_callable(hertz_fit_radius_adhesion, hertz_kwargs),
                # DO NOT pass Sz_meas unless hertz_fit actually uses it
                # Sz_meas_N_per_m=(Sz_arr[load_sl] if Sz_arr is not None else None),
                n_boot=int(getattr(cfg, "hertz_boot_n", 300)),
                seed=int(getattr(cfg, "hertz_boot_seed", 0)),
                keep_frac=float(getattr(cfg, "hertz_boot_keep_frac", 1.0)),
                min_success=int(getattr(cfg, "hertz_boot_min_success", 30)),
                block_size=int(getattr(cfg, "hertz_boot_block", 10)),
            )

            # attach bootstrap summaries into hertz dict
            hertz.update({
                "R_eff_std_m": boot.get("R_eff_std_m", np.nan),
                "R_eff_ci95_lo_m": boot.get("R_eff_ci95_lo_m", np.nan),
                "R_eff_ci95_hi_m": boot.get("R_eff_ci95_hi_m", np.nan),
                "hertz_boot_ok": int(boot.get("ok", 0)),
                "hertz_boot_n_ok": int(boot.get("n_boot_ok", 0)),
                "adhesion_model_used_mode": boot.get("adhesion_model_used_mode", ""),
                "adhesion_model_used_frac": boot.get("adhesion_model_used_frac", np.nan),
            })
        else:
            hertz = {"ok": 0, "reason": "loading segment empty/too short"}


    # ---- Decide on reference area model (after Hertz diagnostic)
    area_mode_selected = getattr(cfg, "area_mode", "nominal")  # default from cfg; can be overridden by gate if live_plots
    if live_plots: # and getattr(cfg, "area_pick_enable", False):
    # default comes from cfg.area_mode
        figs.append(plot_flat_end_fit(P_contact_N[load_sl], Sz_arr[load_sl], flat_fit, E_star_Pa=E_star))
        figs.extend(plot_hertz_diagnostic(
                    h_load, P_load, hertz,
                    title=fp.stem,
                    hardness_Pa=cfg.hardness_Pa,
                    plasticity_p0_frac=cfg.plasticity_p0_frac
                ))
        area_mode_selected = choose_area_mode_gate(figures=figs, default_mode=getattr(cfg, "area_mode", "nominal"))
        # then pass a local cfg-like choice into compute_area_from_choice
        figs = []  # clear figs after decision
    
    ####Decide on reference area: which fit to be trusted:
    A_m2_used, area_mode_used = compute_area_from_choice(
    h_m, P_contact_N, area_mode_selected, cfg=cfg, E_star_Pa=E_star, hertz=hertz, flat_end=flat_fit
    )

    # pick reference area consistently from same ref window
    A_ref = robust_median(A_m2_used[ref_i]) if ref_i.size else np.nan
    if (not np.isfinite(A_ref)) or (A_ref <= 0):
        A_ref = float(A_m2_used[i0]) if np.isfinite(A_m2_used[i0]) else np.nan

    # downstream: use A_m2_used for pressure/shear, and store area_mode_used
    p_ref_Pa = normal_pressure_Pa(P_contact_N, A_m2_used)

    # -----------------------------
    # 8) Uncertainty at reference (mode-aware)
    # -----------------------------
    P_ref = float(robust_median(P_contact_N[ref_i])) if ref_i.size else float(P_contact_N[i0])
    h_ref = float(robust_median(h_m[ref_i])) if ref_i.size else float(h_m[i0])

    # analytic A uncertainty only for nominal geometry
    sigma_A_ref_nominal = None
    A_ref_nominal = None
    if area_mode_used.lower().startswith("nominal"):
        sigma_F_N = cfg.sigma_Fz_N
        sigma_z_m = cfg.sigma_z_m
        sigma_k_frame = cfg.sigma_k_frame_z
        sigma_R_m = cfg.sigma_tip_radius_m

        k_sup_z2, b_sup_z2, sigma_k_sup, sigma_b_sup = fit_support_spring_pre_touch(z_raw_m, Fz_raw_N, touch_i)

        sigma_P_N = sigma_P_contact(
            Fz_raw_N, z_raw_m, k_sup_z2, b_sup_z2,
            sigma_F_N=sigma_F_N,
            sigma_z_m=sigma_z_m,
            sigma_k_sup=sigma_k_sup,
            sigma_b_sup=sigma_b_sup,
        )
        sigma_h_m = sigma_h_contact(
            z_raw_m, touch_i, P_contact_N, sigma_P_N,
            k_frame_z=cfg.k_frame_z,
            sigma_z_m=sigma_z_m,
            sigma_k_frame_z=sigma_k_frame
        )
        sigma_A_m2 = sigma_area_piRh(h_m, sigma_h_m, cfg.tip_radius_m, sigma_R_m)
        sigma_A_ref_nominal = float(robust_median(sigma_A_m2[ref_i])) if ref_i.size else float(sigma_A_m2[i0])
        A_ref_nominal = float(robust_median(A_nominal[ref_i])) if ref_i.size else float(A_nominal[i0])

    # sample A_ref in a way that matches the chosen area model
    A_ref_samples = build_Aref_samples(
        area_mode_used=area_mode_used,
        A_ref=float(A_ref),
        h_ref=float(h_ref),
        P_ref=float(P_ref),
        E_star_Pa=float(E_star),
        cfg=cfg,
        hertz=hertz,
        boot_flat=boot_flat,
        sigma_A_ref=sigma_A_ref_nominal,
        n_fallback=int(getattr(cfg, "ref_unc_n", 2000)),
        seed=int(getattr(cfg, "ref_unc_seed", 0)),
    )

    A_stats = summarize_dist(A_ref_samples)
    p_stats = summarize_dist(P_ref / A_ref_samples)

    # final “reported” reference values
    A_ref = float(A_stats["median"])
    p_report_GPa = float(p_stats["median"] / 1e9)
    pressure_ci95_lo_GPa = float(p_stats["ci95"][0] / 1e9)
    pressure_ci95_hi_GPa = float(p_stats["ci95"][1] / 1e9)

    # useful to store for later propagation
    df2["pressure_ref_GPa"] = p_report_GPa

    # -----------------------------
    # 9) Per-cycle report
    # -----------------------------
    rows: List[Dict] = []
    area_cycles: List[float] = [A_ref]  # area before cycle 1 (reference)
    tr={}
    for b in cycles:
        try:
            # transitions
            tr = detect_stick_slide_transitions(
                df2, b,
                sliding_lateral_stiffness_thresh=cfg.sliding_lateral_stiffness_thresh,
                resticking_lateral_stiffness_thresh=cfg.resticking_lateral_stiffness_thresh,
                frac_up=cfg.trans_frac_up,
                frac_low=cfg.trans_frac_down,
                low_frac_band=cfg.trans_low_band,
                smooth_n=cfg.trans_smooth_n,
            )
        except Exception as e:
            tr = {}

        # interactive per-cycle plots
        if live_plots and getattr(cfg, "plot_cycles", False):
            transitions = plot_check_friction_and_transitions(
                df=df2, cfg=cfg, b=b,
                P_contact_N=P_contact_N,
                A_m2=A_nominal,
                tr=tr,
                title=f"{fp.stem} — cycle {b.cycle}",
            )
            tr.update(transitions)
            # summarize (df, cfg, b, tr, h_ref, A_ref, Sz_ref)
        row = summarize_cycle(
            df2, cfg, b, tr,
            h_ref=h_ref,
            A_ref=A_ref,
            Sz_ref=Sz_initial,
        )
        try:
            # --- derive cycle areas ---
            A_ratio = row.get("A_ratio_to_ref", np.nan)
            if np.isfinite(A_ratio) and np.isfinite(A_ref) and A_ref > 0:
                A_grown = float(A_ratio * A_ref)
            else:
                A_grown = np.nan

            if np.isfinite(A_grown) and A_grown > 0:
                area_cycles.append(A_grown)
            else:
                area_cycles.append(area_cycles[-1])

            A_prev = float(area_cycles[b.cycle - 1])
            A_now  = float(area_cycles[b.cycle])

            # --- build area samples for the cycle (scaled from reference samples) ---
            if np.isfinite(A_ref) and A_ref > 0 and np.isfinite(A_prev) and A_prev > 0:
                A_prev_samples = A_ref_samples * (A_prev / A_ref)
            else:
                A_prev_samples = A_ref_samples

            if np.isfinite(A_ref) and A_ref > 0 and np.isfinite(A_now) and A_now > 0:
                A_now_samples = A_ref_samples * (A_now / A_ref)
            else:
                A_now_samples = A_ref_samples

            # --- tau at stick->slip and re-stick, with CI from area uncertainty ---
            Ft_ss_mN = row.get("Ft_ss_mN", np.nan)
            Ft_rs_mN = row.get("Ft_rs_mN", np.nan)

            if np.isfinite(Ft_ss_mN) and Ft_ss_mN > 0:
                tau_ss = (Ft_ss_mN * 1e-3) / A_prev_samples
                st = summarize_dist(tau_ss)
                row["tau_ss_MPa"] = st["median"] / 1e6
                row["tau_ss_ci95_lo_MPa"] = st["ci95"][0] / 1e6
                row["tau_ss_ci95_hi_MPa"] = st["ci95"][1] / 1e6
            else:
                row["tau_ss_MPa"] = np.nan
                row["tau_ss_ci95_lo_MPa"] = np.nan
                row["tau_ss_ci95_hi_MPa"] = np.nan

            if np.isfinite(Ft_rs_mN) and Ft_rs_mN > 0:
                tau_rs = (Ft_rs_mN * 1e-3) / A_now_samples
                st = summarize_dist(tau_rs)
                row["tau_rs_MPa"] = st["median"] / 1e6
                row["tau_rs_ci95_lo_MPa"] = st["ci95"][0] / 1e6
                row["tau_rs_ci95_hi_MPa"] = st["ci95"][1] / 1e6
            else:
                row["tau_rs_MPa"] = np.nan
                row["tau_rs_ci95_lo_MPa"] = np.nan
                row["tau_rs_ci95_hi_MPa"] = np.nan

            # total sliding metrics (only if indices exist and ordered)
            i_ss = tr.get("i_ss", None)
            i_rs = tr.get("i_rs", None)

            if (i_ss is not None) and (i_rs is not None) and (i_rs > i_ss):
                try:
                    res = total_sliding_cyc_dist_speed(
                        time_s=t,
                        amp=df2["X2_pk_contact_m"].to_numpy(),
                        freq_Hz=cfg.dyn_f2_freq_Hz,
                        start_i=int(i_ss),
                        stop_i=int(i_rs),
                    )
                    row.update(res.get("totals", {}))
                except Exception:
                    row.update({
                        "ok": 0,
                        "total_sliding_time_s": np.nan,
                        "total_osc_cycles": np.nan,
                        "total_slide_dist_m": np.nan,
                        "max_instantaneous_speed_m_per_s": np.nan,
                        "mean_instantaneous_speed_m_per_s": np.nan,
                        "overall_mean_speed_m_per_s": np.nan,
                    })
            else:
                row.update({
                    "ok": 0,
                    "total_sliding_time_s": np.nan,
                    "total_osc_cycles": np.nan,
                    "total_slide_dist_m": np.nan,
                    "max_instantaneous_speed_m_per_s": np.nan,
                    "mean_instantaneous_speed_m_per_s": np.nan,
                    "overall_mean_speed_m_per_s": np.nan,
                })
        except Exception as e:
            row = {"cycle": b.cycle, "ok": 0, "error": str(e)}
### Rows of each cycle friction and area metrics, plus metadata about fits and transitions, get collected into a list of dicts, which is then made into a DataFrame at the end. This is the main per-cycle output of the analysis.
        rows.append(row)
# Ramp-up and Ramp-down Mindlin fits plot function(s) here
        if getattr(cfg, "plot_mindlin", False):
            figs.append(plot_mindlin_fit(
                df=df2, cfg=cfg, b=b, dir=True,
                mind={
                    "a": row.get("mindlin_a_N_per_m", np.nan),
                    "t": row.get("mindlin_t_N", np.nan),
                    "rmse": row.get("mindlin_rmse", np.nan),
                    "ok": row.get("mindlin_ok", 0),
                },
                title=f"{fp.stem} — cycle {b.cycle} Mindlin (up)",
            ))
            figs.append(plot_mindlin_fit(
                df=df2, cfg=cfg, b=b, dir=False,
                mind={
                    "a": row.get("mindlin_a_rd_N_per_m", np.nan),
                    "t": row.get("mindlin_t_rd_N", np.nan),
                    "rmse": row.get("mindlin_rmse_rd", np.nan),
                    "ok": row.get("mindlin_ok_rd", 0),
                },
                title=f"{fp.stem} — cycle {b.cycle} Mindlin (down)",
            ))
            figs = show_and_wait("Mindlin", figs)

    # -----------------------------
    # 9b) Build report (cycle-level table + file-level constants replicated)
    # -----------------------------
    report = pd.DataFrame(rows)
    report.insert(0, "file", fp.name)

    # ---- ensure hertz scalar diagnostics exist even if not run
    rmse_N = float(hertz.get("rmse_N", np.nan))
    if (not np.isfinite(rmse_N)) and np.isfinite(hertz.get("rmse_mN", np.nan)):
        rmse_N = float(hertz["rmse_mN"]) * 1e-3

    adh_used = hertz.get(
        "adhesion_model_used",
        hertz.get("adhesion_model", getattr(cfg, "adhesion_model", ""))
    )

    # ---- per-file constants copied into each cycle row
    report["Sz_initial_N_per_m"] = float(Sz_initial) if np.isfinite(Sz_initial) else np.nan
    report["initial_h_nm"] = float(h_ref * 1e9) if np.isfinite(h_ref) else np.nan
    report["A_ref_um2"] = float(A_ref * 1e12) if np.isfinite(A_ref) else np.nan
    report["area_mode_used"] = str(area_mode_used)

    report["load_max_mN"] = float(load_max_mN) if np.isfinite(load_max_mN) else np.nan
    report["pressure_ref_GPa"] = float(p_report_GPa) if np.isfinite(p_report_GPa) else np.nan
    report["pressure_ref_ci95_lo_GPa"] = float(pressure_ci95_lo_GPa) if np.isfinite(pressure_ci95_lo_GPa) else np.nan
    report["pressure_ref_ci95_hi_GPa"] = float(pressure_ci95_hi_GPa) if np.isfinite(pressure_ci95_hi_GPa) else np.nan

    report["R_from_CSM_a_h_um"] = float(R_csm_a_h * 1e6) if np.isfinite(R_csm_a_h) else np.nan
    report["R_from_CSM_a_P_um"] = float(R_csm_a_P * 1e6) if np.isfinite(R_csm_a_P) else np.nan
    report["A_ref_csm_um2"] = (float(np.nanmedian(A_csm[ref_i])) * 1e12) if (A_csm is not None and ref_i.size) else np.nan
    report["a_ref_csm_um"] = (float(np.nanmedian(a_csm[ref_i])) * 1e6) if (a_csm is not None and ref_i.size) else np.nan

    # ---- Hertz/adhesion outputs
    report["E_star_GPa"] = float(E_star / 1e9) if _finite(E_star) else np.nan
    report["R_eff_um"] = float(hertz.get("R_eff_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_m", np.nan)) else np.nan
    report["hertz_rmse_mN"] = float(rmse_N * 1e3) if np.isfinite(rmse_N) else np.nan
    report["hertz_ok"] = int(hertz.get("ok", 0))
    report["hertz_n_used"] = int(hertz.get("n_used", 0))
    report["adhesion_model"] = str(adh_used)
    report["w_eff_J_per_m2"] = float(hertz.get("w_eff_J_per_m2", np.nan)) if np.isfinite(hertz.get("w_eff_J_per_m2", np.nan)) else np.nan
    report["tabor_mu"] = float(hertz.get("tabor_mu", np.nan)) if np.isfinite(hertz.get("tabor_mu", np.nan)) else np.nan
    report["Fadh_N"] = float(hertz.get("Fadh_N", np.nan)) if np.isfinite(hertz.get("Fadh_N", np.nan)) else np.nan

    report["R_eff_std_um"] = float(hertz.get("R_eff_std_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_std_m", np.nan)) else np.nan
    report["R_eff_ci95_lo_um"] = float(hertz.get("R_eff_ci95_lo_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_ci95_lo_m", np.nan)) else np.nan
    report["R_eff_ci95_hi_um"] = float(hertz.get("R_eff_ci95_hi_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_ci95_hi_m", np.nan)) else np.nan
    report["hertz_boot_ok"] = int(hertz.get("hertz_boot_ok", 0))
    report["hertz_boot_n_ok"] = int(hertz.get("hertz_boot_n_ok", 0))

    # ---- Flat-end fit + bootstrap outputs (fit is optional but useful)
    if isinstance(flat_fit, dict):
        report["flat_end_ok"] = int(flat_fit.get("ok", 0))
        report["flat_end_a_flat_um"] = float(flat_fit.get("a_flat_um", np.nan)) if np.isfinite(flat_fit.get("a_flat_um", np.nan)) else np.nan
        report["flat_end_R_eff_um"] = float(flat_fit.get("R_eff_m", np.nan) * 1e6) if np.isfinite(flat_fit.get("R_eff_m", np.nan)) else np.nan
        report["flat_end_rmse_N_per_m"] = float(flat_fit.get("rmse", np.nan)) if np.isfinite(flat_fit.get("rmse", np.nan)) else np.nan
        report["flat_end_R2"] = float(flat_fit.get("R2", np.nan)) if np.isfinite(flat_fit.get("R2", np.nan)) else np.nan
    else:
        report["flat_end_ok"] = 0
        report["flat_end_a_flat_um"] = np.nan
        report["flat_end_R_eff_um"] = np.nan
        report["flat_end_rmse_N_per_m"] = np.nan
        report["flat_end_R2"] = np.nan

    report["flat_end_boot_ok"] = int(boot_flat.get("ok", 0)) if isinstance(boot_flat, dict) else 0
    report["flat_end_boot_n_success"] = int(boot_flat.get("n_success", 0)) if isinstance(boot_flat, dict) else 0
    report["flat_end_boot_keep_frac"] = float(boot_flat.get("keep_frac", np.nan)) if (isinstance(boot_flat, dict) and np.isfinite(boot_flat.get("keep_frac", np.nan))) else np.nan
    report["flat_end_boot_n_boot"] = int(boot_flat.get("n_boot", 0)) if isinstance(boot_flat, dict) else 0

    # CI summaries (only if ok)
    if isinstance(boot_flat, dict) and int(boot_flat.get("ok", 0)) == 1:
        a_sum = boot_flat.get("a_flat_um", {})
        R_sum = boot_flat.get("R_eff_um", {})
        C_sum = boot_flat.get("C", {})
        S0_sum = boot_flat.get("S0_N_per_m", {})

        report["flat_end_a_flat_med_um"] = float(a_sum.get("median", np.nan))
        report["flat_end_a_flat_std_um"] = float(a_sum.get("std", np.nan))
        report["flat_end_a_flat_ci95_lo_um"] = float(a_sum.get("ci95", (np.nan, np.nan))[0])
        report["flat_end_a_flat_ci95_hi_um"] = float(a_sum.get("ci95", (np.nan, np.nan))[1])

        report["flat_end_R_eff_med_um"] = float(R_sum.get("median", np.nan))
        report["flat_end_R_eff_std_um"] = float(R_sum.get("std", np.nan))
        report["flat_end_R_eff_ci95_lo_um"] = float(R_sum.get("ci95", (np.nan, np.nan))[0])
        report["flat_end_R_eff_ci95_hi_um"] = float(R_sum.get("ci95", (np.nan, np.nan))[1])

        report["flat_end_S0_med_N_per_m"] = float(S0_sum.get("median", np.nan))
        report["flat_end_S0_std_N_per_m"] = float(S0_sum.get("std", np.nan))
        report["flat_end_C_med"] = float(C_sum.get("median", np.nan))
        report["flat_end_C_std"] = float(C_sum.get("std", np.nan))
    else:
        report["flat_end_a_flat_med_um"] = np.nan
        report["flat_end_a_flat_std_um"] = np.nan
        report["flat_end_a_flat_ci95_lo_um"] = np.nan
        report["flat_end_a_flat_ci95_hi_um"] = np.nan
        report["flat_end_R_eff_med_um"] = np.nan
        report["flat_end_R_eff_std_um"] = np.nan
        report["flat_end_R_eff_ci95_lo_um"] = np.nan
        report["flat_end_R_eff_ci95_hi_um"] = np.nan
        report["flat_end_S0_med_N_per_m"] = np.nan
        report["flat_end_S0_std_N_per_m"] = np.nan
        report["flat_end_C_med"] = np.nan
        report["flat_end_C_std"] = np.nan

    # ---- Transfer function (actuator MCK) outputs (dict-style)
    report["mass_kg"] = float(actuator_mck_vertical.get("m_eff", np.nan)) if isinstance(actuator_mck_vertical, dict) else np.nan
    report["damp_act_N_s_per_m"] = float(actuator_mck_vertical.get("c_eff", np.nan)) if isinstance(actuator_mck_vertical, dict) else np.nan
    report["k_act_N_per_m"] = float(actuator_mck_vertical.get("k_eff", np.nan)) if isinstance(actuator_mck_vertical, dict) else np.nan
    report["vert_mck_error"] = str(actuator_mck_vertical.get("error", "")) if isinstance(actuator_mck_vertical, dict) else ""

    report["mass_lat_kg"] = float(actuator_mck_lateral.get("m_eff", np.nan)) if isinstance(actuator_mck_lateral, dict) else np.nan
    report["damp_lat_N_s_per_m"] = float(actuator_mck_lateral.get("c_eff", np.nan)) if isinstance(actuator_mck_lateral, dict) else np.nan
    report["k_lat_N_per_m"] = float(actuator_mck_lateral.get("k_eff", np.nan)) if isinstance(actuator_mck_lateral, dict) else np.nan
    report["lat_mck_error"] = str(actuator_mck_lateral.get("error", "")) if isinstance(actuator_mck_lateral, dict) else ""

    # -----------------------------
    # 10) Summary dict (one row per file; use exact same scalars as report)
    # -----------------------------
    summary = {
        "file": fp.name,
        "n_rows": int(len(df2)),
        "touch_index": int(touch_i),
        "touch_time_s": float(t[touch_i]),
        "n_cycles": int(len(cycles)),

        # core reference quantities (match report)
        "Sz_initial_N_per_m": float(Sz_initial) if np.isfinite(Sz_initial) else np.nan,
        "initial_h_nm": float(h_ref * 1e9) if np.isfinite(h_ref) else np.nan,
        "A_ref_um2": float(A_ref * 1e12) if np.isfinite(A_ref) else np.nan,
        "area_mode_used": str(area_mode_used),
        "load_max_mN": float(load_max_mN) if np.isfinite(load_max_mN) else np.nan,
        "pressure_ref_GPa": float(p_report_GPa) if np.isfinite(p_report_GPa) else np.nan,
        "pressure_ref_ci95_lo_GPa": float(pressure_ci95_lo_GPa) if np.isfinite(pressure_ci95_lo_GPa) else np.nan,
        "pressure_ref_ci95_hi_GPa": float(pressure_ci95_hi_GPa) if np.isfinite(pressure_ci95_hi_GPa) else np.nan,

        # support / calibration meta (handy debug)
        "k_sup_z_N_per_m": float(k_sup_z),
        "b_sup_z_N": float(b_sup_z),
        "k_sup_x_N_per_m": float(df2["kx_sup_est_N_per_m"].iloc[0]) if "kx_sup_est_N_per_m" in df2.columns else np.nan,
        "b_sup_x_N": float(df2["bx_sup_est_N"].iloc[0]) if "bx_sup_est_N" in df2.columns else np.nan,
        "markers_found": ";".join(sorted(markers.keys())) if markers else "",
        "cal_slice_start": int(cal_sl_lat.start) if cal_sl_lat is not None else -1,
        "cal_slice_end": int(cal_sl_lat.stop - 1) if cal_sl_lat is not None else -1,
        "end_of_normal_loading_index": int(i0),
        "start_of_unloading_index": int(i1),
        "end_of_normal_loading_time": float(t[i0]),
        "start_of_unloading_time": float(t[i1]),

        # dynamics / coupling
        "k_sup_z_dyn_N_per_m": float(k_sup_z_dyn) if np.isfinite(k_sup_z_dyn) else np.nan,
        "b_sup_z_dyn_N": float(b_sup_z_dyn) if np.isfinite(b_sup_z_dyn) else np.nan,
        "kzx_dyn_N_per_m": float(cpl.get("kzx", np.nan)),
        "Fz_coupling_r2": float(cpl.get("r2", np.nan)),

        # Hertz/adhesion per-file
        "hertz_ok": int(hertz.get("ok", 0)),
        "hertz_reason": str(hertz.get("reason", "")),
        "E_star_GPa": float(E_star / 1e9) if _finite(E_star) else np.nan,
        "R_eff_um": float(hertz.get("R_eff_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_m", np.nan)) else np.nan,
        "hertz_rmse_mN": float(rmse_N * 1e3) if np.isfinite(rmse_N) else np.nan,
        "hertz_n_used": int(hertz.get("n_used", 0)),
        "adhesion_model": str(adh_used),
        "w_eff_J_per_m2": float(hertz.get("w_eff_J_per_m2", np.nan)) if np.isfinite(hertz.get("w_eff_J_per_m2", np.nan)) else np.nan,
        "tabor_mu": float(hertz.get("tabor_mu", np.nan)) if np.isfinite(hertz.get("tabor_mu", np.nan)) else np.nan,
        "Fadh_N": float(hertz.get("Fadh_N", np.nan)) if np.isfinite(hertz.get("Fadh_N", np.nan)) else np.nan,
        "R_eff_std_um": float(hertz.get("R_eff_std_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_std_m", np.nan)) else np.nan,
        "R_eff_ci95_lo_um": float(hertz.get("R_eff_ci95_lo_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_ci95_lo_m", np.nan)) else np.nan,
        "R_eff_ci95_hi_um": float(hertz.get("R_eff_ci95_hi_m", np.nan) * 1e6) if np.isfinite(hertz.get("R_eff_ci95_hi_m", np.nan)) else np.nan,
        "hertz_boot_ok": int(hertz.get("hertz_boot_ok", 0)),
        "hertz_boot_n_ok": int(hertz.get("hertz_boot_n_ok", 0)),
        "flat_end_ok" : int(flat_fit.get("ok", 0)) if isinstance(flat_fit, dict) else 0,
        "flat_end_a_flat_um" : float(flat_fit.get("a_flat_um", np.nan)) if isinstance(flat_fit, dict) else np.nan,
        "flat_end_boot_ok" : int(boot_flat.get("ok", 0)) if isinstance(boot_flat, dict) else 0,
        "flat_end_boot_n_success" : int(boot_flat.get("n_success", 0)) if isinstance(boot_flat, dict) else 0,
        }
    # Save per-file long report
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        report.to_csv(outdir / f"{fp.stem}_cycle_report.csv", index=False)

    return report, summary

# ============================================================
# 14) Batch + exports
# ============================================================

def analyze_batch(
    input_dir: Path,
    outdir: Path,
    cfg: Config,
    pattern: str,
    live_plots: bool,
    plot_every: int,
    summary_plots: bool,
    origin_csv: bool,
    summary_template: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {input_dir}")
    # ----------------------------
    # 1) Per-file analysis (robust)
    # ----------------------------
    all_cycles: list[pd.DataFrame] = []
    summaries: list[dict] = []
    failed: list[dict] = []

    for i, fp in enumerate(files, start=1):
        do_plots = bool(live_plots) and (plot_every <= 1 or (i % plot_every == 0))

        # IMPORTANT: avoid state leaking (e.g., cfg.area_mode being changed interactively)
        cfg_i = copy.deepcopy(cfg)

        try:
            rep, summ = analyze_one_file(fp, cfg_i, live_plots=do_plots, outdir=outdir)

            # ---- normalize report
            if rep is None or (isinstance(rep, pd.DataFrame) and rep.empty):
                rep = pd.DataFrame([{"file": fp.name, "cycle": np.nan, "ok": 0, "error": "empty_report"}])
            else:
                rep = rep.copy()
                if "file" not in rep.columns:
                    rep.insert(0, "file", fp.name)
                rep["ok_file"] = 1  # file-level ok marker

            # ---- normalize summary
            summ = dict(summ or {})
            summ["file"] = fp.name
            summ["ok"] = 1
            summ["error"] = ""

            all_cycles.append(rep)
            summaries.append(summ)

        except Exception as e:
            tb = traceback.format_exc()

            # file-level failure summary
            summ_fail = {
                "file": fp.name,
                "ok": 0,
                "error": str(e),
                "traceback": tb,
            }
            summaries.append(summ_fail)
            failed.append(summ_fail)

            # OPTIONAL: also add a stub row in cycles so “wide” outputs know file exists but failed
            all_cycles.append(pd.DataFrame([{
                "file": fp.name,
                "cycle": np.nan,
                "ok": 0,
                "error": str(e),
            }]))

    all_cycles_df = pd.concat(all_cycles, ignore_index=True) if all_cycles else pd.DataFrame()
    summaries_df = pd.DataFrame(summaries)

    # optional: save a clean failure report
    if failed:
        pd.DataFrame(failed).to_csv(outdir / "report_failures.csv", index=False)

        # ----------------------------
    # 2) Batch Hertz diagnostic (cross-experiment sanity only)
    # ----------------------------
    if getattr(cfg, "hertz_enable", False):
        hs_m: List[float] = []
        Ps_N: List[float] = []

        for s in summaries:
            if "error" in s:
                continue
            h_nm = s.get("initial_h_nm", np.nan)
            P_mN = s.get("load_max_mN", np.nan)
            if not (np.isfinite(h_nm) and np.isfinite(P_mN)):
                continue
            hs_m.append(float(h_nm) * 1e-9)
            Ps_N.append(float(P_mN) * 1e-3)

        hertz_batch = {"ok": 0, "reason": "not_enough_points"}
        if len(hs_m) >= max(8, int(getattr(cfg, "hertz_min_points", 8))):
            all_depth_m = np.asarray(hs_m, float)
            all_load_N  = np.asarray(Ps_N, float)

            E_star = effective_modulus(cfg.E1_Pa, cfg.nu1, cfg.E2_Pa, cfg.nu2)

            # Use the SAME fitter family as per-file
            hertz_batch = hertz_fit_radius_adhesion(
                all_depth_m, all_load_N,
                E_star_Pa=E_star,
                adhesion_model=getattr(cfg, "adhesion_model", "auto"),
                w_J_per_m2=getattr(cfg, "w_J_per_m2", 0.0),
                sigma_rms_m=getattr(cfg, "sigma_rms_m", None),
                rough_model=getattr(cfg, "rough_model", "none"),
                delta0_m=getattr(cfg, "delta0_m", 0.3e-9),
                z0_m=getattr(cfg, "z0_m", 0.3e-9),
                mu_dmt=getattr(cfg, "mu_dmt", 0.1),
                mu_jkr=getattr(cfg, "mu_jkr", 5.0),
                min_h_m=getattr(cfg, "hertz_min_h_m", 0.0),
                max_frac_of_Pmax=getattr(cfg, "hertz_max_frac_of_Pmax", 1.0),
                min_points=int(getattr(cfg, "hertz_min_points", 8)),
                n_iter=int(getattr(cfg, "hertz_iter", 6)),
                R0_m=getattr(cfg, "tip_radius_m", None),
            )

            if live_plots and getattr(cfg, "hertz_plot", False):
                figs = plot_hertz_diagnostic(
                    all_depth_m, all_load_N, hertz_batch,
                    title=f"{input_dir.name} — batch Hertz (sanity)",
                    hardness_Pa=getattr(cfg, "hardness_Pa", np.nan),
                    plasticity_p0_frac=getattr(cfg, "plasticity_p0_frac", np.nan),
                )
                figs = show_and_wait(f"{input_dir.name} — Hertz diagnostic", figs)

            # Stamp batch Hertz into each successful summary row
            if int(hertz_batch.get("ok", 0)) == 1:
                R_eff_m = float(hertz_batch.get("R_eff_m", np.nan))
                rmse_N  = float(hertz_batch.get("rmse_N", np.nan))
                n_used  = int(hertz_batch.get("n_used", len(hs_m)))
                model_used = str(hertz_batch.get("adhesion_model_used", ""))

                for s in summaries:
                    if "error" in s:
                        continue
                    s["R_global_um_from_batch"] = (R_eff_m * 1e6) if np.isfinite(R_eff_m) else np.nan
                    s["hertz_batch_rmse_mN"] = (rmse_N * 1e3) if np.isfinite(rmse_N) else np.nan
                    s["hertz_batch_n_pairs"] = int(n_used)
                    s["hertz_batch_model_used"] = model_used
        
        # rebuild summaries_df after stamping results
        summaries_df = pd.DataFrame(summaries)

    # ----------------------------
    # 3) Save only TWO outputs
    # ----------------------------

    # (1) short summary (one row per file)
    summary_cols = [c for c in [
        "file", "ok", "error",
        "initial_h_nm", "load_max_mN", "A_ref_um2", "pressure_ref_GPa",
        "pressure_ref_ci95_lo_GPa", "pressure_ref_ci95_hi_GPa",
        "area_mode_used",
        "Sz_initial_N_per_m",
        # Hertz
        "hertz_ok", "R_eff_um", "R_eff_std_um", "R_eff_ci95_lo_um", "R_eff_ci95_hi_um",
        "hertz_rmse_mN", "hertz_n_used",
        "adhesion_model", "w_eff_J_per_m2", "tabor_mu", "Fadh_N",
        # Flat-end (optional)
        "flat_end_boot_ok", "flat_end_a_flat_med_um", "flat_end_R_eff_med_um",
        # Diagnostics
        "R_from_CSM_a_h_um", "R_from_CSM_a_P_um",
    ] if c in summaries_df.columns]

    summaries_df_short = summaries_df.copy()
    if summary_cols:
        summaries_df_short = summaries_df_short[summary_cols].copy()

    summaries_df_short.to_csv(outdir / "summary_short.csv", index=False)

    # (2) detailed dynamic-wide report (one row per file)
    wide_dyn = build_wide_summary_dynamic(
        all_cycles_df=all_cycles_df,
        summaries_df=summaries_df,
        max_cycles=None,   # dynamic: uses max in data
    )
    wide_dyn.to_csv(outdir / "report_detailed_wide.csv", index=False)

    if summary_plots:
        figs = make_folder_summary_plots(all_cycles_df, outdir)
        figs = show_and_wait(figures=figs)

    return wide_dyn, summaries_df_short

# ============================================================
# 15) CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OSM oscillatory shear batch pipeline")

    ap.add_argument("--batch", type=str, default=None, help="Folder containing CSV files (if omitted, a folder picker opens)")
    ap.add_argument("--pattern", type=str, default="*.CSV", help="Glob pattern for CSVs")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory")

    # core physical parameters
    ap.add_argument("--tip_radius_um", type=float, default=5.0, help="Tip radius (µm) for A=pi*h*R")
    ap.add_argument("--E1_GPa", type=float, default=170.0, help="Sample Young's modulus (GPa)")
    ap.add_argument("--nu1", type=float, default=0.27, help="Sample Poisson's ratio")
    ap.add_argument("--E2_GPa", type=float, default=2.5, help="Tip Young's modulus (GPa)")
    ap.add_argument("--nu2", type=float, default=0.35, help="Tip Poisson's ratio")
    ap.add_argument("--hardness_MPa", type=float, default=500.0, help="Sample hardness (MPa) for Hertz fit")
    ap.add_argument("--plasticity_p0_frac", type=float, default=1.0, help="Plasticity parameter p0 fraction for Hertz fit")
    ap.add_argument("--hertz_enable", action="store_true", help="If set, perform Hertzian fit on loading segment")
    ap.add_argument("--hertz_min_h_nm", type=float, default=5.0, help="Minimum depth (nm) for Hertz fit")
    ap.add_argument("--hertz_max_frac_of_Pmax", type=float, default=0.2, help="Maximum fraction of Pmax for Hertz fit")
    ap.add_argument("--hertz_plot", action="store_true", help="If set, plot Hertz diagnostics during live plotting")
    # optional frame stiffness
    ap.add_argument("--k_frame_z", type=float, default=float("nan"), help="Frame stiffness Z (N/m), NaN=off")
    ap.add_argument("--k_frame_x", type=float, default=float("nan"), help="Frame stiffness X (N/m), NaN=off")

    # touch/cycle parameters
    ap.add_argument("--k_touch_min", type=float, default=500.0, help="Touch threshold on Dyn. Stiffness")
    ap.add_argument("--dynF2_active_delta", type=float, default=0.003, help="Active threshold above baseline (RMS units)")
    ap.add_argument("--dynF2_nearzero_delta", type=float, default=0.0005, help="Near-zero boundary above baseline (RMS units)")
    ap.add_argument("--smooth_n", type=int, default=301, help="Rolling median window for cycle detection")
    ap.add_argument("--k_sup_x", type=float, default=float("nan"), help="Fallback lateral support spring stiffness (N/m). Used if calibration slice not found.")
    ap.add_argument("--b_sup_x", type=float, default=0.0, help="Fallback lateral spring intercept (N). Used with --k_sup_x when calibration missing.")
    ap.add_argument("--allow_no_cal", action="store_true", help="If set, do not fail when calibration is missing; use fallback k_sup_x/b_sup_x if provided, else use k_sup_x=0.")
    ap.add_argument("--normal_load_filter_win", type=int, default="101")
    ap.add_argument("--normal_load_sustain_duration", type=float, default="0.1")
    ap.add_argument("--normal_load_rate_th", type=float, default="0.00001")
    # transition detection
    ap.add_argument("--trans_frac_up", type=float, default=0.1, help="K_thresh = trans_frac_up * S_stuck")
    ap.add_argument("--trans_frac_down", type=float, default=0.2, help="K_thresh = trans_frac_down * S_stuck")
    ap.add_argument("--sliding_lateral_stiffness_thresh", type=float, default=500, help="K_thresh minimum for stick->slide detection (N/m)")
    ap.add_argument("--resticking_lateral_stiffness_thresh", type=float, default=1000, help="K_thresh minimum for slide->stick detection (N/m)")
    ap.add_argument("--trans_smooth_n", type=int, default=21, help="Rolling median window for transition detection")

    # plotting + exports
    ap.add_argument("--live_plots", action="store_true", help="Show sanity plots during batch")
    ap.add_argument("--plot_every", type=int, default=5, help="Show plots for every Nth file")
    ap.add_argument("--summary_plots", action="store_true", help="Create folder-level summary plots (saved)")
    ap.add_argument("--origin_csv", action="store_true", help="Export Origin-friendly cycle CSVs (one per file + combined)")
    ap.add_argument("--summary_template", action="store_true", help="Export SummaryNanoRo_like.csv (units row first)")
    ap.add_argument("--expected_cycles", type=int, default=3, help="Expected number of cycles")
    ap.add_argument("--manual_mode", type=str, default="always", choices=["always", "on_fail", "never"], help="Manual repick mode")
    ap.add_argument("--manual_cycle_mode", type=str, default="always", choices=["always", "on_fail", "never"], help="Manual repick mode for cycles")
    ap.add_argument("--plot_mindlin", action="store_true", help="If set, plot Mindlin fits per cycle during live plotting")
    ap.add_argument("--plot_cycles", action="store_true", help="If set, plot per-cycle friction and transitions during live plotting")
    return ap

def main() -> None:
    args = build_argparser().parse_args()

    cfg = Config(
        tip_radius_m=float(args.tip_radius_um) * 1e-6,
        k_frame_z=None if (not np.isfinite(float(args.k_frame_z))) else float(args.k_frame_z),
        k_frame_x=None if (not np.isfinite(float(args.k_frame_x))) else float(args.k_frame_x),
        k_touch_min=float(args.k_touch_min),
        dynF2_active_delta=float(args.dynF2_active_delta),
        dynF2_nearzero_delta=float(args.dynF2_nearzero_delta),
        smooth_n=int(args.smooth_n),
        trans_frac_up=float(args.trans_frac_up),
        trans_frac_down=float(args.trans_frac_down),
        sliding_lateral_stiffness_thresh=float(args.sliding_lateral_stiffness_thresh),
        resticking_lateral_stiffness_thresh=float(args.resticking_lateral_stiffness_thresh),
        trans_smooth_n=int(args.trans_smooth_n),
        k_sup_x_fallback=None if (not np.isfinite(float(args.k_sup_x))) else float(args.k_sup_x),
        b_sup_x_fallback=float(args.b_sup_x),
        allow_no_cal=bool(args.allow_no_cal),
        manual_mode=args.manual_mode,
        manual_cycle_mode=args.manual_cycle_mode,
        expected_cycles=int(args.expected_cycles) if hasattr(args, "expected_cycles") else 3,
        plot_mindlin=bool(args.plot_mindlin),
        hertz_plot=bool(args.hertz_plot), 
        plot_cycles=bool(args.plot_cycles), 
        normal_load_smooth=int(args.normal_load_filter_win),
        normal_load_sustain=float(args.normal_load_sustain_duration),
        loading_rate_threshold=float(args.normal_load_rate_th),
    )
    set_plot_defaults()
    batch_folder = args.batch
    if not batch_folder:
        batch_folder = pick_folder_gui()

    input_dir = Path(batch_folder)

    # If user didn't explicitly set outdir, make it inside the selected folder
    # create results folder inside the chosen folder
    if args.outdir == "results":
        outdir = input_dir / "results"
    else:
        outdir = Path(args.outdir)

    analyze_batch(
        input_dir=input_dir,
        outdir=outdir,
        cfg=cfg,
        pattern=args.pattern,
        live_plots=bool(args.live_plots),
        plot_every=int(args.plot_every),
        summary_plots=bool(args.summary_plots),
        origin_csv=bool(args.origin_csv),
        summary_template=bool(args.summary_template),
    )

    print("Done.")
    print("Outputs in:", outdir)

if __name__ == "__main__":
    main()
