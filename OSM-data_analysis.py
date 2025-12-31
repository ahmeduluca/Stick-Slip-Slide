from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional SciPy for Mindlin fit (preferred)
try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# 1) Units parsing (2nd row in your CSV export)
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
    if np.nanmax(np.abs(phi)) > 7.0:
        return np.deg2rad(phi)
    return phi

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
    scale = np.nanmax(np.abs(dPdt[post])) if post.size else np.nanmax(np.abs(dPdt))
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
    smooth_n: int = 801,
    dpdt_thr_frac: float = 0.02,
    sustain_s: float = 0.3,
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

    w = max(11, int(smooth_n))
    P_sm = pd.Series(P_contact_N).rolling(w, center=True, min_periods=1).median().to_numpy()
    dPdt = np.gradient(P_sm, t)

    start = max(0, int(touch_i))
    if start >= n - 2:
        return (start, n - 1)

    # threshold based on post-touch derivative scale
    post = np.arange(start, n)
    scale = float(np.nanmax(np.abs(dPdt[post]))) if post.size else float(np.nanmax(np.abs(dPdt)))
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
    Pmax = np.nanmax(P[m])
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
    D = float(4.0 * freq_Hz * np.trapz(AA, tt))   # total slide distance over sinusoid
    v_max = float(2.0 * np.pi * freq_Hz * max(AA))  # max(|v|) over a sinusoid
    A_mean_time = float(np.trapz(AA, tt) / total_time)
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


def show_and_wait(fig_title: str = "") -> None:
    import matplotlib.pyplot as plt
    # show the active figure and block until closed
    try:
        if fig_title and plt.get_fignums():
            plt.gcf().canvas.manager.set_window_title(fig_title)
    except Exception:
        pass
    plt.draw()
    plt.pause(0.01)
    plt.show(block=True)  # blocks until the window is closed

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
    dyn_f2_freq_Hz: float = 80.0    # lateral dither frequency

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
    trans_frac_up: float = 0.1              # S_thresh = frac * Sx_stuck
    trans_frac_down: float = 0.15            # S_thresh = frac * Sx_stuck
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

def fit_support_spring_pre_touch(z_m: np.ndarray, F_N: np.ndarray, touch_i: int) -> tuple[float, float]:
    z = z_m[:touch_i]
    F = F_N[:touch_i]
    if np.isfinite(z).sum() < 50 or np.isfinite(F).sum() < 50:
        raise RuntimeError("Not enough pre-touch points to fit support spring.")
    k, b = robust_fit_line(z, F)  # F ≈ k*z + b
    if not np.isfinite(k):
        raise RuntimeError("Support spring fit failed.")
    return k, b

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
    scale = float(np.nanmax(np.abs(da[start_i:end_i+1])))
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

    Ftmax = np.nanmax(Ft_ru) if np.isfinite(Ft_ru).any() else np.nan
    if not (np.isfinite(Ftmax) and Ftmax > 0):
        return {
            "i_ss": None, "i_rs": None,
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
            "i_ss": None, "i_rs": None,
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
    slide_source = "none"

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
    import matplotlib.pyplot as plt

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
) -> None:
    """
    Live sanity plots (shows, does not save):
      (A) P_contact and dP/dt with shear window highlighted
      (B) Dyn. Force 2 (RMS) and smoothed envelope with cycle markers + calibration slice
    """
    import matplotlib.pyplot as plt

    # ----- Plot A: Normal load and derivative -----
    w = max(101, cfg.smooth_n // 2)
    P_sm = pd.Series(P_contact_N).rolling(w, center=True, min_periods=1).median().to_numpy()
    dPdt = np.gradient(P_sm, t)

    plt.figure("Sanity A: Normal load + shear window", figsize=(10, 5))
    plt.clf()
    ax1 = plt.gca()
    ax1.plot(t, P_contact_N * 1e3, label="P_contact (mN)")
    ax1.plot(t, P_sm * 1e3, label="P_contact smoothed (mN)")
    ax1.axvspan(t[i0], t[i1], alpha=0.15, label="shear window")

    # overlay cycle start/end vertical lines
    for cb in cycles:
        ax1.axvline(t[cb.i_start], linestyle="--", linewidth=1)
        ax1.axvline(t[cb.i_end], linestyle="--", linewidth=1)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normal load (mN)")
    ax1.set_title(title or "Normal load and detected shear window")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, dPdt * 1e3, label="dP/dt (mN/s)")
    ax2.set_ylabel("dP/dt (mN/s)")
    ax2.legend(loc="upper right")
    plt.tight_layout()

    # ----- Plot B: Lateral dyn force envelope + cycles + calibration -----
    F2 = _num(df2, cfg.F2_rms_col)
    F2s = pd.Series(np.nan_to_num(F2, nan=0.0)).rolling(cfg.smooth_n, center=True, min_periods=1).median().to_numpy()

    plt.figure("Sanity B: Dyn. Force 2 + cycles", figsize=(10, 5))
    plt.clf()
    ax = plt.gca()
    ax.plot(t, F2, label="Dyn. Force 2 (RMS) raw")
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
        ax.axvline(t[cb.i_end], linestyle="--", linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dyn. Force 2 (RMS, native units)")
    ax.set_title(title or "Lateral envelope and detected cycles")
    ax.legend(loc="upper right")
    plt.tight_layout()

# ----- Plot C: Vertical Stiffness + cycles + calibration -----
    VS = _num(df2, cfg.k_touch_col)

    plt.figure("Sanity C: Vertical Stiffness + cycles", figsize=(10, 5))
    plt.clf()
    ax = plt.gca()
    ax.plot(t, VS, label="Vertical Stiffness raw")

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

def make_folder_summary_plots(all_cycles_df: pd.DataFrame, outdir: Path, show: bool = False) -> None:
    if all_cycles_df.empty:
        return
    import matplotlib.pyplot as plt

    df = all_cycles_df.copy().sort_values(["file", "cycle"])

    # tau vs cycle
    plt.figure(figsize=(8, 5))
    for f, g in df.groupby("file"):
        if "tau_ss_MPa" in g.columns:
            plt.plot(g["cycle"], g["tau_ss_MPa"], marker="o", label=Path(f).stem)
    plt.xlabel("Cycle #")
    plt.ylabel("τ (MPa) at stick→slide")
    plt.title("Folder summary: shear strength at stick→slide")
    plt.tight_layout()
    plt.savefig(outdir / "summary_tau_stick2slide.png", dpi=150)
    if show:
        plt.show(block=False); plt.pause(0.001)
    plt.close()

    # mu vs cycle
    plt.figure(figsize=(8, 5))
    for f, g in df.groupby("file"):
        if "mu_ss" in g.columns:
            plt.plot(g["cycle"], g["mu_ss"], marker="o", label=Path(f).stem)
    plt.xlabel("Cycle #")
    plt.ylabel("μ at stick→slide")
    plt.title("Folder summary: friction coefficient at stick→slide")
    plt.tight_layout()
    plt.savefig(outdir / "summary_mu_stick2slide.png", dpi=150)
    if show:
        plt.show(block=True)
    plt.close()

def plot_cycle_friction_and_transitions(
    df: pd.DataFrame,
    cfg: Config,
    b: CycleBounds,
    P_contact_N: np.ndarray,
    A_m2: np.ndarray,
    tr: dict,
    title: str = "",
) -> None:
    import matplotlib.pyplot as plt

    t = _num(df, cfg.time_col)
    Ft = df["F2_pk_corr_N"].to_numpy()
    Kp = df["Stiffness_lateral"].to_numpy()

    sl = slice(b.i_start, b.i_end + 1)
    tt = t[sl]
    Ftt = Ft[sl]
    Ktt = Kp[sl]

    i_ss = tr.get("i_ss", None)
    i_rs = tr.get("i_rs", None)

    plt.figure(figsize=(10, 6))
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

    if i_ss is not None:
        ax1.axvline(t[i_ss], linewidth=2, label="stick→slide", color="red")
    if i_rs is not None:
        ax1.axvline(t[i_rs], linewidth=2, label="re-stick", color="green")

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

def plot_mindlin_fit(
    df: pd.DataFrame,
    cfg: Config,
    b: CycleBounds, dir:bool,
    mind: dict,
    title: str = "",
) -> None:
    import matplotlib.pyplot as plt

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

    plt.figure(figsize=(8, 6))
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

def plot_hertz_diagnostic(h_m, P_N, fit: dict, title: str = "", hardness_Pa: float = np.nan, plasticity_p0_frac: float = 1.0):
    import matplotlib.pyplot as plt

    mask = fit.get("mask_used", None)
    if mask is None:
        mask = np.isfinite(h_m) & np.isfinite(P_N)

    E_star = fit.get("E_star_Pa", np.nan)
    R_eff = fit.get("R_eff_m", np.nan)
    C = fit.get("C", np.nan)

    # 1) P vs h^(3/2) with fit
    plt.figure(figsize=(9, 6))
    plt.clf()
    x = np.power(np.maximum(0.0, h_m), 1.5)
    plt.plot(x[mask], P_N[mask]*1e3, "o", label="used (mN)")
    if np.isfinite(C):
        xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 300)
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
    plt.figure(figsize=(9, 6))
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
        plt.figure(figsize=(9, 6))
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
        "Sz_sliding": "Vertical stiffness (hold, median)",
        "Sliding_lateral_stiffness_hold": r"$Lateral stiffness S_{x} (hold)$",
        "Ft_hold_mN": "Lateral force amp (hold, corr)",
        "mu_hold": "Friction coefficient (hold)",
        "tau_hold_MPa": "Shear strength (hold)",
        "Ft_ss_mN": "Stick to slide force (corr)",
        "Ft_rs_mN": "Re-stick force (corr)",
        "X_ss_nm": "Slip distance at stick to slide",
        "X_rs_nm": "Slip distance at re-stick",
        "mu_ss": r"${\mu} at stick to slide$",
        "mu_rs": r"${\mu} at re-stick$",
        "tau_ss_MPa": r"${\tau} at stick to slide$",
        "tau_rs_MPa": r"${\tau} at re-stick",
        "A_ratio_to_ref": "Junction growth proxy A/A0",
        "mindlin_a_N_per_m": "Mindlin a (full-stick stiffness)",
        "mindlin_t_N": "Mindlin t (static friction force amp)",
        "mindlin_rmse": "Mindlin fit RMSE",
    }
    units = {
        "P_hold_mN": "mN",
        "Sz_sliding": "N/m",
        "Sliding_lateral_stiffness": "N/m",
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
  "Contact Depth","Initial Contact Area",
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
    Sz_end_of_cycle = (robust_median(Sz[between]) if Sz is not None else np.nan)

    # Junction growth proxies
    K_ratio = (Sz_end_of_cycle / Sz_ref) if (np.isfinite(Sz_end_of_cycle) and np.isfinite(Sz_ref) and Sz_ref != 0) else np.nan
    A_cycle = ((K_ratio)**2*A_ref_m2) if (np.isfinite(A_hold) and np.isfinite(A_ref_m2) and A_ref_m2 > 0) else np.nan

    ## transition values summarized..
    i_ss = tr["i_ss"]
    i_rs = tr["i_rs"]

    Ft_ss_N = tr["Ft_ss_N"]
    X_ss_m  = tr["X_ss_m"]
    Ft_rs_N = tr["Ft_rs_N"]
    X_rs_m  = tr["X_rs_m"]

    # Normal load and area at transitions
    P_ss = float(P_contact_N[i_ss]) if i_ss is not None else np.nan
    A_ss = float(A_m2[i_ss]) if i_ss is not None else np.nan
    mu_ss = (Ft_ss_N / P_ss) if (np.isfinite(Ft_ss_N) and np.isfinite(P_ss) and P_ss > 0) else np.nan
    ##tau_ss_MPa = (Ft_ss_N / A_ss / 1e6) if (np.isfinite(Ft_ss_N) and np.isfinite(A_ss) and A_ss > 0) else np.nan

    P_rs = float(P_contact_N[i_rs]) if i_rs is not None else np.nan
    A_rs = float(A_m2[i_rs]) if i_rs is not None else np.nan
    mu_rs = (Ft_rs_N / P_rs) if (np.isfinite(Ft_rs_N) and np.isfinite(P_rs) and P_rs > 0) else np.nan
    ##tau_rs_MPa = (Ft_rs_N / A_rs / 1e6) if (np.isfinite(Ft_rs_N) and np.isfinite(A_rs) and A_rs > 0) else np.nan

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
    df, units_map, scale = read_csv_with_units(fp)

    # required columns
    required = [cfg.time_col, cfg.Fz_raw_col, cfg.z_raw_col, cfg.F2_rms_col, cfg.X2_rms_col, cfg.PH2_col]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    #time and markers extraction
    markers = extract_markers(df, cfg.markers_col)
    t = _num(df, cfg.time_col)

    # normal raw to SI
    Fz_raw_N = _num(df, cfg.Fz_raw_col) * scale[cfg.Fz_raw_col]
    z_raw_m  = _num(df, cfg.z_raw_col)  * scale[cfg.z_raw_col]

    # vertical stiffness channel (if present)
    Sz_arr = None
    Sz_initial = np.nan
    if cfg.Sz_col in df.columns:
        Sz_arr = _num(df, cfg.Sz_col) * scale[cfg.Sz_col]  # typically N/m already
        Sz_arr = vertical_stiffness_frame_corrected(Sz_arr, cfg.k_frame_z)
    else:
        Sz_arr = None

    # touch
    touch_i = detect_touch_index(df, cfg, markers)

    # support spring fit and corrected normal load
    k_sup_z, b_sup_z = fit_support_spring_pre_touch(z_raw_m, Fz_raw_N, touch_i)
    P_contact_N = corrected_normal_load(Fz_raw_N, z_raw_m, k_sup_z, b_sup_z)

    # depth & area
    h_m = contact_depth_h_m(z_raw_m, touch_i, P_contact_N, cfg.k_frame_z)
    A_m2 = area_pi_h_R(h_m, cfg.tip_radius_m)

    # calibration for lateral spring subtraction
    cal_sl_lat, cal_sl_vert = find_calibration_slices_pre_touch(df, cfg, markers, touch_i)

    k_sup_z_dyn, b_sup_z_dyn = fit_vertical_dynamic_support_spring(df, cfg, scale, cal_sl_vert)
    cpl = fit_vertical_dynamic_coupling(df, cfg, scale, cal_sl_vert)

    df2 = compute_lateral_corrected(df, cfg, scale, cal_sl_lat)
    i0, i1 = find_shear_window_from_normal_load_v2(t, P_contact_N, touch_i)
    cycles = detect_cycles(df2, cfg, start_i=i0, end_i=i1)

    # Hertz diagnostics
    hertz = {"ok": 0}
    if cfg.hertz_enable:
        E_star = effective_modulus(cfg.E1_Pa, cfg.nu1, cfg.E2_Pa, cfg.nu2)

        load_sl = slice(touch_i, i0 + 1)
        h_load = h_m[load_sl]
        P_load = P_contact_N[load_sl]

        hertz = hertz_fit_radius(
            h_load, P_load,
            E_star_Pa=E_star,
            hardness_Pa=cfg.hardness_Pa,
            plasticity_p0_frac=cfg.plasticity_p0_frac,
            min_h_m=cfg.hertz_min_h_m,
            max_frac_of_Pmax=cfg.hertz_max_frac_of_Pmax,
            min_points=cfg.hertz_min_points,
            n_iter=cfg.hertz_iter,
        )

        if live_plots and cfg.hertz_plot:
            plot_hertz_diagnostic(h_load, P_load, hertz,
                                  title=fp.stem,
                                  hardness_Pa=cfg.hardness_Pa,
                                  plasticity_p0_frac=cfg.plasticity_p0_frac)
            show_and_wait(f"{fp.stem} — Hertz diagnostic")

    if live_plots:
        sanity_plot_window_cycles(
            df2=df2,
            cfg=cfg,
            t=t,
            P_contact_N=P_contact_N,
            i0=i0,
            i1=i1,
            cycles=cycles,
            cal_sl=cal_sl_lat,
            title=fp.stem,
        )
        show_and_wait(f"{fp.stem} — approve window/cycles")

    # attach normal/depth/area + metadata
    df2["P_contact_N"] = P_contact_N
    df2["Sz_corrected"] = Sz_arr
    df2["h_m"] = h_m
    df2["A_m2"] = A_m2
    df2["touch_index"] = touch_i
    df2["k_sup_z_N_per_m"] = k_sup_z
    df2["b_sup_z_N"] = b_sup_z
    df2["k_sup_z_dyn_N_per_m"] = k_sup_z_dyn
    df2["b_sup_z_dyn_N"] = b_sup_z_dyn
    df2["kzz_fit_N_per_m"] = cpl.get("kzz", np.nan)
    df2["kzx_fit_N_per_m"] = cpl.get("kzx", np.nan)
    df2["Fz_coupling_r2"] = cpl.get("r2", np.nan)

    if not cycles:
        raise RuntimeError("No cycles detected.")

    # Reference window = loading-finished plateau (same anchor for Sz_initial, h_ref, A_ref)
    # ---------------------------
    ref_i = window_idx_fw(t, i0, cfg.ref_window_s)  # <-- center i0 to an averaging window for stable initial loading values.

    # Initial vertical stiffness (Sz_initial)
    if Sz_arr is not None and ref_i.size:
        Sz_initial = robust_median(Sz_arr[ref_i])
    else:
        Sz_initial = np.nan

    # Initial depth/area from same window
    h_ref = robust_median(h_m[ref_i]) if ref_i.size else np.nan
    A_ref = robust_median(A_m2[ref_i]) if ref_i.size else np.nan

    # fallbacks
    if (not np.isfinite(A_ref)) or (A_ref <= 0):
        A_ref = float(A_m2[i0]) if np.isfinite(A_m2[i0]) else np.nan

    # live plots (optional)
    maybe_live_plots(df2, cfg, live_plots, title_prefix=fp.stem)

    # per-cycle report
    rows = []
    area_cycles = []
    area_cycles.append(A_ref)
    for b in cycles:
        tr = detect_stick_slide_transitions(
            df2, b, sliding_lateral_stiffness_thresh=cfg.sliding_lateral_stiffness_thresh,
            resticking_lateral_stiffness_thresh=cfg.resticking_lateral_stiffness_thresh,
            frac_up=cfg.trans_frac_up,
            frac_low=cfg.trans_frac_down,
            low_frac_band=cfg.trans_low_band,
            smooth_n=cfg.trans_smooth_n
            )
    # transitions (needed for plotting)

    # compute row (includes mindlin results internally)
        row = summarize_cycle(
            df2, cfg, b, tr,
            h_ref=h_ref,
            A_ref=A_ref,
            Sz_ref=Sz_initial)
        
        # compute shear strengths based on area initial and grown for diffeerent phases (stick to slide and restick).
        area_cycles.append(row["A_ratio_to_ref"] * A_ref)
        tau_ss_MPa = (row["Ft_ss_mN"] / area_cycles[b.cycle-1] / 1e9) if (np.isfinite(row["Ft_ss_mN"]) and np.isfinite(area_cycles[b.cycle-1]) and area_cycles[b.cycle-1] > 0) else np.nan
        tau_rs_MPa = (row["Ft_rs_mN"] / area_cycles[b.cycle] / 1e9) if (np.isfinite(row["Ft_rs_mN"]) and np.isfinite(area_cycles[b.cycle]) and area_cycles[b.cycle] > 0) else np.nan
        row["tau_ss_MPa"] = tau_ss_MPa
        row["tau_rs_MPa"] = tau_rs_MPa
        rows.append(row)
        
        ##total sliding and speed calcs for the cycle; sliding defined between ss and rs indices.

        res = total_sliding_cyc_dist_speed(
            time_s=t,
            amp=df2["X2_pk_contact_m"].to_numpy(),   # peak amplitude in meters
            freq_Hz=cfg.dyn_f2_freq_Hz,
            start_i=tr["i_ss"],
            stop_i=tr["i_rs"]
            )

        # totals
        print(res["totals"])
        row.update(res["totals"])
        # cycle inspection plots (block until closed)
        if live_plots:
            plot_cycle_friction_and_transitions(
                df=df2, cfg=cfg, b=b,
                P_contact_N=P_contact_N,
                A_m2=A_m2,
                tr=tr,
                title=f"{fp.stem} — cycle {b.cycle}"
            )
            show_and_wait()

            plot_mindlin_fit(
                df=df2, cfg=cfg, b=b, dir=True,
                mind={
                    "a": row.get("mindlin_a_N_per_m", np.nan),
                    "t": row.get("mindlin_t_N", np.nan),
                    "rmse": row.get("mindlin_rmse", np.nan),
                    "ok": row.get("mindlin_ok", 0),
                },
                title=f"{fp.stem} — cycle {b.cycle} Mindlin"
            )
            plot_mindlin_fit(
                df=df2, cfg=cfg, b=b, dir=False,
                mind={
                    "a": row.get("mindlin_a_rd_N_per_m", np.nan),
                    "t": row.get("mindlin_t_rd_N", np.nan),
                    "rmse": row.get("mindlin_rmse_rd", np.nan),
                    "ok": row.get("mindlin_ok_rd", 0),
                },
                title=f"{fp.stem} — cycle {b.cycle} Mindlin"
            )
            show_and_wait()

    report = pd.DataFrame(rows)
    report.insert(0, "file", fp.name)

    # store Sz_initial in all cycle rows for wide-summary convenience
    report["Sz_initial_N_per_m"] = Sz_initial
    report.update([("E_star_GPa", float(hertz.get("E_star_Pa", np.nan) / 1e9)),
    ("R_eff_um", float(hertz.get("R_eff_m", np.nan) * 1e6)),
    ("hertz_rmse_mN", float(hertz.get("rmse_N", np.nan) * 1e3)),
    ("hertz_n_used", int(hertz.get("n_used", 0))),
    ("hertz_ok", int(hertz.get("ok", 0)))])

    summary = {
        "file": fp.name,
        "n_rows": int(len(df2)),
        "touch_index": int(touch_i),
        "touch_time_s": float(t[touch_i]),
        "k_sup_z_N_per_m": float(k_sup_z),
        "b_sup_z_N": float(b_sup_z),
        "k_sup_x_N_per_m": float(df2["kx_sup_est_N_per_m"].iloc[0]) if "kx_sup_est_N_per_m" in df2.columns else np.nan,
        "b_sup_x_N": float(df2["bx_sup_est_N"].iloc[0]) if "bx_sup_est_N" in df2.columns else np.nan,
        "n_cycles": int(len(cycles)),
        "inital_h_nm": float(h_ref * 1e9) if np.isfinite(h_ref) else np.nan,
        "load_max_mN": robust_median(P_contact_N[ref_i])*1e3,
        "A_ref_um2": float(A_ref * 1e12) if np.isfinite(A_ref) else np.nan,
        "Sz_initial_N_per_m": float(Sz_initial) if np.isfinite(Sz_initial) else np.nan,
        "markers_found": ";".join(sorted(markers.keys())) if markers else "",
        "cal_slice_start": int(cal_sl_lat.start) if cal_sl_lat is not None else -1,
        "cal_slice_end": int(cal_sl_lat.stop - 1) if cal_sl_lat is not None else -1,
        "end_of_normal_loading_index": i0,
        "start_of_unloading_index": i1,
        "end_of_normal_loading_time": float(t[i0]),
        "start_of_unloading_time": float(t[i1]),
        "k_sup_z_dyn_N_per_m": float(k_sup_z_dyn) if np.isfinite(k_sup_z_dyn) else np.nan,
        "b_sup_z_dyn_N": float(b_sup_z_dyn) if np.isfinite(b_sup_z_dyn) else np.nan,
        "kzx_dyn_N_per_m": float(cpl.get("kzx", np.nan)),
        "Fz_coupling_r2": float(cpl.get("r2", np.nan)),
    }

    # save per-file long report
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

    all_cycles: List[pd.DataFrame] = []
    summaries: List[Dict] = []

    for i, fp in enumerate(files, start=1):
        try:
            do_plots = live_plots and (plot_every <= 1 or (i % plot_every == 0))
            rep, summ = analyze_one_file(fp, cfg, live_plots=do_plots, outdir=outdir)
            all_cycles.append(rep)
            summaries.append(summ)
        except Exception as e:
            summaries.append({"file": fp.name, "error": str(e)})

    # Hertz diagnostics-overall batch
    hertz = {"ok": 0}
    if cfg.hertz_enable:
        E_star = effective_modulus(cfg.E1_Pa, cfg.nu1, cfg.E2_Pa, cfg.nu2)

    pairs = []
    for s in summaries:
        if ("initial_h_nm" in s) and ("load_max_mN" in s):
            if np.isfinite(s["initial_h_nm"]) and np.isfinite(s["load_max_mN"]):
                pairs.append((s["initial_h_nm"] * 1e-9, s["load_max_mN"] * 1e-3))

    if len(pairs) >= 2:
        all_depth_m, all_load_N = map(np.asarray, zip(*pairs))
        hertz = hertz_fit_radius(
            all_depth_m, all_load_N,
            E_star_Pa=E_star,
            hardness_Pa=cfg.hardness_Pa,
            plasticity_p0_frac=cfg.plasticity_p0_frac,
            min_h_m=cfg.hertz_min_h_m,
            max_frac_of_Pmax=cfg.hertz_max_frac_of_Pmax,
            min_points=2,
            n_iter=cfg.hertz_iter,
        )

        if live_plots and cfg.hertz_plot:
            plot_hertz_diagnostic(all_depth_m, all_load_N, hertz, title=f"{input_dir.name} — batch Hertz",
                                  hardness_Pa=cfg.hardness_Pa,
                                  plasticity_p0_frac=cfg.plasticity_p0_frac)
            show_and_wait(f"{input_dir.name} — Hertz diagnostic")

        if int(hertz.get("ok", 0)) == 1:
        # store global result into summaries_df so it appears in report_summaries.csv
            summaries["R_global_um_from_batch"] = hertz["R_global_um"]
            summaries["hertz_batch_rmse_mN"] = hertz["rmse_mN"]
    
    all_cycles_df = pd.concat(all_cycles, ignore_index=True) if all_cycles else pd.DataFrame()
    summaries_df = pd.DataFrame(summaries)

    all_cycles_df.to_csv(outdir / "report_all_cycles.csv", index=False)
    summaries_df.to_csv(outdir / "report_summaries.csv", index=False)

    # folder summary plots (saved)
    if summary_plots:
        make_folder_summary_plots(all_cycles_df, outdir, show=live_plots)

    # Origin-friendly cycle exports (one per file + combined)
    if origin_csv and (not all_cycles_df.empty):
        ln, un = origin_long_names_and_units_cycles()
        export_origin_csv(all_cycles_df, outdir / "origin_ALL_cycles.csv", ln, un)
        if "file" in all_cycles_df.columns:
            for fname, g in all_cycles_df.groupby("file"):
                safe = Path(fname).stem.replace(" ", "_")
                export_origin_csv(g, outdir / f"origin_{safe}_cycles.csv", ln, un)

    # SummaryNanoRo-like wide export (exact columns + units row first)
    if summary_template and (not all_cycles_df.empty):
        wide = build_wide_summary_like_template(all_cycles_df)
        wide.to_csv(outdir / "SummaryNanoRo_like_raw.csv", index=False)  # header only, no units row
        export_like_summarynanoro(wide, outdir / "SummaryNanoRo_like.csv")

    return all_cycles_df, summaries_df


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
    )

    batch_folder = args.batch
    if not batch_folder:
        batch_folder = pick_folder_gui()

    input_dir = Path(batch_folder)

    # If user didn't explicitly set outdir, make it inside the selected folder
    # (matches your request: create results folder inside the chosen folder)
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
