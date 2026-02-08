import inspect
from typing import List, Tuple
import numpy as np
import pandas as pd

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

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

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

def rms_to_peak(x_rms: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0) * x_rms

def phase_to_rad(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    if safe_nanmax(np.abs(phi)) > 7.0:
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

def window_idx(t: np.ndarray, center_i: int, halfwidth_s: float) -> np.ndarray:
    t0 = t[center_i]
    return np.where((t >= t0 - halfwidth_s) & (t <= t0 + halfwidth_s))[0]

def window_idx_fw(t: np.ndarray, start_i: int, width_s: float) -> np.ndarray:
    t0 = t[start_i]
    return np.where((t >= t0) & (t <= t0 + width_s))[0]

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

def mindlin_model(Q: np.ndarray, a: float, t: float) -> np.ndarray:
    return a * np.power(np.maximum(1e-30, 1.0 - (Q / t)), 1.0 / 3.0)

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

def summarize_dist(x: np.ndarray, ci: float = 0.95) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": np.nan, "mean": np.nan, "std": np.nan, "ci95": (np.nan, np.nan)}
    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "ci95": (float(np.quantile(x, lo)), float(np.quantile(x, hi))),
    }