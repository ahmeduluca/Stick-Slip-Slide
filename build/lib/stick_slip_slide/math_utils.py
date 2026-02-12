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
    return a * np.cbrt(np.maximum(1e-30, 1.0 - (Q / t)))

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

def keep_longest_contiguous(mask: np.ndarray, *, min_len: int) -> np.ndarray:
    """
    Given a boolean mask over an array, keep only the longest contiguous True block.
    Returns a new boolean mask (same shape). If no block >= min_len, returns all-False.
    """
    mask = np.asarray(mask, bool)
    n = mask.size
    if n == 0:
        return mask

    # Find starts/ends of True runs
    d = np.diff(mask.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, n]

    if starts.size == 0:
        return np.zeros_like(mask, dtype=bool)

    lengths = ends - starts
    j = int(np.argmax(lengths))
    if lengths[j] < int(min_len):
        return np.zeros_like(mask, dtype=bool)

    out = np.zeros_like(mask, dtype=bool)
    out[starts[j]:ends[j]] = True
    return out

EPS_A = 1e-24  # m^2 ~picometer minimum
Z95 = 1.959963984540054  # 95% Gaussian factor

def tau_from_samples(Ft_mN: float, A_samples: np.ndarray, eps: float = EPS_A) -> np.ndarray:
    """Return finite tau samples (Pa) from force (mN) and area samples (m^2)."""
    if (not np.isfinite(Ft_mN)) or (Ft_mN <= 0) or (A_samples is None):
        return np.asarray([], dtype=float)
    A_safe = np.where(A_samples > eps, A_samples, np.nan)
    tau = (Ft_mN * 1e-3) / A_safe  # Pa
    tau = tau[np.isfinite(tau)]
    return tau

def tau_nominal_with_sigma(Ft_mN, A_m2, sigma_A_m2=None):
    if (not np.isfinite(Ft_mN)) or (Ft_mN <= 0) or \
    (not np.isfinite(A_m2)) or (A_m2 <= EPS_A):
        return (np.nan, np.nan)

    tau_MPa = (Ft_mN * 1e-3) / A_m2 / 1e6

    if np.isfinite(sigma_A_m2) and sigma_A_m2 > 0:
        sigma_tau = tau_MPa * (sigma_A_m2 / A_m2)
    else:
        sigma_tau = np.nan
    return (tau_MPa, sigma_tau)

# ---------- sigma -> CI95 ----------
def sigma_to_ci95(mean: float, sigma: float):
    """
    Convert symmetric 1-sigma (Gaussian) to 95% CI.
    Returns (ci_lo, ci_hi).
    """
    if not (np.isfinite(mean) and np.isfinite(sigma) and sigma > 0):
        return (np.nan, np.nan)

    delta = Z95 * sigma
    return (mean - delta, mean + delta)

# ---------- CI95 -> symmetric sigma ----------
def ci95_to_sigma(mean: float, ci_lo: float, ci_hi: float):
    """
    Infer symmetric 1-sigma from 95% CI.
    Uses total CI width.
    """
    if not (
        np.isfinite(mean)
        and np.isfinite(ci_lo)
        and np.isfinite(ci_hi)
        and ci_hi > ci_lo
    ):
        return np.nan

    return (ci_hi - ci_lo) / (2.0 * Z95)

def _fit_single_tone(t: np.ndarray, x: np.ndarray, f_hz: float) -> tuple[float, float]:
    """
    Fit x(t) ≈ a*cos(wt) + b*sin(wt) + c by least squares.
    Returns (A_pk, phi_rad) where:
      x_tone(t) = A_pk * cos(wt + phi)
    Phase is referenced to cosine.
    """
    w = 2.0 * np.pi * float(f_hz)
    ct = np.cos(w * t)
    st = np.sin(w * t)
    G = np.column_stack([ct, st, np.ones_like(t)])

    # Solve least squares
    beta, *_ = np.linalg.lstsq(G, x, rcond=None)
    a, b, _c = beta

    A_pk = float(np.hypot(a, b))
    phi = float(np.arctan2(-b, a))  # because a*cos + b*sin = A*cos(wt+phi) with phi = atan2(-b,a)
    return A_pk, phi

def add_harmonics_from_raw_x(
    df: pd.DataFrame,
    time_col: str,
    x_raw_col: str,
    f1_hz: float,
    n_cycles_window: float = 5.0, #similar to time constant (n*period collected)
    min_points: int = 200,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Sliding-window extraction of 1st and 2nd harmonics from raw displacement x_raw(t).

    Window length = n_cycles_window / f1_hz seconds centered at each row time.
    Needs df to contain time_col and x_raw_col (raw time series at the SAME sampling as df).
    Adds:
      - X1st_pk, phi1sr_rad, X1st_rms
      - X2nd_pk, phi2nd_rad, X2nd_rms
    """
    out = df.copy()
    t_all = np.asarray(out[time_col], float)
    x_all = np.asarray(out[x_raw_col], float)

    f1 = float(f1_hz)
    f2 = 2.0 * f1
    halfW = 0.5 * (float(n_cycles_window) / f1)  # seconds

    X1st_pk = np.full(len(out), np.nan)
    phi1st = np.full(len(out), np.nan)
    X2nd_pk = np.full(len(out), np.nan)
    phi2nd = np.full(len(out), np.nan)

    # Two-pointer window for O(n) scanning
    j0 = 0
    j1 = 0
    n = len(out)
    for i in range(n):
        tc = t_all[i]
        t_lo = tc - halfW
        t_hi = tc + halfW

        while j0 < n and t_all[j0] < t_lo:
            j0 += 1
        if j1 < j0:
            j1 = j0
        while j1 < n and t_all[j1] <= t_hi:
            j1 += 1

        if (j1 - j0) < min_points:
            continue

        t = t_all[j0:j1]
        x = x_all[j0:j1]

        # Detrend (remove mean + linear drift) to stabilize harmonic fits
        tt = t - t[0]
        A = np.column_stack([tt, np.ones_like(tt)])
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        x_d = x - (A @ coef)

        try:
            a1, p1 = _fit_single_tone(t, x_d, f1)
            a2, p2 = _fit_single_tone(t, x_d, f2)
        except Exception:
            continue

        X1st_pk[i], phi1st[i] = a1, p1
        X2nd_pk[i], phi2nd[i] = a2, p2

    out["X1st_pk"] = X1st_pk
    out["phi1st_rad"] = phi1st
    out["X1st_rms"] = X1st_pk / np.sqrt(2.0)

    out["X2nd_pk"] = X2nd_pk
    out["phi2nd_rad"] = phi2nd
    out["X2nd_rms"] = X2nd_pk / np.sqrt(2.0)
    return out

def _fit_single_tone_cosref(t: np.ndarray, x: np.ndarray, f_hz: float) -> tuple[float, float]:
    """
    Fit x(t) ≈ a*cos(wt) + b*sin(wt) + c (least squares).
    Returns (A_pk, phi_rad) where x_tone(t) = A_pk*cos(wt + phi), phi referenced to cosine.
    """
    w = 2.0 * np.pi * float(f_hz)
    ct = np.cos(w * t)
    st = np.sin(w * t)
    G = np.column_stack([ct, st, np.ones_like(t)])
    beta, *_ = np.linalg.lstsq(G, x, rcond=None)
    a, b, _c = beta
    A_pk = float(np.hypot(a, b))
    phi = float(np.arctan2(-b, a))  # a*cos + b*sin = A*cos(wt+phi), phi=atan2(-b,a)
    return A_pk, phi


def _detrend_linear(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove mean + linear drift: x_d = x - (m*(t-t0) + c)."""
    tt = t - t[0]
    A = np.column_stack([tt, np.ones_like(tt)])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - (A @ coef)


def add_energy_totals_and_xraw_harmonics(
    df: pd.DataFrame,
    *,
    time_col: str,
    f1_hz: float,
    x_raw_col: str,
    E_cycle_col: str = "E_diss_J_per_cycle",
    n_cycles_window: float = 5.0, #similar to time constant usage in lockin-> n*period = window.
    min_points: int = 200,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Adds:
      Energy on fundamental frq via Lock-in output:
        - P_diss_W
        - n_cycles_inc
        - dE_diss_J
        - E_diss_total_J
      Harmonics from raw -lateral-displacement:
        - x1sr_pk_m, x1st_rms_m, phi_x1sr_rad
        - x2nd_pk_m, x2nd_rms_m, phi_x2nd_rad
    then the distortion and augmented energy calcualtions, finally total dissipated enrgy.

    Assumes df contains DAQ-rate time and raw x (e.g., 500 Hz).
    Uses a sliding window centered at each row time of length n_cycles_window/f1_hz seconds.
    """
    out = df.copy()

    # ---------- total dissipated energy ----------
    t = np.asarray(out[time_col], float)
    Ecyc = np.asarray(out[E_cycle_col], float)
    f1 = float(f1_hz)

    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        pos = dt[1:][dt[1:] > 0]
        med = float(np.nanmedian(pos)) if pos.size else 0.0
        if dt[0] <= 0 and med > 0:
            dt[0] = med

    ncyc = f1 * dt
    P = Ecyc * f1
    dE = Ecyc * ncyc
    dE_safe = np.where(np.isfinite(dE), dE, 0.0)
    Etot = np.cumsum(dE_safe)

    #dissipations at fundamental harmonic -direct lock-in output
    out["P_diss_fn_W"] = P
    out["n_cycles_inc"] = ncyc
    out["dE_diss_fn_J"] = dE
    out["E_diss_fn_total_J"] = Etot

    # ---------- harmonics from x_raw ----------
    x_all = np.asarray(out[x_raw_col], float)
    f2 = 2.0 * f1
    halfW = 0.5 * (float(n_cycles_window) / f1)

    X1st_pk = np.full(len(out), np.nan)
    phi1st = np.full(len(out), np.nan)
    X2nd_pk = np.full(len(out), np.nan)
    phi2nd = np.full(len(out), np.nan)

    # O(n) window scan (assumes time is monotonic increasing)
    j0 = 0
    j1 = 0
    n = len(out)

    for i in range(n):
        tc = t[i]
        t_lo = tc - halfW
        t_hi = tc + halfW

        while j0 < n and t[j0] < t_lo:
            j0 += 1
        if j1 < j0:
            j1 = j0
        while j1 < n and t[j1] <= t_hi:
            j1 += 1

        if (j1 - j0) < min_points:
            continue

        tw = t[j0:j1]
        xw = x_all[j0:j1]

        if not np.all(np.isfinite(xw)) or not np.all(np.isfinite(tw)):
            continue

        # Remove mean + linear drift
        xwd = _detrend_linear(tw, xw)

        try:
            a1, p1 = _fit_single_tone_cosref(tw, xwd, f1)
            a2, p2 = _fit_single_tone_cosref(tw, xwd, f2)
        except Exception:
            continue

        X1st_pk[i], phi1st[i] = a1, p1
        X2nd_pk[i], phi2nd[i] = a2, p2
    out["X1st_pk"] = X1st_pk
    out["phi1st_rad"] = phi1st
    out["X1st_rms"] = X1st_pk / np.sqrt(2.0)

    out["X2nd_pk"] = X2nd_pk
    out["phi2nd_rad"] = phi2nd
    out["X2nd_rms"] = X2nd_pk / np.sqrt(2.0)
    
    # Harmonic distortion metrics (dimensionless)
    out["X2nd_over_X1st"] = out["X2nd_pk"] / np.maximum(1e-30, out["X1st_pk"])
    
    # --- Augment dissipation using X1st/X2nd (linear damping assumption) + totals + flag ---
    t = np.asarray(out[time_col], float)
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        pos = dt[1:][dt[1:] > 0]
        med = float(np.nanmedian(pos)) if pos.size else 0.0
        if dt[0] <= 0 and med > 0:
            dt[0] = med

    f1 = float(f1_hz)
    ncyc = f1 * dt

    Kpp = np.abs(np.asarray(out["Damping_lateral"], float))  # |Im(K*)| at fundamental
    X1 = np.asarray(out["X1st_pk"], float)
    X2 = np.asarray(out["X2nd_pk"], float)

    # Nonlinearity indicator (displacement distortion)
    out["X2nd_over_X1st"] = X2 / np.maximum(1e-30, X1)
    out["nonlinear_flag"] = out["X2nd_over_X1st"] > 0.05  # tweak threshold in case e.g.5%

    # Per-cycle energies
    E1 = np.asarray(out["E_diss_J_per_cycle"], float)                 # lock-in (fundamental)
    Eaug = np.pi * Kpp * (X1**2 + X2**2)                               # augmented (1st+2nd via X harmonics)
    out["E_diss_aug12_J_per_cycle"] = Eaug
    out["E_diss_extra2_J_per_cycle"] = np.pi * Kpp * (X2**2)
    out["E_extra2_over_E1"] = out["E_diss_extra2_J_per_cycle"] / np.maximum(1e-30, E1)

    # Totals (integrate over time)
    dE1 = E1 * ncyc
    dEaug = Eaug * ncyc
    out["E_diss_total_J"] = np.cumsum(np.where(np.isfinite(dE1), dE1, 0.0))
    out["E_diss_aug12_total_J"] = np.cumsum(np.where(np.isfinite(dEaug), dEaug, 0.0))

    return out

