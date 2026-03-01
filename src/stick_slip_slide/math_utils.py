import inspect
from typing import List, Tuple, Any
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

def robust_mad(x: np.ndarray, *, scale_to_sigma: bool = True) -> float:
    """
    Robust scatter from Median Absolute Deviation (MAD).

    If scale_to_sigma=True (default), returns an estimate comparable to 1σ
    for Gaussian-like noise: sigma ≈ 1.4826 * MAD.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if scale_to_sigma:
        return float(1.4826 * mad)
    return float(mad)

def lockin_regime_stats(
    *,
    sl: slice,
    prefix: str = "",
    # Any lock-in channels
    F_pk: np.ndarray | None = None,
    X_pk: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    Kp: np.ndarray | None = None,
    Kpp: np.ndarray | None = None,
    # Optional: allow additional named channels without changing signature
    extras: dict[str, np.ndarray] | None = None,
    # Optional: compute derived ratio stats if possible
    compute_ratio_R: bool = True,
) -> dict[str, Any]:
    """
    Robust per-slice statistics for lock-in channels.

    - Works with any subset of provided arrays.
    - Returns only computed keys.
    - Uses MAD-based robust sigma: sigma ~= 1.4826 * median(|x - median(x)|).
    - If both F_pk and X_pk provided, can compute ratio R=F/X and its propagated sigma
      (based on slice-level sigma_F and sigma_X, assuming independence).

    Keys (if inputs exist):
      <prefix>F_pk_med, <prefix>sigma_F_pk
      <prefix>X_pk_med, <prefix>sigma_X_pk
      <prefix>phi_med, <prefix>sigma_phi
      <prefix>Kp_med, <prefix>sigma_Kp
      <prefix>Kpp_med, <prefix>sigma_Kpp
      <prefix>R_med, <prefix>sigma_R   (if compute_ratio_R and F & X exist)
      <prefix>n_used (always, if slice valid)
    """

    def _p(k: str) -> str:
        return f"{prefix}_{k}" if prefix else k

    out: dict[str, Any] = {}

    if sl is None:
        return out

    i0 = int(sl.start) if sl.start is not None else 0
    i1 = int(sl.stop) if sl.stop is not None else 0
    if i1 <= i0:
        return out

    idx = np.arange(i0, i1)

    def _robust_med_sigma(a: np.ndarray | None) -> tuple[float, float, int]:
        if a is None:
            return (np.nan, np.nan, 0)
        x = np.asarray(a, float)
        if x.size == 0:
            return (np.nan, np.nan, 0)
        # slice safely
        idx_clip = idx[(idx >= 0) & (idx < x.size)]
        if idx_clip.size < 3:
            return (np.nan, np.nan, int(idx_clip.size))
        xs = x[idx_clip]
        xs = xs[np.isfinite(xs)]
        if xs.size < 3:
            return (np.nan, np.nan, int(xs.size))
        med = float(np.median(xs))
        mad = float(np.median(np.abs(xs - med)))
        sig = float(1.4826 * mad)  # Gaussian-equivalent sigma
        return (med, sig, int(xs.size))

    # Collect core stats
    n_used_list = []

    for name, arr in [
        ("F_pk", F_pk),
        ("X_pk", X_pk),
        ("phi", phi),
        ("Kp", Kp),
        ("Kpp", Kpp),
    ]:
        if arr is None:
            continue
        med, sig, n_used = _robust_med_sigma(arr)
        out[_p(f"{name}_med")] = med
        out[_p(f"sigma_{name}")] = sig
        n_used_list.append(n_used)

    # Extras: any other channels (e.g., "F_rms", "X_rms", "Sx", "Dx", etc.)
    if extras:
        for key, arr in extras.items():
            med, sig, n_used = _robust_med_sigma(arr)
            out[_p(f"{key}_med")] = med
            out[_p(f"sigma_{key}")] = sig
            n_used_list.append(n_used)

    # Always report how many finite samples were used (best-effort)
    if n_used_list:
        out[_p("n_used")] = int(np.max(n_used_list))
    else:
        # nothing finite provided
        out[_p("n_used")] = 0
        return out

    # Derived ratio R = F/X (optional)
    if compute_ratio_R and (F_pk is not None) and (X_pk is not None):
        Fm = out.get(_p("F_pk_med"), np.nan)
        Xm = out.get(_p("X_pk_med"), np.nan)
        sF = out.get(_p("sigma_F_pk"), np.nan)
        sX = out.get(_p("sigma_X_pk"), np.nan)

        if np.isfinite(Fm) and np.isfinite(Xm) and (Xm != 0):
            Rm = float(Fm / Xm)
            # Propagate using slice-level sigmas (independence assumption)
            sR = np.sqrt((sF / Xm) ** 2 + ((Fm * sX) / (Xm ** 2)) ** 2) if (np.isfinite(sF) and np.isfinite(sX)) else np.nan
            out[_p("R_med")] = Rm
            out[_p("sigma_R")] = float(sR) if np.isfinite(sR) else np.nan

    return out

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

def robust_fit_line_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ## this is used only for support spring linear term, no offsets allowed.
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[m]; y = y[m]
    if x.size < 10:
        return (np.nan, 0.0)

    qlo, qhi = np.quantile(x, [0.05, 0.95])
    mm = (x >= qlo) & (x <= qhi)
    if mm.sum() >= 10:
        x2, y2 = x[mm], y[mm]
    else:
        x2, y2 = x, y

    # Least-squares slope through origin: a = (x·y)/(x·x)
    denom = float(np.dot(x2, x2))
    a = float(np.dot(x2, y2) / denom) if denom > 0 else np.nan

    if np.isfinite(a) and a < 0:
        a = np.nan

    return float(a), 0.0

def robust_fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, float)[m]
    y = np.asarray(y, float)[m]

    if x.size < 10:
        return (np.nan, np.nan)

    # Trim extremes in x
    qlo, qhi = np.quantile(x, [0.05, 0.95])
    mm = (x >= qlo) & (x <= qhi)

    if mm.sum() >= 10:
        x2, y2 = x[mm], y[mm]
    else:
        x2, y2 = x, y

    # Degenerate guard: if x barely varies, slope is meaningless
    x_span = float(np.nanmax(x2) - np.nanmin(x2))
    if (not np.isfinite(x_span)) or (x_span <= 0):
        return (np.nan, np.nan)

    # polyfit can warn/overflow on ill-conditioned inputs; catch & return nan
    try:
        a, b = np.polyfit(x2, y2, 1)
    except Exception:
        return (np.nan, np.nan)

    a = float(a); b = float(b)
    if (not np.isfinite(a)) or (not np.isfinite(b)):
        return (np.nan, np.nan)

    return (a, b)

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

EPS_A = 1e-24  # m^2 (small-area guard)
EPS_F = 1e-30  # mN  (small-force guard for relative error)

def tau_nominal_with_sigma(Ft_mN, A_m2, sigma_A_m2=None, sigma_Ft_mN=None):
    """
    Nominal shear stress tau = Ft/A in MPa, with 1-sigma via standard propagation.

    Inputs
    ------
    Ft_mN : float
        Tangential force in mN
    A_m2 : float
        Area in m^2.
    sigma_A_m2 : float or None
        1-sigma uncertainty of area in m^2.
    sigma_Ft_mN : float or None
        1-sigma uncertainty of force in mN.

    Returns
    -------
    (tau_MPa, sigma_tau_MPa)
    """
    # basic validity guards
    if (not np.isfinite(Ft_mN)) or (not np.isfinite(A_m2)) or (A_m2 <= EPS_A):
        return (np.nan, np.nan)

    # nominal tau
    tau_MPa = (Ft_mN * 1e-3) / A_m2 / 1e6  # = Ft_mN * 1e-9 / A_m2

    # If tau is ~0, relative propagation is ill-conditioned; return nan
    if not np.isfinite(tau_MPa):
        return (np.nan, np.nan)

    # build relative variance from whichever sigmas are provided
    rel_var = 0.0
    any_sigma = False

    if (sigma_Ft_mN is not None) and np.isfinite(sigma_Ft_mN) and (sigma_Ft_mN > 0):
        denomF = max(abs(Ft_mN), EPS_F)
        rel_var += (sigma_Ft_mN / denomF) ** 2
        any_sigma = True

    if (sigma_A_m2 is not None) and np.isfinite(sigma_A_m2) and (sigma_A_m2 > 0):
        rel_var += (sigma_A_m2 / A_m2) ** 2
        any_sigma = True

    sigma_tau = abs(tau_MPa) * np.sqrt(rel_var) if any_sigma else np.nan
    return (tau_MPa, sigma_tau)

def inflate_tau_uncertainty_with_force(
    *,
    tau_med_Pa: float,
    ci95_lo_Pa: float,
    ci95_hi_Pa: float,
    Ft_mN: float,
    sigma_Ft_mN: float | None,
    ci95_to_sigma,
    sigma_to_ci95,
):
    """
    Combine area-derived tau uncertainty (from ci95) with independent force uncertainty.

    Assumes:
      - tau distribution already includes A uncertainty (via sampling A).
      - force uncertainty is symmetric (1-sigma) and independent of A.

    Returns
    -------
    sigma_total_Pa, (ci95_lo2_Pa, ci95_hi2_Pa)
    """
    # area-only sigma inferred from existing CI
    sigma_Aonly_Pa = ci95_to_sigma(tau_med_Pa, ci95_lo_Pa, ci95_hi_Pa)
    sigma_total_Pa = sigma_Aonly_Pa

    if (sigma_Ft_mN is not None) and np.isfinite(sigma_Ft_mN) and (sigma_Ft_mN > 0) and np.isfinite(Ft_mN):
        Fabs = abs(Ft_mN)
        if Fabs > EPS_F and np.isfinite(tau_med_Pa):
            sigma_Fterm_Pa = abs(tau_med_Pa) * (sigma_Ft_mN / Fabs)
            sigma_total_Pa = float(np.sqrt(sigma_Aonly_Pa**2 + sigma_Fterm_Pa**2))

    ci_lo2_Pa, ci_hi2_Pa = sigma_to_ci95(tau_med_Pa, sigma_total_Pa)
    return sigma_total_Pa, (ci_lo2_Pa, ci_hi2_Pa)

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

