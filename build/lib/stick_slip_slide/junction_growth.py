from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class JunctionGrowthFitResult:
    i_end_local: int
    slope: float
    intercept: float
    r2_linear: float
    x: np.ndarray
    y: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    K0: float


def _moving_average(y: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return y.copy()
    n = int(max(1, n))
    if n % 2 == 0:
        n += 1
    pad = n // 2
    yp = np.pad(y, pad_width=pad, mode="edge")
    ker = np.ones(n, dtype=float) / n
    return np.convolve(yp, ker, mode="valid")


def _r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _fit_line_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Robust simple linear fit y = m*x + b without lstsq/SVD.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xm = float(np.mean(x))
    ym = float(np.mean(y))

    dx = x - xm
    dy = y - ym

    sxx = float(np.sum(dx * dx))
    if (not np.isfinite(sxx)) or (sxx <= 0.0):
        raise ValueError("Linear fit failed: x has zero or invalid variance in fit window.")

    sxy = float(np.sum(dx * dy))
    if not np.isfinite(sxy):
        raise ValueError("Linear fit failed: invalid covariance in fit window.")

    m = sxy / sxx
    b = ym - m * xm

    if (not np.isfinite(m)) or (not np.isfinite(b)):
        raise ValueError("Linear fit failed: non-finite slope/intercept.")

    return float(m), float(b)


def fit_junction_growth_from_stiffness(
    Ft_N: np.ndarray,
    Pz_N: np.ndarray,
    K_Npm: np.ndarray,
    *,
    i_end_local: int,
    K0_mode: str = "median_first",   # "median_first" or "first"
    n_ref: int = 5,
    smooth_n: int = 1,
    min_fit_pts: int = 5,
    eps_Pz: float = 1e-30,
    eps_K: float = 1e-30,
    plot: bool = False,
) -> JunctionGrowthFitResult:
    """
    Simple junction-growth fitter with user-supplied fit end index.

    Assumes the arrays are already the correct sliced branch.
    No internal masking, no index remapping, no turn-point detection.

    Fits:
        y = (K/K0)^4 - 1
    versus
        x = (Ft/Pz)^2

    from index 0 to i_end_local (inclusive).
    """
    Ft = np.asarray(Ft_N, dtype=float).copy()
    Pz = np.asarray(Pz_N, dtype=float).copy()
    K = np.asarray(K_Npm, dtype=float).copy()

    if not (Ft.ndim == Pz.ndim == K.ndim == 1):
        raise ValueError("Ft_N, Pz_N, and K_Npm must be 1D arrays.")

    n = len(Ft)
    if not (len(Pz) == n and len(K) == n):
        raise ValueError("Ft_N, Pz_N, and K_Npm must have the same length.")

    if n == 0:
        raise ValueError("Empty input arrays.")

    if not (0 <= i_end_local < n):
        raise ValueError(f"i_end_local={i_end_local} is out of bounds for data length {n}.")

    j = i_end_local + 1
    if j < min_fit_pts:
        raise ValueError(f"Only {j} points up to i_end_local, but min_fit_pts={min_fit_pts}.")

    # Fit window only
    Ft_fit_raw = Ft[:j]
    Pz_fit_raw = Pz[:j]
    K_fit_raw = K[:j]

    # Hard sanity checks before forming x and y
    bad = (
        ~np.isfinite(Ft_fit_raw)
        | ~np.isfinite(Pz_fit_raw)
        | ~np.isfinite(K_fit_raw)
        | (np.abs(Pz_fit_raw) <= eps_Pz)
        | (K_fit_raw <= eps_K)
    )

    if np.any(bad):
        ibad = np.where(bad)[0]
        preview = ibad[:10].tolist()
        raise ValueError(
            "Invalid values in fit window. "
            f"Bad local indices (first up to 10 shown): {preview}. "
            "Check NaN/inf, near-zero Pz, or non-positive K."
        )

    n_ref_eff = min(max(1, n_ref), j)
    if K0_mode == "first":
        K0 = float(K_fit_raw[0])
    elif K0_mode == "median_first":
        K0 = float(np.median(K_fit_raw[:n_ref_eff]))
    else:
        raise ValueError("K0_mode must be 'first' or 'median_first'.")

    if (not np.isfinite(K0)) or (K0 <= eps_K):
        raise ValueError(f"Invalid K0={K0}. Check early stiffness points.")

    # Build full x,y for returned arrays / plotting
    bad_all = (
        ~np.isfinite(Ft)
        | ~np.isfinite(Pz)
        | ~np.isfinite(K)
        | (np.abs(Pz) <= eps_Pz)
        | (K <= eps_K)
    )
    if np.any(bad_all):
        # keep this strict so downstream behavior is predictable
        ibad = np.where(bad_all)[0]
        preview = ibad[:10].tolist()
        raise ValueError(
            "Invalid values in provided arrays. "
            f"Bad local indices (first up to 10 shown): {preview}. "
            "Since this fitter assumes upstream cleaning, please clean/slice before calling."
        )

    x = (Ft / Pz) ** 2
    y_raw = (K / K0) ** 4 - 1.0
    y = _moving_average(y_raw, smooth_n)

    x_fit = x[:j]
    y_fit_data = y[:j]

    if (not np.all(np.isfinite(x_fit))) or (not np.all(np.isfinite(y_fit_data))):
        raise ValueError("Non-finite x or y values produced in fit window.")

    # Require some variation in x
    if np.ptp(x_fit) <= 0.0:
        raise ValueError("Fit window has no variation in x = (Ft/Pz)^2.")

    slope, intercept = _fit_line_closed_form(x_fit, y_fit_data)
    y_fit = slope * x_fit + intercept
    r2_linear = _r2_score(y_fit_data, y_fit)

    if plot:
        plt.figure(figsize=(6, 4.5))
        plt.plot(x, y, "k.-", lw=1, ms=4, label="data")
        plt.plot(x_fit, y_fit, "r-", lw=2, label="linear fit")
        plt.axvline(x[i_end_local], color="r", ls="--", lw=1, label="fit end")
        plt.xlabel(r"$(F_t/P_z)^2$")
        plt.ylabel(r"$(K/K_0)^4 - 1$")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    return JunctionGrowthFitResult(
        i_end_local=int(i_end_local),
        slope=float(slope),
        intercept=float(intercept),
        r2_linear=float(r2_linear),
        x=x,
        y=y,
        x_fit=x_fit,
        y_fit=y_fit,
        K0=float(K0),
    )