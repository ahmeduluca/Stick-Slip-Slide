
"""
roughness_adhesion_pack.py
=========================

A lightweight toolkit to connect AFM roughness (height maps) to contact-regime indicators for
diamond | fused silica contacts:

- Combine rms roughness from two surfaces.
- Compute isotropic PSD C(q) from AFM height maps (FFT-based), and moments:
    sigma (rms height), m (rms slope), kappa (rms curvature).
- Estimate adhesion regime tendency (Tabor parameter mu_T) for macro sphere contacts.
- Provide simple "effective adhesion" reductions gamma_eff using either rms height (sigma) or rms slope (m).
- Provide Hertz contact radius a_H and approach/indentation depth delta_H vs load and plot delta_H/ sigma.

Designed to be easy to extend later (e.g., Persson renormalization, Pastewka-Robbins stickiness maps).

Dependencies: numpy, matplotlib. (Optional: imageio for reading TIFF/PNG height images)

Notes on AFM input
------------------
Best input: an AFM height map exported as a *grid* (CSV/ASCII) where each cell is a height value.
You must supply pixel spacing (dx, dy). Heights can be in nm; set units accordingly.

If you only have an image (PNG/TIF) where grayscale encodes height, you can load it too, but you
must provide the scale: height_nm_per_gray and optional offset_nm.

Author: Ahmed Uluca (generated with ChatGPT)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Core contact parameters
# ----------------------------

@dataclass
class MaterialPair:
    """Material pair properties for reduced modulus and adhesion parameters."""
    E_star: float = 69.6e9   # Pa, diamond|silica default
    z0: float = 0.3e-9       # m, adhesive interaction range (order atomic)

def hertz_a(P: np.ndarray, R: float, E_star: float) -> np.ndarray:
    """Hertz contact radius a for sphere radius R [m], load P [N]."""
    return (3.0 * P * R / (4.0 * E_star)) ** (1.0 / 3.0)

def hertz_delta(P: np.ndarray, R: float, E_star: float) -> np.ndarray:
    """Hertz approach/indentation depth delta = a^2 / R [m]."""
    a = hertz_a(P, R, E_star)
    return (a * a) / R

def tabor_mu(R: float, w: float, E_star: float, z0: float) -> float:
    """Tabor parameter mu_T for smooth sphere contact."""
    return (R * w * w / (E_star * E_star * z0 ** 3.0)) ** (1.0 / 3.0)

def combine_rms(sigma1: float, sigma2: float) -> float:
    """Quadrature combine rms heights (same units)."""
    return float(np.sqrt(sigma1 * sigma1 + sigma2 * sigma2))

def hertz_P(delta, R, E_star):
    """Hertz load vs indentation depth delta (sphere on flat)."""
    return (4.0/3.0) * E_star * np.sqrt(R) * delta**1.5

def dmt_P(delta, R, E_star, w):
    """DMT: Hertz load minus constant adhesive force 2πRw."""
    F_ad = 2.0*np.pi*R*w
    return hertz_P(delta, R, E_star) - F_ad

def pull_off_forces(R, w):
    """Pull-off forces (negative)."""
    return {
        "DMT": -2.0*np.pi*R*w,
        "JKR": -1.5*np.pi*R*w,
    }

def jkr_a_from_P(P, R, E_star, w):
    """JKR contact radius a(P) (classic closed form)."""
    P = np.asarray(P, dtype=float)
    term = 6.0*np.pi*R*w*P + (3.0*np.pi*R*w)**2
    term = np.maximum(term, 0.0)
    a3 = (3.0*R/(4.0*E_star)) * (P + 3.0*np.pi*R*w + np.sqrt(term))
    a3 = np.maximum(a3, 0.0)
    return a3**(1.0/3.0)

def jkr_delta_from_a(a, R, E_star, w):
    """JKR indentation relation."""
    a = np.asarray(a, dtype=float)
    return (a*a)/R - np.sqrt(np.maximum((8.0*np.pi*w*a)/(3.0*E_star), 0.0))

def jkr_P_delta_curve(delta, R, E_star, w):
    """
    JKR load as a function of delta by parameterizing via P and interpolating.
    """
    P_po = -1.5*np.pi*R*w
    P_max = max(5e-3, 50*abs(P_po))  # heuristic upper limit
    P_grid = np.linspace(P_po, P_max, 6000)

    a = jkr_a_from_P(P_grid, R, E_star, w)
    d = jkr_delta_from_a(a, R, E_star, w)

    order = np.argsort(d)
    d_sorted = d[order]
    P_sorted = P_grid[order]

    delta = np.asarray(delta, dtype=float)
    delta_clip = np.clip(delta, d_sorted[0], d_sorted[-1])
    return np.interp(delta_clip, d_sorted, P_sorted)

def w_eff_from_sigma(w0, sigma_m, z0_m=0.3e-9, C=1.0):
    # roughness kills adhesion (height-only heuristic)
    return w0 * np.exp(-(C*sigma_m/z0_m)**2)

def w_eff_from_slope(w0, m, m0=0.2):
    # slope-based heuristic (often more realistic than sigma-only)
    return w0 * np.exp(-(m/m0)**2)

def w_use_vs_load(P, w0, w_eff, Pc):
    # P in N; Pc in N
    return w_eff + (w0 - w_eff) * (1.0 - np.exp(-P/Pc))

# ----------------------------
# PSD + moments from AFM map
# ----------------------------

def _hann2(ny: int, nx: int) -> np.ndarray:
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    return wy[:, None] * wx[None, :]

def psd_isotropic_from_map(h: np.ndarray, dx: float, dy: float, detrend: bool = True, window: bool = True,
                           nbins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic (radially averaged) PSD C(q) from a 2D height map h[y,x] in meters.

    Returns:
        q_centers: [1/m]
        Cq: [m^4] isotropic PSD

    Convention:
        Uses FFT-based estimate; suitable for regime indicators and relative comparisons.
    """
    h = np.array(h, dtype=float)
    ny, nx = h.shape
    if detrend:
        h = h - np.mean(h)
        # optional plane detrend (fast):
        yy, xx = np.mgrid[0:ny, 0:nx]
        A = np.c_[xx.ravel(), yy.ravel(), np.ones(nx * ny)]
        coeff, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
        plane = (coeff[0] * xx + coeff[1] * yy + coeff[2])
        h = h - plane

    if window:
        w = _hann2(ny, nx)
        h = h * w
        # compensate mean power loss of Hann window approximately
        win_norm = np.mean(w ** 2)
    else:
        win_norm = 1.0

    # FFT
    H = np.fft.rfft2(h)
    # spatial frequencies
    qx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)   # [1/m]
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)    # [1/m]
    Qx, Qy = np.meshgrid(qx, qy)
    q = np.sqrt(Qx ** 2 + Qy ** 2)

    # 2D PSD estimate (units m^4)
    Lx = nx * dx
    Ly = ny * dy
    # Parseval-consistent scaling:
    C2 = (dx * dy / (Lx * Ly)) * (np.abs(H) ** 2) / win_norm

    # Radial binning
    q_flat = q.ravel()
    C_flat = C2.ravel()

    qmax = np.max(q_flat)
    bins = np.linspace(0.0, qmax, nbins + 1)
    inds = np.digitize(q_flat, bins) - 1
    Cq = np.zeros(nbins)
    qsum = np.zeros(nbins)
    cnt = np.zeros(nbins)

    for i in range(nbins):
        mask = inds == i
        if np.any(mask):
            Cq[i] = np.mean(C_flat[mask])
            qsum[i] = np.mean(q_flat[mask])
            cnt[i] = np.sum(mask)

    valid = cnt > 0
    return qsum[valid], Cq[valid]

def moments_from_isotropic_psd(q: np.ndarray, Cq: np.ndarray) -> tuple[float, float, float]:
    """
    Compute sigma (rms height), m (rms slope), kappa (rms curvature) from isotropic PSD C(q).

    For isotropic PSD C(q) [m^4]:
        sigma^2 = 2π ∫ q C(q) dq
        m^2     = 2π ∫ q^3 C(q) dq
        kappa^2 = 2π ∫ q^5 C(q) dq
    """
    q = np.asarray(q)
    Cq = np.asarray(Cq)
    # ensure increasing q
    order = np.argsort(q)
    q = q[order]
    Cq = Cq[order]
    dq = np.gradient(q)

    sigma2 = 2.0 * np.pi * np.sum(q * Cq * dq)
    m2     = 2.0 * np.pi * np.sum(q ** 3 * Cq * dq)
    k2     = 2.0 * np.pi * np.sum(q ** 5 * Cq * dq)

    return float(np.sqrt(max(sigma2, 0.0))), float(np.sqrt(max(m2, 0.0))), float(np.sqrt(max(k2, 0.0)))

# ----------------------------
# Simple gamma_eff models
# ----------------------------

def gamma_eff_from_sigma(gamma0: float, sigma: float, z0: float, C: float = 1.0) -> float:
    """
    Heuristic "roughness kills adhesion" via rms height sigma.
    gamma_eff = gamma0 * exp(-(C*sigma/z0)^2)
    """
    x = C * sigma / z0
    return float(gamma0 * np.exp(-(x * x)))

def gamma_eff_from_slope(gamma0: float, m: float, m0: float = 0.2) -> float:
    """
    Heuristic adhesion reduction via rms slope m (dimensionless).
    gamma_eff = gamma0 * exp(-(m/m0)^2)
    """
    x = m / m0
    return float(gamma0 * np.exp(-(x * x)))

# ----------------------------
# IO helpers
# ----------------------------

def load_height_csv(path: Path, unit: str = "nm") -> np.ndarray:
    """
    Load a grid CSV/ASCII file into a 2D height map.
    Assumes file contains only numeric grid values.

    unit: 'm', 'nm', 'um'
    """
    arr = np.loadtxt(path, delimiter=",")
    scale = {"m": 1.0, "nm": 1e-9, "um": 1e-6}[unit]
    return arr * scale

def load_height_image(path: Path, height_nm_per_gray: float, offset_nm: float = 0.0) -> np.ndarray:
    """
    Load an image (PNG/TIF) where grayscale encodes height.
    Requires 'imageio' installed. Heights returned in meters.

    height_nm_per_gray: nm per grayscale level (0..255) or per normalized intensity (0..1)
    """
    try:
        import imageio
    except ImportError as e:
        raise ImportError("imageio is required for image loading. Install: pip install imageio") from e

    im = imageio.imread(path)


    im = imageio.imread(path)
    if im.ndim == 3:
        im = im[..., 0]  # take first channel
    im = im.astype(float)
    # If 0..255
    if im.max() > 1.5:
        gray = im / 255.0
    else:
        gray = im
    h_nm = offset_nm + height_nm_per_gray * gray
    return h_nm * 1e-9

# ----------------------------
# Plotting
# ----------------------------

def plot_delta_vs_load(material: MaterialPair, radii_um: list[float], sigma_nm: float,
                       P_min_uN: float = 10.0, P_max_mN: float = 50.0, out: Path | None = None):
    """
    Plot Hertz approach delta_H vs load P and right axis delta_H/sigma (log 10^n).
    """
    P = np.logspace(np.log10(P_min_uN * 1e-6), np.log10(P_max_mN * 1e-3), 300)  # N
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
    ax.set_xscale("log"); ax.set_yscale("log")

    colors = ["#222222", "#D00000", "#00509D", "#00A896", "#6A00F4"]
    for i, R_um in enumerate(radii_um):
        R = R_um * 1e-6
        d_nm = hertz_delta(P, R, material.E_star) * 1e9
        ax.plot(P * 1e3, d_nm, lw=2.4, color=colors[i % len(colors)], label=fr"$R={R_um:g}\,\mu$m")

    ax.set_xlabel("Normal load P [mN]")
    ax.set_ylabel(r"Hertz approach $\delta_H$ [nm]")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(framealpha=0.95, fontsize=9, loc="lower right")

    # right axis
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ymin_nm, ymax_nm = ax.get_ylim()
    ax2.set_ylim(ymin_nm / sigma_nm, ymax_nm / sigma_nm)
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax2.yaxis.set_major_locator(LogLocator(base=10))
    ax2.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax2.set_ylabel(fr"Scaled approach $\delta_H/\sigma$ (σ={sigma_nm:g} nm)")

    fig.tight_layout()
    if out:
        fig.savefig(out.with_suffix(".png"), dpi=300)
        fig.savefig(out.with_suffix(".svg"))
    return fig

def plot_psd(q: np.ndarray, Cq: np.ndarray, out: Path | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(q, Cq, lw=2.0)
    ax.set_xlabel(r"wavenumber $q$ [1/m]")
    ax.set_ylabel(r"Isotropic PSD $C(q)$ [m$^4$]")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    if out:
        fig.savefig(out.with_suffix(".png"), dpi=300)
        fig.savefig(out.with_suffix(".svg"))
    return fig
import matplotlib.pyplot as plt
from pathlib import Path

def plot_load_depth_and_pulloff(outdir, E_star, radii_um, w_use, delta_max_nm=200.0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    delta = np.logspace(-2, np.log10(delta_max_nm), 500)*1e-9  # m

    # Save pull-off table
    lines = ["R_um, Fpo_DMT_mN, Fpo_JKR_mN\n"]
    for R_um in radii_um:
        R = R_um*1e-6
        fpo = pull_off_forces(R, w_use)
        lines.append(f"{R_um:.6g}, {fpo['DMT']*1e3:.6g}, {fpo['JKR']*1e3:.6g}\n")
    (outdir/"pull_off_table.csv").write_text("".join(lines))

    # Plot load-depth curves
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=200)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)  # allows negative loads around pull-off

    for R_um in radii_um:
        R = R_um*1e-6
        P_h = hertz_P(delta, R, E_star)
        P_d = dmt_P(delta, R, E_star, w_use)
        P_j = jkr_P_delta_curve(delta, R, E_star, w_use)

        ax.plot(delta*1e9, P_h*1e3, lw=1.6, alpha=0.7, label=f"Hertz, R={R_um:g} µm")
        ax.plot(delta*1e9, P_d*1e3, lw=2.0, linestyle="--", label=f"DMT, R={R_um:g} µm")
        ax.plot(delta*1e9, P_j*1e3, lw=2.0, linestyle="-.", label=f"JKR, R={R_um:g} µm")

    ax.axhline(0, color="0.4", lw=1)
    ax.set_xlabel("Indentation depth δ [nm]")
    ax.set_ylabel("Normal load P [mN]")
    ax.set_title(f"Normal load–depth curves: Hertz vs DMT vs JKR (w={w_use:g} J/m²)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, ncol=2, framealpha=0.95)
    fig.tight_layout()

    fig.savefig(outdir/"load_depth_Hertz_DMT_JKR.png", dpi=300)
    fig.savefig(outdir/"load_depth_Hertz_DMT_JKR.svg")

# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="AFM roughness → PSD moments → adhesion/contact regime helpers")
    p.add_argument("--csv", type=str, default="", help="Path to AFM height grid CSV (values only).")
    p.add_argument("--csv-unit", type=str, default="nm", choices=["m","nm","um"], help="Units of CSV heights.")
    p.add_argument("--image", type=str, default="", help="Path to AFM height image (PNG/TIF) with grayscale->height.")
    p.add_argument("--height-nm-per-gray", type=float, default=1.0, help="For --image: nm per intensity (0..1).")
    p.add_argument("--offset-nm", type=float, default=0.0, help="For --image: offset in nm.")
    p.add_argument("--dx-nm", type=float, required=False, default=10.0, help="Pixel spacing dx in nm.")
    p.add_argument("--dy-nm", type=float, required=False, default=10.0, help="Pixel spacing dy in nm.")
    p.add_argument("--nbins", type=int, default=250, help="Radial bins for isotropic PSD.")
    p.add_argument("--gamma0", type=float, default=0.05, help="Reference work of adhesion gamma0 [J/m^2].")
    p.add_argument("--sigma-silica-nm", type=float, default=0.5, help="Silica rms height [nm] (if no AFM map).")
    p.add_argument("--sigma-diamond-nm", type=float, default=1.0, help="Diamond rms height [nm] (if no AFM map).")
    p.add_argument("--outdir", type=str, default="roughness_pack_out", help="Output directory for plots.")
    p.add_argument("--w0", type=float, default=0.05, help="Baseline work of adhesion w0 [J/m^2] (clean smooth-sphere upper bound).")
    p.add_argument("--use-w-eff", action="store_true", help="Use roughness-reduced effective adhesion w_eff instead of w0.")
    p.add_argument("--C-gamma", type=float, default=1.0, help="Heuristic factor in w_eff = w0*exp(-(C*sigma/z0)^2).")
    p.add_argument("--w-min", type=float, default=1e-6, help="Floor for w_use [J/m^2] to avoid zeros.")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    material = MaterialPair()

    # If AFM map provided, compute PSD + moments; otherwise use rms inputs.
    sigma_map = None
    m_map = None
    if args.csv:
        h = load_height_csv(Path(args.csv), unit=args.csv_unit)
        dx = args.dx_nm * 1e-9
        dy = args.dy_nm * 1e-9
        q, Cq = psd_isotropic_from_map(h, dx, dy, nbins=args.nbins)
        sigma_map, m_map, kappa_map = moments_from_isotropic_psd(q, Cq)
        print(f"[AFM CSV] sigma={sigma_map*1e9:.3g} nm, m={m_map:.3g}, kappa={kappa_map:.3g} 1/m")
        plot_psd(q, Cq, out=outdir/"psd")
    elif args.image:
        h = load_height_image(Path(args.image), args.height_nm_per_gray, args.offset_nm)
        dx = args.dx_nm * 1e-9
        dy = args.dy_nm * 1e-9
        q, Cq = psd_isotropic_from_map(h, dx, dy, nbins=args.nbins)
        sigma_map, m_map, kappa_map = moments_from_isotropic_psd(q, Cq)
        print(f"[AFM IMG] sigma={sigma_map*1e9:.3g} nm, m={m_map:.3g}, kappa={kappa_map:.3g} 1/m")
        plot_psd(q, Cq, out=outdir/"psd")
    else:
        print("[No AFM map] Using provided rms heights only.")

    # Combined sigma (use AFM sigma if given for one surface; otherwise combine silica+diamond)
    sigma_comb_nm = combine_rms(args.sigma_silica_nm, args.sigma_diamond_nm)
    sigma_use_nm = sigma_comb_nm if sigma_map is None else (sigma_map*1e9)  # choose map sigma if present
    print(f"Combined/used sigma ≈ {sigma_use_nm:.3g} nm")

    # gamma_eff estimates
    gamma_eff_sig = gamma_eff_from_sigma(args.gamma0, sigma_use_nm*1e-9, material.z0, C=1.0)
    print(f"gamma0={args.gamma0:.3g} J/m^2 -> gamma_eff(sigma)≈{gamma_eff_sig:.3g} J/m^2")

    if m_map is not None:
        gamma_eff_m = gamma_eff_from_slope(args.gamma0, m_map, m0=0.2)
        print(f"gamma_eff(slope)≈{gamma_eff_m:.3g} J/m^2  (m0=0.2 heuristic)")

    # Tabor parameters for your radii under gamma0 and gamma_eff (smooth-sphere indicator)
    for R_um in [0.593, 13.434, 81.285]:
        R = R_um * 1e-6
        mu0 = tabor_mu(R, args.gamma0, material.E_star, material.z0)
        mue = tabor_mu(R, max(gamma_eff_sig, 1e-12), material.E_star, material.z0)
        print(f"R={R_um:g} µm: mu_T(gamma0)={mu0:.3g}, mu_T(gamma_eff)={mue:.3g}")

    # Hertz delta vs load plot with right axis delta/sigma
    plot_delta_vs_load(material, radii_um=[0.593, 13.434, 81.285], sigma_nm=float(sigma_use_nm),
                       out=outdir/"delta_vs_load")
    # Determine w_use
    w0 = args.w0
    if m_map is not None:          # PSD-based
        w_eff = w_eff_from_slope(w0, m_map, m0=0.2)
    else:                          # rms-only fallback
        sigma_m = sigma_use_nm * 1e-9
        w_eff = w_eff_from_sigma(w0, sigma_m, z0_m=material.z0, C=args.C_gamma)
    w_use = max(w_eff, args.w_min) if args.use_w_eff else max(w0, args.w_min)
    
    plot_load_depth_and_pulloff(
    outdir="out_pulloff",
    E_star=material.E_star,
    radii_um=[0.593, 13.434, 81.285],
    w_use=w_use,
    delta_max_nm=200.0
    )
    print(f"w0 = {w0:.3g} J/m^2")
    print(f"w_eff = {w_eff:.3g} J/m^2 (roughness reduced)")
    print(f"w_use = {w_use:.3g} J/m^2 (used in models)")

    print(f"Saved plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
