from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import Config
from ..cycle_types import CycleBounds
from ..math_utils import (
    _num, safe_numeric, median_dt, contiguous_regions, safe_nanmax, safe_nanmin
)
from ..mechanics import hertz_apparent_radius_R_of_h
from ..math_utils import mindlin_model  # after moving it there


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

    df["cycle"] = safe_numeric(df["cycle"]).astype("Int64")
    df = df.dropna(subset=["cycle"])
    df["cycle"] = df["cycle"].astype(int)

    if max_cycle_to_plot is None:
        max_cycle_to_plot = int(df["cycle"].max())
    else:
        max_cycle_to_plot = int(max(1, max_cycle_to_plot))

    # Helpful base columns (may not exist in all runs)
    for c in ["mu_ss", "tau_ss_MPa", "pressure_ref_GPa", "A_ref_um2", "A_ratio_to_ref", "mu_hold", "mindlin_t_N", "mindlin_a_N_per_m"]:
        if c in df.columns:
            df[c] = safe_numeric(df[c])

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
