# Schematic-friction_scaling_laws_PRB.py
# Minimal, PRB-clean schematic figure: pressure scaling (top) + size scaling (bottom)

from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def prb_rcparams() -> None:
    mpl.rcParams.update({
        # typography (PRB-like)
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # lines / axes
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,

        # save
        "savefig.dpi": 300,
    })


def tau_corrugation_model(p: np.ndarray, tau0: float, k: float, p0: float, m: float) -> np.ndarray:
    """
    Schematic pressure-induced corrugation / adhesion form:
        tau(p) = tau0 + k * (p/p0)^m
    """
    return tau0 + k * (p / p0) ** m


def tau_dislocation_size_model(a: np.ndarray, b: float, tau_coherent: float, a0: float, exponent: float,
                               tau_large_plateau: float) -> np.ndarray:
    """
    Schematic HK/Gao-like: tau ~ (a/b)^(-1/2) until it hits a large-contact plateau.
    Uses a smooth "cap" at small a and "floor" at large a.

    exponent: typically 1/2 for HK/Gao.
    """
    x = np.maximum(a / b, 1.0)
    tau = tau_coherent * (x / (a0 / b)) ** (-exponent)
    tau = np.minimum(tau, tau_coherent)          # coherent upper bound
    tau = np.maximum(tau, tau_large_plateau)     # large-contact plateau
    return tau


def make_anchor_box(entries: list[dict]) -> str:
    lines = ["Anchors:"]
    for d in entries:
        lines.append(f"{d['id']:>2d}  {d['label']}")
    return "\n".join(lines)


def plot_numbered_points(ax, x, y, ids, *, ms=9):
    # Plot a marker, then the number on top of it (white), PRB-clean.
    ax.scatter(x, y, s=(ms * 10) ** 2 / 10, zorder=5)
    for xi, yi, idx in zip(x, y, ids):
        ax.text(xi, yi, str(idx), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=6)


def main(
    *,
    out_png: str = "fig1_landscape_prb_clean.png",
    out_pdf: str = "fig1_landscape_prb_clean.pdf",
):
    prb_rcparams()

    # --- user-tunable constants ---
    b_m = 0.5e-9  # Burgers vector / atomic length scale for normalization (schematic)
    # Plateaus and magnitudes are schematic; you can tune to your preferred "tau/G*" scales.
    tau_coherent_over_G = 8e-2     # coherent upper bound ~ O(1e-1)
    tau_large_over_G = 8e-4        # large-contact plateau ~ O(1e-3)

    # Pressure model (schematic): match your idea: modest slope + possible upturn
    # Here we keep it as a single curve (no duplicate elastic/plastic curves),
    # and show yield region via shaded band.
    tau0_Pa = 35e6     # baseline shear strength at low p (schematic)
    k_Pa = 8e6         # amplitude for pressure term (schematic)
    p0_Pa = 1e9        # 1 GPa scale
    m = 0.8            # sublinear -> gentle rise until high p

    # Pressure range: keep to what you said is useful (no need kPa–10 GPa if you don't want)
    # Here: 1 MPa to 10 GPa, but the axis will focus on 1e6–1e10.
    p = np.logspace(6, 10, 600)

    # Size range: 1 nm to 100 um
    a = np.logspace(-9, -4, 700)

    # --- schematic model curves ---
    # Top: tau_max vs pbar
    tau_p = tau_corrugation_model(p, tau0=tau0_Pa, k=k_Pa, p0=p0_Pa, m=m)

    # Bottom: tau/G* vs a/b (HK/Gao)
    tau_over_G = tau_dislocation_size_model(
        a=a,
        b=b_m,
        tau_coherent=tau_coherent_over_G,
        a0=2e-9,                 # sets where the coherent cap starts (schematic)
        exponent=0.5,            # HK/Gao ~ a^{-1/2}
        tau_large_plateau=tau_large_over_G
    )

    # Add two additional schematic single-asperity scalings you asked for:
    # (i) repulsive commensurate mapped to tau ~ a^{-1/3}
    # (ii) adhesive incommensurate mapped to tau ~ a^{-3/2}
    # Both with caps/floors so they don't blow up.
    def capped_power(a_arr, a_ref, y_ref, exp, y_min, y_max):
        y = y_ref * (a_arr / a_ref) ** (-exp)
        y = np.clip(y, y_min, y_max)
        return y

    tau_over_G_rep = capped_power(a, a_ref=1e-9, y_ref=8e-3, exp=1/3, y_min=6e-4, y_max=1e-1)
    tau_over_G_adh_inc = capped_power(a, a_ref=1e-9, y_ref=2e-3, exp=1.5, y_min=1e-4, y_max=1e-1)

    # --- anchors (from PDFs; fill missing p and a as you prefer) ---
    # Keep exactly your dict style as source-of-truth for plotting.
    anchors = [
        dict(id=1, label="AFM Pt/mica", a_m=13.7e-9, p_Pa=np.nan, tau_Pa=910e6, G_Pa=22.3e9),
        dict(id=2, label="AFM SiNx/mica", a_m=8.4e-9, p_Pa=np.nan, tau_Pa=52e6,  G_Pa=np.nan),
        dict(id=3, label="AFM WC/diamond", a_m=1.1e-9, p_Pa=np.nan, tau_Pa=238e6, G_Pa=np.nan),

        dict(id=4, label="2D indenter diamond/silica", a_m=np.nan, p_Pa=np.nan, tau_Pa=0.3e9, G_Pa=np.nan),
        dict(id=5, label="2D indenter diamond/silica", a_m=np.nan, p_Pa=np.nan, tau_Pa=1.7e9, G_Pa=np.nan),

        dict(id=6, label="MoNI (HOPG)", a_m=np.nan, p_Pa=np.nan, tau_Pa=2.8e6,  G_Pa=np.nan),
        dict(id=7, label="MoNI (1L graphene)", a_m=np.nan, p_Pa=np.nan, tau_Pa=23.3e6, G_Pa=np.nan),
    ]

    # For plotting: choose representative x for anchors on each panel.
    # Panel (a) needs pressure p̄: if absent, place them at a representative location just to show regime.
    # Panel (b) needs a and tau/G*: if G* absent, just plot tau/G using a nominal G=30 GPa (schematic).
    G_nom = 30e9

    # --- figure layout ---
    fig = plt.figure(figsize=(6.6, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.12)
    axP = fig.add_subplot(gs[0, 0])
    axA = fig.add_subplot(gs[1, 0])

    # Panel labels
    axP.text(0.01, 0.98, "(a)", transform=axP.transAxes, ha="left", va="top", fontsize=11)
    axA.text(0.01, 0.98, "(b)", transform=axA.transAxes, ha="left", va="top", fontsize=11)

    # --- (a) tau_max vs pressure ---
    axP.set_xscale("log")
    axP.set_yscale("log")
    axP.plot(p, tau_p, label=r"$\tau=\tau_0+k\bar{p}^m$")

    # Yield / plasticity band (schematic): around 0.5–10 GPa (adjust as you like)
    axP.axvspan(5e8, 1e10, alpha=0.10, zorder=0)

    # Plot anchors on pressure panel:
    # Put ones with known p where you later fill; for now place them cleanly without text overlap:
    xP, yP, idsP = [], [], []
    for d in anchors:
        # only show "pressure-type" anchors here (Pharr 2025 + Tosatti σ is not pressure; but you may still show it)
        # We'll show Pharr 2025 τc point at 2 GPa nominal just as a marker.
        if "Pharr 2025" in d["label"]:
            xP.append(2e9)
            yP.append(d["tau_Pa"])
            idsP.append(d["id"])
        if "Brazil&Pharr" in d["label"]:
            xP.append(8e8)   # nominal placement
            yP.append(d["tau_Pa"])
            idsP.append(d["id"])
    if xP:
        plot_numbered_points(axP, np.array(xP), np.array(yP), idsP)

    axP.set_xlim(1e6, 1e10)
    axP.set_ylim(1e6, 5e9)
    axP.set_ylabel(r"shear strength $\tau_{\max}$ (Pa)")

    # --- (b) tau/G* vs a/b ---
    axA.set_xscale("log")
    axA.set_yscale("log")

    x_ab = a / b_m
    axA.plot(x_ab, tau_over_G, label=r"$\tau/G^* \propto (a/b)^{-1/2}$")
    axA.plot(x_ab, tau_over_G_rep, linestyle="--", label=r"$\tau/G^* \propto a^{-1/3}$")
    axA.plot(x_ab, tau_over_G_adh_inc, linestyle=":", label=r"$\tau/G^* \propto a^{-3/2}$")

    # Add coherent upper bound + large-contact plateau (explicit)
    axA.hlines(tau_coherent_over_G, x_ab.min(), x_ab.max(), linestyle=":", linewidth=1.8,
               label="coherent/atomic upper bound (schematic)")
    axA.hlines(tau_large_over_G, x_ab.min(), x_ab.max(), linestyle="--", linewidth=1.8,
               label="large-contact plateau (schematic)")

    # Shade "this work" size window (edit to your actual a-range)
    axA.axvspan((50e-9)/b_m, (2000e-9)/b_m, alpha=0.10, zorder=0)

    # Plot anchors on size panel (those with a_m)
    xA, yA, idsA = [], [], []
    for d in anchors:
        if np.isfinite(d["a_m"]):
            G_eff = d["G_Pa"] if np.isfinite(d["G_Pa"]) else G_nom
            xA.append(d["a_m"] / b_m)
            yA.append(d["tau_Pa"] / G_eff)
            idsA.append(d["id"])
    if xA:
        plot_numbered_points(axA, np.array(xA), np.array(yA), idsA)

    axA.set_xlim(1, 2e5)
    axA.set_ylim(1e-5, 2e-1)
    axA.set_xlabel(r"normalized contact size $a/b$  (schematic; $b=0.5$ nm)")
    axA.set_ylabel(r"effective shear strength $\tau/G^*$")

    # --- legends: keep only curve legend + separate anchor box to avoid collisions ---
    leg1 = axP.legend(loc="lower right", frameon=True, framealpha=1.0)
    leg2 = axA.legend(loc="lower left", frameon=True, framealpha=1.0)

    # Anchor list box (single place; no overlapping point labels)
    anchor_box = make_anchor_box(anchors)
    axA.text(0.985, 0.02, anchor_box, transform=axA.transAxes,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7"))

    # Axis cosmetics: ticks on all sides, no heavy grid
    for ax in (axP, axA):
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(False)

    fig.tight_layout(pad=0.6)

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
