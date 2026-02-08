from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from .config import Config
from .math_utils import robust_median, _num
from .signal_processing import window_idx_fw
from .cycle_types import CycleBounds
from .mechanics import (normal_pressure_Pa, shear_stress_Pa, estimate_lockin_lag_force_sigma,
                          sigma_shear_strength, area_pi_h_R)
from .fitting import mindlin_fit

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

    # --- guard: nothing to widen ---
    if all_cycles_df.empty or ("cycle" not in all_cycles_df.columns):
        # return something sensible: just the per-file summary if present
        return summaries_df.copy() if summaries_df is not None else pd.DataFrame()
    
    # ensure numeric cycle + drop NaNs
    df = all_cycles_df.copy()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df = df.dropna(subset=["cycle"])
    if df.empty:
        return summaries_df.copy() if summaries_df is not None else pd.DataFrame()

    df["cycle"] = df["cycle"].astype(int)

    # dynamic max cycles
    if max_cycles is None:
        mc = df["cycle"].max()
        if not np.isfinite(mc) or mc <= 0:
            return summaries_df.copy() if summaries_df is not None else pd.DataFrame()
        max_cycles = int(mc)

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
        ## mindlin ramp-down
        "mindlin_a_rd_N_per_m": float(mind_rd.get("a", np.nan)),
        "mindlin_t_rd_N": float(mind_rd.get("t", np.nan)),
        "mindlin_rmse_rd": float(mind_rd.get("rmse", np.nan)),
        "mindlin_n_rd": int(mind_rd.get("n", 0)),
        "mindlin_ok_rd": int(mind_rd.get("ok", 0)),
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
                or from R_eff_m samples if you want “Hertz-equivalent” path.

    Important: We treat h_ref and P_ref as fixed scalars here.
               You can extend later to sample h_ref too (but keep it stable first).
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
