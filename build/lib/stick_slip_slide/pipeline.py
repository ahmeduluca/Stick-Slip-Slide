from __future__ import annotations

import argparse
import copy
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .cycle_types import CycleBounds

# === IO ===
from .io import (
    read_csv_with_units,
    extract_markers,
)

# === Math/Signal utilities ===
from .math_utils import (
    _num,
    robust_median,
    summarize_dist,
    window_idx_fw,
    tau_from_samples,
    tau_nominal_with_sigma,
    ci95_to_sigma,
    sigma_to_ci95,
    add_harmonics_from_raw_x,
    add_energy_totals_and_xraw_harmonics,
    robust_mad,
    inflate_tau_uncertainty_with_force,
    EPS_A
)

# === Signal detection ===
from .signal_processing import (
    detect_touch_index,
    manual_pick_touch,
    recompute_touch_meta_from_index,
    detect_cycles,
    manual_pick_shear_window,
    manual_pick_cycles,
    detect_stick_slide_transitions,
    find_calibration_slices_pre_touch,
    find_shear_window_from_normal_load_v2,
    approve_or_repick_gate,
)

# === Mechanics/physics ===
from .mechanics import (
    corrected_normal_load,
    correct_contact_force,
    vertical_stiffness_frame_corrected,
    contact_depth_h_m,
    area_pi_h_R,
    normal_pressure_Pa,
    compute_lateral_corrected,
    effective_modulus,
    total_sliding_cyc_dist_speed,
    fit_actuator_mck,
    a_from_stiffness_Sneddon,
    a_from_depth_sphere,
    estimate_R_from_a_and_h,
    estimate_R_from_a_and_P,
    A_from_a,
    sigma_P_contact,
    sigma_h_contact,
    sigma_area_piRh,
    compute_area_from_choice,
    prop_div,
    _finite
)

# === Fitting ===
from .fitting import (hertz_fit_radius_adhesion,
    fit_vertical_dynamic_support_spring,
    fit_vertical_dynamic_coupling,
    fit_support_spring_pre_touch,
    fit_flat_end_stiffness,
    filter_kwargs_for_callable,
    bootstrap_hertz_radius_uncertainty,
    bootstrap_flat_end_stiffness_uncertainty,
)

# === Plotting/UI ===
from .plotting import (
    sanity_plot_window_cycles,
    show_and_wait,
    plot_normal_loading_depth_stiffness,
    plot_contact_radius_sanity,
    plot_flat_end_fit,
    plot_hertz_diagnostic,
    choose_area_mode_gate,
    plot_mindlin_fit,
    make_folder_summary_plots,
    plot_check_friction_and_transitions,
)

# === Reporting ===
from .reporting import (
    build_Aref_samples,
    build_wide_summary_dynamic,
    summarize_cycle,
)


def analyze_one_file(fp: Path, cfg, live_plots: bool, outdir: Optional[Path]) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean, single-pass analyze_one_file with:
      - correct ordering (no undefined vars)
      - optional approval + manual repick gate
      - no overwriting of manually-picked indices
      - consistent reference window for Sz_ref, h_ref, A_ref
      - safe handling of missing transitions / sliding metrics
      - fixed tau unit conversion
      - Hertz diagnostic on the loading segment (touch -> end-of-loading)

    Assumptions:
      - cfg.manual_mode in {"never","on_fail","always"}
      - manual_pick_touch(df, cfg) -> int
      - manual_pick_shear_window(t, P_contact_N, F2_rms) -> (i0:int, i1:int)
      - manual_pick_cycles(df2, cfg, i0, i1) -> List[CycleBounds]
      - approve_or_repick_gate(fig_title) -> "accept"|"repick" (raises on pass/abort)
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
    print(live_plots)
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
    touch_meta = {}
    auto_ok_touch = True
    err_touch = None
    try:
        touch_i, touch_meta = detect_touch_index(df, cfg, scale, markers)

        F0 = touch_meta["F0_mN"]
        z0 = touch_meta["z0_nm"]

        sigma_F_point = touch_meta["sigma_F_point_used_mN"]   # 10 nN
        sigma_F0 = touch_meta["sigma_F0_mN"]
        sigma_P = np.sqrt(sigma_F_point**2 + sigma_F0**2)

        sigma_z_point = touch_meta["sigma_z_point_used_nm"]
        sigma_z0 = touch_meta["sigma_z0_nm"]
        sigma_h = np.sqrt(sigma_z_point**2 + sigma_z0**2)    
    except Exception as e:
        auto_ok_touch = False
        err_touch = str(e)

    # If manual always: we still want a first guess for plotting, but we can repick later.
    # If auto failed and manual disabled -> error later.

    # -----------------------------
    # Helper: compute normal signals given touch_i
    # -----------------------------
    def _compute_normal_signals(_touch_i: int):
        k_sup_z, b_sup_z, _, _, meta_vfit = fit_support_spring_pre_touch(z_raw_m, Fz_raw_N, _touch_i, cfg)
        print(meta_vfit)
        P_contact_N = corrected_normal_load(Fz_raw_N, z_raw_m, k_sup_z, b_sup_z)
        h_m = contact_depth_h_m(z_raw_m, touch_i, P_contact_N, cfg.k_frame_z)
        return k_sup_z, b_sup_z, P_contact_N, h_m

    # We cannot proceed without a touch index (unless user repicks)
    k_sup_z = b_sup_z = None
    P_contact_N = None
    h_m = None
    if touch_i is not None:
        k_sup_z, b_sup_z, P_contact_N, h_m = _compute_normal_signals(touch_i)

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
    if df2 is not None:
        df2 = add_harmonics_from_raw_x(df=df2, time_col=cfg.time_col, x_raw_col = cfg.x_raw_col, f1_hz = cfg.dyn_f2_freq_Hz)
        df2 = add_energy_totals_and_xraw_harmonics(df = df2, time_col=cfg.time_col, f1_hz = cfg.dyn_f2_freq_Hz, x_raw_col = cfg.x_raw_col)
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
            if not cycles or len(cycles) == 0:
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
    print(errors) if errors else print("Auto-analysis successful, no errors detected.")
    figs:List[plt.Figure] = []
    if need_approval:
        # Plot current best guess (may be partial)
        try:
            figs=sanity_plot_window_cycles(
                df2=df2 if df2 is not None else df,   # df2 may be None, plot what we have
                cfg=cfg,
                t=t,
                P_contact_N=P_contact_N if P_contact_N is not None else np.zeros_like(t),
                depth=h_m if h_m is not None else np.zeros_like(t),
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
                    df2=df2, cfg=cfg, t=t, P_contact_N=P_contact_N, depth=h_m if h_m is not None else np.zeros_like(t),
                    i0=win[0], i1=win[1], cycles=cycles, cal_sl=cal_sl_lat, title=fp.stem
                )
        decision = approve_or_repick_gate(figs, "approve (a)/ repick touch (t) / repick window (w) / repick cycles (c) / pass (p) / abort (esc)")
        figs = []
    else:
        decision = "approve"  # no manual mode
        if live_plots:
            figs = sanity_plot_window_cycles( df2=df2, cfg=cfg, t=t, P_contact_N=P_contact_N, depth=h_m if h_m is not None else np.zeros_like(t), 
                                      i0=win[0], i1=win[1], cycles=cycles, cal_sl=cal_sl_lat, title=fp.stem)
            figs = show_and_wait(f"{fp.stem} — auto analysis complete", figures=figs)
    print(decision)
    if decision == "repick_touch":
        touch_i = manual_pick_touch(df2, cfg, initial=touch_i)
        # recompute spring fit + P_contact_N + h/A etc as needed
        k_sup_z, b_sup_z, P_contact_N, h_m = _compute_normal_signals(touch_i)
        # recompute cal dynamics + df2
        df2, cal_sl_lat, cal_sl_vert, k_sup_z_dyn, b_sup_z_dyn, cpl = _compute_df2(df, cfg, markers, scale, touch_i)
        # RECOMPUTE
        touch_meta.update(recompute_touch_meta_from_index(
            t_s=t,
            F_mN=Fz_raw_N * 1e3,
            z_nm=z_raw_m * 1e9,
            touch_i=touch_i,
            daq_hz=getattr(cfg, "daq_hz", 500.0),
            offset_seconds=getattr(cfg, "touch_offset_seconds", 2.0),
            offset_margin_s=getattr(cfg, "touch_offset_margin_s", 0.5),
            sigma_F_point_mN=getattr(cfg, "sigma_Fz_N", 10e-9) * 1e3,  # 10 nN -> 0.01 mN
            sigma_z_point_nm=(getattr(cfg, "sigma_z_m", np.nan) * 1e9 if np.isfinite(getattr(cfg, "sigma_z_m", np.nan)) else None),
        ))
        # recompute shear window + cycles
        try:
            i0, i1 = find_shear_window_from_normal_load_v2(t, P_contact_N, touch_i,cfg)
            win = (int(i0), int(i1))
        except:
            i0, i1 = None, None
        i0, i1 = manual_pick_shear_window(t, P_contact_N, F2_rms=df2[cfg.F2_rms_col], initial=[i0,i1])
        win = (int(i0), int(i1))
        try:
            cycles = detect_cycles(df2, cfg, start_i=i0, end_i=i1)  # or manual_pick_cycles
        except Exception as e:
            cycles = []
        cycles = manual_pick_cycles(df2, cfg, i0, i1, initial=cycles, n_cycles=cfg.expected_cycles)

    elif decision == "repick_window":
        i0, i1 = manual_pick_shear_window(t, P_contact_N, F2_rms=df2[cfg.F2_rms_col], initial=[i0,i1])
        win = (int(i0), int(i1))
        try:
            cycles = detect_cycles(df2, cfg, start_i=i0, end_i=i1)  # or manual_pick_cycles
        except Exception as e:
            cycles = []
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
    except Exception as e:
        actuator_mck_vertical = {"m_eff": np.nan, "c_eff": np.nan, "k_eff": np.nan, "error": str(e)}
    try:
        actuator_mck_lateral = fit_actuator_mck(t[cal_sl_lat], df2["X2_pk_m"][cal_sl_lat], df2["F2_pk_N"][cal_sl_lat])
        F2_pk_cor_mck = correct_contact_force(t, df2["X2_pk_m"], df2["F2_pk_N"], float(actuator_mck_lateral.get("m_eff", np.nan)),
                                            actuator_mck_lateral.get("c_eff", np.nan), actuator_mck_lateral.get("k_eff", np.nan))
    except Exception as e:
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

    if live_plots and getattr(cfg, "final_approve_plot", False):
        figs = sanity_plot_window_cycles(
            df2=df2, cfg=cfg, t=t, P_contact_N=P_contact_N, depth=h_m if h_m is not None else np.zeros_like(t),
            i0=i0, i1=i1, cycles=cycles, cal_sl=cal_sl_lat, title=fp.stem
        )
        figs = show_and_wait(f"{fp.stem} — final indices", figs)

    # -----------------------------
    # 6) Reference window (end of normal loading plateau)
    # -----------------------------
    ref_i = window_idx_fw(t, i0, cfg.ref_window_s)  # forward/centered window
    if ref_i.size == 0:
        ref_i = np.array([i0], dtype=int)

    Sz_ref = None
    sigma_Sz_ref = None
    if Sz_arr is not None and ref_i.size:
        Sz_ref = float(robust_median(Sz_arr[ref_i])) if ref_i.size else float(Sz_arr[i0])
        sigma_Sz_ref = float(robust_mad(Sz_arr[ref_i])) if ref_i.size else float(robust_mad(Sz_arr[max(0,i0-50):i0+50]))
    elif Sz_arr is not None and i0 is not None:
        Sz_ref = float(Sz_arr[i0]) if np.isfinite(Sz_arr[i0]) else np.nan
        sigma_Sz_ref = float(robust_mad(Sz_arr[max(0, i0-50): i0+50])) if np.isfinite(Sz_arr[i0]) else np.nan

    h_ref = robust_median(h_m[ref_i]) if ref_i.size else np.nan
    A_ref_nom = robust_median(A_nominal[ref_i]) if ref_i.size else np.nan
    
    ## energy refs ~comparable with later cyclic
    try:
        X_ref_1st = robust_median(df2["X1st_pk"][ref_i]) if ref_i.size and df2["X1st_pk"].size else np.nan
        X_ref_2nd = robust_median(df2["X2nd_pk"][ref_i]) if ref_i.size and df2["X2nd_pk"].size else np.nan
        E_ref_total = robust_median(df2["E_diss_fn_total_J"][ref_i]) if ref_i.size and df2["E_diss_fn_total_J"].size else np.nan
        P_ref_total = robust_median(df2["P_diss_fn_W"][ref_i]) if ref_i.size and df2["P_diss_fn_W"].size else np.nan
    except:
        E_ref_total=np.nan
        P_ref_total=np.nan
        X_ref_1st=np.nan
        X_ref_2nd=np.nan

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
        a_csm = a_from_stiffness_Sneddon(Sz_arr, float(E_star))
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
            n_iter=int(getattr(cfg, "flat_iter", 50)),
            clip_sigma=float(getattr(cfg, "flat_clip_sigma", 3.0)),
            min_points=int(getattr(cfg, "flat_min_points", 30)),
        )
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
        hertz = hertz_fit_radius_adhesion(h_load, P_load, **filter_kwargs_for_callable(hertz_fit_radius_adhesion, hertz_kwargs))
    else:
        hertz = {"ok": 0, "reason": "loading segment empty/too short"}
    boot_hertz = {"ok": 0}
    # ---- Decide on reference area model (after Hertz diagnostic)
    area_mode_selected = getattr(cfg, "area_mode", "nominal")  # default from cfg; can be overridden by gate if live_plots
    if live_plots and load_sl is not None: # and getattr(cfg, "area_pick_enable", False):
    # default comes from cfg.area_mode
        figs.append(plot_flat_end_fit(P_contact_N[load_sl], Sz_arr[load_sl], flat_fit, E_star_Pa=E_star) if flat_fit.get("ok", 0) else None)
        figs.extend(plot_hertz_diagnostic(
                    h_m[load_sl], P_contact_N[load_sl], hertz,
                    title=fp.stem,
                    hardness_Pa=cfg.hardness_Pa,
                    plasticity_p0_frac=cfg.plasticity_p0_frac
                ) if hertz.get("ok", 0) else [])
        area_mode_selected = choose_area_mode_gate(figures=figs, default_mode=getattr(cfg, "area_mode", "nominal")) if figs else area_mode_selected
        # then pass a local cfg-like choice into compute_area_from_choice
        figs = []  # clear figs after decision
    
    ####Decide on reference area: which fit to be trusted:
    A_m2_used, area_mode_used = compute_area_from_choice(
        h_m, P_contact_N, area_mode_selected,
        cfg=cfg, E_star_Pa=E_star, hertz=hertz, flat_end=flat_fit,
        Sz_meas_N_per_m=Sz_arr,   # NEW
    )        

    # --- bootstrap fit ---
    if area_mode_used == "fit_hertz":        
        boot_hertz = bootstrap_hertz_radius_uncertainty(
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
    elif area_mode_used.startswith("flat"):
        boot_flat = bootstrap_flat_end_stiffness_uncertainty(
        P_contact_N[load_sl], Sz_arr[load_sl],
        fit_fn=fit_flat_end_stiffness,
        fit_kwargs=dict(
            E_star_Pa=E_star,
            P_min_N=getattr(cfg, "flat_Pmin_N", None),
            P_max_N=getattr(cfg, "flat_Pmax_N", None),
            robust=True,
            n_iter=int(getattr(cfg, "flat_iter", 50)),
            clip_sigma=float(getattr(cfg, "flat_clip_sigma", 3.0)),
            min_points=int(getattr(cfg, "flat_min_points", 30)),
        ),
        n_boot=int(getattr(cfg, "flat_boot_n", 400)),
        seed=int(getattr(cfg, "flat_boot_seed", 0)),
        keep_frac=float(getattr(cfg, "flat_boot_keep_frac", 1.0)),
        min_success=int(getattr(cfg, "flat_boot_min_success", 50)),
        block_size=int(getattr(cfg, "flat_boot_block", 10)),
    )
    df2["A_m2"] = A_m2_used
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
    sigma_k_frame = cfg.sigma_k_frame_z
    sigma_R_m = cfg.sigma_tip_radius_m
    # Prefer touch-derived channel noise/offsets if available
    sigma_F_N = float(touch_meta.get("sigma_F_point_used_mN", 1e3 * cfg.sigma_Fz_N)) * 1e-3
    sigma_z_m = float(touch_meta.get("sigma_z_point_used_nm", 1e9 * cfg.sigma_z_m)) * 1e-9

    z0_m = 1e-9 * float(touch_meta.get("z0_nm", np.nan))
    sigma_z0_m = 1e-9 * float(touch_meta.get("sigma_z0_nm", 0.0))

    k_sup_z2, b_sup_z2, sigma_k_sup, sigma_b_sup, meta_vfit = fit_support_spring_pre_touch(z_raw_m, Fz_raw_N, touch_i, cfg)

    sigma_P_N = sigma_P_contact(
        Fz_raw_N, z_raw_m, k_sup_z2, b_sup_z2,
        sigma_F_N=sigma_F_N,
        sigma_z_m=sigma_z_m,
        sigma_k_sup=sigma_k_sup,
        sigma_b_sup=sigma_b_sup,
        relative_to_touch=False,   # because P_contact_N comes from corrected_normal_load
    )

    sigma_h_m = sigma_h_contact(
        z_raw_m, touch_i, P_contact_N, sigma_P_N,
        k_frame_z=cfg.k_frame_z,
        sigma_z_m=sigma_z_m,
        sigma_k_frame_z= sigma_k_frame,
        z0_m=z0_m,
        sigma_z0_m=sigma_z0_m,
    )

    if area_mode_used.lower().startswith("nominal"):
        sigma_A_m2 = sigma_area_piRh(h_m, sigma_h_m, cfg.tip_radius_m, sigma_R_m)
        sigma_A_ref_nominal = float(robust_median(sigma_A_m2[ref_i])) if ref_i.size else float(sigma_A_m2[i0])
        A_ref_nominal = float(robust_median(A_nominal[ref_i])) if ref_i.size else float(A_nominal[i0])

    # sample A_ref in a way that matches the chosen area model
    mode = (area_mode_used or "").lower()

    # Defaults passed into build_Aref_samples
    sigma_A_use = None

    if mode.startswith("nominal_depth") or mode == "nominal":
        # existing analytic depth-based sigma
        sigma_A_use = sigma_A_ref_nominal

    A_ref_samples, diag = build_Aref_samples(
        area_mode_used=area_mode_used,
        A_ref=float(A_ref),
        h_ref=float(h_ref),
        P_ref=float(P_ref),
        E_star_Pa=float(E_star),
        cfg=cfg,
        hertz=hertz,
        boot_hertz=boot_hertz,   # will often be None now
        boot_flat=boot_flat,
        sigma_A_ref=sigma_A_use,  # only for nominal_depth
        n_fallback=int(getattr(cfg, "ref_unc_n", 2000)),
        seed=int(getattr(cfg, "ref_unc_seed", 0)),
        # NEW optional args:
        Sz_ref=Sz_ref,
        sigma_Sz_ref=sigma_Sz_ref,
    )

    A_stats = summarize_dist(A_ref_samples)
    p_stats = summarize_dist(P_ref / A_ref_samples)

    # final “reported” reference values
    A_ref = float(A_stats["median"])
    sigma_A_ref = ci95_to_sigma(A_stats["mean"], A_stats["ci95"][0], A_stats["ci95"][1])
    p_report_GPa = float(p_stats["median"] / 1e9)
    pressure_ci95_lo_GPa = float(p_stats["ci95"][0] / 1e9)
    pressure_ci95_hi_GPa = float(p_stats["ci95"][1] / 1e9)

    # useful to store for later propagation
    df2["pressure_ref_GPa"] = p_report_GPa

    for k, v in diag.items():
        df2[f"aref_diag_{k}"] = v

    # -----------------------------
    # 9) Per-cycle report
    # -----------------------------
    rows: List[Dict] = []
    area_cycles: List[float] = [A_ref]  # area before cycle 1 (reference)
    tr={}
    Sz_ref0 = Sz_ref
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
                cfg=cfg
            )
            trans = tr
            print({
                "mode": trans["trans_mode_used"],
                "i_ss": trans["i_ss"],
                "i_rs": trans["i_rs"],
                "i_ss_stiff": trans["i_ss_stiff"],
                "i_rs_stiff": trans["i_rs_stiff"],
                "i_ss_phase": trans["i_ss_phase"],
                "i_rs_phase": trans["i_rs_phase"],
                "phi_ss_peak_rad": trans["phi_ss_peak_rad"],
                "phi_rs_peak_rad": trans["phi_rs_peak_rad"],
            })
        except Exception as e:
            tr = {}
            tb = traceback.format_exc()
            print(tb)

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
            Sz_ref=Sz_ref0,
            outdir=outdir
        )
        try:
            # ---- robust divide-by-zero / tiny-area guards ----
            try:
                # --- derive cycle areas ---
                A_ratio = row.get("A_ratio_to_ref", np.nan)
                if np.isfinite(A_ratio) and np.isfinite(A_ref) and (A_ref > EPS_A):
                    A_grown = float(A_ratio * A_ref)
                else:
                    A_grown = np.nan

                if np.isfinite(A_grown) and (A_grown > EPS_A):
                    area_cycles.append(A_grown)
                else:
                    area_cycles.append(area_cycles[-1])

                A_prev = float(area_cycles[b.cycle - 1])
                A_now  = float(area_cycles[b.cycle])

                # --- build area samples for the cycle (scaled from reference samples) ---
                if np.isfinite(A_ref) and (A_ref > EPS_A) and np.isfinite(A_prev) and (A_prev > EPS_A):
                    A_prev_samples = A_ref_samples * (A_prev / A_ref)
                else:
                    A_prev_samples = np.full_like(A_ref_samples, np.nan)

                if np.isfinite(A_ref) and (A_ref > EPS_A) and np.isfinite(A_now) and (A_now > EPS_A):
                    A_now_samples = A_ref_samples * (A_now / A_ref)
                else:
                    A_now_samples = np.full_like(A_ref_samples, np.nan)

                # --- tau at stick->slip and re-stick, with CI from area uncertainty ---
                Ft_ss_mN = row.get("Ft_ss_mN", np.nan)
                Ft_rs_mN = row.get("Ft_rs_mN", np.nan)
                sigma_Fss = row.get("sigma_Ft_ss_uN", 0) * 1e-3
                sigma_Frs =  row.get("sigma_Ft_rs_uN", 0) * 1e-3
                sigma_mu_ss = prop_div(Ft_ss_mN, sigma_Fss, load_max_mN, np.nanmedian(sigma_P_N[ref_i]))
                sigma_mu_rs = prop_div(Ft_rs_mN, sigma_Frs, load_max_mN, np.nanmedian(sigma_P_N[ref_i]))
                row["sigma_mu_ss"] = sigma_mu_ss
                row["sigma_mu_rs"] = sigma_mu_rs

                tau_ss = tau_from_samples(Ft_ss_mN, A_prev_samples, eps=EPS_A)
                if tau_ss.size > 0:
                    st = summarize_dist(tau_ss)
                    tau_med_Pa = st["median"]
                    ci_lo_Pa, ci_hi_Pa = st["ci95"][0], st["ci95"][1]

                    sigma_Pa, (ci_lo2_Pa, ci_hi2_Pa) = inflate_tau_uncertainty_with_force(
                        tau_med_Pa=tau_med_Pa,
                        ci95_lo_Pa=ci_lo_Pa,
                        ci95_hi_Pa=ci_hi_Pa,
                        Ft_mN=Ft_ss_mN,
                        sigma_Ft_mN=sigma_Fss,
                        ci95_to_sigma=ci95_to_sigma,
                        sigma_to_ci95=sigma_to_ci95,
                    )

                    row["tau_ss_MPa"] = tau_med_Pa / 1e6
                    row["tau_ss_sigma_MPa"] = sigma_Pa / 1e6
                    row["tau_ss_ci95_lo_MPa"] = ci_lo2_Pa / 1e6
                    row["tau_ss_ci95_hi_MPa"] = ci_hi2_Pa / 1e6

                else:
                    tau_nom, sigma_tau = tau_nominal_with_sigma(Ft_ss_mN,A_ref_nominal, sigma_A_m2=sigma_A_ref_nominal, sigma_Ft_mN = sigma_Fss)
                    ci_lo, ci_hi = sigma_to_ci95(tau_nom, sigma_tau)
                    row["tau_ss_MPa"] = tau_nom
                    row["tau_ss_sigma_MPa"] = sigma_tau
                    row["tau_ss_ci95_lo_MPa"] = ci_lo
                    row["tau_ss_ci95_hi_MPa"] = ci_hi

                tau_rs = tau_from_samples(Ft_rs_mN, A_now_samples, eps=EPS_A)
                if tau_rs.size > 0:
                    st = summarize_dist(tau_rs)
                    tau_med_Pa = st["median"]
                    ci_lo_Pa, ci_hi_Pa = st["ci95"][0], st["ci95"][1]

                    sigma_Pa, (ci_lo2_Pa, ci_hi2_Pa) = inflate_tau_uncertainty_with_force(
                        tau_med_Pa=tau_med_Pa,
                        ci95_lo_Pa=ci_lo_Pa,
                        ci95_hi_Pa=ci_hi_Pa,
                        Ft_mN=Ft_rs_mN,
                        sigma_Ft_mN=sigma_Frs,
                        ci95_to_sigma=ci95_to_sigma,
                        sigma_to_ci95=sigma_to_ci95,
                    )
                    row["tau_rs_MPa"] = tau_med_Pa / 1e6
                    row["tau_rs_sigma_MPa"] = sigma_Pa / 1e6
                    row["tau_rs_ci95_lo_MPa"] = ci_lo2_Pa / 1e6
                    row["tau_rs_ci95_hi_MPa"] = ci_hi2_Pa / 1e6
                else:
                    tau_nom, sigma_tau = tau_nominal_with_sigma(Ft_rs_mN,A_ref_nominal,
                                                                 sigma_A_m2=sigma_A_ref_nominal, sigma_Ft_mN = sigma_Frs)
                    ci_lo, ci_hi = sigma_to_ci95(tau_nom, sigma_tau)
                    row["tau_rs_MPa"] = tau_nom
                    row["tau_rs_sigma_MPa"] = sigma_tau
                    row["tau_rs_ci95_lo_MPa"] = ci_lo
                    row["tau_rs_ci95_hi_MPa"] = ci_hi
            except Exception:
                row["tau_ss_MPa"] = np.nan
                row["tau_ss_ci95_lo_MPa"] = np.nan
                row["tau_ss_ci95_hi_MPa"] = np.nan
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
        Sz_ref0 = row.get("Sz_end_cyc", Sz_ref)
        print(Sz_ref)
        print(Sz_ref0)
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
    report["Sz_ref_N_per_m"] = float(Sz_ref) if np.isfinite(Sz_ref) else np.nan
    report["initial_h_nm"] = float(h_ref * 1e9) if np.isfinite(h_ref) else np.nan
    report["A_ref_um2"] = float(A_ref * 1e12) if np.isfinite(A_ref) else np.nan
    report["sigma_A_ref"] = float(sigma_A_ref * 1e12) if np.isfinite(sigma_A_ref) else np.nan
    report["area_mode_used"] = str(area_mode_used)
    try:
        report["initial_hold_time"] = t[cycles[0].i_start] - t[i0] if cycles is not None and i0 is not None else 0
    except:
        report["initial_hold_time"] = 0.

    report["load_max_mN"] = float(load_max_mN) if np.isfinite(load_max_mN) else np.nan
    report["pressure_ref_GPa"] = float(p_report_GPa) if np.isfinite(p_report_GPa) else np.nan
    report["pressure_ref_ci95_lo_GPa"] = float(pressure_ci95_lo_GPa) if np.isfinite(pressure_ci95_lo_GPa) else np.nan
    report["pressure_ref_ci95_hi_GPa"] = float(pressure_ci95_hi_GPa) if np.isfinite(pressure_ci95_hi_GPa) else np.nan

    report["R_from_CSM_a_h_um"] = float(R_csm_a_h * 1e6) if np.isfinite(R_csm_a_h) else np.nan
    report["R_from_CSM_a_P_um"] = float(R_csm_a_P * 1e6) if np.isfinite(R_csm_a_P) else np.nan
    try:
        report["A_ref_csm_um2"] = (float(np.nanmedian(A_csm[ref_i])) * 1e12) if (A_csm is not None and ref_i.size) else np.nan
        report["a_ref_csm_um"] = (float(np.nanmedian(a_csm[ref_i])) * 1e6) if (a_csm is not None and ref_i.size) else np.nan
    except:
        report["A_ref_csm_um2"] = np.nan
        report["a_ref_csm_um"] =  np.nan
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
    # ---- Hertz bootstrap outputs (from bootstrap_hertz_radius_uncertainty)
    report["hertz_boot_ok"] = int(boot_hertz.get("ok", 0)) if isinstance(boot_hertz, dict) else 0
    report["hertz_boot_n_ok"] = int(boot_hertz.get("n_boot_ok", 0)) if isinstance(boot_hertz, dict) else 0
    report["hertz_boot_n_boot"] = int(boot_hertz.get("n_boot", 0)) if isinstance(boot_hertz, dict) else 0
    report["hertz_boot_keep_frac"] = float(boot_hertz.get("keep_frac", np.nan)) if (isinstance(boot_hertz, dict) and np.isfinite(boot_hertz.get("keep_frac", np.nan))) else np.nan
    report["hertz_boot_block_size"] = int(boot_hertz.get("block_size", 0)) if isinstance(boot_hertz, dict) and boot_hertz.get("block_size", None) is not None else 0

    if isinstance(boot_hertz, dict) and int(boot_hertz.get("ok", 0)) == 1:
        report["R_eff_std_um"] = float(boot_hertz.get("R_eff_std_m", np.nan) * 1e6) if np.isfinite(boot_hertz.get("R_eff_std_m", np.nan)) else np.nan
        report["R_eff_ci95_lo_um"] = float(boot_hertz.get("R_eff_ci95_lo_m", np.nan) * 1e6) if np.isfinite(boot_hertz.get("R_eff_ci95_lo_m", np.nan)) else np.nan
        report["R_eff_ci95_hi_um"] = float(boot_hertz.get("R_eff_ci95_hi_m", np.nan) * 1e6) if np.isfinite(boot_hertz.get("R_eff_ci95_hi_m", np.nan)) else np.nan

        # model-switch checkpoint (very useful)
        report["adhesion_model_used_mode"] = str(boot_hertz.get("adhesion_model_used_mode", ""))
        report["adhesion_model_used_frac"] = float(boot_hertz.get("adhesion_model_used_frac", np.nan)) if np.isfinite(boot_hertz.get("adhesion_model_used_frac", np.nan)) else np.nan

        # (optional) how many valid paired samples were actually stored
        s = boot_hertz.get("samples", {}) or {}
        report["hertz_boot_samples_n"] = int(np.size(s.get("R_eff_m", [])))
    else:
        report["R_eff_std_um"] = np.nan
        report["R_eff_ci95_lo_um"] = np.nan
        report["R_eff_ci95_hi_um"] = np.nan
        report["adhesion_model_used_mode"] = ""
        report["adhesion_model_used_frac"] = np.nan
        report["hertz_boot_samples_n"] = 0

    if isinstance(boot_hertz, dict) and int(boot_hertz.get("ok", 0)) == 1:
        report["R_eff_std_um"] = float(boot_hertz.get("R_eff_std_m", np.nan) * 1e6) if np.isfinite(boot_hertz.get("R_eff_std_m", np.nan)) else np.nan
        report["R_eff_ci95_lo_um"] = float(boot_hertz.get("R_eff_ci95_lo_m", np.nan) * 1e6) if np.isfinite(boot_hertz.get("R_eff_ci95_lo_m", np.nan)) else np.nan
        report["R_eff_ci95_hi_um"] = float(boot_hertz.get("R_eff_ci95_hi_m", np.nan) * 1e6) if np.isfinite(boot_hertz.get("R_eff_ci95_hi_m", np.nan)) else np.nan
    else:
        report["R_eff_std_um"] = np.nan
        report["R_eff_ci95_lo_um"] = np.nan
        report["R_eff_ci95_hi_um"] = np.nan
    report["hertz_boot_ok"] = int(boot_hertz.get("ok", 0)) if isinstance(boot_hertz, dict) else 0
    report["hertz_boot_n_ok"] = int(boot_hertz.get("n_boot_ok", 0)) if isinstance(boot_hertz, dict) else 0
    report["hertz_boot_n_boot"] = int(boot_hertz.get("n_boot", 0)) if isinstance(boot_hertz, dict) else 0


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
    report["flat_end_boot_n_success"] = int(boot_flat.get("n_boot_ok", 0)) if isinstance(boot_flat, dict) else 0
    report["flat_end_boot_keep_frac"] = float(boot_flat.get("keep_frac", np.nan)) if (isinstance(boot_flat, dict) and np.isfinite(boot_flat.get("keep_frac", np.nan))) else np.nan
    report["flat_end_boot_n_boot"] = int(boot_flat.get("n_boot", 0)) if isinstance(boot_flat, dict) else 0

    # CI summaries (only if ok)
    if isinstance(boot_flat, dict) and int(boot_flat.get("ok", 0)) == 1:
        a_sum = boot_flat.get("a_flat_um", {})      # summary dict
        R_sum = boot_flat.get("R_eff_um", {})       # summary dict
        C_sum = boot_flat.get("C", {})              # summary dict
        S0_sum = boot_flat.get("S0_N_per_m", {})    # summary dict

        report["flat_end_a_flat_med_um"] = float(a_sum.get("median", np.nan))
        report["flat_end_a_flat_std_um"] = float(a_sum.get("std", np.nan))
        report["flat_end_a_flat_ci95_lo_um"] = float(a_sum.get("ci95_lo", np.nan))
        report["flat_end_a_flat_ci95_hi_um"] = float(a_sum.get("ci95_hi", np.nan))

        report["flat_end_R_eff_med_um"] = float(R_sum.get("median", np.nan))
        report["flat_end_R_eff_std_um"] = float(R_sum.get("std", np.nan))
        report["flat_end_R_eff_ci95_lo_um"] = float(R_sum.get("ci95_lo", np.nan))
        report["flat_end_R_eff_ci95_hi_um"] = float(R_sum.get("ci95_hi", np.nan))

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

    report["E_ref_tot"] = E_ref_total
    report["P_ref_tot"] = P_ref_total
    report["X_ref_1st"] = X_ref_1st
    report["X_ref_2nd"] = X_ref_2nd

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
        "Sz_ref_N_per_m": float(Sz_ref) if np.isfinite(Sz_ref) else np.nan,
        "initial_h_nm": float(h_ref * 1e9) if np.isfinite(h_ref) else np.nan,
        "sigma_depth_nm": np.nanmedian(sigma_h_m[ref_i]) * 1e9 if sigma_h_m.size else np.nan,
        "A_ref_um2": float(A_ref * 1e12) if np.isfinite(A_ref) else np.nan,
        "area_mode_used": str(area_mode_used),
        "load_max_mN": float(load_max_mN) if np.isfinite(load_max_mN) else np.nan,
        "sigma_load_mN": np.nanmedian(sigma_P_N[ref_i]) * 1e3 if sigma_P_N.size else np.nan,
        "pressure_ref_GPa": float(p_report_GPa) if np.isfinite(p_report_GPa) else np.nan,
        "pressure_ref_ci95_lo_GPa": float(pressure_ci95_lo_GPa) if np.isfinite(pressure_ci95_lo_GPa) else np.nan,
        "pressure_ref_ci95_hi_GPa": float(pressure_ci95_hi_GPa) if np.isfinite(pressure_ci95_hi_GPa) else np.nan,

        # support / calibration meta (handy debug)
        "k_sup_z_N_per_m": float(k_sup_z),
        "b_sup_z_N": float(b_sup_z),
        "k_sup_x_N_per_m": float(df2["kx_sup_est_N_per_m"].iloc[0]) if "kx_sup_est_N_per_m" in df2.columns else np.nan,
        "c_sup_x_Ns_per_m": float(df2["cx_sup_est_Ns_per_m"].iloc[0]) if "cx_sup_est_Ns_per_m" in df2.columns else np.nan,
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
        **touch_meta,
        **meta_vfit,
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
    cfg,
    pattern: str,
    live_plots: bool,
    plot_every: int,
    summary_plots: bool,
    origin_csv: bool,
    summary_template: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    print(type(pattern), pattern)
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
            print(tb)
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
        #spring calibrations
        "k_sup_z_N_per_m", "b_sup_z_Ns_per_m", "k_sup_x_N_per_m", "b_sup_x_Ns_per_m",
        # dynamics / coupling
        "k_sup_z_dyn_N_per_m","b_sup_z_dyn_N", "kzx_dyn_N_per_m",
        # touch index meta data output
        "method", "dt_s","nmin","win","sigma_F_point_known_mN","daq_hz","F0_mN","F0_n_used","sigma_F_point_est_mN",
        "sigma_F_point_used_mN","sigma_F0_mN","z0_nm","z0_n_used","sigma_z_point_est_nm","sigma_z_point_used_nm",
        "sigma_z0_nm","F0_mN","sigma_F0_mN","z0_nm","sigma_z0_nm","touch_idx_mc_ok","touch_idx_ci95_lo",
        "touch_idx_ci95_hi","touch_idx_sigma","sigma_F_point_used_mN","sigma_z_point_used_nm", "touch_mc_seed",
        # initial-max- depth, load and area calculation
        "initial_h_nm", "sigma_depth_nm", "load_max_mN", "sigma_load_mN", "area_mode_used",
        "Sz_ref_N_per_m", "A_ref_um2", 
        # pressure
        "pressure_ref_GPa", "pressure_ref_ci95_lo_GPa", "pressure_ref_ci95_hi_GPa",
        #area mode diagnostics..
        # Hertz
        "hertz_ok", "R_eff_um", "R_eff_std_um", "R_eff_ci95_lo_um", "R_eff_ci95_hi_um",
        "hertz_rmse_mN", "hertz_n_used",
        "adhesion_model", "w_eff_J_per_m2", "tabor_mu", "Fadh_N",
        # Flat-end (optional)
        "flat_end_boot_ok", "flat_end_a_flat_med_um", "flat_end_R_eff_med_um",
        # Diagnostics by direct stiffness channel and measured-depth or applied-load
        "R_from_CSM_a_h_um", "R_from_CSM_a_P_um",
    ] if c in summaries_df.columns]

    summaries_df_short = summaries_df.copy()
    if summary_cols:
        summaries_df_short = summaries_df_short[summary_cols].copy()

    summaries_df_short.to_csv(outdir / "summary_short.csv", index=False)

    if all_cycles_df is None or all_cycles_df.empty:
        print("[sss] No cycles collected across files; skipping wide summary.")
    else:
        if "cycle" in all_cycles_df.columns:
            print("[sss] cycles collected:", all_cycles_df["cycle"].dropna().astype(int).max() if all_cycles_df["cycle"].notna().any() else "none")

    # (2) detailed dynamic-wide report (one row per file)
    wide_dyn = build_wide_summary_dynamic(
        all_cycles_df=all_cycles_df,
        summaries_df=summaries_df,
        max_cycles=None,   # dynamic: uses max in data
    )
    wide_dyn.to_csv(outdir / "report_detailed_wide.csv", index=False)

    if summary_plots:
        figs = make_folder_summary_plots(all_cycles_df, outdir)
        figs = show_and_wait(fig_title="folder summaries", figures=figs)

    return wide_dyn, summaries_df_short

# ============================================================