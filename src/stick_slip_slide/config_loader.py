from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict
import yaml
import numpy as np
import argparse

from .config import Config  # wherever your Config dataclass lives


# ---- unit conversion helpers ----
def _maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _to_none_if_nan(x: Any) -> float | None:
    v = _maybe_float(x)
    if v is None:
        return None
    return None if (not np.isfinite(v)) else float(v)

def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def load_config_yaml(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # allow nesting but also allow flat keys
    core = raw.get("core", {})
    touch = raw.get("touch", {})
    cycles = raw.get("cycles", {})
    trans = raw.get("transition", {})
    hertz = raw.get("hertz", {})
    mats = raw.get("materials", {})
    adhesion = raw.get("adhesion", {})
    paths = raw.get("paths", {})

    # start from defaults
    cfg = Config()

    # --- map YAML to Config fields (only what you want configurable) ---
    updates: dict[str, Any] = {}

    # paths (not in dataclass by default; only include if you added them)
    # updates["batch_folder"] = paths.get("batch", None)

    # core physical
    if "tip_radius_um" in core:
        updates["tip_radius_m"] = float(core["tip_radius_um"]) * 1e-6
        print("Loaded tip_radius_m, by YAML")

    # frame stiffness
    if "k_frame_z" in core:
        updates["k_frame_z"] = _to_none_if_nan(core["k_frame_z"])
    if "k_frame_x" in core:
        updates["k_frame_x"] = _to_none_if_nan(core["k_frame_x"])
    if "pattern" in core:
        updates["pattern"] = core["pattern"]
    if "batch" in core:
        updates["batch"] = core["batch"]

    # touch
    if "k_touch_min" in touch:
        updates["k_touch_min"] = float(touch["k_touch_min"])
    if "k_touch_min_duration_s" in touch:
        updates["k_touch_min_duration_s"] = float(touch["k_touch_min_duration_s"])

    # shear window / normal-load window
    if "loading_rate_threshold" in touch:
        updates["loading_rate_threshold"] = float(touch["loading_rate_threshold"])
    if "normal_load_sustain" in touch:
        updates["normal_load_sustain"] = float(touch["normal_load_sustain"])
    if "normal_load_smooth" in touch:
        updates["normal_load_smooth"] = int(touch["normal_load_smooth"])

    # cycles
    for k in [
        "smooth_n", "dynF2_baseline_q", "dynF2_active_delta", "dynF2_nearzero_delta",
        "hold_top_frac", "hold_min_s", "min_cycle_points",
        "dfdt_smooth_n", "dfdt_thr_frac", "dfdt_hold_frac", "min_ramp_s", "min_hold_s",
        "expected_cycles"
    ]:
        if k in cycles:
            updates[k] = cycles[k]

    # transition
    for k in [
        "trans_frac_up", "trans_frac_down",
        "sliding_lateral_stiffness_thresh", "resticking_lateral_stiffness_thresh",
        "trans_low_band", "trans_smooth_n"
    ]:
        if k in trans:
            updates[k] = trans[k]

    # hertz
    if "hertz_enable" in hertz:
        updates["hertz_enable"] = bool(hertz["hertz_enable"])
    for k in [
        "hardness_Pa", "plasticity_p0_frac",
        "hertz_min_h_m", "hertz_max_frac_of_Pmax", "hertz_min_points",
        "hertz_iter", "hertz_plot", "area_mode"
    ]:
        if k in hertz:
            v = hertz[k]
            if k in {"hardness_Pa","plasticity_p0_frac","hertz_min_h_m","hertz_max_frac_of_Pmax"}:
                updates[k] = float(v)
            elif k in {"hertz_min_points","hertz_iter"}:
                updates[k] = int(v)
            else:
                updates[k] = v

    # materials (accept GPa in YAML, convert to Pa)
    if "E1_GPa" in mats:
        updates["E1_Pa"] = float(mats["E1_GPa"]) * 1e9
    if "E2_GPa" in mats:
        updates["E2_Pa"] = float(mats["E2_GPa"]) * 1e9
    if "nu1" in mats:
        updates["nu1"] = float(mats["nu1"])
    if "nu2" in mats:
        updates["nu2"] = float(mats["nu2"])

    # adhesion
    for k in [
        "adhesion_model", "w_J_per_m2", "sigma_rms_m", "rough_model",
        "delta0_m", "z0_m", "mu_dmt", "mu_jkr",
        "min_h_m", "max_frac_of_Pmax", "min_points", "n_iter"
    ]:
        if k in adhesion:
            updates[k] = adhesion[k]

    # plotting / manual modes
    plotting = raw.get("plotting", {})
    if "manual_mode" in plotting:
        updates["manual_mode"] = str(plotting["manual_mode"])
        print("Loaded manual_mode, by YAML")
    if "manual_cycle_mode" in plotting:
        updates["manual_cycle_mode"] = str(plotting["manual_cycle_mode"])
    if "plot_mindlin" in plotting:
        updates["plot_mindlin"] = bool(plotting["plot_mindlin"])
    if "plot_cycles" in plotting:
        updates["plot_cycles"] = bool(plotting["plot_cycles"])
    if "live_plots" in plotting:
        updates["live_plots"] = bool(plotting["live_plots"])
        print("Loaded live_plots:", bool(plotting["live_plots"]))
    if "plot_every" in plotting:
        updates["plot_every"] = int(plotting["plot_every"])

    # finally: only keep keys that exist in Config (protect against typos)
    cfg_fields = {f.name for f in fields(Config)}
    clean_updates = {k: v for k, v in updates.items() if k in cfg_fields}

    return replace(cfg, **clean_updates)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("sss")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (optional)")
    p.add_argument("--batch", type=str, default=None, help="Folder containing CSV files (if omitted, a folder picker opens)")
    p.add_argument("--pattern", type=str, default="*.CSV", help="Glob pattern for CSVs")
    p.add_argument("--outdir", type=str, default="results", help="Output directory")

    # # core physical parameters
    p.add_argument("--tip_radius_um", type=float, default=5.0, help="Tip radius (µm) for A=pi*h*R")
    p.add_argument("--E1_GPa", type=float, default=170.0, help="Sample Young's modulus (GPa)")
    p.add_argument("--nu1", type=float, default=0.27, help="Sample Poisson's ratio")
    p.add_argument("--E2_GPa", type=float, default=2.5, help="Tip Young's modulus (GPa)")
    p.add_argument("--nu2", type=float, default=0.35, help="Tip Poisson's ratio")
    p.add_argument("--hardness_MPa", type=float, default=500.0, help="Sample hardness (MPa) for Hertz fit")
    p.add_argument("--plasticity_p0_frac", type=float, default=1.0, help="Plasticity parameter p0 fraction for Hertz fit")
    p.add_argument("--hertz_enable", type=bool, default=False, help="If set, perform Hertzian fit on loading segment")
    p.add_argument("--hertz_min_h_nm", type=float, default=5.0, help="Minimum depth (nm) for Hertz fit")
    p.add_argument("--hertz_max_frac_of_Pmax", type=float, default=0.2, help="Maximum fraction of Pmax for Hertz fit")
    p.add_argument("--hertz_plot", type=bool, default=False, help="If set, plot Hertz diagnostics during live plotting")
    # # optional frame stiffness
    p.add_argument("--k_frame_z", type=float, default=float("nan"), help="Frame stiffness Z (N/m), NaN=off")
    p.add_argument("--k_frame_x", type=float, default=float("nan"), help="Frame stiffness X (N/m), NaN=off")

    # # touch/cycle parameters
    p.add_argument("--k_touch_min", type=float, default=500.0, help="Touch threshold on Dyn. Stiffness")
    p.add_argument("--dynF2_active_delta", type=float, default=0.003, help="Active threshold above baseline (RMS units)")
    p.add_argument("--dynF2_nearzero_delta", type=float, default=0.0005, help="Near-zero boundary above baseline (RMS units)")
    p.add_argument("--smooth_n", type=int, default=301, help="Rolling median window for cycle detection")
    p.add_argument("--k_sup_x", type=float, default=float("nan"), help="Fallback lateral support spring stiffness (N/m). Used if calibration slice not found.")
    p.add_argument("--b_sup_x", type=float, default=0.0, help="Fallback lateral spring intercept (N). Used with --k_sup_x when calibration missing.")
    p.add_argument("--allow_no_cal", type=bool, default=False, help="If set, do not fail when calibration is missing; use fallback k_sup_x/b_sup_x if provided, else use k_sup_x=0.")
    p.add_argument("--normal_load_filter_win", type=int, default=101)
    p.add_argument("--normal_load_sustain_duration", type=float, default=0.1)
    p.add_argument("--normal_load_rate_th", type=float, default=0.00001)
    # # transition detection
    p.add_argument("--trans_frac_up", type=float, default=0.1, help="K_thresh = trans_frac_up * S_stuck")
    p.add_argument("--trans_frac_down", type=float, default=0.2, help="K_thresh = trans_frac_down * S_stuck")
    p.add_argument("--sliding_lateral_stiffness_thresh", type=float, default=500, help="K_thresh minimum for stick->slide detection (N/m)")
    p.add_argument("--resticking_lateral_stiffness_thresh", type=float, default=1000, help="K_thresh minimum for slide->stick detection (N/m)")
    p.add_argument("--trans_smooth_n", type=int, default=21, help="Rolling median window for transition detection")

    # # plotting + exports
    p.add_argument("--live_plots", type=bool, default=False, help="Show sanity plots during batch")
    p.add_argument("--plot_every", type=int, default=5, help="Show plots for every Nth file")
    p.add_argument("--summary_plots", type=bool, default=False, help="Create folder-level summary plots (saved)")
    p.add_argument("--origin_csv", type=bool, default=False, help="Export Origin-friendly cycle CSVs (one per file + combined)")
    p.add_argument("--summary_template", type=bool, default=False, help="Export SummaryNanoRo_like.csv (units row first)")
    p.add_argument("--expected_cycles", type=int, default=3, help="Expected number of cycles")
    p.add_argument("--manual_mode", type=str, default="always", choices=["always", "on_fail", "never"], help="Manual repick mode")
    p.add_argument("--manual_cycle_mode", type=str, default="always", choices=["always", "on_fail", "never"], help="Manual repick mode for cycles")
    p.add_argument("--plot_mindlin", type=bool, default=False, help="If set, plot Mindlin fits per cycle during live plotting")
    p.add_argument("--plot_cycles", type=bool, default=False, help="If set, plot per-cycle friction and transitions during live plotting")
    return p

def load_config():
    args = build_parser().parse_args()
    # load cfg (YAML if provided, else defaults)
    if args.config:
        cfg = load_config_yaml(args.config)
    else:
        cfg = Config()
    return cfg
    