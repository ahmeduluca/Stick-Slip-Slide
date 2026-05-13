from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, get_args, get_origin

import numpy as np
import yaml

from .config import Config

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


def _to_none_if_nan(x: Any) -> Any:
    try:
        v = float(x)
    except Exception:
        return x
    return None if not np.isfinite(v) else v


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return bool(v)


def _cast_value(value: Any, target_type: Any) -> Any:
    """
    Best-effort caster for dataclass field types.
    Handles Optional[T], tuple, int, float, bool, str.
    """
    if value is None:
        return None

    origin = get_origin(target_type)
    args = get_args(target_type)

    # Optional[T] or Union[T, None]
    if origin is not None and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _cast_value(value, non_none[0])

    # tuple[...] or plain tuple
    if origin is tuple or target_type is tuple:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return tuple(value)

    if target_type is bool:
        return _as_bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)

    # fallback
    return value


def load_config_yaml(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # nested sections (still allow flat keys too)
    sections = {
        "core": raw.get("core", {}),
        "touch": raw.get("touch", {}),
        "cycles": raw.get("cycles", {}),
        "transition": raw.get("transition", {}),
        "hertz": raw.get("hertz", {}),
        "materials": raw.get("materials", {}),
        "adhesion": raw.get("adhesion", {}),
        "plotting": raw.get("plotting", {}),
        "uncertainty": raw.get("uncertainty", {}),
        "channels": raw.get("channels", {}),
        "markers": raw.get("markers", {}),
        "calibration": raw.get("calibration", {}),
        "reporting": raw.get("reporting", {}),
        "paths": raw.get("paths", {}),
    }

    cfg = Config()
    cfg_fields = {f.name: f.type for f in fields(Config)}
    updates: dict[str, Any] = {}

    def set_if_present(
        cfg_key: str,
        *yaml_keys: str,
        section_names: tuple[str, ...] | None = None,
        transform=None,
    ) -> None:
        """
        Look for yaml_keys first in given sections, then at top-level raw.
        First match wins.
        """
        nonlocal updates

        if section_names is None:
            section_names = tuple(sections.keys())

        found = False
        value = None

        # search in requested sections
        for sec_name in section_names:
            sec = sections.get(sec_name, {})
            for yk in yaml_keys:
                if isinstance(sec, dict) and yk in sec:
                    value = sec[yk]
                    found = True
                    break
            if found:
                break

        # fallback to top-level flat keys
        if not found:
            for yk in yaml_keys:
                if yk in raw:
                    value = raw[yk]
                    found = True
                    break

        if not found:
            return

        value = _to_none_if_nan(value)
        if transform is not None and value is not None:
            value = transform(value)

        if cfg_key in cfg_fields:
            updates[cfg_key] = _cast_value(value, cfg_fields[cfg_key])

    # ------------------------------------------------------------------
    # CORE / PATHS
    # ------------------------------------------------------------------
    set_if_present("pattern", "pattern", section_names=("core", "paths"))
    set_if_present("batch", "batch", section_names=("core", "paths"))
    set_if_present("outdir", "outdir", section_names=("core", "paths"))
    set_if_present("time_col", "time_col", section_names=("core", "channels"))
    set_if_present("markers_col", "markers_col", section_names=("core", "channels"))
    set_if_present("daq_hz", "daq_hz", "fs_hz", section_names=("core",))

    # tip radius aliases
    set_if_present(
        "tip_radius_m",
        "tip_radius_m",
        section_names=("core", "hertz"),
    )
    set_if_present(
        "tip_radius_m",
        "tip_radius_um",
        section_names=("core", "hertz"),
        transform=lambda v: float(v) * 1e-6,
    )

    # ------------------------------------------------------------------
    # CHANNELS
    # ------------------------------------------------------------------
    set_if_present("Fz_raw_col", "Fz_raw_col", section_names=("channels", "core"))
    set_if_present("z_raw_col", "z_raw_col", section_names=("channels", "core"))
    set_if_present("x_raw_col", "x_raw_col", section_names=("channels", "core"))

    set_if_present("Sz_col", "Sz_col", section_names=("channels", "core"))
    set_if_present("Fz_dyn_rms_col", "Fz_dyn_rms_col", section_names=("channels", "core"))
    set_if_present("Z_dyn_rms_col", "Z_dyn_rms_col", section_names=("channels", "core"))
    set_if_present("PHI_col", "PHI_col", section_names=("channels", "core"))

    set_if_present("k_touch_col", "k_touch_col", section_names=("touch", "channels"))
    set_if_present("F2_rms_col", "F2_rms_col", section_names=("channels", "core"))
    set_if_present("X2_rms_col", "X2_rms_col", section_names=("channels", "core"))
    set_if_present("PH2_col", "PH2_col", section_names=("channels", "core"))

    # ------------------------------------------------------------------
    # VERTICAL SUPPORT SPRING / CALIBRATION
    # ------------------------------------------------------------------
    set_if_present("k_sup_z_fallback", "k_sup_z_fallback", section_names=("calibration", "core"))
    set_if_present("b_sup_z_fallback", "b_sup_z_fallback", section_names=("calibration", "core"))
    set_if_present("allow_no_cal_z", "allow_no_cal_z", section_names=("calibration", "core"))

    # ------------------------------------------------------------------
    # TOUCH
    # ------------------------------------------------------------------
    for key in [
        "k_touch_min",
        "k_touch_min_duration_s",
        "touch_mc_n",
        "marker_surface",
        "touch_slope_window_s",
        "touch_baseline_frac",
        "touch_offset_seconds",
        "touch_offset_margin_s",
        "touch_ignore_first_s",
        "touch_baseline_window_s",
        "touch_k_nsigma",
        "touch_require_monotonic_z",
    ]:
        set_if_present(key, key, section_names=("touch",))

    # ------------------------------------------------------------------
    # LATERAL LOCK-IN / CALIBRATION
    # ------------------------------------------------------------------
    set_if_present("dyn_f2_freq_Hz", "dyn_f2_freq_Hz", section_names=("core", "calibration"))
    set_if_present("lockin_slope_hw", "lockin_slope_hw", section_names=("core", "calibration"))

    set_if_present("marker_cal_up", "marker_cal_up", section_names=("markers", "calibration", "core"))
    set_if_present("marker_cal_dn", "marker_cal_dn", section_names=("markers", "calibration", "core"))
    set_if_present("k_sup_x_fallback", "k_sup_x_fallback", "k_sup_x", section_names=("calibration", "core"))
    set_if_present("c_sup_x_fallback", "c_sup_x_fallback", "c_sup_x", section_names=("calibration", "core"))
    set_if_present("allow_no_cal", "allow_no_cal", section_names=("calibration", "core"))
    set_if_present("cal_force_thr_rms", "cal_force_thr_rms", section_names=("calibration", "core"))
    set_if_present("cal_min_points", "cal_min_points", section_names=("calibration", "core"))

    # frame stiffness
    set_if_present("k_frame_z", "k_frame_z", section_names=("core", "calibration"))
    set_if_present("k_frame_x", "k_frame_x", section_names=("core", "calibration"))

    # ------------------------------------------------------------------
    # SHEAR WINDOW / CYCLES
    # ------------------------------------------------------------------
    for key in [
        "loading_rate_threshold",
        "normal_load_sustain",
        "normal_load_smooth",
    ]:
        set_if_present(key, key, section_names=("touch", "cycles"))

    for key in [
        "smooth_n",
        "dynF2_baseline_q",
        "dynF2_active_delta",
        "dynF2_nearzero_delta",
        "hold_top_frac",
        "hold_min_s",
        "min_cycle_points",
        "dfdt_smooth_n",
        "dfdt_thr_frac",
        "dfdt_hold_frac",
        "min_ramp_s",
        "min_hold_s",
        "expected_cycles",
    ]:
        set_if_present(key, key, section_names=("cycles",))

    # ------------------------------------------------------------------
    # UNCERTAINTY
    # ------------------------------------------------------------------
    for key in [
        "sigma_Fz_N",
        "sigma_z_m",
        "sigma_Ft_N",
        "sigma_Sz_N_per_m",
        "sigma_tip_radius_m",
        "sigma_Estar_Pa",
        "sigma_k_frame_z",
        "lockin_tau_s",
        "lockin_force_noise_N",
    ]:
        set_if_present(key, key, section_names=("uncertainty", "core", "calibration"))

    # ------------------------------------------------------------------
    # REPORTING WINDOWS
    # ------------------------------------------------------------------
    for key in [
        "pre_window_s",
        "post_window_s",
        "ref_window_s",
    ]:
        set_if_present(key, key, section_names=("reporting", "core"))

    # ------------------------------------------------------------------
    # TRANSITION
    # ------------------------------------------------------------------
    for key in [
        "trans_frac_up",
        "trans_frac_down",
        "sliding_lateral_stiffness_thresh",
        "resticking_lateral_stiffness_thresh",
        "trans_low_band",
        "trans_smooth_n",
        "trans_mode",
        "trans_skip_start_frac_up",
        "trans_skip_end_frac_dn",
        "trans_min_valid_Ft_frac",
        "phase_near_max_frac",
    ]:
        set_if_present(key, key, section_names=("transition",))

    # ------------------------------------------------------------------
    # MINDLIN
    # ------------------------------------------------------------------
    for key in [
        "mindlin_min_frac_of_maxF",
        "mindlin_max_frac_of_maxF",
        "mindlin_min_points",
    ]:
        set_if_present(key, key, section_names=("transition", "hertz"))

    # ------------------------------------------------------------------
    # HERTZ
    # ------------------------------------------------------------------
    set_if_present("hertz_enable", "hertz_enable", section_names=("hertz",))
    for key in [
        "hardness_Pa",
        "plasticity_p0_frac",
        "hertz_min_h_m",
        "hertz_max_frac_of_Pmax",
        "hertz_min_points",
        "hertz_iter",
        "hertz_plot",
        "area_mode",
    ]:
        set_if_present(key, key, section_names=("hertz",))

    # ------------------------------------------------------------------
    # MATERIALS
    # ------------------------------------------------------------------
    set_if_present("E1_Pa", "E1_Pa", section_names=("materials",))
    set_if_present("E2_Pa", "E2_Pa", section_names=("materials",))
    set_if_present("E1_Pa", "E1_GPa", section_names=("materials",), transform=lambda v: float(v) * 1e9)
    set_if_present("E2_Pa", "E2_GPa", section_names=("materials",), transform=lambda v: float(v) * 1e9)
    set_if_present("nu1", "nu1", section_names=("materials",))
    set_if_present("nu2", "nu2", section_names=("materials",))

    # ------------------------------------------------------------------
    # PLOTTING / MANUAL
    # ------------------------------------------------------------------
    for key in [
        "manual_mode",
        "manual_cycle_mode",
        "plot_mindlin",
        "plot_cycles",
        "live_plots",
        "plot_every",
        "summary_plots",
        "origin_csv",
        "summary_template",
    ]:
        set_if_present(key, key, section_names=("plotting",))

    # ------------------------------------------------------------------
    # ADHESION
    # ------------------------------------------------------------------
    for key in [
        "adhesion_model",
        "w_J_per_m2",
        "sigma_rms_m",
        "rough_model",
        "delta0_m",
        "z0_m",
        "mu_dmt",
        "mu_jkr",
        "min_h_m",
        "max_frac_of_Pmax",
        "min_points",
        "n_iter",
    ]:
        set_if_present(key, key, section_names=("adhesion",))

    return replace(cfg, **updates)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("sss")

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    p.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Folder containing CSV files; overrides YAML if given"
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern for CSV files; overrides YAML if given"
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory; overrides YAML if given"
    )

    # optional convenience overrides
    p.add_argument(
        "--live_plots",
        action="store_true",
        help="Enable live plots for this run"
    )
    p.add_argument(
        "--summary_plots",
        action="store_true",
        help="Enable summary plots for this run"
    )

    return p

def load_config() -> Config:
    args = build_parser().parse_args()

    # start from YAML or defaults
    if args.config:
        cfg = load_config_yaml(args.config)
    else:
        cfg = Config()

    # apply CLI overrides only when explicitly provided
    updates = {}
    if args.batch is not None:
        updates["batch"] = args.batch
    if args.pattern is not None:
        updates["pattern"] = args.pattern
    if args.outdir is not None:
        updates["outdir"] = args.outdir
    if args.live_plots:
        updates["live_plots"] = True
    if args.summary_plots:
        updates["summary_plots"] = True

    if updates:
        cfg = replace(cfg, **updates)

    return cfg
    