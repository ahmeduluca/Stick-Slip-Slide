# signal.py
from __future__ import annotations

from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from .io import extract_markers
from dataclasses import dataclass
from .plotting import pick_indices_from_plot

from .cycle_types import CycleBounds
from .math_utils import (
    _num, median_dt, rolling_median, safe_nanmax, contiguous_regions
)

def detect_touch_index(df: pd.DataFrame, cfg: Config, markers: Dict[str, int]) -> int:
    if cfg.marker_surface in markers:
        return int(markers[cfg.marker_surface])

    t = _num(df, cfg.time_col)
    k = _num(df, cfg.k_touch_col)
    dt = np.nanmedian(np.diff(t))
    nmin = max(1, int(cfg.k_touch_min_duration_s / max(dt, 1e-12)))

    above = np.isfinite(k) & (k > cfg.k_touch_min)
    idxs = np.where(above)[0]
    for i in idxs:
        if i + nmin < len(above) and np.all(above[i:i+nmin]):
            return int(i)
    raise RuntimeError("Touch not found (no Surface Index marker and stiffness criterion failed).")

def manual_pick_touch(df: pd.DataFrame, cfg: Config, initial: Optional[int] = None) -> int:
    t = _num(df, cfg.time_col)
    k = _num(df, cfg.k_touch_col)
    picks = pick_indices_from_plot(
        t,
        series=[("Dyn. Stiffness", k)],
        prompts=["Click TOUCH point (first contact)."],
        n_clicks=1,
        predefined_picks=initial if initial is not None else [],
        title="Pick touch index"
    )
    return picks[0]

def detect_cycles(
    df: pd.DataFrame,
    cfg,
    start_i: int = 0,
    end_i: Optional[int] = None
) -> List[CycleBounds]:
    t = _num(df, cfg.time_col)
    a = np.nan_to_num(_num(df, cfg.F2_rms_col), nan=0.0)
    cycles: List[CycleBounds] = []

    n = len(a)
    if n < 10:
        raise RuntimeError("Too few samples for cycle detection.")

    if end_i is None:
        end_i = n - 1

    start_i = int(np.clip(start_i, 0, n - 2))
    end_i = int(np.clip(end_i, start_i + 1, n - 1))

    dt = median_dt(t)
    ramp_min_pts = max(3, int(cfg.min_ramp_s / max(dt, 1e-12)))
    hold_min_pts = max(3, int(cfg.min_hold_s / max(dt, 1e-12)))

    # Smooth envelope for amplitude
    a_s = a #rolling_median(a, cfg.smooth_n)

    # Extra smoothing for derivative stability
    a_sd = a_s #rolling_median(a, cfg.dfdt_smooth_n)
    da = np.gradient(a_sd, t)

    # zero out outside shear window
    a_s[:start_i] = 0.0
    a_s[end_i + 1:] = 0.0
    a_sd[:start_i] = 0.0
    a_sd[end_i + 1:] = 0.0
    da[:start_i] = 0.0
    da[end_i + 1:] = 0.0

    # Baseline and amplitude thresholds (secondary checks)
    base = float(np.quantile(a_s[start_i:end_i + 1], cfg.dynF2_baseline_q))
    thr_active = base + cfg.dynF2_active_delta
    thr_nz = base + cfg.dynF2_nearzero_delta

    # Derivative thresholds
    scale = float(safe_nanmax(np.abs(da[start_i:end_i + 1])))
    if not np.isfinite(scale) or scale <= 0:
        return cycles #raise RuntimeError("Cycle detection: derivative scale is zero; check signal/window.")

    dthr = cfg.dfdt_thr_frac * scale
    dhold = cfg.dfdt_hold_frac * dthr

    ramp_up = da > dthr
    ramp_dn = da < -dthr
    hold_like = np.abs(da) <= dhold

    up_regs = [(s, e) for (s, e) in contiguous_regions(ramp_up) if (e - s + 1) >= ramp_min_pts]

    cursor = start_i
    cyc = 0

    if not up_regs:
         return cycles  # No ramp-up regions found; return empty list
    
    for (us, ue) in up_regs:
        if us < cursor:
            continue

        # backtrack to near-zero to define start
        i_start = us
        while i_start > start_i and a_s[i_start] > thr_nz:
            i_start -= 1

        # find first sustained ramp-down after this ramp-up
        dn_after = [
            (s, e) for (s, e) in contiguous_regions(ramp_dn)
            if (e - s + 1) >= ramp_min_pts and s > ue
        ]

        search_end = end_i
        if dn_after:
            first_dn_s, _ = dn_after[0]
            search_end = min(search_end, first_dn_s)

        if search_end <= ue:
            continue

        seg_pk = slice(us, search_end + 1)
        i_peak = int(us + np.argmax(a_s[seg_pk]))
        amax = float(a_s[i_peak])

        # reject tiny bumps
        if not (amax > thr_active and (amax - base) > 2.0 * cfg.dynF2_active_delta):
            continue

        # hold plateau near the top
        near_top = (a_s >= cfg.hold_top_frac * amax) & hold_like
        near_top[:us] = False
        near_top[end_i + 1:] = False
        if dn_after:
            near_top[dn_after[0][0]:] = False

        top_regs = [(s, e) for (s, e) in contiguous_regions(near_top) if (e - s + 1) >= hold_min_pts]
        if top_regs:
            rs, re = min(top_regs, key=lambda r: abs((r[0] + r[1]) / 2 - i_peak))
            i_hold0, i_hold1 = int(rs), int(re)
        else:
            i_hold0, i_hold1 = i_peak, i_peak

        # end: after ramp-down, go until near-zero
        if dn_after:
            i_end = int(dn_after[0][1])
        else:
            i_end = int(i_peak)

        while i_end < end_i and a_s[i_end] > thr_nz:
            i_end += 1

        if not (i_start < i_peak < i_end):
            continue

        cyc += 1
        cycles.append(CycleBounds(cyc, i_start, i_peak, i_hold0, i_hold1, i_end))
        cursor = i_end + 1

    return cycles

def manual_pick_shear_window(t: np.ndarray, P_contact_N: np.ndarray, F2_rms: np.ndarray, initial: Optional[List[int]] = None) -> Tuple[int, int]:
    picks = pick_indices_from_plot(
        t,
        series=[("P_contact (N)", P_contact_N), ("Dyn Force 2 RMS", F2_rms)],
        prompts=["Click shear window START.", "Click shear window END."],
        n_clicks=2,
        predefined_picks=initial,
        title="Pick shear window bounds"
    )
    i0, i1 = sorted(picks)
    return i0, i1

def manual_pick_one_cycle(t: np.ndarray, F2_env: np.ndarray, initial: CycleBounds, title: str = "") -> "CycleBounds":
    picks = pick_indices_from_plot(
        t,
        series=[("Dyn Force 2 envelope", F2_env)],
        prompts=[
            "Click cycle START (near-zero before ramp-up).",
            "Click cycle PEAK (max amplitude).",
            "Click HOLD END (plateau end).",
            "Click cycle END (back to near-zero).",
            "If no cycle to select, click f.",
        ],
        predefined_picks=[initial.i_start if initial.i_start is not None else 0
                          , initial.i_peak if initial.i_peak is not None else 0
                          , initial.i_hold1 if initial.i_hold1 is not None else 0
                          , initial.i_end if initial.i_end is not None else 0],
        n_clicks=4,

        title=title or "Pick cycle bounds"
    )
    i_start, i_peak, i_hold1, i_end = picks
    i_hold0=i_peak+1
    cycle = 1  # dummy cycle number; will be replaced by caller
    # basic ordering guard
    if not (i_start < i_peak < i_end):
        cycle, i_start, i_end, i_peak, i_hold0, i_hold1 = 1,0,0,0,0,0
    if i_peak < i_hold0:
        i_hold0, i_peak = i_peak, i_hold0
    if i_hold1 < i_hold0:
        i_hold0, i_hold1 = i_hold1, i_hold0
    return CycleBounds(cycle, i_start=i_start, i_peak=i_peak, i_hold0=i_hold0, i_hold1=i_hold1, i_end=i_end)

def manual_pick_cycles(df2: pd.DataFrame, cfg: Config, i0: int, i1: int, initial: List[CycleBounds], n_cycles: int = 3) -> List["CycleBounds"]:
    t = _num(df2, cfg.time_col)
    F2 = np.nan_to_num(_num(df2, cfg.F2_rms_col), nan=0.0)
    F2s = pd.Series(F2).rolling(cfg.smooth_n, center=True, min_periods=1).median().to_numpy()
    F2s[:i0] = 0.0
    F2s[i1+1:] = 0.0

    cycles = []
    for c in range(1, n_cycles + 1):
        cb = manual_pick_one_cycle(t, F2s, initial[c-1] if initial is not None and c-1 < len(initial) else CycleBounds(cycle=c, i_start=i0, i_peak=0, i_hold0=0, i_hold1=0, i_end=i1), title=f"Pick cycle {c}")
        if cb.cycle == 0:
            continue
        if cb.i_start == 0:
            break
        cb = CycleBounds(cycle=c, i_start=cb.i_start, i_peak=cb.i_peak, i_hold0=cb.i_hold0, i_hold1=cb.i_hold1, i_end=cb.i_end)
        cycles.append(cb)
    return cycles

# ============================================================
# Transition detection: stick->slide and re-stick
# ============================================================
def detect_stick_slide_transitions(
    df: pd.DataFrame,
    b: CycleBounds,
    sliding_lateral_stiffness_thresh: float,    # user threshold for stick->slide (Sx drops below)
    resticking_lateral_stiffness_thresh: float, # user threshold for re-stick (Sx rises above)
    frac_up: float,                             # fallback fraction for stick->slide: S_slide = frac_up * Sx_stuck
    frac_low: float,                            # fallback fraction for re-stick:  S_re   = frac_low * Sx_stuck
    low_frac_band: tuple[float, float],         # band of Ft/Ftmax to estimate Sx_stuck
    smooth_n: int,
) -> dict:
    """
    stick->slide (ramp-up): first index where Sx falls below S_slide
    re-stick    (ramp-down): after hold ends, first index where Sx rises above S_re,
                             but ONLY after it has gone below S_slide at least once.

    Sx_stuck estimated as median Sx in early ramp-up band: Ft in [low*Ftmax, high*Ftmax].
    """
    Ft = df["F2_pk_corr_N"].to_numpy()
    Xc = df["X2_pk_contact_m"].to_numpy()
    Sx = df["Stiffness_lateral"].to_numpy()

    # ---- ramp-up slice ----
    ru0 = int(b.i_start)
    ru1 = int(b.i_peak)
    ru = slice(ru0, ru1 + 1)

    # ---- ramp-down slice: START AFTER HOLD ----
    # This is the key fix: do not allow re-stick detection during peak/hold region.
    rd0 = int(max(b.i_hold1, b.i_peak))
    rd1 = int(b.i_end)
    rd = slice(rd0, rd1 + 1)

    # Smooth Sx within each slice
    Sx_ru_s = pd.Series(Sx[ru]).rolling(smooth_n, center=True, min_periods=1).median().to_numpy()
    Sx_rd_s = pd.Series(Sx[rd]).rolling(smooth_n, center=True, min_periods=1).median().to_numpy()

    Ft_ru = Ft[ru]

    Ftmax = safe_nanmax(Ft_ru) if np.isfinite(Ft_ru).any() else np.nan
    if not (np.isfinite(Ftmax) and Ftmax > 0):
        return {
            "i_ss": np.nan, "i_rs": np.nan,
            "Sx_stuck": np.nan,
            "Sx_slide_used": np.nan,
            "Sx_restick_used": np.nan,
            "Ft_ss_N": np.nan, "X_ss_m": np.nan,
            "Ft_rs_N": np.nan, "X_rs_m": np.nan,
            "rd0": rd0,
            "went_low_first": 0,
        }

    # ---- estimate Sx_stuck from early ramp-up ----
    lo = low_frac_band[0] * Ftmax
    hi = low_frac_band[1] * Ftmax
    m_stuck = np.isfinite(Ft_ru) & np.isfinite(Sx_ru_s) & (Ft_ru >= lo) & (Ft_ru <= hi)

    if m_stuck.sum() < 10:
        idxs = np.where(np.isfinite(Sx_ru_s))[0][:max(10, min(30, len(Sx_ru_s)))]
        Sx_stuck = float(np.nanmedian(Sx_ru_s[idxs])) if idxs.size else np.nan
    else:
        Sx_stuck = float(np.nanmedian(Sx_ru_s[m_stuck]))

    if not (np.isfinite(Sx_stuck) and Sx_stuck > 0):
        return {
            "i_ss": np.nan, "i_rs": np.nan,
            "Sx_stuck": Sx_stuck,
            "Sx_slide_used": np.nan,
            "Sx_restick_used": np.nan,
            "Ft_ss_N": np.nan, "X_ss_m": np.nan,
            "Ft_rs_N": np.nan, "X_rs_m": np.nan,
            "rd0": rd0,
            "went_low_first": 0,
        }

    # ---- decide thresholds: user-provided takes priority if finite ----
    # stick->slide threshold (must be BELOW stuck)
    Sx_slide = float(sliding_lateral_stiffness_thresh)
    if not (np.isfinite(Sx_slide) and (Sx_slide < Sx_stuck)):
        Sx_slide = float(frac_up) * Sx_stuck
        slide_source = "frac"
    else:
        slide_source = "user"

    # ---- stick->slide: first crossing below Sx_slide on ramp-up ----
    i_ss_rel = None
    for i, val in enumerate(Sx_ru_s):
        if np.isfinite(val) and (val < Sx_slide):
            i_ss_rel = i
            slide_source = "user"
            break
    if i_ss_rel is None:
        # fallback: fraction of Sx_stuck
        Sx_slide = float(frac_up) * Sx_stuck
        for i, val in enumerate(Sx_ru_s):
            if np.isfinite(val) and (val < Sx_slide):
                i_ss_rel = i
                slide_source = "frac"
                break

    i_ss = (ru0 + i_ss_rel) if i_ss_rel is not None else None

    # ---- re-stick: AFTER HOLD, require it went low first, then crosses above Sx_re ----
    went_low = False
    i_rs_rel = None
    re_source = "none"
    Sx_re = float(resticking_lateral_stiffness_thresh)
    for j, val in enumerate(Sx_rd_s):
        if not np.isfinite(val):
            continue
        # Step 1: detect "went low" (i.e., definitely in sliding regime)
        if not went_low:
            if val < Sx_slide:
                went_low = True
            continue
        # Step 2: first time it rises above re-stick threshold
        if val > Sx_re:
            i_rs_rel = j
            re_source = "user"
            break
    
    if i_rs_rel is None:
        Sx_re = float(frac_low) * Sx_stuck
        went_low = False
        # fallback: fraction of Sx_stuck
        for j, val in enumerate(Sx_rd_s):
            if not np.isfinite(val):
                continue
        # Step 1: detect "went low" (i.e., definitely in sliding regime)
            if not went_low:
                if val < Sx_slide:
                    went_low = True
                continue
        # Step 2: first time it rises above re-stick threshold
            if val > Sx_re:
                i_rs_rel = j
                re_source = "frac"
                break
    i_rs = (rd0 + i_rs_rel) if i_rs_rel is not None else None

    out = {
        "i_ss": i_ss,
        "i_rs": i_rs,
        "Sx_stuck": float(Sx_stuck),
        "Sx_slide_used": float(Sx_slide),
        "Sx_restick_used": float(Sx_re),
        "Sx_slide_source": slide_source,
        "Sx_restick_source": re_source,
        "rd0": int(rd0),
        "went_low_first": int(went_low),
    }

    if i_ss is not None:
        out["Ft_ss_N"] = float(Ft[i_ss])
        out["X_ss_m"] = float(Xc[i_ss])
    else:
        out["Ft_ss_N"] = np.nan
        out["X_ss_m"] = np.nan

    if i_rs is not None:
        out["Ft_rs_N"] = float(Ft[i_rs])
        out["X_rs_m"] = float(Xc[i_rs])
    else:
        out["Ft_rs_N"] = np.nan
        out["X_rs_m"] = np.nan

    return out

def want_manual(cfg: Config, mode: str, failed: bool) -> bool:
    if mode == "never": return False
    if mode == "always": return True
    return bool(failed)  # "on_fail"

def approve_or_repick_gate(figures, fig_title: str = "") -> str:
    """
    Returns: "accept" | "repick_touch" | "repick_window" | "repick_cycles"
    Raises on pass/abort.
    """
    decision = {"val": None}

    def on_key(event):
        k = (event.key or "").lower()
        if k == "a":
            decision["val"] = "accept"
        elif k == "t":
            decision["val"] = "repick_touch"
        elif k == "w":
            decision["val"] = "repick_window"
        elif k == "c":
            decision["val"] = "repick_cycles"
        elif k == "p":
            decision["val"] = "pass"
        elif k == "escape":
            decision["val"] = "abort"

        if decision["val"] is not None:
            plt.close("all")
    for fig in figures:
        try:
            fig.canvas.mpl_connect("key_press_event", on_key)
            if fig_title:
                fig.canvas.manager.set_window_title(fig_title)
        except Exception:
            pass
    plt.show(block=True)

    if decision["val"] == "accept":
        return "accept"
    if decision["val"] in {"repick_touch","repick_window","repick_cycles"}:
        return decision["val"]
    if decision["val"] == "pass":
        raise RuntimeError("User passed file.")
    raise RuntimeError("User aborted.")
# ============================================================
# ============================================================
# Calibration slice, cycle detection
# ============================================================
def find_calibration_slices_pre_touch(
    df: pd.DataFrame,
    cfg: Config,
    markers: Dict[str, int],
    touch_i: int,
) -> tuple[Optional[slice], Optional[slice]]:
    """
    Returns (cal_sl_lat, cal_sl_vert).
    cal_sl_lat: slice for lateral calibration (used for spring subtraction fit)
    cal_sl_vert: slice for vertical dynamic calibration bump (sanity check only)
    """
    # Markers (if present) win
    if cfg.marker_cal_up in markers and cfg.marker_cal_dn in markers:
        i0 = int(markers[cfg.marker_cal_up])
        i1 = int(markers[cfg.marker_cal_dn])
        if i1 <= i0:
            raise RuntimeError("Calibration markers out of order.")
        return slice(i0, i1 + 1), None

    # Pre-touch search window
    pre_end = max(0, int(touch_i))
    if pre_end < 50:
        # too short to detect bumps robustly
        return None, None

    idx = np.arange(pre_end)

    # Lateral bump (primary)
    y_lat = _num(df, cfg.F2_rms_col)
    cal_lat = find_first_bump_region(
        y=y_lat, idx=idx,
        smooth_n=101, thr_frac=0.08, pad=20, min_points=40
    )

    # Vertical bump (optional sanity)
    cal_vert = None
    if getattr(cfg, "Fz_dyn_rms_col", None) and cfg.Fz_dyn_rms_col in df.columns:
        y_v = _num(df, cfg.Fz_dyn_rms_col)
        cal_vert = find_first_bump_region(
            y=y_v, idx=idx,
            smooth_n=101, thr_frac=0.08, pad=20, min_points=40
        )

    return cal_lat, cal_vert
# ============================================================
def find_shear_window_from_normal_load_v2(
    t: np.ndarray,
    P_contact_N: np.ndarray,
    touch_i: int,
    cfg: Config
) -> tuple[int, int]:
    """
    Robust window:
      i0 = after loading is finished (just before/around max load plateau)
      i1 = before unloading begins
    Anchors everything around the max of smoothed P after touch.
    """
    n = len(t)
    dt = median_dt(t)
    if n < 10:
        return (0, n - 1)
    
    smooth_n = cfg.normal_load_smooth
    dpdt_thr_frac = cfg.loading_rate_threshold
    sustain_s = cfg.normal_load_sustain

    w = max(11, int(smooth_n))
    P_sm = pd.Series(P_contact_N).rolling(w, center=True, min_periods=1).median().to_numpy()
    dPdt = np.gradient(P_sm, t)
    dPdt = pd.Series(dPdt).rolling(w, center=True, min_periods=1).median().to_numpy()

    start = max(0, int(touch_i))
    if start >= n - 2:
        return (start, n - 1)

    # threshold based on post-touch derivative scale
    post = np.arange(start, n)
    scale = float(safe_nanmax(np.abs(dPdt[post]))) if post.size else float(safe_nanmax(np.abs(dPdt)))
    if not np.isfinite(scale) or scale <= 0:
        return (start, n - 1)

    thr = dpdt_thr_frac * scale
    sustain_pts = max(3, int(sustain_s / max(dt, 1e-12)))

    # locate max load AFTER touch (on smoothed)
    iPmax = int(start + np.nanargmax(P_sm[start:]))

    # sustained +slope and -slope masks
    pos = dPdt > thr
    neg = dPdt < -thr
    stable = np.abs(dPdt) <= thr

    # ---- find i0: first stable point AFTER the last sustained positive region before max ----
    pos_regs = [(s, e) for (s, e) in contiguous_regions(pos) if (e - s + 1) >= sustain_pts]
    # pick the last pos-reg that ends before iPmax
    pos_before = [r for r in pos_regs if r[1] < iPmax]
    if pos_before:
        last_pos_end = pos_before[-1][1]
        # from there, find the next stable region
        stable_regs = [(s, e) for (s, e) in contiguous_regions(stable) if (e - s + 1) >= sustain_pts and s > last_pos_end]
        i0 = stable_regs[0][0] if stable_regs else max(start, last_pos_end + 1)
    else:
        # fallback: just use a bit before max plateau
        i0 = max(start, iPmax - 5 * sustain_pts)

    # ---- find i1: first sustained negative region AFTER max ----
    neg_regs = [(s, e) for (s, e) in contiguous_regions(neg) if (e - s + 1) >= sustain_pts and s > iPmax]
    i1 = neg_regs[0][0] if neg_regs else (n - 1)

    # guards
    i0 = int(np.clip(i0, start, n - 2))
    i1 = int(np.clip(i1, i0 + 1, n - 1))
    return i0, i1

def find_first_bump_region(y: np.ndarray, idx: np.ndarray, smooth_n: int = 101, thr_frac: float = 0.08, 
                           pad: int = 20, min_points: int = 40) -> Optional[slice]:
    """
    Finds the FIRST contiguous region where smoothed y rises above baseline+thr_frac*span.
    Returns a slice in ORIGINAL indices (not local).
    """
    if idx.size < 10:
        return None

    y0 = np.nan_to_num(y[idx], nan=0.0)
    ys = pd.Series(y0).rolling(int(smooth_n), center=True, min_periods=1).median().to_numpy()

    base = float(np.quantile(ys, 0.10))
    span = float(np.quantile(ys, 0.95) - base)
    if not np.isfinite(span) or span <= 0:
        return None

    thr = base + float(thr_frac) * span
    regs = contiguous_regions(ys > thr)
    if not regs:
        return None

    s, e = regs[0]
    s = max(0, s - int(pad))
    e = min(len(ys) - 1, e + int(pad))
    if (e - s) < int(min_points):
        return None

    i0 = int(idx[s])
    i1 = int(idx[e])
    return slice(i0, i1 + 1)

def window_idx(t: np.ndarray, center_i: int, halfwidth_s: float) -> np.ndarray:
    t0 = t[center_i]
    return np.where((t >= t0 - halfwidth_s) & (t <= t0 + halfwidth_s))[0]

def window_idx_fw(t: np.ndarray, start_i: int, width_s: float) -> np.ndarray:
    t0 = t[start_i]
    return np.where((t >= t0) & (t <= t0 + width_s))[0]
