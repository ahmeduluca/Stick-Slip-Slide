from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Config:
    # core
    pattern: str = "*.CSV"
    batch: str = None
    outdir: str = "results"
    time_col: str = "Time"
    markers_col: str = "Markers"

    # raw normal channels
    Fz_raw_col: str = "Force"
    z_raw_col: str = "Displacement"

    # vertical stiffness
    Sz_col: str = "Dyn. Stiffness"
    # optional vertical lock-in (RMS) if available for calibration sanity check
    Fz_dyn_rms_col: Optional[str] = "Dyn. Force" 
    Z_dyn_rms_col: Optional[str] = "Dyn. Disp."    

    # touch detection
    k_touch_col: str = "Dyn. Stiffness"
    k_touch_min: float = 500.0
    k_touch_min_duration_s: float = 0.1
    marker_surface: str = "Surface Index"

    # lateral lock-in channels (RMS)
    F2_rms_col: str = "Dyn. Force 2"
    X2_rms_col: str = "Dyn. Disp. 2"
    PH2_col: str = "Dyn. Phase 2"  # displacement relative to force
    dyn_f2_freq_Hz: float = 80.0    # lateral CSM frequency
    lockin_slope_hw: float = 0.05  # Hz, half-width for slope calc

    # detect shear window parameters
    ## !not absolute loading rate but the max rate's percentage after the touch pt as the stopping pt of loading.
    loading_rate_threshold: float = 0.001 #threshold for finding loading/unloading and shear window by normal load gradient max' percentage

    normal_load_sustain: float = 0.5 #the minimum time (s) for indication of hold has started-to avoid measurement and gradient instability
    normal_load_smooth: int = 201 ##smoothing on the normal load-rate channel.

    # cycle detection from Dyn Force 2 RMS envelope
    smooth_n: int = 501
    dynF2_baseline_q: float = 0.05
    dynF2_active_delta: float = 0.03
    dynF2_nearzero_delta: float = 0.01
    hold_top_frac: float = 0.98
    hold_min_s: float = 0.2
    min_cycle_points: int = 200
    # derivative-based cycle detection
    dfdt_smooth_n: int = 301          # extra smoothing for derivative stability
    dfdt_thr_frac: float = 0.001       # derivative threshold as fraction of max |dF/dt| in shear window
    dfdt_hold_frac: float = 0.9      # hold condition: |dF/dt| <= hold_frac * dfdt_thr
    min_ramp_s: float = 2.0           # minimum duration of ramp-up/down (seconds)
    min_hold_s: float = 0.5           # minimum duration of hold plateau (seconds)

    # --- Uncertainty inputs (1-sigma) ---
    # normal channels
    sigma_Fz_N: float = 5e-9          # example: 5 nN
    sigma_z_m: float = 1e-9           # example: 1 nm

    # lateral amplitude / friction force channel
    sigma_Ft_N: float = 10e-9         # ~10 nN typical noise per lock-in τ

    # vertical stiffness channel
    sigma_Sz_N_per_m: float = 50.0    # example; set from hold scatter

    # model parameters
    sigma_tip_radius_m: float = 0.5e-6   # e.g. ±0.5 µm
    sigma_Estar_Pa: float = 2e9          # e.g. ±2 GPa
    sigma_k_frame_z: float = 0.0         # if you want it; else 0

    # lock-in lag model (for ramped envelopes)
    lockin_tau_s: float = 0.050
    lockin_force_noise_N: float = 10e-9

    # reporting windows around cycles
    pre_window_s: float = 1.0
    post_window_s: float = 3.0
    ref_window_s: float = 1.0

    # lateral calibration markers (preferred)
    marker_cal_up: str = "dynLRampUp"
    marker_cal_dn: str = "dynLRampDown"
    k_sup_x_fallback: Optional[float] = None  # N/m
    b_sup_x_fallback: float = 0.0             # N
    allow_no_cal: bool = False

    # fallback lateral calibration heuristic if markers missing
    cal_force_thr_rms: float = 0.01
    cal_min_points: int = 400

    # frame stiffness corrections (optional)
    k_frame_z: Optional[float] = 1000000   # N/m
    k_frame_x: Optional[float] = 500000   # N/m

    # tip radius for A = pi*h*R
    tip_radius_m: float = 50e-6

    # transition detection (stick->slide and re-stick)
    trans_frac_up: float = 0.1              # S_thresh = frac * Sx_stuck (stick to slide)
    trans_frac_down: float = 0.15            # S_thresh = frac * Sx_stuck (restick)
    sliding_lateral_stiffness_thresh: float = 500.0    # N/m minimum S_thresh for stick->slide
    resticking_lateral_stiffness_thresh: float = 1000.0 # N/m minimum S_thresh for slide->stick
    trans_low_band: tuple[float, float] = (0.05, 0.20)  # early ramp-up force band to estimate S_stuck
    trans_smooth_n: int = 21

    # Mindlin fit K(Q)=a*(1-Q/t)^(1/3) on ramp-up
    mindlin_min_frac_of_maxF: float = 0.1
    mindlin_max_frac_of_maxF: float = 0.99
    mindlin_min_points: int = 30

# ------------------------------
    # Hertz diagnostics (normal F vs h)
    # ------------------------------
    hertz_enable: bool = True

    # material constants (Pa)
    E1_Pa: float = 70e9          # fused silica ~ 70 GPa
    nu1: float = 0.18            # fused silica ~ 0.18
    E2_Pa: float = 1140e9        # diamond ~ 1140 GPa
    nu2: float = 0.07            # diamond ~ 0.07

    hardness_Pa: float = 10.0e9   # optional; set None/NaN to disable plasticity filtering
    plasticity_p0_frac: float = 1.0  # require max Hertz pressure p0 <= frac * hardness -silica yields close to hardness C~1-1.50 GPa

    # data selection
    hertz_min_h_m: float = 1e-9      # ignore ultra-small depths (noise/offset), e.g. 5 nm
    hertz_max_frac_of_Pmax: float = 1  # fit only up to this fraction of peak load in loading
    hertz_min_points: int = 50

    # robust / iteration
    hertz_iter: int = 3              # iterate fit-filter-fit using p0 criterion
    hertz_plot: bool = True         # show diagnostic plot per file when live_plots is True
    area_mode: str = "nominal"   # "nominal" | "fit_hertz" "flat_end"
    
    # manual selection policy
    manual_mode: str = "always"   # "never" | "on_fail" | "always"
    manual_cycle_mode: str = "always"  # same idea, but for per-cycle indices
    expected_cycles: Optional[int] = None  # for manual cycle picking prompt and validation later..
    plot_mindlin: bool = True
    plot_cycles: bool = True
    live_plots: bool = False
    plot_every: int = 0
    summary_plots: bool = True
    origin_csv: bool = True
    summary_template: bool = True

    ##Lock-in amplifier EG7280-used parameters-
    lockin_tau_s: float = 0.050
    lockin_force_noise_N: float = 10e-9
    lockin_pre_s: float = 0.30
    lockin_guard_s: float = 0.25

    adhesion_model: str = "auto"   # "hertz"|"dmt"|"jkr or "auto"
    w_J_per_m2: float = 0.5       # user-set (e.g. silica/diamond range)->via atomistic, 
    sigma_rms_m: float | None = 0.5  # RMS Roughness 
    rough_model: str = "exp"       # "none"|"exp"|user set effective adhesion by roughness
    delta0_m: float = 0.3e-9 #cut off for exponential Persson
    z0_m: float = 0.3e-9    # L-J minimum, stable pt.
    mu_dmt: float = 1 #Tabor parameter for DMT-upper limit
    mu_jkr: float = 5.0 #Tabor parameter for JKR--lower limit
    min_h_m: float = 1e-10#minimum contact depth for hertz fit,
    max_frac_of_Pmax: float = 1  # fit only up to this fraction of peak load in loading.
    min_points: int = 10
    n_iter: int = 100