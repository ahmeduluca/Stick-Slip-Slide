# src/stick_slip_slide/cli.py
from __future__ import annotations

from pathlib import Path

from .config_loader import load_config
from .io import pick_folder_gui
from .plotting import set_plot_defaults
from .pipeline import analyze_batch

def main() -> None:
    cfg = load_config()
    print(cfg.live_plots)

    set_plot_defaults()

    #batch folder (GUI picker if missing)
    batch_folder = cfg.batch or pick_folder_gui()
    input_dir = Path(batch_folder)

    # 4) outdir behavior (match monolith)
    outdir = (input_dir / "results") if (cfg.outdir == "results") else Path(cfg.outdir)

    analyze_batch(
        input_dir=input_dir,
        outdir=outdir,
        cfg=cfg,
        pattern=cfg.pattern,
        live_plots=cfg.live_plots,
        plot_every=cfg.plot_every,
        summary_plots=bool(cfg.summary_plots),
        origin_csv=bool(cfg.origin_csv),
        summary_template=bool(cfg.summary_template),
    )

    print("Done.")
    print("Outputs in:", outdir)
