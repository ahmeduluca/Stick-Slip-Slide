import re
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

# ============================================================
# 1) Units parsing (2nd row in CSV files)
# ============================================================

PREFIX = {
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}

def pick_folder_gui() -> str:
    """
    Opens a native folder picker (Windows/macOS/Linux) using tkinter.
    Returns the selected folder path, or raises if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(
            "tkinter is not available in this Python environment. "
            "Install/enable tkinter or pass --batch <folder>."
        ) from e

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select folder containing CSV files")
    root.destroy()

    if not folder:
        raise RuntimeError("No folder selected (cancelled).")
    return folder

def clean_unit_str(u: str) -> str:
    if u is None:
        return ""
    u = str(u).strip()
    u = u.replace("Â", "")      # fix ÂµN artifacts
    u = u.replace("μ", "µ")     # normalize mu variants
    return u

def parse_simple_unit(token: str) -> tuple[float, str]:
    token = clean_unit_str(token)
    if token == "" or token.lower() in {"none", "nan"}:
        return (1.0, "")
    if token in {"C", "°C"}:
        return (1.0, "C")

    m = re.fullmatch(r"([nµumkMG]?)([A-Za-z]+)", token)
    if not m:
        return (1.0, token)
    pref, base = m.group(1), m.group(2)
    if pref == "u":
        pref = "µ"
    return (PREFIX.get(pref, 1.0), base)

def parse_compound_unit(u: str) -> tuple[float, str]:
    u = clean_unit_str(u)
    if u == "" or u.lower() in {"none", "nan"}:
        return (1.0, "")

    if "/" in u:
        num, den = u.split("/", 1)
        s_num, base_num = parse_simple_unit(num)
        s_den, base_den = parse_simple_unit(den)
        unit_str = f"{base_num}/{base_den}".strip("/")
        return (s_num / s_den, unit_str)

    s, base = parse_simple_unit(u)
    return (s, base)

def read_csv_with_units(filepath: Path) -> tuple[pd.DataFrame, dict, dict]:
    """
    Assumes:
      Row 0: headers
      Row 1: units row
      Row 2+: numeric data
    Returns:
      df_data (units row removed)
      units_map[col] = unit string (cleaned)
      scale_to_SI[col] = multiplier to SI base units
    """
    raw = pd.read_csv(filepath, header=0, low_memory=False)
    if len(raw) < 2:
        raise RuntimeError("CSV too short: missing units row / data.")

    units_row = raw.iloc[0].to_dict()
    units_map = {c: clean_unit_str(units_row.get(c, "")) for c in raw.columns}
    scale_to_SI = {c: parse_compound_unit(units_map[c])[0] for c in raw.columns}

    df = raw.iloc[1:].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.reset_index(drop=True, inplace=True)
    return df, units_map, scale_to_SI

def extract_markers(df: pd.DataFrame, markers_col: str) -> Dict[str, int]:
    if markers_col not in df.columns:
        return {}
    out: Dict[str, int] = {}
    m = df[markers_col]
    mask = m.notna()
    # store first occurrence of each marker string
    for i in np.where(mask)[0]:
        name = str(m.iloc[i])
        out.setdefault(name, int(i))
    return out


def load_experiment(filepath: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Compatibility shim: same as read_csv_with_units."""
    return read_csv_with_units(filepath)


