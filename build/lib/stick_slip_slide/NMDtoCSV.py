import struct
import xml.etree.ElementTree as ET
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd


# ---------------------------
# UI: file selector
# ---------------------------

def pick_nmd_file() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select Nanoindenter .NMD file",
        filetypes=[("Nanoindenter data", "*.NMD *.nmd"), ("All files", "*.*")]
    )
    root.destroy()

    if not file_path:
        raise SystemExit("No file selected.")

    return Path(file_path)


# ---------------------------
# NMD parsing helpers
# ---------------------------

def find_sample_xml_block(file_path: Path, scan_bytes: int = 5_000_000):
    with file_path.open("rb") as f:
        head = f.read(scan_bytes)

    arr = np.frombuffer(head, dtype=np.uint8)
    is_ascii = ((arr >= 32) & (arr < 127)) | (arr == 9) | (arr == 10) | (arr == 13)

    max_len = 0
    max_start = 0
    run_start = None

    for i, ok in enumerate(is_ascii):
        if ok and run_start is None:
            run_start = i
        if (not ok or i == len(is_ascii) - 1) and run_start is not None:
            run_end = i if not ok else i + 1
            run_len = run_end - run_start
            if run_len > max_len:
                max_len = run_len
                max_start = run_start
            run_start = None

    ascii_block = head[max_start:max_start + max_len]
    end_tag = b"</SAMPLE>"
    end_pos = ascii_block.rfind(end_tag)
    if end_pos < 0:
        raise RuntimeError("Could not locate </SAMPLE> in metadata block.")

    xml_bytes = ascii_block[:end_pos + len(end_tag)]
    return xml_bytes, max_start, max_start + end_pos + len(end_tag)


def parse_channels(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
    chans = [ch.attrib for ch in root.findall(".//CHANNEL")]
    df = pd.DataFrame(chans).drop_duplicates()
    df["DATAINDEX"] = df["DATAINDEX"].astype(int)
    return df


def parse_tests(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
    return pd.DataFrame([t.attrib for t in root.findall(".//TEST")])


def read_data_blocks(file_path: Path, offset: int, max_blocks: int = 3000):
    blocks = {}
    with file_path.open("rb") as f:
        f.seek(offset)

        for idx in range(max_blocks):
            raw_len = f.read(4)
            if len(raw_len) < 4:
                break

            (n,) = struct.unpack("<I", raw_len)
            if n == 0 or n > 50_000_000:
                break

            raw = f.read(n * 8)
            if len(raw) < n * 8:
                break

            blocks[idx] = np.frombuffer(raw, dtype="<f8")

    return blocks


def build_column_map(channels: pd.DataFrame):
    col_map = {}
    for _, row in channels.iterrows():
        idx = row["DATAINDEX"]
        label = row.get("DISPLAYNAME") or row.get("NAME") or f"DATAINDEX_{idx}"
        if idx not in col_map:
            col_map[idx] = f"{label} [idx={idx}]"
    return col_map


# ---------------------------
# Main extractor
# ---------------------------
def extract_nmd(nmd_path: Path):
    print(f"Reading: {nmd_path}")

    out_dir = nmd_path.parent / f"{nmd_path.stem}_NMD_CSVs"
    out_dir.mkdir(exist_ok=True)

    xml_bytes, _, xml_end = find_sample_xml_block(nmd_path)

    channels = parse_channels(xml_bytes)
    tests = parse_tests(xml_bytes)

    channels.to_csv(out_dir / "channels.csv", index=False)
    tests.to_csv(out_dir / "tests.csv", index=False)

    blocks = read_data_blocks(nmd_path, xml_end)
    col_map = build_column_map(channels)

    data = {}
    for idx, arr in blocks.items():
        if idx in col_map:
            data[col_map[idx]] = arr
        else:
            data[f"UNMAPPED_DATAINDEX_{idx}"] = arr

    lengths = [len(v) for v in data.values()]
    nmin = min(lengths)
    if len(set(lengths)) != 1:
        data = {k: v[:nmin] for k, v in data.items()}

    df = pd.DataFrame(data)
    df.to_csv(out_dir / "data_test1.csv", index=False)

    print("Extraction complete.")
    print(f"CSV files written to: {out_dir}")


# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    nmd_file = pick_nmd_file()
    extract_nmd(nmd_file)
