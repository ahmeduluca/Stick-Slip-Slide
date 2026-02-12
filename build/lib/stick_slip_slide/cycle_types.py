# cycle_types.py
from dataclasses import dataclass

@dataclass(frozen=True)
class CycleBounds:
    cycle: int
    i_start: int
    i_peak: int
    i_hold0: int
    i_hold1: int
    i_end: int