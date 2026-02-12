"""
Stick–Slip–Slide

Core package for processing oscillatory shearing experiments into
physically comparable quantities (contact area, pressure, shear strength,
friction, dissipation, transitions).
"""

# Delayed imports: avoid importing heavy submodules at package import time
# to prevent circular import issues during development. Import submodules
# explicitly (e.g. `from stick_slip_slide import pipeline`) when needed.

__version__ = "0.1.0"
