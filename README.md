# Stick-Slip-Slide (Oscillatory Shearing Method: Data Analysis)

Automated data analysis pipeline for oscillatory shearing experiments (2D indenter / nanoindentation + lateral actuation) targeting:
- pre-sliding (Mindlin partial slip) behavior
- stick–slip / transition to gross sliding
- friction scaling (shear strength vs pressure/area/contact radius)
- adhesion + roughness corrections and pull-off diagnostics (for specifically Hertzian contacts)

> This repository is part of Ahmed Uluca’s PhD work in Physics.  
> Status: research code (actively evolving).

---
**Physics-based analysis of oscillatory stick–slip and sliding experiments**

This repository provides a modular, reproducible analysis framework for **oscillatory shear experiments** used to study friction, partial slip, and sliding at small contact sizes.

The code is designed for **real experimental data**, emphasizing:
- contact mechanics consistency,
- machine compliance correction,
- stiffness-based stick–slip detection,
- and transparent, inspectable analysis decisions.

---

## 🔬 Scientific motivation

In nano- and micro-scale friction experiments, lateral force alone is not sufficient to characterize interfacial slip.  
This framework extracts **contact-level quantities** such as:

- contact radius and area regarding **adhesion-roughness** included Hertzian calculations /or **flattened sphere** by V-CSM,
- stick → slip → re-stick transitions: user-selectable from graphs (by picker functions),
- local shear strength (τ),
- pressure scaling,
- pre-sliding distance,
- dissipated energy per oscillation cycle.

The analysis is particularly suited for:
- 2D indentation / oscillatory shear setups, using simultaneous vertical and lateral CSM modes. (KLA Gemini 2D indenter)
---

## 🧠 Analysis philosophy

The pipeline follows the **physical sequence of the experiment**, not arbitrary signal processing steps:

1. Detect first contact
2. Correct machine compliance
3. Identify shear window
4. Detect oscillation cycles
5. Detect stick–slip transitions from stiffness loss
6. Infer contact area using elastic contact mechanics
7. Compute frictional and energetic quantities

Every step:
- has diagnostic plots,
- supports manual override,
- and fails explicitly when assumptions are violated.

---

## 📊 Analysis flow

```mermaid
flowchart TD
    A[Raw CSV files] --> B[Touch detection]
    B --> C[Normal load correction]
    C --> D[Shear window detection]
    D --> E[Cycle detection]
    E --> F[Stick–slip transitions]
    F --> G[Contact mechanics]
    G --> H[Friction & energy]
    H --> I[Reports & summaries]
...
````
## Repository structure:
```
Stick-Slip-Slide/
│
├─ src/stick_slip_slide/
│  ├─ cli.py               # Command-line entry point
│  ├─ __main__.py          # python -m stick_slip_slide
│  ├─ pipeline.py          # Main analysis orchestration
│  ├─ io.py                # CSV reading, units, GUI folder picker
│  ├─ signal.py            # Touch detection, cycle detection, transitions
│  ├─ mechanics.py         # Contact mechanics & physical calculations
│  ├─ fitting.py           # Hertz, adhesion, flat-end, Mindlin fits
│  ├─ plotting.py          # Diagnostic and summary plots
│  ├─ reporting.py         # Per-cycle and batch summaries
│  ├─ math_utils.py        # Numerical helpers
│  ├─ config.py            # Config dataclass (defaults)
│  └─ config_loader.py     # YAML → Config mapping
│
├─ config.example.yaml     # Example analysis configuration
├─ OSM-data_analysis.py    # Legacy monolithic script (reference only)
├─ pyproject.toml
└─ README.md
...
```

