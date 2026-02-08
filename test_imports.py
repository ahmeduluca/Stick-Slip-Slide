#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')

print("Testing imports from fitting.py")
print("=" * 60)

# Test each import statement from fitting.py one by one
print("\n1. Basic imports (numpy, pandas):")
try:
    import numpy as np
    import pandas as pd
    from typing import Dict, Optional, Callable, Any
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ {e}")

print("\n2. Config import (circular potential):")
try:
    from stick_slip_slide.config import Config
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ {e}")

print("\n3. math_utils imports:")
try:
    from stick_slip_slide.math_utils import (
        _num, robust_fit_line, rms_to_peak,
        safe_nanmax, safe_nanmin, filter_kwargs_for_callable,
        mindlin_model,
    )
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ {e}")

print("\n4. mechanics imports (circular!):")
try:
    from stick_slip_slide.mechanics import (
        hertz_load_from_h,
        jkr_P_from_h,
        tabor_mu,
        c_from_tabor,
        auto_model_from_mu,
        w_eff_from_roughness,
    )
    print("   ✓ Success")
    print(f"   Got {sum([hertz_load_from_h, jkr_P_from_h, tabor_mu, c_from_tabor, auto_model_from_mu, w_eff_from_roughness] != (None,))} functions")
except Exception as e:
    print(f"   ✗ {e}")

print("\n5. SciPy import (optional):")
try:
    from scipy.optimize import curve_fit
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ {e}")

print("\n6. Now try to import fitting module itself:")
try:
    import stick_slip_slide.fitting as fit
    print("   ✓ Module imported")
    print(f"   __dict__ has {len(fit.__dict__)} items")
    funcs = [x for x in dir(fit) if callable(getattr(fit, x)) and not x.startswith('_')]
    print(f"   Public callables: {len(funcs)}")
except Exception as e:
    print(f"   ✗ {e}")
    import traceback
    traceback.print_exc()
