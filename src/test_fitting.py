#!/usr/bin/env python
import sys
import importlib.util

print("=" * 60)
print("Testing fitting.py import...")
print("=" * 60)

# First, try normal import
try:
    import stick_slip_slide.fitting as fit
    fit_attrs = [x for x in dir(fit) if not x.startswith('_')]
    print(f"\n1. Normal import successful")
    print(f"   Public attributes: {fit_attrs[:10] if fit_attrs else 'NONE'}")
    print(f"   Total public: {len(fit_attrs)}")
except Exception as e:
    print(f"\n1. Normal import FAILED: {e}")
    import traceback
    traceback.print_exc()

# Second, test if fitting.py file can be compiled
print("\n" + "=" * 60)
print("Testing fitting.py syntax...")
try:
    with open('stick_slip_slide/fitting.py', 'r') as f:
        code = compile(f.read(), 'fitting.py', 'exec')
    print("2. fitting.py compiles successfully")
except SyntaxError as e:
    print(f"2. fitting.py has syntax error: {e}")

# Third, check if mechanics.py has circular import issues
print("\n" + "=" * 60)
print("Testing mechanics.py import...")
try:
    import stick_slip_slide.mechanics as mech
    mech_attrs = [x for x in dir(mech) if not x.startswith('_') and callable(getattr(mech, x))]
    print(f"3. mechanics import successful")
    print(f"   Callable attributes: {len(mech_attrs)}")
except Exception as e:
    print(f"3. mechanics import FAILED: {e}")
    import traceback
    traceback.print_exc()

# Fourth, check pipeline
print("\n" + "=" * 60)
print("Testing pipeline.py import...")
try:
    import stick_slip_slide.pipeline as pipeline
    print("4. pipeline import successful")
except Exception as e:
    print(f"4. pipeline import FAILED: {e}")
    import traceback
    traceback.print_exc()
