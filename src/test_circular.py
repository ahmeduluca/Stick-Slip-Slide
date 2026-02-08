#!/usr/bin/env python
import sys
import traceback

print("=" * 60)
print("Testing circular import issue")
print("=" * 60)

# Test 1: Import mechanics first (no fitting dependency)
print("\n1. Importing mechanics alone...")
try:
    import stick_slip_slide.mechanics as mech
    print("   SUCCESS")
    print(f"   Has {len([x for x in dir(mech) if not x.startswith('_') and callable(getattr(mech, x))])} public functions")
except Exception as e:
    print(f"   FAILURE: {e}")
    traceback.print_exc()

# Test 2: Import fitting (depends on mechanics)
print("\n2. Importing fitting alone...")
try:
    import stick_slip_slide.fitting as fit
    print("   SUCCESS")
    public_funcs = [x for x in dir(fit) if not x.startswith('_') and callable(getattr(fit, x))]
    print(f"   Has {len(public_funcs)} public functions")
    if public_funcs:
        print(f"   Functions: {public_funcs[:5]}")
except Exception as e:
    print(f"   FAILURE: {e}")
    traceback.print_exc()

# Test 3: Check what fitting module has
print("\n3. Checking fitting module contents...")
try:
    import stick_slip_slide.fitting as fit
    print(f"   fit.__dict__ keys: {list(fit.__dict__.keys())}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Try accessing a function directly
print("\n4. Trying to access mindlin_fit from fitting...")
try:
    import stick_slip_slide.fitting as fit
    func = getattr(fit, 'mindlin_fit', None)
    if func:
        print(f"   SUCCESS: mindlin_fit = {func}")
    else:
        print(f"   FAILURE: mindlin_fit not found")
except Exception as e:
    print(f"   ERROR: {e}")
    traceback.print_exc()
