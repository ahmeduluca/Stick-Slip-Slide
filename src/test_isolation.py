#!/usr/bin/env python
import sys
import astimport traceback

# Test: manually load the module to see where it errors
module_code = None
with open('stick_slip_slide/fitting.py', 'r') as f:
    module_code = f.read()

print("=== Testing compilation ===")
try:
    compile(module_code, 'fitting.py', 'exec')
    print("✓ Compiles OK")
except SyntaxError as e:
    print(f"✗ Compile error: {e}")
    sys.exit(1)

print("\n=== Testing import in isolation ===")
# Isolate fitting import in a fresh namespace
namespace = {}
try:
    import sys
    sys.path.insert(0, '.')
    import stick_slip_slide.mechanics  # load mechanics first
    print("✓ mechanics imported OK")
except Exception as e:
    print(f"✗ mechanics failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    import stick_slip_slide.fitting  # now load fitting
    print("✓ fitting imported OK")
except Exception as e:
    print(f"✗ fitting failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Testing attributes ===")
import stick_slip_slide.fitting as fit
functions = [x for x in dir(fit) if callable(getattr(fit, x)) and not x.startswith('_')]
print(f"Public functions found: {len(functions)}")
if functions:
    print(f"First 5: {functions[:5]}")
else:
    print("✗ NO FUNCTIONS FOUND")
    print(f"\nModule __dict__ keys: {list(fit.__dict__.keys())}")
