#!/usr/bin/env python
import sys
import traceback

sys.path.insert(0, 'src')

print("Testing exec with proper globals context")
print("=" * 60)

code = open('src/stick_slip_slide/fitting.py').read()

# Try exec without any globals (bare namespace)
print("\n1. Exec with empty namespace:")
namespace1 = {}
try:
    exec(code, namespace1)
    print("   Completed without exception")
except Exception as e:
    print(f"   Exception: {e}")

functions1 = [k for k in namespace1.keys() if callable(namespace1[k]) and not k.startswith('_')]
print(f"   Functions: {len(functions1)}")

# Try exec with __builtins__ + imports
print("\n2. Exec with proper globals (imports first):")
namespace2 = {'__builtins__': __builtins__, '__name__': '__main__'}

# Pre-import dependencies
try:
    import numpy as np
    import pandas as pd
    namespace2['np'] = np
    namespace2['pd'] = pd
    print("   ✓ numpy, pandas available")
except Exception as e:
    print(f"   ✗ Failed to import numpy/pandas: {e}")

# Now try exec
try:
    exec(code, namespace2)
    print("   Completed without exception")
except Exception as e:
    print(f"   Exception: {type(e).__name__}: {e}")
    traceback.print_exc()

functions2 = [k for k in namespace2.keys() if callable(namespace2[k]) and not k.startswith('_') and k != '__builtins__']
print(f"   Functions found: {len(functions2)}")
if functions2:
    print(f"   Examples: {functions2[:5]}")

# Try actual import to see if it works
print("\n3. Testing actual module import:")
try:
    import stick_slip_slide.fitting as fit
    print("   ✓ Import successful")
    funcs = [x for x in dir(fit) if callable(getattr(fit, x)) and not x.startswith('_')]
    print(f"   Functions from import: {len(funcs)}")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    traceback.print_exc()
