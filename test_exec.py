#!/usr/bin/env python
import sys
import traceback

sys.path.insert(0, 'src')

print("=" * 60)
print("Method 1: Direct exec with error capture")
print("=" * 60)

code = open('src/stick_slip_slide/fitting.py').read()
namespace = {}

try:
    exec(code, namespace)
    print("✓ Exec completed without exception")
except Exception as e:
    print(f"✗ Exception during exec:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {e}")
    traceback.print_exc()

functions_in_namespace = [k for k in namespace.keys() if callable(namespace[k]) and not k.startswith('_')]
print(f"\nFunctions in exec namespace: {len(functions_in_namespace)}")
if functions_in_namespace:
    print(f"Functions: {functions_in_namespace[:5]}")

print("\n" + "=" * 60)
print("Method 2: Import with linecache to see source")
print("=" * 60)

try:
    import stick_slip_slide.fitting as fit_module
    print("✓ Module imported")
    attrs = dir(fit_module)
    public = [a for a in attrs if not a.startswith('_')]
    print(f"Public attributes: {len(public)}")
    
    # Try to manually get a function
    try:
        mindlin = getattr(fit_module, 'mindlin_fit')
        print(f"✓ Got mindlin_fit: {mindlin}")
    except AttributeError as e:
        print(f"✗ AttributeError: {e}")
        
except Exception as e:
    print(f"✗ Import failed: {e}")
    traceback.print_exc()
