#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')

print("Detailed module inspection")
print("=" * 60)

import stick_slip_slide.fitting as fit

print(f"\nfitting.__dict__ keys: {list(fit.__dict__.keys())}")
print(f"\nTotal items in __dict__: {len(fit.__dict__)}")

# Check the actual content
for key in fit.__dict__:
    val = fit.__dict__[key]
    print(f"  {key}: {type(val)} = {repr(val)[:60]}")

# Now check what dir() says
print(f"\ndir(fitting):")
d = dir(fit)
print(f"Total: {len(d)}")
print(f"Non-private: {len([x for x in d if not x.startswith('_')])}")
print(f"Non-private items: {[x for x in d if not x.startswith('_')]}")

# Check if imports are in the namespace
print(f"\nChecking for imported names:")
print(f"  'np' in __dict__: {'np' in fit.__dict__}")
print(f"  'pd' in __dict__: {'pd' in fit.__dict__}")
print(f"  'Config' in __dict__: {'Config' in fit.__dict__}")
print(f"  'hertz_load_from_h' in __dict__: {'hertz_load_from_h' in fit.__dict__}")
print(f"  'mindlin_fit' in __dict__: {'mindlin_fit' in fit.__dict__}")
