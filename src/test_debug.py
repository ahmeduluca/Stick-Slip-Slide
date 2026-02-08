#!/usr/bin/env python
import sys
import traceback

print("Testing with debug output in fitting.py")

# Temporarily modify fitting.py to see where it stops
fitting_path = 'stick_slip_slide/fitting.py'

with open(fitting_path, 'r') as f:
    content = f.read()

# Check if debug lines are already there
if '# DEBUG:' not in content:
    # Add debug output after the imports section (around line 32)
    lines = content.split('\n')
    
    # Find the end of imports (look for first function def)
    insert_line = 0
    for i, line in enumerate(lines):
        if line.startswith('def '):
            insert_line = i
            break
    
   if insert_line > 0:
        lines.insert(insert_line, '# DEBUG: About to define functions')
        lines.insert(insert_line + 1, 'print("DEBUG: fitting.py reached function definitions", file=__import__("sys").stderr)')
        
        # Re-write the file
        with open(fitting_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print("Added debug output to fitting.py")

# Now try importing
print("\nImporting fitting with debug output...")
try:
    import stick_slip_slide.fitting as fit
    print("Import successful")
    print(f"Functions found: {len([x for x in dir(fit) if not x.startswith('_') and callable(getattr(fit, x))])}")
except Exception as e:
    print(f"Import failed: {e}")
    traceback.print_exc()
