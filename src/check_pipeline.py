import sys, traceback, importlib
sys.path.insert(0, '.')
importlib.invalidate_caches()

try:
    import stick_slip_slide.pipeline as pipeline
    print('Imported pipeline OK')
    print('analyze_one_file present:', hasattr(pipeline, 'analyze_one_file'))
    print('analyze_batch present:', hasattr(pipeline, 'analyze_batch'))
except Exception as e:
    print('IMPORT ERROR:', e)
    traceback.print_exc()
    raise
