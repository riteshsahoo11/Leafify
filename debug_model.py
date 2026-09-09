import sys
sys.path.append('backend')
try:
    from backend.main import load_model
    load_model()
except Exception as e:
    import traceback
    traceback.print_exc()
