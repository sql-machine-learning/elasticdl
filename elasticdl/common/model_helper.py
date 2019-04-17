# common helper methods for model manipulation.
import os
from contextlib import contextmanager

def load_user_model(model_file, model_class):
    with _add_to_path(os.path.dirname(absolute_path)):
        spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, model_class)

@contextmanager
def _add_to_path(p):
    import sys

    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path
