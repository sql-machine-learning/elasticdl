# common helper methods for model manipulation.
import importlib.util


def load_user_model(model_file):
    spec = importlib.util.spec_from_file_location(model_file, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
