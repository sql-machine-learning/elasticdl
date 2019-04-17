# common helper methods for model manipulation.
import os

def load_user_model(model_file, model_class):
    with add_to_path(os.path.dirname(absolute_path)):
        spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, model_class)
