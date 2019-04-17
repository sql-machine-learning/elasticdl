# common helper methods for model manipulation.
import importlib.util
import sys
import os


def load_user_model(model_file, model_class):
    print('-------')
    print(model_file)
    print(model_class)
    spec = importlib.util.spec_from_file_location(model_file, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, model_class)
