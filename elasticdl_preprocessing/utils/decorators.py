def model_input_name(*args):
    def decorator(clz):
        input_names = list(args)
        if not input_names:
            raise ValueError("Input names should not be empty")

        if not all(isinstance(name, str) for name in input_names):
            raise ValueError("Input names should be string type")

        setattr(clz, "_model_input_names", input_names)

        return clz

    return decorator
