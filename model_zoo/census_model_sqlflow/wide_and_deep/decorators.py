def model_input_name(input_names):
    def decorator(clz):
        if isinstance(input_names, str):
            setattr(clz, "_model_input_names", [input_names])
        elif isinstance(input_names, list) and all(isinstance(name, str) for name in names):
            setattr(clz, "_model_input_names", input_names)

        return clz

    return decorator
