"""
The decorator to declare the input tensors of a keras model.
For keras subclass model, it has a core method `def call(self, inputs)`.
But we don't know how many tensors the model accepts just from `inputs`.
To solve this, we can use this decorators just as follows:

Example:
@declare_model_inputs("wide_embeddings,", "deep_embeddings")
class WideAndDeepClassifier(tf.keras.Model):
    def __init__(self):
        pass

    def call(self, inputs):
        pass

And then we can get the input tensor names from the property of the model class
`WideAndDeepClassifier._model_inputs` => ["wide_embeddings", "deep_embeddings"]
"""
def declare_model_inputs(*args):
    def decorator(func):
        input_names = list(args)
        if not input_names:
            raise ValueError("Model input names should not be empty.")

        if not all(isinstance(name, str) for name in input_names):
            raise ValueError("Model input names should be string type.")

        setattr(func, "_model_inputs", input_names)

        return func

    return decorator
