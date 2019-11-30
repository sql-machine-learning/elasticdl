import itertools
import os
import unittest

from elasticdl.python.tests.test_utils import (
    DatasetName,
    distributed_train_and_evaluate,
)

_model_zoo_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
)


class ExampleTest(unittest.TestCase):
    def test_deepfm_functional_train(self):
        model_params = (
            "input_dim=5383;embedding_dim=4;input_length=10;fc_unit=4"
        )
        use_asyncs = [False, True]
        for use_async in use_asyncs:
            distributed_train_and_evaluate(
                10,
                _model_zoo_path,
                "deepfm_functional_api.deepfm_functional_api.custom_model",
                model_params=model_params,
                training=True,
                dataset_name=DatasetName.FRAPPE,
                use_async=use_async,
            )

if __name__ == "__main__":
    unittest.main()
