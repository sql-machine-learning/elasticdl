import itertools
import os
import unittest

from elasticdl.python.tests.test_utils import (
    DatasetName,
    distributed_train_and_evaluate,
    create_pserver,
)

_model_zoo_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
)


class ExampleTest(unittest.TestCase):
    def _test_train(self, feature_shape, model_def, model_params="", dataset_name=DatasetName.IMAGE_DEFAULT):
        num_ps_pods = 2
        use_asyncs = [False, True]
        model_versions = []
        for use_async in use_asyncs:
            grads_to_wait = 1 if use_async else 2
            ps_channels, pservers = create_pserver(_model_zoo_path, model_def, grads_to_wait, use_async, num_ps_pods)
            try:
                model_version = distributed_train_and_evaluate(
                    feature_shape,
                    _model_zoo_path,
                    model_def,
                    model_params=model_params,
                    training=True,
                    dataset_name=dataset_name,
                    use_async=use_async,
                    ps_channels=ps_channels,
                    pservers=pservers,
                )
            finally:
                for pserver in pservers:
                    pserver.server.stop(0)
            model_versions.append(model_version)
        return model_versions

    def _test_evaluate(self, feature_shape, model_def, model_params="", dataset_name=DatasetName.IMAGE_DEFAULT):
        num_ps_pods = 2
        grads_to_wait = 1
        tf.keras.backend.clear_session()
        ps_channels, pservers = create_pserver(_model_zoo_path, model_def, grads_to_wait, False, num_ps_pods)
        try:
            model_version = distributed_train_and_evaluate(
                feature_shape, 
                _model_zoo_path,
                model_def,
                model_params=model_params,
                training=False,
                dataset_name=dataset_name,
                ps_channels=ps_channels,
                pservers=pservers,
            )
        finally:
            for pserver in pservers:
                pserver.server.stop(0)
        return model_version

    def test_deepfm_functional_train(self):
        self._test_train(
            10,
            "deepfm_functional_api.deepfm_functional_api.custom_model",
            "input_dim=5383;embedding_dim=4;input_length=10;fc_unit=4",
            dataset_name=DatasetName.FRAPPE,
        )

    def test_deepfm_functional_evaluate(self):
        self._test_evaluate(
            10,
            "deepfm_functional_api.deepfm_functional_api.custom_model",
            "input_dim=5383;embedding_dim=4;input_length=10;fc_unit=4",
            dataset_name=DatasetName.FRAPPE,
        )

    def test_mnist_train(self):
        model_defs = [
            "mnist_functional_api.mnist_functional_api.custom_model",
            "mnist_subclass.mnist_subclass.CustomModel",
        ]

        model_versions = []
        for model_def in model_defs:
            versions = self._test_train(
                feature_shape=[28,28],
                model_def=model_def,
            )

            model_versions.extend(versions)
        # async model version = sync model version * 2
        self.assertEqual(model_versions[0] * 2, model_versions[1])
        self.assertEqual(model_versions[2] * 2, model_versions[3])


    def _test_mnist_train(self):
        model_defs = [
            "mnist_functional_api.mnist_functional_api.custom_model",
            "mnist_subclass.mnist_subclass.CustomModel",
        ]
        use_asyncs = [False, True]
        configs = list(itertools.product(model_defs, use_asyncs))

        model_versions = []
        for config in configs:
            model_version = distributed_train_and_evaluate(
                [28, 28],
                _model_zoo_path,
                config[0],
                training=True,
                use_async=config[1],
            )
            model_versions.append(model_version)
        # async model version = sync model version * 2
        self.assertEqual(model_versions[0] * 2, model_versions[1])
        self.assertEqual(model_versions[2] * 2, model_versions[3])

if __name__ == "__main__":
    unittest.main()
