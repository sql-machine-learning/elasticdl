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

    def test_deepfm_functional_evaluate(self):
        model_params = (
            "input_dim=5383;embedding_dim=4;input_length=10;fc_unit=4"
        )
        distributed_train_and_evaluate(
            10,
            _model_zoo_path,
            "deepfm_functional_api.deepfm_functional_api.custom_model",
            model_params=model_params,
            training=False,
            dataset_name=DatasetName.FRAPPE,
        )

    def test_mnist_train(self):
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

    def test_mnist_evaluate(self):
        model_defs = [
            "mnist_functional_api.mnist_functional_api.custom_model",
            "mnist_subclass.mnist_subclass.CustomModel",
        ]
        for model_def in model_defs:
            distributed_train_and_evaluate(
                [28, 28], _model_zoo_path, model_def, training=False
            )

    def test_cifar10_train(self):
        model_defs = [
            "cifar10_functional_api.cifar10_functional_api.custom_model",
            "cifar10_subclass.cifar10_subclass.CustomModel",
        ]
        use_asyncs = [False, True]
        configs = list(itertools.product(model_defs, use_asyncs))

        model_versions = []
        for config in configs:
            model_version = distributed_train_and_evaluate(
                [32, 32, 3],
                _model_zoo_path,
                config[0],
                training=True,
                use_async=config[1],
            )
            model_versions.append(model_version)
        # async model version = sync model version * 2
        self.assertEqual(model_versions[0] * 2, model_versions[1])
        self.assertEqual(model_versions[2] * 2, model_versions[3])

    def test_cifar10_evaluate(self):
        model_defs = [
            "cifar10_functional_api.cifar10_functional_api.custom_model",
            "cifar10_subclass.cifar10_subclass.CustomModel",
        ]
        for model_def in model_defs:
            distributed_train_and_evaluate(
                [32, 32, 3], _model_zoo_path, model_def, training=False
            )

    def test_resnet50_subclass_train(self):
        use_asyncs = [False, True]
        for use_async in use_asyncs:
            distributed_train_and_evaluate(
                [224, 224, 3],
                _model_zoo_path,
                "resnet50_subclass.resnet50_subclass.CustomModel",
                training=True,
                dataset_name=DatasetName.IMAGENET,
                use_async=use_async,
            )

    def test_resnet50_subclass_evaluate(self):
        distributed_train_and_evaluate(
            [224, 224, 3],
            _model_zoo_path,
            "resnet50_subclass.resnet50_subclass.CustomModel",
            model_params='num_classes=10;dtype="float32"',
            training=False,
            dataset_name=DatasetName.IMAGENET,
        )


if __name__ == "__main__":
    unittest.main()
