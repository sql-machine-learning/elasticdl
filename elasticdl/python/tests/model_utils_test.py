import os
import unittest

from elasticdl.python.common.model_utils import (
    _get_spec_value,
    get_dict_from_params_str,
    get_model_spec,
    get_module_file_path,
)

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


class ModelHelperTest(unittest.TestCase):
    def test_get_model_spec(self):
        (
            model,
            dataset_fn,
            loss,
            optimizer,
            eval_metrics_fn,
            prediction_outputs_processor,
            custom_data_reader,
        ) = get_model_spec(
            model_zoo=_model_zoo_path,
            model_def="test_module.custom_model",
            dataset_fn="dataset_fn",
            loss="loss",
            optimizer="optimizer",
            eval_metrics_fn="eval_metrics_fn",
            model_params="",
            prediction_outputs_processor="PredictionOutputsProcessor",
            custom_data_reader="custom_data_reader",
        )

        self.assertTrue(model is not None)
        self.assertTrue(dataset_fn is not None)
        self.assertTrue(loss is not None)
        self.assertTrue(optimizer is not None)
        self.assertTrue(eval_metrics_fn is not None)
        self.assertTrue(prediction_outputs_processor is not None)
        self.assertTrue(custom_data_reader is not None)
        self.assertRaisesRegex(
            Exception,
            "Cannot find the custom model function/class "
            "in model definition files",
            get_model_spec,
            model_zoo=_model_zoo_path,
            model_def="test_module.undefined",
            dataset_fn="dataset_fn",
            loss="loss",
            optimizer="optimizer",
            eval_metrics_fn="eval_metrics_fn",
            model_params="",
            prediction_outputs_processor="PredictionOutputsProcessor",
            custom_data_reader="custom_data_reader",
        )

    def test_get_module_file_path(self):
        self.assertEqual(
            get_module_file_path(_model_zoo_path, "test_module.custom_model"),
            os.path.join(_model_zoo_path, "test_module.py"),
        )

    def test_get_spec_value(self):
        self.assertTrue(
            _get_spec_value(
                "custom_model", _model_zoo_path, {"custom_model": 1}
            )
            is not None
        )
        self.assertTrue(
            _get_spec_value("test_module.custom_model", _model_zoo_path, {})
            is not None
        )
        self.assertTrue(
            _get_spec_value("test_module.unknown_model", _model_zoo_path, {})
            is None
        )
        self.assertRaisesRegex(
            Exception,
            "Missing required spec key unknown_model "
            "in the module: test_module.unknown_model",
            _get_spec_value,
            "test_module.unknown_model",
            _model_zoo_path,
            {},
            True,
        )

    def test_get_dict_from_params_str(self):
        self.assertEqual(
            get_dict_from_params_str('ls=["a", "b"]'), {"ls": ["a", "b"]}
        )
        self.assertEqual(
            get_dict_from_params_str('ls=["a", "b"]; d={"a": 3}'),
            {"ls": ["a", "b"], "d": {"a": 3}},
        )
        self.assertEqual(
            get_dict_from_params_str('ls=["a", "b"];partition=dt=20190011'),
            {"ls": ["a", "b"], "partition": "dt=20190011"},
        )
        self.assertEqual(get_dict_from_params_str(""), {})


if __name__ == "__main__":
    unittest.main()
