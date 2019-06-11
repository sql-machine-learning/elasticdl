import os
import logging
import traceback

import tensorflow as tf
assert tf.executing_eagerly()

import recordio
import numpy as np

from contextlib import closing
from collections import defaultdict
from elasticdl.python.elasticdl.common.ndarray import tensor_to_ndarray
from elasticdl.python.elasticdl.common.model_helper import (
    load_user_model,
    build_model,
    load_from_checkpoint_file,
)
from elasticdl.python.data.codec import TFExampleCodec
from elasticdl.python.data.codec import BytesCodec


class Evaluator(object):
    """ElasticDL evaluator"""

    def __init__(self,
                 model_file,
                 trained_model,
                 data_dir,
                 codec_type=None,
                 batch_size=10):
        """
        Arguments:
            model_file: A module to define the model
            max_minibatch_retry_num: The maximum number of a minibatch retry as its results
                (e.g. gradients) are not accepted by master.
        """
        self._logger = logging.getLogger(__name__)
        model_module = load_user_model(model_file)
        self._model = model_module.model
        self._trained_model = trained_model
        self._data_dir = data_dir
        self._feature_columns = model_module.feature_columns()
        build_model(self._model, self._feature_columns)
        self._input_fn = model_module.input_fn 
        self._opt_fn = model_module.optimizer
        self._loss = model_module.loss
        self._eval_metrics_fn = model_module.eval_metrics_fn
        all_columns = self._feature_columns + model_module.label_columns()
        if codec_type == "tf_example":
            self._codec = TFExampleCodec(all_columns)
        elif codec_type == "bytes":
            self._codec = BytesCodec(all_columns)
        else:
            raise ValueError("invalid codec_type: " + codec_type)

        self._codec_type = codec_type
        self._batch_size = batch_size
        self._loss = 0
        self._accuracy = 0
        self._num_samples = 0

    def _initialize_model(self):
        pb_model = load_from_checkpoint_file(self._trained_model)
        for var in self._model.trainable_variables:
            var.assign(tensor_to_ndarray(pb_model.param[var.name]))

    @staticmethod
    def _get_batch(reader, batch_size, decode):
        res = []
        for i in range(batch_size):
            record = reader.record()
            if record is None:
                break
            res.append(decode(record))
        return res

    def _get_features_and_labels_from_record(self, record_buf):
        batch_input_data, batch_labels = self._input_fn(record_buf)
        features = [batch_input_data[f_col.key] for f_col in self._feature_columns]
        if len(features) == 1:
            features = features[0]
        return features, batch_labels

    def run(self):
        # Initialize model from checkpoint
        self._initialize_model()
        for file_name in os.listdir(self._data_dir):
            self._logger.info("evaluating file " + file_name)            
            with closing(recordio.Scanner(self._data_dir + "/" + file_name)) as reader:
                while True:
                    record_buf = self._get_batch(reader, self._batch_size, self._codec.decode)
                    if not record_buf:
                        break
                    features, labels = self._get_features_and_labels_from_record(record_buf)
                    outputs = self._model.call(features, training=False)
                    evaluation_metrics = self._eval_metrics_fn(outputs, labels)
                    self._loss = self._loss + evaluation_metrics['loss']
                    self._accuracy = self._accuracy + evaluation_metrics['acc']
                    self._num_samples = self._num_samples + 1
        avg_loss = self._loss / self._num_samples
        avg_accuracy = self._accuracy / self._num_samples
        self._logger.info("Model loss: %f accuracy: %f" % (avg_loss, avg_accuracy))
