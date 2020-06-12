# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import shutil
import tempfile

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.tensor_utils import (
    pb_to_indexed_slices,
    pb_to_ndarray,
)
from elasticdl.python.ps.embedding_table import create_embedding_table
from elasticdl.python.ps.parameters import Parameters


def save_pb_to_file(pb_obj, file_name):
    """Save a protobuf object to file"""
    encoded_model = pb_obj.SerializeToString()
    with open(file_name, "wb") as f:
        f.write(encoded_model)


def load_pb_from_file(pb_obj, file_name):
    """Load a protobuf object from a file"""
    with open(file_name, "rb") as f:
        pb_obj.ParseFromString(f.read())
    return pb_obj


def _get_params_shard_from_pb(model_pb, shard_index, shard_num):
    """Get parameters including variables values and embedding table
    from a model protobuf.
    Args:
        model_pb: A Model protobuf instance.
        shard_index: Model shard index.
        shard_num: The total number of model shards.
    Return:
        non_embedding_vars: A Python dict in which the key is a variable
            name and the value is a `tf.Variable` object.
        embedding_table_values: A Python dict in which the key is an embedding
            table name and the value is a tuple with 2 elements. The value[0]
            is indices and value[1] is the corresponding embedding vector.
    """
    non_embedding_vars = {}
    embedding_table_values = {}

    for name, pb in model_pb.dense_parameters.items():
        if string_to_id(name, shard_num) == shard_index:
            non_embedding_vars[name] = tf.Variable(
                initial_value=pb_to_ndarray(pb), trainable=True
            )
    for name, pb in model_pb.embedding_tables.items():
        embedding_table_values.setdefault(name, ([], []))
        t = pb_to_indexed_slices(pb)
        for embedding_id, vector in zip(t.indices, t.values):
            if int_to_id(embedding_id, shard_num) == shard_index:
                embedding_table_values[name][0].append(embedding_id)
                embedding_table_values[name][1].append(vector)
    return non_embedding_vars, embedding_table_values


def save_checkpoint_without_embedding(model, checkpoint_dir, version=100):
    checkpoint_saver = CheckpointSaver(checkpoint_dir, 0, 0, False)
    params = Parameters()
    for var in model.trainable_variables:
        params.non_embedding_params[var.name] = var
    params.version = version
    model_pb = params.to_model_pb()
    checkpoint_saver.save(version, model_pb, False)


class Checkpoint(object):
    def __init__(self, version, file):
        self.version = version
        self.file = file


class CheckpointSaver(object):
    """Checkpoint Saver implementation"""

    def __init__(
        self,
        checkpoint_dir,
        checkpoint_steps,
        keep_checkpoint_max,
        include_evaluation,
    ):
        """
        Arguments:
            checkpoint_dir: The directory to store the checkpoint files.
                            Directory will be created if not exist.
            checkpoint_steps: Save checkpoint every this many steps.
            keep_checkpoint_max: The maximum number of recent checkpoint
                                 files to keep.
        """
        self._directory = checkpoint_dir
        self._steps = checkpoint_steps
        self._max_versions = keep_checkpoint_max
        if not self._directory:
            self._directory = os.getcwd() + "/checkpoint_dir"
        if self._steps:
            os.makedirs(self._directory, exist_ok=True)
        self._checkpoint_dir_list = []
        self._include_evaluation = include_evaluation
        self._eval_checkpoint_dir = (
            tempfile.mkdtemp() if include_evaluation else ""
        )

    def _get_checkpoint_file(
        self, version, is_eval_checkpoint=False, shard_index=0, shard_num=1
    ):
        checkpoint_dir = (
            self._eval_checkpoint_dir
            if is_eval_checkpoint
            else self._directory
        )
        checkpoint_version_dir = os.path.join(
            checkpoint_dir, "version-%s" % str(version)
        )
        with contextlib.suppress(FileExistsError):
            os.makedirs(checkpoint_version_dir, exist_ok=True)
        return "%s/variables-%s-of-%s.ckpt" % (
            checkpoint_version_dir,
            str(shard_index),
            str(shard_num),
        )

    def is_enabled(self):
        """Checkpoint is enabled or not"""
        return self._steps

    def need_to_checkpoint(self, version):
        """Check if the given model version needs to be checkpointed"""
        return self.is_enabled() and version % self._steps == 0

    def save(
        self, version, model, is_eval_checkpoint, shard_index=0, shard_num=1
    ):
        """Checkpoint the given model

        Args:
            version (int): iteration steps
            model: a model protobuf instance
            is_eval_checkpoint (bool): if True, the model will be saved to
                a temporary directory.
            shard_index (int): default 0. The shard index in all
                model shard files, e.g. the shard_index is PS instance index
                using ParameterServerStrategy.
            shard_number (int): default 1. The number of model shards,
                e.g. shard_number is the number of PS instances using
                ParameterServerStrategy.
        """
        filename = self._get_checkpoint_file(
            version, is_eval_checkpoint, shard_index, shard_num
        )
        save_pb_to_file(model, filename)
        if not is_eval_checkpoint:
            self._checkpoint_dir_list.append(os.path.dirname(filename))
            if self._max_versions:
                self._delete_old_checkpoints_if_needed()

    def _delete_old_checkpoints_if_needed(self):
        """Delete the oldest checkpoint files and keep the number of
        checkpoints is not beyond max_version.
        """
        if len(self._checkpoint_dir_list) > self._max_versions:
            old_version_dir = self._checkpoint_dir_list[0]

            # Some PS instances have not saved checkpoint shard files of
            # the version if invalid. And the slowest PS will remove the
            # old version checkpoint.
            if self.check_checkpoint_valid(old_version_dir):
                self._checkpoint_dir_list.pop(0)
                with contextlib.suppress(FileNotFoundError):
                    shutil.rmtree(old_version_dir)

    @staticmethod
    def get_valid_lastest_version_dir(checkpoint_dir):
        """Get the valid and lastest version checkpoint directory"""
        if not checkpoint_dir or not os.path.exists(checkpoint_dir):
            return None

        version_folders = os.listdir(checkpoint_dir)
        if not version_folders:
            return None
        version_num = [int(v.split("-")[-1]) for v in version_folders]
        version_folder_pairs = sorted(
            zip(version_num, version_folders), reverse=True
        )
        for version, folder in version_folder_pairs:
            folder_dir = os.path.join(checkpoint_dir, folder)
            if CheckpointSaver.check_checkpoint_valid(folder_dir):
                return folder_dir
        return None

    @staticmethod
    def check_checkpoint_valid(checkpoint_dir):
        """Check whether the checkpoint directory is valid. The filename template
        in the checkpoint directory like "variables-{i}-of-{N}.ckpt". We will
        parse any filename to get N which is the total number of parameters
        shards. It is valid if the number of files in the directory N.
        """
        if not os.path.exists(checkpoint_dir):
            return False

        shard_files = os.listdir(checkpoint_dir)
        if not shard_files:
            return False

        shard_file_prefix = shard_files[0].split(".")[0]
        expected_shard_num = int(shard_file_prefix.split("-")[-1])
        return expected_shard_num == len(shard_files)

    @staticmethod
    def restore_params_from_checkpoint(checkpoint_dir, shard_index, shard_num):
        """Restore a shard parameters from the checkpoint directory.
        If shard_num=1, a entire model parameters will be restored.

        Args:
            checkpoint_dir: a directory with checkpoint files.
            shard_index: Model shard index, e.g. the PS instance index
                using ParameterServerStrategy with multiple PS instances.
            shard_num: The total number of model shards, e.g. the total PS
                instancecount using ParameterServerStrategy with multiple
                PS instances.

        Return:
            parameters: A Parameter object which contains model version,
                non-embedding parameters and embedding tables for the
                PS instance with ps_id.
        """

        variable_shard_files = os.listdir(checkpoint_dir)
        non_embedding_vars = {}
        embedding_tables = {}
        version = None
        for shard_file in variable_shard_files:
            shard_file_path = os.path.join(checkpoint_dir, shard_file)
            model_pb = elasticdl_pb2.Model()
            model_pb = load_pb_from_file(model_pb, shard_file_path)
            if version is None:
                version = model_pb.version
            elif version != model_pb.version:
                raise ValueError(
                    "The versions in model shards are not consistent"
                )

            for embedding_info_pb in model_pb.embedding_table_infos:
                embedding_table = create_embedding_table(embedding_info_pb)
                embedding_tables.setdefault(
                    embedding_table.name, embedding_table
                )

            (
                shard_non_embedding_vars,
                shard_embedding_table_values,
            ) = _get_params_shard_from_pb(model_pb, shard_index, shard_num)

            non_embedding_vars.update(shard_non_embedding_vars)
            for name, pair in shard_embedding_table_values.items():
                embedding_tables[name].set(pair[0], pair[1])

        parameters = Parameters()
        parameters.non_embedding_params.update(non_embedding_vars)
        parameters.embedding_params.update(embedding_tables)
        parameters.version = version
        return parameters

    @staticmethod
    def get_version_from_checkpoint(checkpoint_dir):
        """Get model version from the checkpoint. There may be several shard
        files in the checkpoint directory. The model versions of shard files
        are same, so we only need to read one shard file to get model version.
        """
        variable_shard_files = os.listdir(checkpoint_dir)
        shard_file_path = os.path.join(checkpoint_dir, variable_shard_files[0])
        model_pb = elasticdl_pb2.Model()
        model_pb = load_pb_from_file(model_pb, shard_file_path)
        return model_pb.version
