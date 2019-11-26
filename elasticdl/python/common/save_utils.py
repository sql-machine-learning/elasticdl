import os
import tempfile

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.tensor import Tensor
from elasticdl.python.ps.parameters import Parameters


def save_pb_to_file(pb_model, file_name):
    encoded_model = pb_model.SerializeToString()
    with open(file_name, "wb") as f:
        f.write(encoded_model)


def load_pb_from_file(file_name):
    pb_model = elasticdl_pb2.Model()
    with open(file_name, "rb") as f:
        pb_model.ParseFromString(f.read())
    return pb_model


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

    for tensor_pb in model_pb.param:
        tensor = Tensor.from_tensor_pb(tensor_pb)
        if tensor.indices is not None:
            embedding_table_values.setdefault(tensor.name, ([], []))
            for embedding_id, vector in zip(tensor.indices, tensor.values):
                if int_to_id(embedding_id, shard_num) == shard_index:
                    embedding_table_values[tensor.name][0].append(embedding_id)
                    embedding_table_values[tensor.name][1].append(vector)
        else:
            if string_to_id(tensor.name, shard_num) == shard_index:
                non_embedding_vars[tensor.name] = tf.Variable(
                    initial_value=tensor.values, trainable=True
                )
    return non_embedding_vars, embedding_table_values


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
        self._checkpoint_list = []
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
            model: a pb_model
            is_eval_checkpoint (bool): if True, the model will be saved to
                a temporary directory.
            shard_index (int): default 0. The shard index in all
                model shard files, e.g. the shard_index is PS instance index
                using ParameterServerStrategy.
            shard_number (int): default 1. The number of model shards,
                e.g. shard_number is the number of PS instances using
                ParameterServerStrategy.
        """
        file = self._get_checkpoint_file(
            version, is_eval_checkpoint, shard_index, shard_num
        )
        save_pb_to_file(model, file)
        if not is_eval_checkpoint:
            self._checkpoint_list.append(Checkpoint(version, file))
            if self._max_versions:
                while len(self._checkpoint_list) > self._max_versions:
                    file_to_delete = self._checkpoint_list.pop(0).file
                    os.remove(file_to_delete)
                    # Remove the directory if empty
                    delete_dir_name = os.path.dirname(file_to_delete)
                    if not os.listdir(delete_dir_name):
                        try:
                            os.rmdir(delete_dir_name)
                        except Exception:
                            pass

    def get_latest_checkpoint_version(self):
        """Get the latest checkpointed model version"""
        if not self._checkpoint_list:
            raise RuntimeError("No model checkpoint available")
        return self._checkpoint_list[-1].version

    @staticmethod
    def get_valid_lastest_version_dir(checkpoint_dir):
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
        from elasticdl.python.ps.embedding_table import create_embedding_table

        variable_shard_files = os.listdir(checkpoint_dir)
        non_embedding_vars = {}
        embedding_tables = {}
        version = None
        for shard_file in variable_shard_files:
            shard_file_path = os.path.join(checkpoint_dir, shard_file)
            model_pb = load_pb_from_file(shard_file_path)
            if version is None:
                version = model_pb.version
            elif version != model_pb.version:
                raise ValueError(
                    "The versions in model shards are not consistency"
                )

        for embedding_info_pb in model_pb.embedding_table_info:
            embedding_table = create_embedding_table(embedding_info_pb)
            embedding_tables.setdefault(embedding_table.name, embedding_table)

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
