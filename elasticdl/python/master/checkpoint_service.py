import os
import tempfile

from elasticdl.python.common.model_utils import (
    load_from_checkpoint_file,
    save_checkpoint_to_file,
)


class Checkpoint(object):
    def __init__(self, version, file):
        self.version = version
        self.file = file


class CheckpointService(object):
    """Checkpoint Service implementation"""

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
        return "%s/variables-%s-of-%s.chkpt" % (
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
        save_checkpoint_to_file(model, file)
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

    def remove_eval_checkpoint(self, version):
        chkpt_file = self._get_checkpoint_file(
            version, is_eval_checkpoint=True
        )
        os.remove(chkpt_file)

    def get_checkpoint_path(self, version):
        """Get the full path of the given checkpoint version"""
        chkpt_file = self._get_checkpoint_file(
            version, is_eval_checkpoint=False
        )
        if os.path.isfile(chkpt_file):
            return chkpt_file
        chkpt_file = self._get_checkpoint_file(
            version, is_eval_checkpoint=True
        )
        if os.path.isfile(chkpt_file):
            return chkpt_file
        return ""

    def get_checkpoint_model(self, version):
        """Read checkpoint using model version"""
        file = self.get_checkpoint_path(version)
        try:
            return load_from_checkpoint_file(file)
        except Exception:
            raise RuntimeError(
                "Failed to read model checkpoint from file " + file
            )

    def get_latest_checkpoint_version(self):
        """Get the latest checkpointed model version"""
        if not self._checkpoint_list:
            raise RuntimeError("No model checkpoint available")
        return self._checkpoint_list[-1].version


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
        if check_checkpoint_valid(folder_dir):
            return folder_dir
    return None


def check_checkpoint_valid(folder_dir):
    shard_files = os.listdir(folder_dir)
    if not shard_files:
        return False
    shard_file_prefix = shard_files[0].split(".")[0]
    expected_shard_num = int(shard_file_prefix.split("-")[-1])
    return expected_shard_num == len(shard_files)
