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

    def _get_checkpoint_file(self, version, is_eval_checkpoint=False):
        return "%s/model_v%s.chkpt" % (
            self._eval_checkpoint_dir
            if is_eval_checkpoint
            else self._directory,
            str(version),
        )

    def is_enabled(self):
        """Checkpoint is enabled or not"""
        return self._steps

    def need_to_checkpoint(self, version):
        """Check if the given model version needs to be checkpointed"""
        return self.is_enabled() and version % self._steps == 0

    def save(self, version, model, is_eval_checkpoint):
        """Checkpoint the given model"""
        file = self._get_checkpoint_file(version, is_eval_checkpoint)
        save_checkpoint_to_file(model, file)
        if not is_eval_checkpoint:
            self._checkpoint_list.append(Checkpoint(version, file))
            if self._max_versions:
                while len(self._checkpoint_list) > self._max_versions:
                    file_to_delete = self._checkpoint_list.pop(0).file
                    os.remove(file_to_delete)

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
