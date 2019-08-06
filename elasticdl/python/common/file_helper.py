import os
import shutil

from elasticdl.python.common.log_util import default_logger as logger


def copy_if_not_exists(src, dst, is_dir):
    if os.path.exists(dst):
        logger.info(
            "Skip copying from %s to %s since the destination already exists"
            % (src, dst)
        )
    else:
        if is_dir:
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
