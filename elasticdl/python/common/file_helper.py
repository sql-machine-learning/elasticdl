import os
import shutil


def copy_if_not_exists(src, dst, is_dir):
    if os.path.exists(dst):
        print(
            "Skip copying from %s to %s since the destination already exists"
            % (src, dst)
        )
    else:
        if is_dir:
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
