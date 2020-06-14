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

import os
import shutil

from elasticdl.python.common.log_utils import default_logger as logger


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
