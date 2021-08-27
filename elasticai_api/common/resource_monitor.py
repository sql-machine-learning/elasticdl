# Copyright 2021 The ElasticDL Authors. All rights reserved.
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

import threading
import time

import psutil

from elasticai_api.common.master_client import GlobalMasterClient


def get_process_cpu_percent():
    """Get the cpu percent of the current process.
    """
    try:
        procTotalPercent = 0
        result = {}
        proc_info = []
        # 分析依赖文件需要获取 memory_maps
        # 使用进程占用的总CPU计算系统CPU占用率
        for proc in psutil.process_iter(
            ["pid", "ppid", "name", "username", "cmdline"]
        ):
            proc_percent = proc.cpu_percent()
            procTotalPercent += proc_percent
            proc.info["cpu_percent"] = round(proc_percent, 2)
            proc_info.append(proc.info)
        # 暂时不上报进程数据，看下数据量的情况
        result["proc_info"] = proc_info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = round(procTotalPercent / cpu_count, 2)
    except Exception:
        cpu_percent = 0.0
    return cpu_percent


def get_used_memory():
    mem = psutil.virtual_memory()
    used_mem = mem.used
    if hasattr(mem, "cached"):
        used_mem += mem.cached
    return used_mem


class ResourceMonitor(object):
    def __init__(self):
        """
        The monitor samples the used memory and cpu percent
        reports the used memory and cpu percent to the ElasticDL master.
        """
        self._max_memory = 0
        self._max_cpu_percent = 0.0
        threading.Thread(target=self._update_resource, daemon=True).start()

    def _update_resource(self):
        while True:
            used_mem = get_used_memory()
            self._max_memory = (
                used_mem if used_mem > self._max_memory else self._max_memory
            )
            cpu_percent = get_process_cpu_percent()
            self._max_cpu_percent = (
                cpu_percent
                if cpu_percent > self._max_cpu_percent
                else self._max_cpu_percent
            )
            time.sleep(5)

    def report_resource(self):
        try:
            GlobalMasterClient.MASTER_CLIENT.report_used_resource(
                self._max_memory, self._max_cpu_percent
            )
        except Exception:
            pass


class GlobalResourceMonitor(object):
    RESOURCE_MONITOR = ResourceMonitor()
