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

import time

from elasticai_api.common.master_client import GlobalMasterClient


def worker_sync(sync_name, timeout=None):
    """
    Input:
        sync_name: a name (string) for this sync
        timeout: if not None, timeout in seconds
    Return:
        True if sync successfully. False if timeout.
    """
    master_client = GlobalMasterClient.MASTER_CLIENT
    while True:
        res = master_client.worker_sync(sync_name)
        if res.ready:
            return True
        time.sleep(1)
        if timeout is not None:
            timeout -= 1
            if timeout <= 0:
                return False


def wait_worker_sync(sync_name, notify=False, timeout=None):
    """
    Input:
        sync_name: a name (string) for this sync
        notify: if False, wait this sync. If True, sync is done
                and those who wait for this sync finish waiting.
        timeout: if not None, timeout in seconds if notify=False
    """
    master_client = GlobalMasterClient.MASTER_CLIENT
    if notify:
        master_client.wait_worker_sync(sync_name, notify)
        return True
    else:
        while True:
            res = master_client.wait_worker_sync(sync_name, notify)
            if res.ready:
                return True
            time.sleep(1)
            if timeout is not None:
                timeout -= 1
                if timeout <= 0:
                    return False


def delete_worker_sync(sync_name):
    """
    Delete the worker sync with name=sync_name if exists,
    so sync_name can be reused in another worker sync.
    """
    master_client = GlobalMasterClient.MASTER_CLIENT
    master_client.delete_worker_sync(sync_name)


def delete_all_worker_sync():
    """
    Delete all worker syncs ,
    so they can be reused later for new worker sync.
    """
    master_client = GlobalMasterClient.MASTER_CLIENT
    master_client.delete_all_worker_sync()
