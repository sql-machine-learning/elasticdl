"""
ElasticDL Worker Manager
"""

import logging
from .k8s import Client

class WorkerManager(object):
    def __init__(self, max_workers, **kwargs):
        self._logger = logging.getLogger("WorkerManager")
        self._max_workers = max_workers
        self._num_workers = 0
        self._k8s_client = Client(event_callback=self.event_callback, **kwargs)

    def event_callback(self, event):
        self._logger.warning(event)
        if 
