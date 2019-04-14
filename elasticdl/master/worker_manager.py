"""
ElasticDL Worker Manager
"""
from .k8s import Client

class WorkerManager(object):
    def __init__(self, max_workers, ):
        self._max_workers = max_workers
        self._num_workers = 0

    def run(self, )
