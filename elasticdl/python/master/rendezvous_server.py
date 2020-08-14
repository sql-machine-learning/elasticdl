from horovod.runner.http.http_server import RendezvousServer
from horovod.runner.common.util.hosts import get_host_assignments, parse_hosts


_WORKER_SLOT_NUMBER = 1
_HOST_SEP = ","


class HorovodRendezvousServer(object):
    def __init__(self, server_host):
        self._rendezvous_host = server_host
        self._rendezvous_id = 0
        self._worker_hosts = []
        self._rendezvous_server = RendezvousServer(verbose=True)
        self._rendezvous_port = self._rendezvous_server.start()

    def set_worker_hosts(self, worker_hosts):
        """
        Set worker hosts into RendezvousServer

        Args:
            worker_hosts: String, multiple hosts are separated by ","
        """
        if sorted(worker_hosts) == sorted(self._worker_hosts):
            return

        self._rendezvous_id += 1
        self._worker_hosts = worker_hosts
        host_alloc_plan = self._get_host_plan()
        self._rendezvous_server.init(host_alloc_plan)

    def _get_host_plan(self):
        hosts = []
        for host in self._worker_hosts:
            hosts.append(host + ":" + str(_WORKER_SLOT_NUMBER))

        host_infos = parse_hosts(_HOST_SEP.join(hosts))
        host_alloc_plan = get_host_assignments(host_infos, len(host_infos))
        return host_alloc_plan

    def get_server_host(self):
        return self._rendezvous_host
