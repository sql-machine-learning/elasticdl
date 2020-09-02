try:
    from horovod.runner.common.util.hosts import (
        get_host_assignments,
        parse_hosts,
    )

    print(
        "Found hosts.get_host_assignments",
        str(get_host_assignments),
        str(parse_hosts),
    )
except ImportError:
    print("Cannot find hosts.get_host_assignments")

try:
    from horovod.runner.http.http_server import RendezvousServer

    print("Found http_server.RendezvousServer", str(RendezvousServer))
except ImportError:
    print("Cannot find http_server.RendezvousServer")
