import re


memory_multipliers = {
    "k": 1000,
    "M": 1000 ** 2,
    "G": 1000 ** 3,
    "T": 1000 ** 4,
    "P": 1000 ** 5,
    "E": 1000 ** 6,
    "Ki": 1024,
    "Mi": 1024 ** 2,
    "Gi": 1024 ** 3,
    "Ti": 1024 ** 4,
    "Pi": 1024 ** 5,
    "Ei": 1024 ** 6,
}


def _parse_cpu(cpu_str):
    """Parse k8s cpu resource.

    Args:
        cpu_str: The string representation of the CPU resource.

    Return:
        CPU count in float.
    """
    m = re.compile("([0-9]+)m").match(cpu_str)
    if m:
        return float(m.group(1)) / 1000
    else:
        return float(cpu_str)


def _parse_memory(mem_str):
    """Parse k8s memory resource into bytes.

    Args:
        mem_str: The string representation of the memory resource.

    Return:
        Memory in bytes.
    """
    m = re.compile("([0-9]+)([A-Za-z]{1,2})").match(mem_str)
    if m:
        return int(m.group(1)) * memory_multipliers[m.group(2)]
    elif mem_str == "0":
        # Special case
        return 0
    else:
        print("Error: unable to parse the memory string '%s'" % mem_str)


def parse_resource(resource_str):
    kvs = resource_str.split(',')
    parsed_res_dict = {}
    for kv in kvs:
        k, v = kv.split('=')
        if k in ['gpu', 'memory', 'disk']:
            v = int(v)
        elif k == 'cpu':
            v = float(v)
        parsed_res_dict[k.lower()] = v
    return parsed_res_dict
