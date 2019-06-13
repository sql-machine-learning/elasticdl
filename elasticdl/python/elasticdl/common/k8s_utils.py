import re


_ALLOWED_RESOURCE_TYPES = ["memory", "disk", "ephemeral-storage", "cpu", "gpu"]


def _is_numeric(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def _valid_processing_units_spec(pu_str):
    regexp = re.compile("([1-9]{1})([0-9]*)m$")
    if not regexp.match(pu_str) and not _is_numeric(pu_str):
        raise ValueError(
            "invalid processing units (cpu or gpu) request spec: " + pu_str
        )
    return pu_str


def _valid_mem_spec(mem_str):
    regexp = re.compile("([1-9]{1})([0-9]*)(E|P|T|G|M|K|Ei|Pi|Ti|Gi|Mi|Ki)$")
    if not regexp.match(mem_str):
        raise ValueError("invalid memory request spec: " + mem_str)
    return mem_str


def parse_resource(resource_str):
    """Parse combined k8s resource string into a dict.

    Args:
        resource_str: The string representation for k8s resource,
            e.g. "cpu=100m,memory=1024Mi,disk=1024Mi,gpu=1024Mi".

    Return:
        A Python dictionary parsed from the given resource string.
    """
    kvs = resource_str.split(",")
    resource_names = []
    parsed_res_dict = {}
    for kv in kvs:
        k, v = kv.split("=")
        if k not in resource_names:
            resource_names.append(k)
        else:
            raise ValueError(
                "The resource string contains duplicate resource names: %s" % k
            )
        if k in ["memory", "disk", "ephemeral-storage"]:
            _valid_mem_spec(v)
        elif k in ["cpu", "gpu"]:
            _valid_processing_units_spec(v)
        else:
            raise ValueError(
                "%s is not in the allowed list of resource types: %s" % (
                    k, _ALLOWED_RESOURCE_TYPES
                )
            )
        parsed_res_dict[k.lower()] = v
    return parsed_res_dict
