import re


_ALLOWED_RESOURCE_TYPES = ["memory", "disk", "ephemeral-storage", "cpu", "gpu"]
# Any domain name is (syntactically) valid if it's a dot-separated list of
# identifiers, each no longer than 63 characters, and made up of letters,
# digits and dashes (no underscores).
_GPU_VENDOR_REGEX_STR = "^[a-zA-Z\d-]{,63}(\.[a-zA-Z\d-]{,63})*/gpu$"


def _is_numeric(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def _valid_gpu_spec(gpu_str):
    if not gpu_str.isnumeric():
        raise ValueError("invalid gpu request spec: " + gpu_str)
    return gpu_str


def _valid_cpu_spec(cpu_str):
    regexp = re.compile("([1-9]{1})([0-9]*)m$")
    if not regexp.match(cpu_str) and not _is_numeric(cpu_str):
        raise ValueError("invalid cpu request spec: " + cpu_str)
    return cpu_str


def _valid_mem_spec(mem_str):
    regexp = re.compile("([1-9]{1})([0-9]*)(E|P|T|G|M|K|Ei|Pi|Ti|Gi|Mi|Ki)$")
    if not regexp.match(mem_str):
        raise ValueError("invalid memory request spec: " + mem_str)
    return mem_str


def parse_resource(resource_str):
    """Parse combined k8s resource string into a dict.

    Args:
        resource_str: The string representation for k8s resource,
            e.g. "cpu=250m,memory=32Mi,disk=64Mi,gpu=1,ephemeral-storage=32Mi".

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
        elif k == "cpu":
            _valid_cpu_spec(v)
        elif "gpu" in k:
            if k == "gpu":
                k = "nvidia.com/gpu"
            elif not re.compile(_GPU_VENDOR_REGEX_STR).match(k):
                raise ValueError(
                    "gpu resource name does not have a valid vendor name: %s"
                    % k
                )
            _valid_gpu_spec(v)
        else:
            raise ValueError(
                "%s is not in the allowed list of resource types: %s"
                % (k, _ALLOWED_RESOURCE_TYPES)
            )
        parsed_res_dict[k] = v
    return parsed_res_dict
