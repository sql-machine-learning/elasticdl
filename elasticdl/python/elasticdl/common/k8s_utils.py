import re


def _valid_cpu_spec(cpu_str):
    regexp = re.compile("([1-9]{1})([0-9]*)m$")
    if not regexp.match(cpu_str):
        raise ValueError("invalid cpu request spec: " + cpu_str)
    return cpu_str


def _valid_mem_spec(mem_str):
    regexp = re.compile("([1-9]{1})([0-9]*)(E|P|T|G|M|K|Ei|Pi|Ti|Gi|Mi|Ki)$")
    if not regexp.match(mem_str):
        raise ValueError("invalid memory request spec: " + mem_str)
    return mem_str


def parse_resource(resource_str):
    kvs = resource_str.split(',')
    parsed_res_dict = {}
    for kv in kvs:
        k, v = kv.split('=')
        if k in ['gpu', 'memory', 'disk']:
            _valid_mem_spec(v)
        elif k == 'cpu':
            _valid_cpu_spec(v)
        parsed_res_dict[k.lower()] = v
    return parsed_res_dict
