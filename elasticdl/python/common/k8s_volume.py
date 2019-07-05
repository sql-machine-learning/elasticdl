_ALLOWED_VOLUME_KEYS = ["claim_name", "volume_name", "mount_path"]


def parse(volume_str):
    """Parse combined k8s volume string into a dict.

    Args:
        volume_str: The string representation for k8s volume,
            e.g. "claim_name=c1,volume_name=v1,mount_path=/path1".

    Return:
        A Python dictionary parsed from the given volume string.
    """
    kvs = volume_str.split(",")
    volume_keys = []
    parsed_volume_dict = {}
    for kv in kvs:
        k, v = kv.split("=")
        if k not in volume_keys:
            volume_keys.append(k)
        else:
            raise ValueError(
                "The volume string contains duplicate volume key: %s" % k
            )
        if k not in _ALLOWED_VOLUME_KEYS:
            raise ValueError(
                "%s is not in the allowed list of volume keys: %s"
                % (k, _ALLOWED_VOLUME_KEYS)
            )
        parsed_volume_dict[k] = v
    return parsed_volume_dict
