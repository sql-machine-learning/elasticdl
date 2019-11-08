_ALLOWED_VOLUME_KEYS = ["claim_name", "host_path", "type", "mount_path"]


def parse(volume_str):
    """Parse combined k8s volume strings separated by
    semicolons to Python dictionaries.

    Args:
        volume_str: The string representation for k8s volume,
        e.g. "host_path=c0,mount_path=/path0;claim_name=c1,mount_path=/path1".

    Return:
        A Python list which contains dictionaries and each dictionary is
        representation for a k8s volume.
    """
    volumes = volume_str.strip().split(";")
    volume_mount_pairs = []
    for volume_str in volumes:
        if volume_str:
            volume_mount_pairs.append(parse_single_volume(volume_str))
    return volume_mount_pairs


def parse_single_volume(volume_str):
    """Parse combined k8s volume string into a dict.

    Args:
        volume_str: The string representation for k8s volume,
            e.g. "claim_name=c1,mount_path=/path1".

    Return:
        A Python dictionary parsed from the given volume string.
    """
    kvs = volume_str.strip().split(",")
    volume_keys = []
    parsed_volume_dict = {}
    for kv in kvs:
        k, v = kv.strip().split("=")
        k = k.strip()
        v = v.strip()
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
