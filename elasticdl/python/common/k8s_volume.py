from kubernetes import client

_ALLOWED_VOLUME_KEYS = ["claim_name", "host_path", "type", "mount_path"]


def parse_volume_and_mount(volume_conf, pod_name):
    """Get k8s volumes list and volume mounts list from
    the volume config string.

    Args:
        volume_conf (string): the volumes config string,
        e.g. "host_path=c0,mount_path=/path0;claim_name=c1,mount_path=/path1".
        pod_name (string): the pod name

    Return:
        volumes (List): a Python list contains k8s volumes.
        volume_mounts (List): a Python list contains k8s volume mounts.
    """
    volumes = []
    volume_mounts = []
    volume_dicts = parse(volume_conf)
    for i, volume_dict in enumerate(volume_dicts):
        volume_name = pod_name + "-volume-%d" % i
        if "claim_name" in volume_dict:
            pvc_volume_source = client.V1PersistentVolumeClaimVolumeSource(
                claim_name=volume_dict["claim_name"], read_only=False
            )
            volume = client.V1Volume(
                name=volume_name, persistent_volume_claim=pvc_volume_source
            )
        elif "host_path" in volume_dict:
            volume = client.V1Volume(
                name=volume_name,
                host_path=client.V1HostPathVolumeSource(
                    path=volume_dict["host_path"],
                    type=volume_dict.get("type", None),
                ),
            )
        volumes.append(volume)
        volume_mounts.append(
            client.V1VolumeMount(
                name=volume_name, mount_path=volume_dict["mount_path"]
            )
        )
    return volumes, volume_mounts


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
