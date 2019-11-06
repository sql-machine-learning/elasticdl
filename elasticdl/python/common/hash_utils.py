import hashlib


def string_to_id(name, num):
    h = hashlib.sha256(name.encode("utf-8"))
    return int(h.hexdigest(), base=32) % num
