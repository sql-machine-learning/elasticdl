import hashlib


def string_to_id(name, bucket_num):
    h = hashlib.sha256(name.encode("utf-8"))
    return int(h.hexdigest(), base=32) % bucket_num


def int_to_id(number, bucket_num):
    return number % bucket_num
