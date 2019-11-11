import hashlib


def string_to_id(name, num):
    h = hashlib.sha256(name.encode("utf-8"))
    return int(h.hexdigest(), base=32) % num


def int_to_id(index, num):
    return index % num


def scatter_embedding_vector(values, indices, num):
    ps_ids = {}
    indices_list = indices.tolist()
    for i, item_id in enumerate(indices_list):
        ps_id = int_to_id(item_id, num)
        if ps_id not in ps_ids:
            ps_ids[ps_id] = [(i, item_id)]
        else:
            ps_ids[ps_id].append((i, item_id))
    results = {}
    for ps_id, i_item_id in ps_ids.items():
        i = [v[0] for v in i_item_id]
        item_id = [v[1] for v in i_item_id]
        results[ps_id] = (values[i, :], item_id)
    return results
