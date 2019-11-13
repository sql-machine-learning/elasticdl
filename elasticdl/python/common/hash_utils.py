import hashlib


def string_to_id(name, bucket_num):
    h = hashlib.sha256(name.encode("utf-8"))
    return int(h.hexdigest(), base=32) % bucket_num


def int_to_id(number, bucket_num):
    return number % bucket_num


def scatter_embedding_vector(values, indices, bucket_num):
    """
    Scatter embedding vectors to different parameter servers.
    There are two steps to process <id, embedding vector> pairs:
    1. hash the item id to a parameter server id
    2. put corresponding item embedding vector to the parameter server id

    For example, we scatter following embedding vectors into two parameter
    servers:
        values = np.array([[1, 2], [3, 4], [5, 6]])
        indices = np.array([8, 1, 7])

    ID 8 will be hashed to parameter server 0, ID 1 and ID 7 will be hashed to
    parameter server 1. So, parameter server 0 has embedding vectors
    np.array([[1, 2]]), parameter server 1 has embedding vectors,
    np.array([[3, 4], [5, 6]).

    The function return a dictionary:
    {
        0: (np.array([[1, 2]]), [8]),
        1: (np.array([[3, 4], [5, 6]]), [1, 7])
    }
    """
    ps_ids = {}
    indices_list = indices.tolist()
    for i, item_id in enumerate(indices_list):
        ps_id = int_to_id(item_id, bucket_num)
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
