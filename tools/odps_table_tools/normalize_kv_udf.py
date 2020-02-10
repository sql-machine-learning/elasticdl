# coding:utf-8
# qinlong.wql
from odps.udf import BaseUDTF


def parse_kv_string_to_dict(
    kvs_string, output_key_names, inter_kv_sep, intra_kv_sep
):
    """ Parse kv string to a dict like "k1:v1,k2:v2" to {k1:v1,k2:v2}
    Args:
        kvs_string: Key-value pairs string
        output_key_names: All key names to be saved in the output dict
        inter_kv_sep: Inter separator in the Key-value pair string
        intra_kv_sep: Intra separator int Ker-value pairs string.

    Returns:
        dict
    """
    kv_dict = {}
    kv_pairs = kvs_string.split(inter_kv_sep)
    for kv in kv_pairs:
        key_and_value = kv.split(intra_kv_sep)
        if len(key_and_value) == 2:
            kv_dict[key_and_value[0]] = key_and_value[1]
    values = []
    for name in output_key_names:
        values.append(kv_dict.get(name, ""))

    return values


class KVFlatter(BaseUDTF):
    """Split string by separator to values
    """

    def process(self, *args):
        """
        Args:
            args (list): args[0] is kv column value, args[-3] is the feature
            names string other values in args is the append column values.
            args[-1] is the intra key-value pair separator and args[-2]
            it the inter key-value sparator.
        """
        if len(args) < 4:
            raise ValueError("The input values number can not be less than 4")
        feature_names = args[-3].split(",")
        inter_kv_sep = args[-2]
        intra_kv_sep = args[-1]
        feature_values = parse_kv_string_to_dict(
            args[0], feature_names, inter_kv_sep, intra_kv_sep
        )
        for value in args[1:-3]:
            feature_values.append(str(value))
        self.forward(*feature_values)
