import numpy as np


class BytesCodec(object):
    def __init__(self, feature_columns):
        self._feature_columns = feature_columns
        self._col_id = {
            c.name: order for order, c in enumerate(feature_columns)
        }

    def encode(self, data):
        # Rearrange the data in order of the columns.
        values = [None] * len(self._feature_columns)
        for f_name, f_value in data:
            col_id = self._col_id[f_name]
            column = self._feature_columns[col_id]
            if column.dtype != f_value.dtype or column.shape != f_value.shape:
                raise ValueError(
                    "Input data doesn't match column %s definition: column: (%s, %s) data: (%s, %s)" % (
                    f_name, column.dtype, column.shape, f_value.dtype, f_value.shape)
                )
            values[col_id] = f_value.tobytes()
        for id, value in enumerate(values):
            if value is None:
                raise ValueError(
                    "Missing value for column: %s",
                    self._col_id[id].name
                )
        return b"".join(values)

    def decode(self, record):
        offset = 0
        res = {}
        for c in self._feature_columns:
            count = np.prod(c.shape)
            res[c.name] = np.frombuffer(
                record,
                dtype=c.dtype.as_numpy_dtype,
                count=count,
                offset=offset).reshape(c.shape)
            offset += count * c.dtype.size
        return res
