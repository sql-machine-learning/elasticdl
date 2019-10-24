from elasticdl.python.ps.embedding_table import create_embedding_table


class Parameters(object):
    def __init__(self):
        self.init_status = False
        self.non_embedding_params = {}
        self.embedding_params = {}

    def get_embedding_param(self, name, indices):
        if name not in self.embedding_params:
            raise ValueError(
                "Please initialize embedding param %s first!", name
            )
        return self.embedding_params[name].get(indices)

    def set_embedding_param(self, name, indices, values):
        if name not in self.embedding_params:
            raise ValueError(
                "Please initialize embedding param %s first!", name
            )
        self.embedding_params[name].set(indices, values)

    def set_non_embedding_params(self, variables):
        self.non_embedding_params = variables

    def get_non_embedding_params(self):
        return self.non_embedding_params

    def init_from_model_pb(self, model_pb):
        # TODO(qijun) waiting for Tensor/Model proto message definition
        pass

    def _init_non_embedding_param(self, variables_pb):
        # TODO(qijun) waiting for Tensor/Model proto message definition
        pass

    def _init_embedding_param(self, embedding_table_info):
        table = create_embedding_table(embedding_table_info)
        self.embedding_params[table.name] = table

    def clear(self):
        self.non_embedding_params.clear()
        self.embedding_params.clear()
