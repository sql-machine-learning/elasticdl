"""
A wrapper RecordIO file that supporting customized encoding/decoding
"""
import recordio

class File(recordio.File):

    def __init__(self, *args, decoder=None, encoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._decoder = decoder if decoder else lambda x : x
        self._encoder = encoder if encoder else lambda x : x

    def write(self, content):
        super().write(self._encoder(content))


    def get_reader(self, *args, **kwargs):
        return map(self._decoder, super().get_reader(*args, **kwargs))
