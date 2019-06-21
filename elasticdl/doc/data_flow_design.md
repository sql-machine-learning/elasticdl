# Data Flow Design
### Some issues in the current system:
1. There are two codecs in the system, which has two downsides:
    1. We need to implement codec-related logic separately for each one
    2. When training the model, users need to know what codec is used in the RecordIO data

2. Currently, Feature Columns are used for [data encoding/decoding](https://github.com/wangkuiyi/elasticdl/blob/f83834067fbc610fa7a7261a758c9d13aff09a0c/elasticdl/python/data/codec/tf_example_codec.py#L5-L11) and [model building](https://github.com/wangkuiyi/elasticdl/blob/f83834067fbc610fa7a7261a758c9d13aff09a0c/elasticdl/python/elasticdl/worker/worker.py#L52). But actually what these parts essentially need are the follows:

    1. data encoding: A [example](https://github.com/tensorflow/tensorflow/blob/93dd14dce2e8751bcaab0a0eb363d55eb0cc5813/tensorflow/core/example/example.proto#L88-L90) object, which is essentially a map from `feature_name` to one of `BytesList`, `FloatList` and `Int64List` ([here](https://github.com/tensorflow/tensorflow/blob/93dd14dce2e8751bcaab0a0eb363d55eb0cc5813/tensorflow/core/example/feature.proto#L76-L88))

    2. data decoding: A map from `feature_name` to either `tf.FixedLenFeature` or `tf.VarLenFeature` ([here](https://www.tensorflow.org/tutorials/load_data/tf_records#read_the_tfrecord_file))

    3. model building: The shape of the model input ([here](https://github.com/wangkuiyi/elasticdl/blob/f83834067fbc610fa7a7261a758c9d13aff09a0c/elasticdl/python/elasticdl/common/model_helper.py#L20)). Please note that the `model input` here doesn't necessarily mean the data in RecordIO -- we may need to do some transformation for the data in RecordIO (e.g: normalize the size of the images in RecordIO) before we feed them into the data.

	So there are three groups of objects mentioned here: A.`Feature Column(e.g, numeric_column/bucketized_column/...)`, B.`BytesList, FloatList and Int64List` and C.`tf.FixedLenFeature/tf.VarLenFeature`. Except we can map from group A to group B with [make_parse_example_spec()](https://www.tensorflow.org/api_docs/python/tf/feature_column/make_parse_example_spec), there's not mapping between each other. While in our current system, we map them in some hacky ways(e.g, we map A to B [here](elasticdl/python/data/recordio_gen/convert_numpy_to_recordio.py), which won't work at all if `Feature Column` is not `numeric_column`). So I'd like to propose a way to decouple all of them.

### New Design:
In the new design, I propose:
1. Remove the `byte_codec` and only keep `tfexample_codec`

2. Get rid of the concept of `Feature Columns` in the system, and refactor the components in our current system that currently need `Feature Columns` as follow:
    1. Raw data to RecordIO:
    ```python
    def prepare_data_for_a_single_file(filename, file_obj):
        ```
        param filename: Current filename
        param file_obj: A file object to read the file # because we directly read from tar file, we can't get this file_obj from filename
        return: An example object https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/core/example/example.proto#L88-L90
        ```
        # Users must implement the logic here
        return example
    ```

    ```python
    def convert_examples_to_recordio(examples, <some_recordio_parameters>):
        ```
        Write examples to RecordIO
        ```
        # This is the common logic. So we provide the implementation here.
    ```

    2. RecordIO to the data that is ready to feed into the model:
    This is done in `input_fn(records, decode_fn)` defined in model file.
    ```python
    def decode(raw):
        ```
        Take an encoded string of tf example object and return a tf example object
        ```
    ```
    ```python
    def input_fn(records, decode_fn):
        ```
        Take a list of record and the decode function, and return a tuple (feature_tensor_list, label_numpy_array)
        ```
    ```