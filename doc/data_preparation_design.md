
# Data Preparation Design Doc

## Background
Currently, ElasticDL requires the input data in [RecordIO]( https://github.com/elasticdl/recordio) format. This project is to help users convert raw training data to the required RecordIO format.

## Design
The system is to use Spark to prepare the data in parallel either in a container (`local` mode) or in a Spark cluster (`cluster` mode). We'll provide users with a docker image `data_preparation_image`. And the users can use it like this:
```bash
docker run data_preparation_image \
    <user_defined_model_file> \
    <raw_data_directory> \
    <output_recordio_data_directory> \
    <running_mode> \
    <other_mode_related_arguments>
```
The general idea is to give users enough flexibility to prepare the data. The only responsibilty of our docker image is to run whatever user-defined data preparation logic in parallel. Below is the description for each argument:


### `user_defined_model_file`

This file defines the [model](https://github.com/sql-machine-learning/elasticdl/blob/develop/doc/model_building.md) that is going to be trained. In this file, there are two parts that are needed by data preparation:

1. Feature Column. In order to convert data into RecordIO format, we have to know the corresponding feature columns. Given that the feature columns are already defined in the current model file (e.g. [MNIST functional API example](https://github.com/sql-machine-learning/elasticdl/blob/0b7d75fd5073802f33e192244283b86ccf2684e0/elasticdl/python/examples/mnist_functional_api.py#L18-L24)), we can just leverage the existing model file in order to keep the feature columns consistent between data preparation and model training.

1. Data Preparation Function For a Single File: The signature for this function is:
    ```python
    def prepare_data_for_a_single_file(filename):
        '''
        This function is to read a single file and do whatever 
        user-defined logic to prepare the data (e.g, IO from the user's file system, 
        feature engineering), and return a tuple of numpy array, which should 
        be compatible with the feature column above.
        '''
        pass
    ```
    This part is not in our current model definition yet and is not limited to data preparation. It is also needed for online serving because the online input is always raw (e.g, text) and this provides the functionality to bind the feature transformation with the model. The user may need to install dependencies that the user needs to prepare the data in a new image, which can be derived from the image we provide.


### `raw_data_directory`
This directory should include all training data files and will be traversed recursively to get all training data by our Spark job.

### `output_recordio_data_directory`
The output directory for prepared data in RecordIO format.

### `running_mode`
The Spark job can be run in either `local` or `cluster` mode. If there's no Spark cluster available for the user, the user can still use `local` mode to process the data in the container which runs the image we provide.

### `other_mode_related_arguments`
Other arguments that are related to the `running_mode`. For example, we need to provide `num_workers` no matter which mode Spark runs in, and we need to provide more cluster-related arguments if we want to run it in a cluster.
