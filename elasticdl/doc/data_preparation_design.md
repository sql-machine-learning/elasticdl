
# Data Preparation Design Doc

 ## Background

 The input of ElasticDL is in [RecordIO](https://github.com/ElasticDL/pyrecordio) format. This project is to create an easy-to-use system to convert raw training data to RecordIO format.

 ## Design
The system is to use Spark to prepare the data parallelly in a container. We'll provide the user with a docker image `data_preparation_image`. And the user can use it like this:
```bash
docker run --rm data_preparation_image \
    <user_defined_model_file> \
    <raw_data_directory> \
    <running_mode> \
    <other_mode_related_arguments>
```
The general idea is to give user enough flexibility to prepare the data. The only responsibilty of our docker image is to run whatever user-defined data preparation logic paralelly. The explaination of the arguments above is here:


 ### `user_defined_model_file`

 This file defines the model that is going to be trained. In this file, there are 2 parts that are needed by data preparation:

 1. Feature Column: In order to convert data into RecordIO format, we have to know the feature column. Given that the current model file has already implemented this (e.g: https://github.com/wangkuiyi/elasticdl/blob/0b7d75fd5073802f33e192244283b86ccf2684e0/elasticdl/python/examples/mnist_functional_api.py#L18-L24), in order to keep the feature column consistency between data preparation and model training, we should get the feature column from this file directly.

 2. Data Preparation Function For a Single File: The signature for this function is:
    ```python
    def prepare_data_for_a_single_file(filename):
    '''
    This function is to read a single file and do whatever 
    user-defined logic to prepare the data (e.g: IO from the user's file system, 
    feature engineering), and return a tuple of numpy array, which should 
    be compatible with the feature column above.
    '''
    pass
    ```
    This part is not in our current model definition and is to be implemented. I think this part is not for data preparation only. It is needed for online serving anyway because the online input is always raw (e.g: text) and this provides the functionality to bind the feature transformation with the model.


 ### `raw_data_directory`
This directory should include all training data files. The directory will be traversed recursively to get all training data by our Spark job.


 ### `running_mode`
The Spark job can be run in either `local` or `cluster`. If there's no Spark cluster available for the user, the user can still use `local` mode to process the data in the container which runs the image we provide.

 ### `other_mode_related_arguments`
Some arguments relate to the `running_mode`. For example, we need to provide `num_works` no matter which mode Spark runs in, and we need to provide more cluster-related arguments if we want to run it in cluster.
