
# RecordIO Data Preparation Tutorial


## Background
Currently, ElasticDL requires the input data in [RecordIO](https://github.com/wangkuiyi/recordio) format. This tutorial is to help users convert raw training data to the required RecordIO format. The RecordIO API is written in Golang and you can see how to use that [here](https://github.com/wangkuiyi/recordio/blob/develop/recordio_test.go). Because we process our data via PySpark job, what we use is the [python wrapper](https://github.com/wangkuiyi/recordio/tree/develop/python) outside its Golang implementation.

This tutorial provides three approaches to process the data: [local Python script](#python-script), [local PySpark job](#local-pyspark-job) and [PySpark job running on Google Cloud](#pyspark-job-on-google-cloud).


## Python Script
If your data amount is small and it locates in your local disk, this is the approach you want to use. [Here](https://github.com/wangkuiyi/elasticdl/blob/develop/elasticdl/python/data/recordio_gen/mnist/gen_data.py) is a sample program to convert the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset into RecordIO format. It should be straightforward to write your own script by mimicking the sample program.


## Local PySpark Job
If your data amount is large but can fit in your local disk, and you want to process your data in parallel, this is the approach you want to use. You can [set up your own Spark environment locally](https://www.tutorialkart.com/apache-spark/how-to-install-spark-on-mac-os/). But we recommend to run the PySpark job in the Docker container. Here are the steps to run our sample PySpark job, which processes the MNIST data, in the Docker container:
0. Download MNIST sampled training data in `jpg` format [here](https://www.kaggle.com/scolianni/mnistasjpg/downloads/trainingSample.zip/1), and unzip it to your training data directory.

1. Build Docker image:
    ```bash
    docker build -t elasticdl:data_process \
        -f elasticdl/docker/Dockerfile.data_process .
    ```

2. Run PySpark job in Docker Container:
    ```bash
    OUTPUT_DIR=~/Desktop/sample_recordio_output
    TRAINING_DATA_DIR=~/Desktop/training_data
    MODEL_FILE=/elasticdl/python/examples/mnist_functional_api.py

    docker run --rm -v $OUTPUT_DIR:/output_dir \
        -v $TRAINING_DATA_DIR:/training_data_dir \
        elasticdl:data_process_new \
        /elasticdl/python/data/recordio_gen/sample_pyspark_recordio_gen/spark_gen_recordio.py \
        --training_data_dir=/training_data_dir/ \
        --output_dir=/output_dir/  \
        --model_file=$MODEL_FILE \
        --records_per_file=200
    ```
    After the job is finished, you should see your data named in the format of `data-<worker_num_that_generate_this_data>-<chunk_num>` located in `OUTPUT_DIR`.
    
    Notes:
    1. If your PySpark job needs other dependencies in the image, you can create your own image derived from the sample Dockerfile.
    2. You need to provide your own [model file](https://github.com/wangkuiyi/elasticdl/blob/0b7d75fd5073802f33e192244283b86ccf2684e0/elasticdl/doc/model_building.md), from which we need the [feature](https://github.com/wangkuiyi/elasticdl/blob/develop/elasticdl/doc/model_building.md#feature_columns) and [label](https://github.com/wangkuiyi/elasticdl/blob/develop/elasticdl/doc/model_building.md#label_columns) columns, as well as the [user-defined data processing logic](prepare_data_for_a_single_file).
    3. There are some other arguments you can pass to our backbone PySpark file as you can see [here](TODO: add the link here once this PR is merged).


## PySpark Job on Google Cloud
TODO.