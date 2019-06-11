
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
        -f elasticdl/python/data/recordio_gen/sample_pyspark_recordio_gen/Docker/Dockerfile .
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
If your data amount is huge so that it can't fit into your local disk, this is the approach you want to use. In this tutorial we use [Google Filestore](https://cloud.google.com/filestore/) as our training data storage. We also tried [Google Cloud Storage](https://cloud.google.com/storage/), which is not a good fit for our use case(see [here](https://github.com/wangkuiyi/elasticdl/issues/381#issuecomment-500686228)). Here are the steps to run our PySpark job on Google Cloud:
1. Set up the Google Cloud SDK and project following [here](https://cloud.google.com/sdk/docs/quickstarts) based on your OS.

2. Upload the [initialization script](TODO: add the link here once this PR is merged), which will install all dependencies we need to Spark cluster, to your Google Cloud Storage:
```bash
LOCAL_INIT_SCRIPT=elasticdl/python/data/recordio_gen/sample_pyspark_recordio_gen/go-pip-install.sh
GS_INIT_SCRIPT=gs://elasticdl.appspot.com/go-pip-install.sh

gsutil cp $LOCAL_INIT_SCRIPT $GS_INIT_SCRIPT
```

3. [Create a Dataproc cluster](https://cloud.google.com/dataproc/docs/guides/create-cluster) with the [initialization actions](https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/init-actions#using_initialization_actions) by the script above:
```bash
CLUSTER_NAME=test-cluster

gcloud beta dataproc clusters create $CLUSTER_NAME \
--image-version=preview \
--optional-components=ANACONDA \
--initialization-actions $GS_INIT_SCRIPT
```

4. [Create a Google Filestore instance](https://cloud.google.com/filestore/docs/creating-instances#create-instance-gcloud), which is used to store our training data:
```bash
PROJECT_NAME=elasticdl
FILESTORE_NAME=elasticdl

gcloud filestore instances create $FILESTORE_NAME $PROJECT_NAME \
    --location=us-west1-a \
    --file-share=name="elasticdl",capacity=1TB \
    --network=name="default"
```

5. Mount the Filestore to every node of your Spark cluster per [here](https://cloud.google.com/filestore/docs/quickstart-gcloud#mount-filestore-fileshare). In this tutorial, I mounted it to `/filestore_mnt`.

6. [Copy the training data from local to Filestore](https://cloud.google.com/filestore/docs/copying-data#computer-to-fileshare):
```bash
gcloud compute scp $TRAINING_DATA_DIR --recurse \
    test-cluster-m:/filestore_mnt/$TRAINING_DATA_DIR \
    --project elasticdl --zone us-west1-a
```

7. Zip the `elasticdl` folder as the dependency for the PySpark job, which will be submitted together with PySpark job in the next step:
```bash
zip -r elasticdl.zip elasticdl
```

8. [Submit the PySpark job](https://cloud.google.com/sdk/gcloud/reference/dataproc/jobs/submit/pyspark):
```bash
gcloud dataproc jobs submit pyspark \
    elasticdl/python/data/recordio_gen/sample_pyspark_recordio_gen/spark_gen_recordio.py \
    --cluster=$CLUSTER_NAME --region=global --py-files=elasticdl.zip \
    --files=elasticdl/python/examples/mnist_functional_api.py \
    -- --training_data_dir=/filestore_mnt/$TRAINING_DATA_DIR \
    --output_dir=/filestore_mnt --model_file=$MODEL_FILE --records_per_file=200
```

Then you can see four generated RecordIO files `data-0-0000`, `data-0-0001`, `data-1-0000` and `data-1-0001` located in the mounted directory `/filestore_mnt`.
