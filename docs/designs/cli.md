# ElasticDL Command-line Client Tool

## Background

ElasticDL is a Kubernetes-Native deep learning framework. As it runs
distributed training/prediction/evaluation jobs in a cluster, we need a client
to submit the jobs to the cluster. The main functionality of the client is
*building image for ElasticDL job* and *submitting ElasticDL job*.

Currently we have a client but it's tightly coupled with the main package. It's
too heavy that users need pip install the whole elasticdl package and lots of
dependencies such as TensorFlow, grpcio, etc.

To improve the user experience, we need a light-weight client CLI of ElasticDL.
It only has depedency on docker and Kubernetes Api. In this doc, we are
discusssing about this command-line client tool.

## User Story

1. Users develop model and the directory structure of model definition files
   is as follows:

    ```TEXT
    a_directory
    - wide_and_deep.py
    requirements.txt
    ```

2. Generate a Dockerfile.  

    Input the command:

    ```bash
    cd ${model_root_path}
    elasticdl zoo init [base_image_name]
    ```

    `base_image_name` is optional and the default value is `python`.
    The generated Dockerfile example is:

    ```Dockerfile
    FROM python
    COPY . /model_zoo
    RUN pip install -r /model_zoo/requirements.txt
    RUN pip install elasticdl
    ```

    Users can make additional updates on the Dockerfile if necessary.

3. Build the Docker image for an ElasticDL job.

    ```bash
    elasticdl zoo build --image=reg.docker.alibaba-inc.com/bright/elasticdl-wnd:1.0 .
    ```

4. Push the Docker image to a remote registry (optional)

    ```bash
    elasticdl zoo push reg.docker.alibaba-inc.com/bright/elasticdl-wnd:1.0
    ```

5. Submit a model training/prediction/evaluation job.

    ```bash
    elasticdl train \
        --image=reg.docker.alibaba-inc.com/bright/elasticdl-wnd:1.0 \
        --model_def=a_directory.wide_and_deep.custom_model \
        --training_data=/data/mnist/train \
        --validation_data=/data/mnist/test \
        --num_epochs=2 \
        --minibatch_size=64 \
        --num_ps_pods=1 \
        --num_workers=1 \
        --evaluation_steps=50 \
        --job_name=test-mnist \
        --distribution_strategy=ParameterServerStrategy \
        --master_resource_request="cpu=0.2,memory=1024Mi" \
        --master_resource_limit="cpu=1,memory=2048Mi" \
        --worker_resource_request="cpu=0.4,memory=1024Mi" \
        --worker_resource_limit="cpu=1,memory=2048Mi" \
        --ps_resource_request="cpu=0.2,memory=1024Mi" \
        --ps_resource_limit="cpu=1,memory=2048Mi"
    ```
