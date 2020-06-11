# ElasticDL Command-line Client Tool

## Background

ElasticDL is a Kubernetes-Native deep learning framework. As it runs
distributed training/prediction/evaluation jobs in a cluster, we need a client
to submit the job to the cluster. Currently we have one but it's tightly
coupled with the main package. User need pip install the whole elasticdl
package and lots of dependencies and it's too heavy.

The main functionality of the client is *building image for ElasticDL job* and
*submitting ElasticDL job*. It only has depedency on docker and Kubernetes Api.
In this doc, we are focusing on a light-weight client CLI of ElasticDL.

## User Story

1. User develops model and the directory structure is as follows:

    ```TEXT
    a_directory
    - layers
    - models
    - utils
    requirements.txt
    ```

2. Generate an Dockerfile.  

    Input the command:

    ```bash
    cd ${model_repo_root_path}
    elasticdl zoo init
    ```

    The example Dockerfile is:

    ```Dockerfile
    FROM elastcdl:latest
    COPY . /models
    RUN pip install -r /models/requirements.txt
    ```

    User can make additional update on the Dockerfile if necessary.

3. Build the docker image for ElasticDL job.

    ```bash
    elasticdl zoo build --image=reg.docker.alibaba-inc.com/bright/elasticdl-wnd:1.0 .
    ```

4. Push the docker image to a remote registry (optional)

    ```bash
    elasticdl zoo push reg.docker.alibaba-inc.com/bright/elasticdl-wnd:1.0
    ```

5. Submit a model training/prediction/evalution job.

    ```bash
    elasticdl train \
        --image=reg.docker.alibaba-inc.com/bright/elasticdl-wnd:1.0 \
        --model_def=my_model_repo.models.wnd.custom_model \
        --training_data=/data/mnist/train \
        --validation_data=/data/mnist/test \
        --num_epochs=2 \
        --minibatch_size=64 \
        --num_minibatches_per_task=2 \
        --num_ps_pods=1 \
        --num_workers=1 \
        --evaluation_steps=50 \
        --grads_to_wait=1 \
        --job_name=test-mnist \
        --distribution_strategy=ParameterServerStrategy \
        --master_resource_request="cpu=0.2,memory=1024Mi" \
        --master_resource_limit="cpu=1,memory=2048Mi" \
        --worker_resource_request="cpu=0.4,memory=1024Mi" \
        --worker_resource_limit="cpu=1,memory=2048Mi" \
        --ps_resource_request="cpu=0.2,memory=1024Mi" \
        --ps_resource_limit="cpu=1,memory=2048Mi"
    ```
