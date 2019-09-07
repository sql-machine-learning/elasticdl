#!/usr/bin/env bash

JOB_TYPE=$1

if [[ "$JOB_TYPE" == "train" ]]; then
    elasticdl train \
      --image_base=elasticdl:ci \
      --model_zoo=model_zoo \
      --model_def=mnist_functional_api.mnist_functional_api.custom_model \
      --training_data_dir=/data/mnist/train \
      --evaluation_data_dir=/data/mnist/test \
      --num_epochs=2 \
      --master_resource_request="cpu=400m,memory=1024Mi" \
      --master_resource_limit="cpu=1,memory=2048Mi" \
      --worker_resource_request="cpu=400m,memory=2048Mi" \
      --worker_resource_limit="cpu=1,memory=3072Mi" \
      --minibatch_size=64 \
      --records_per_task=100 \
      --num_workers=2 \
      --checkpoint_steps=10 \
      --evaluation_steps=15 \
      --grads_to_wait=2 \
      --job_name=test-mnist-train \
      --log_level=INFO \
      --image_pull_policy=Never \
      --output=model_output
elif [[ "$JOB_TYPE" == "evaluate" ]]; then
    elasticdl evaluate \
      --image_base=elasticdl:ci \
      --model_zoo=model_zoo \
      --model_def=mnist_functional_api.mnist_functional_api.custom_model \
      --checkpoint_filename_for_init=elasticdl/python/tests/testdata/mnist_functional_api_model_v110.chkpt \
      --evaluation_data_dir=/data/mnist/test \
      --num_epochs=1 \
      --master_resource_request="cpu=400m,memory=1024Mi" \
      --master_resource_limit="cpu=1,memory=2048Mi" \
      --worker_resource_request="cpu=400m,memory=2048Mi" \
      --worker_resource_limit="cpu=1,memory=3072Mi" \
      --minibatch_size=64 \
      --records_per_task=100 \
      --num_workers=2 \
      --evaluation_steps=15 \
      --job_name=test-mnist-evaluate \
      --log_level=INFO \
      --image_pull_policy=Never
elif [[ "$JOB_TYPE" == "predict" ]]; then
    elasticdl predict \
      --image_base=elasticdl:ci \
      --model_zoo=model_zoo \
      --model_def=mnist_functional_api.mnist_functional_api.custom_model \
      --checkpoint_filename_for_init=elasticdl/python/tests/testdata/mnist_functional_api_model_v110.chkpt \
      --prediction_data_dir=/data/mnist/test \
      --master_resource_request="cpu=400m,memory=1024Mi" \
      --master_resource_limit="cpu=1,memory=2048Mi" \
      --worker_resource_request="cpu=400m,memory=2048Mi" \
      --worker_resource_limit="cpu=1,memory=3072Mi" \
      --minibatch_size=64 \
      --records_per_task=100 \
      --num_workers=2 \
      --job_name=test-mnist-predict \
      --log_level=INFO \
      --image_pull_policy=Never
else
    echo "Unsupported job type specified: $JOB_TYPE"
fi
