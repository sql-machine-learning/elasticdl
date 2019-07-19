set -x

elasticdl train \
  --image_base=elasticdl:ci \
  --model_zoo=/model_zoo \
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
  --job_name=test-mnist \
  --log_level=INFO \
  --image_pull_policy=Never
