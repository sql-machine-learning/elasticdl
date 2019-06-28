set -x
python -m elasticdl.python.client.client train \
  --image_base=elasticdl:ci \
  --model_file=$(pwd)/elasticdl/python/examples/mnist_functional_api.py \
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
  --grads_to_wait=2 \
  --job_name=test-mnist \
  --image_name=elasticdl:dev \
  --log_level=INFO \
  --image_pull_policy=Never
