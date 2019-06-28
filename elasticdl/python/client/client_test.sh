python -m elasticdl.python.client.client train \
  --image_base=elasticdl:ci \
  --model_file=/Users/l.zou/git/elasticdl/elasticdl/python/examples/mnist_subclass.py \
  --training_data_dir=/data/mnist/train \
  --evaluation_data_dir=/data/mnist/test \
  --num_epochs=1 \
  --master_resource_request="cpu=1,memory=512Mi" \
  --master_resource_limit="cpu=1,memory=512Mi" \
  --worker_resource_request="cpu=1,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=1024Mi" \
  --minibatch_size=10 \
  --records_per_task=100 \
  --num_workers=1 \
  --checkpoint_steps=2 \
  --grads_to_wait=2 \
  --job_name=test \
  --image_name=elasticdl:dev \
  --log_level=INFO \
  --image_pull_policy=Never

