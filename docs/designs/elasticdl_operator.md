# Design for ElasticDL Operator

## Motivation

ElasticDL uses master-worker architecture.
Each ElasticDL job has a unique master pod.
The master pod manages the lifecycle of worker pods and controls the training process.

ElasticDL provides a command-line [client tool](https://github.com/sql-machine-learning/elasticdl/blob/develop/docs/designs/client_tool.md)
to submit a job to a Kubernetes cluster.
At first, a master pod will be launched.
Then, the master pod launches worker pods and PS pods if necessary.
The training process begins once a worker pod becomes ready.

When making ElasticDL as a product of cloud computing,
we find that we have to address the following two points:

- Job monitoring and management. ElasticDL client tool only launches a job.
We have to write extra scripts to monitor the pods' status
and clean pods when a job completes.

- Product compatibility. Current products have deployed some operators of [Kubeflow](https://www.kubeflow.org/),
such as TF operator and PyTorch operator.
Many development works have been done when integrating the operators,
including dashboard, command-line tool, and controllers with rich monitoring functions.
It's better to reuse the work.

So, we decide to apply the
[operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
to ElasticDL as well.
We introduce a [CRD](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
to define the workload of ElasticDL jobs.
Then, we describe each ElasticDL job with a YAML file according to the CRD.
The custom controller handles the request from the YAML file.

Please note the fact that the controller only launches
the master pod, and monitors the job.
The worker pods are still mananged by the master pod.
The controller does not take part in fault-torance and elastic scheduling features.

## Case study: Describing a MNIST Training Job

Let's use a real case to drive the design of ElasticDL CRD.
Following is the master pod YAML file of a MNIST job
dumped from the ElasticDL client tool.
It contains all the needed information.

```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    app: elasticdl
    elasticdl-job-name: test-mnist
    elasticdl-replica-index: '0'
    elasticdl-replica-type: master
  name: elasticdl-test-mnist-master
  namespace: default
spec:
  containers:
  - args:
    - -c
    - set -o pipefail; python -m elasticdl.python.master.main --worker_image 'elasticdl:test'
      --model_zoo 'model_zoo' --cluster_spec '' --minibatch_size '64' --log_level
      'INFO' --feed 'feed' --loss 'loss' --optimizer 'optimizer' --callbacks
      'callbacks' --eval_metrics_fn 'eval_metrics_fn' --custom_data_reader 'custom_data_reader'
      --model_def 'mnist.mnist_functional_api.custom_model' --model_params
      '' --get_model_steps '1' --data_reader_params '' --distribution_strategy 'ParameterServerStrategy'
      --checkpoint_steps '0' --checkpoint_dir '' --keep_checkpoint_max '0' --output
      '' --image_name 'elasticdl:test' --job_name 'test-mnist' --master_resource_request
      'cpu=1,memory=1024Mi' --master_resource_limit 'cpu=1,memory=2048Mi' --num_workers
      '8' --worker_resource_request 'cpu=2,gpu=1,memory=2048Mi' --worker_resource_limit
      'cpu=2,gpu=1,memory=2048Mi' --master_pod_priority '' --worker_pod_priority 'high=0.5' --num_ps_pods
      '1' --ps_resource_request 'cpu=2,memory=1024Mi' --ps_resource_limit 'cpu=2,memory=2048Mi'
      --ps_pod_priority 'high' --volume 'host_path=/data,mount_path=/data' --image_pull_policy
      'Never' --restart_policy 'Never' --envs '' --extra_pypi_index 'https://pypi.org/simple'
      --namespace 'default' --num_minibatches_per_task '2' --aux_params '' --log_file_path '' --tensorboard_log_dir
      '' --num_epochs '2' --grads_to_wait '1' --training_data '/data/mnist/train'
      --validation_data '' --evaluation_steps '0' --evaluation_start_delay_secs '100'
      --evaluation_throttle_secs '0' --checkpoint_dir_for_init '' --sync_version_tolerance
      '0' --log_loss_steps '100' --use_async 'False' --lr_staleness_modulation 'False'
    command:
    - /bin/bash
    env:
    - name: MY_POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
    image: elasticdl:test
    imagePullPolicy: Never
    name: elasticdl-test-mnist-master
    resources:
      limits:
        cpu: '1'
        memory: 2048Mi
      requests:
        cpu: '1'
        memory: 1024Mi
    volumeMounts:
    - mountPath: /data
      name: elasticdl-test-mnist-master-volume-0
  priorityClassName: ''
  restartPolicy: Never
  volumes:
  - hostPath:
      path: /host_data
    name: elasticdl-test-mnist-master-volume-0
```

We could rewrite it as a custom ElasticDLJob object
after the ElasticDL CRD is created.
The following is a sample:

```yaml
apiVersion: "elasticdl.org/v1"
kind: "ElasticAIJob"
metadata:
  name: "test-mnist"
spec:
  jobArgs:
  - "--model_zoo /model_zoo"
  - "--model_def mnist.mnist_functional_api.custom_model"
  - "--training_data /data/mnist/train"
  - "--valiation_data /data/mnist/val"
  - "--output /data/output"
  - "--minibatch_size 64"
  - "--num_minibatches_per_task 2"
  - "--evaluation_step 1000"
  master:
    image: elasticdl-mnist
    priority: high
    resource_request: "cpu=1,memory=1024Mi"
    volume: "host_path=/host_data,mount_path=/data"
  ps:
    count: 2
    priority: high
    image: elasticdl-ps
    resource_request: "cpu=1,memory=1024Mi"
  worker:
    count: 10
    priority: high=0.5
    image: elasticdl-worker
    resource_request: "cpu=4,gpu=1,memory=2048Mi"
    volume: "host_path=/host_data,mount_path=/data"
```

## ElasticDL CRD

This is a reference CRD design.
Each ElasticDL operator implementation may have its own design.

```yaml
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: eaijobs.elasticdl.org
spec:
  group: elasticdl.org
  scope: Namespaced
  versions:
  - name: v1alpha1
    served: true
    storaged: true
  names:
    kind: ElasticAIJob
    listKind: ElasticAIJobList
    singular: elasticaijob
    plural: elasticaijobs
    shortNames:
    - eaijob
  subresources:
    status: {}
  validation:
    openAPIV3Schema:
      properties:
        spec:
          properties:
            jobArgs:
              type: array
              items:
                type: string
                pattern: '^--([a-z0-9_]+)\s([a-z0-9_]+)$'
            Master:
              properties:
                image: string
                priority: string
                resource_request: string
                volume: string
            PS:
              properties:
                count: integer
                  minium: 0
                image: string
                priority: string
                resource_request: string
                volume: string
            Worker:
               properties:
                count: integer
                  minium: 1
                image: string
                priority: string
                resource_request: string
                volume: string
```

## ElasticDL Controller

Currently, we do not have a plan to give an
ElasticDL controller implementation in the community.
