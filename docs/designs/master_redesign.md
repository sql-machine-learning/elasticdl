# Master Design

This document is a new design of the ElasticDL master.

## Motivation

We plan to provide elastic training feature to customized training loop for
AllReduce and other deep learning frameworks. Currently the master is designed
for ElasticDL's own training framework. This requires us to re-design the
master. After this re-design, the master consists of a few modular components
and only launches required components when providing elastic training feature
in different scenarios.

## Master Components

The master has four modular components: elastic pod manager, task manager,
rendezvous server and ElasticDL training/evaluation service.

### Elastic Pod Manager

The elastic pod manager is responsible for:

1. Create PS/worker pods;
2. Monitor pod events, keep track all pods status;
3. Relaunch failed worker pods when needed.

### Task Manager

The task manager is responsible for creating tasks and dispatching tasks to
workers. Workers will ask the task manager for new tasks and report task
completion status to the task manager when needed. Optionally, the task
mamager supports task fault-tolerance by recovering un-completed tasks from
failed workers and re-assigning to other workers.

For training jobs, the task manager supports dynamic sharding and partitions
all training data into shards. Each task contains a data shard.

### Rendezvous Server

The rendezvous server supports for elastic AllReduce. After detecting worker
pod number change, it creates a new worker host plan so that worker pods can
re-initialize Horovod.

### ElasticDL Training/Evaluation Service

All other current master functions except the above three components. ElasticDL
 raining/evaluation service supports for ElasticDL native training. It includes
evaluation service, supporting for model saving and checkpoint, etc.

Thus, we can launch the master in seven component combinations:

1. Task manager only (without fault-tolerance);
2. Elastic pod manager only;
3. Elastic pod manager + Rendezvous server;
4. Elastic pod manager + Task manager;
5. Elastic pod manager + Rendezvous server + Task manager;
6. Elastic pod manager + Task manager + ElasticDL training/evaluation service;
7. Elastic pod manager + Task manager + Rendezvous server + ElasticDL
training/evaluation service.

Case | Elastic pod manager | Task manager | Rendezvous server | ElasticDL training/evaluation service | Usage
---|---|---|---|---|---
1 | | Y (without fault-tolerance) | | | Dynamic sharding service (rarely used alone)
2 | Y | | | | Provide elastic training to other DL frameworks for PS-based training. DL frameworks need to have their own dynamic sharding implementations.
3 | Y | | Y | | Provide elastic training to other DL frameworks for AllReduce training. DL frameworks need to have their own dynamic sharding implementations.
4 | Y | Y | | | Provide elastic training and dynamic shading to other DL frameworks for PS-based training.
5 | Y | Y | Y | | Provide elastic training and dynamic shading to other DL frameworks for AllReduce training.
6 | Y | Y | | Y | ElasticDL PS-based training.
7 | Y | Y | Y | Y | ElasticDL AllReduce training.

## Implementation

Current version of the master

```python
Init
- get master addr/port, job_type
- create HorovodRendezvousServer if needed.
- load model_inst/optimizer/custom_data_reader
  - model_inst used for tensorflow.python.keras.callbacks
  - optimizer to get info for go-PS.
  - custom_data_reader to get create_data_reader_fn for create_shard()
- init callbacks
- get model version from checkpoint
- create task_d (task_dispatcher)
- deferred_callback_create_train_end_task
- create evaluation service if needed
- create instance manager
- create master service
- timeout task check thread (recover task, remove worker if timeout)

Prepare
- start evaluation service if needed.
- start rpc server
- rendezvous service start if needed.
- start instance_manager
  - start PS
  - start workers

Run:
  loop until no more tasks, or all workers completed.

```

We need to add more arguments

- need\_pod\_manager (default:True)
- need\_task\_manager (default: True)
- need\_training\_service (default: True)
- task\_fault\_tolerance (default: True)
- worker\_command (provided by ElasticDL operator, if none, using default
`python -m elasticdl.python.worker.main args`)

```python
Init
 args = parse_args()

 # create pod manager if args.need_pod_manager
 pod_manager = create_pod_manager_if_needed(args)

 # create task_manager if args. need_task_manager
 task_manager = create_task_manager_if_needed(args, pod_manager)

 # create rendezvous server if args.distribution_strategy==AllreduceStrategy
 rendezvous_server = create_rendezvous_server_if_needed(args, pod_manager)

 # init ElasticDL training/evaluation service if args.need_training_service
 training_service = init_training_service_if_needed(args)

 # create master rpc service
 rpc_server = create_rpc_service()


Prepare
  rpc_server.start()
  if pod_manager:
    pod_manager.start()
  if task_manager:
    task_manager.start()
  if rendezvous_server:
    rendezvous_server.start()
  if training_service:
     training_service.start()


Run
  loop until no more tasks, or all workers completed.

```

Note that k8s client with watch is included in the pod manager, which is using
an event callback defined inside the pod manager. Thus, both the task manager
with fault-tolerance and the rendezvous server need to enable its callback
functions in that event callback.

### Implementation Steps

1. Create the above framework, but put everything in the current master version
into training_service. We will remove most components out of it after
implementing them as modular components.
2. Move rpc server as a modular components.
3. Implement pod_manager.
4. Implement task_manager.
5. Implement rendezvous_server.

Step 2 to 5 can be implemented in parallel.
