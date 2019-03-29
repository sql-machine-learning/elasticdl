```python
# Basic design philosophy:
#
# - Workers are gRPC client, the master is the gRPC server.
# - The master keeps the three task queues: todo, doing, and done.
# - The master keeps the current global model and the model ID is the hash of the model parameters.

#------- worker.py -------#

model_params = NULL
model_version = NULL

module = parse_command_line_for_class_name()(
    parse_command_line_for_ctor_params())

master = grpc.create_client(getenv("MASTER_ADDR"))

while True:
    # Claim a task from the master. A task consists of a data segment and the
    # model id.  The meaning of a task is "to update the specified model with
    # the given data segment".
    task, err = master.GetTask()
    if err != NULL:
        continue # Retry to get task.

    # If the worker doesn't have the model to be updated, it downloads the
    # model from the master.
    if task.model_version != model_version:
        model_params, model_version, err = master.GetModel(task.model_version)
        if err != NULL:
            # Tell the master that the task is not completed, so the master
            # could move the task from the doing queue back to the todo queue.
            master.ReportResult(task, FAILED) 
            continue

    try:
        data = local_data(task.data_segment)
        cost = module.forward(data, model_params)
        gradients = module.backward(cost, model_params)
    except:
        master.ReportResult(task, FAILED)
        continue
    else:
        master.ReportResult(task, gradients)


#------- master.py -------#

inputs = parse_command_line().recordio_files() # No online learning in this version.
todo = partition_data_into_tasks(inputs)
doing = Queue()
done = Queue()

module = parse_command_line_for_class_name()(
    parse_command_line_for_ctor_params())

model_params = module.create_parameters().random_initialize()
model_version = 0

gradients = []

@grpc
def GetModel():
    return model_params, model_version


@grpc
def GetTask():
    task = todo.pop()
    doing.push(task)
    return task, model_version


@grpc
def ReportResult(task, result):
    if task.model_version != model_version:
        return # Ignore the report.

    if result == FAILED:
        # Move the failed task from doing back to todo.
        find_and_remove_task_from(doing, task)
        todo.push(task)
        return
    
    gradients = [gradients, result]
    find_and_remove_task_from(doing, task)
    done.push(task)
    if gradients.length() >= num_gradients_sufficient_to_update_model():
        model_params = optimize_model(model_params, gradients)
        model_version = model_version + 1
        gradients = [] # Clear out the buffer.
```
