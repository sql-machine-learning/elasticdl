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

    task_status = SUCCEED
    for minibatch in read_data(task):
        try:
            # If the current model_version on the worker is older than the model
            # on the master, master.UpdateLocalModel updates model_version and 
            # model_params; otherwise, it leaves these two variables unchanged.
            master.UpdateModel(&model_version, &model_params)
            cost = module.forward(data, model_params)
            gradients = module.backward(cost, model_params)
        except:
            task_status = FAILED
            break
        else:
            master.ReportGradients(task, gradients)
    master.ReportTask(task, task_status)
    

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
def UpdateModel(mv, mp):
    if model_version != mv:
        copy(*mv, model_version)
        copy(*mp, model_params)


@grpc
def GetTask():
    task = todo.pop()
    doing.push(task)
    return task


@grpc
def ReportGradients(task, result):
    if task.model_version != model_version:
        return # Ignore the report.

    gradients = [gradients, result]
    if len(gradients) >= num_gradients_sufficient_to_update_model():
        model_params = optimize_model(model_params, gradients)
        model_version = model_version + 1
        gradients = [] # Clear out the buffer.


@grpc
def ReportTask(task, status):
    if status == FAILED:
        move_task(task, doing, todo) # Move the task from doing back to todo
    else:
        move_task(task, doing, done) # Move the task from doing to done
```
