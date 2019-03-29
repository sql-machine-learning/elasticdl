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
    if err == NO_MORE_TASK:
        break    # Training completed.
    if err != NULL:
        continue # Retry to get task.

    task_status = SUCCEED
    for minibatch in read_data(task):
        accepted = False
        report_count = 0
        while not accepted:
            try:
                # If the current model_version on the worker is older than the model
                # on the master, this call updates model_version and 
                # model_params; otherwise, it leaves these two variables unchanged.
                master.UpdateModelIfOutOfDate(&model_version, &model_params)
                cost = module.forward(data, model_params)
                gradients = module.backward(cost, model_params)
            except:
                task_status = FAILED
                break
            else:
                # If the reported gradients are not accepted by the master due to old model_version,
                # try the minibatch again with the updated model in the next while loop.
                # Fail the task if the minibatch report count exceeds a predefined threshold.
                accepted = master.ReportGradients(model_version, gradients)
                if not accepted:
                    report_count += 1
                    if report_count == PREDEFINED_MAX_REPORT_COUNT:
                        task_status = FAILED
                        break
        if task_status == FAILED:
            break
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
def UpdateModelIfOutOfDate(mv, mp):
    if model_version != mv:
        copy(*mv, model_version)
        copy(*mp, model_params)


@grpc
def GetTask():
    task = todo.pop()
    doing.push(task)
    return task


@grpc
def ReportGradients(mv, grads):
    accepted = False
    if mv == model_version:
        gradients += grads
        accepted = True
        if len(gradients) >= num_gradients_sufficient_to_update_model():
            model_params = optimize_model(model_params, gradients)
            model_version = model_version + 1
            gradients = [] # Clear out the buffer.
    return accepted


@grpc
def ReportTask(task, status):
    if status == FAILED:
        move_task(task, doing, todo) # Move the task from doing back to todo
    else:
        move_task(task, doing, done) # Move the task from doing to done
```
