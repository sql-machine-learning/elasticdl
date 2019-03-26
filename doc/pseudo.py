
def BuildTasks(shard_names, num_record_per_shard, num_record_per_task):
    # Returns a list of (shard_name, start, end) tuple

class TaskQueue(object):
    def __init__(self, shard_names, num_record_per_shard, num_epoch, num_record_per_task):
        self._num_epoch = num_epoch
        self._cur_epoch = 0
        self._dispatched_tasks = []
        self._tasks = BuildTasks(shard_names, num_record_per_shard, num_record_per_task)
        random.shuffle(self._tasks)
        
    def next(self):
        if not self._tasks:
            if self._cur_epoch >= self._num_epoch:
                raise StopIteration()
            self._tasks = self._dispatched_tasks
            self._dispatched_tasks = []
            random.shuffle(self._tasks)
            self._cur_epoch += 1
        
        task = self._tasks.pop()
        self._dispatched_tasks.append(task)
        return task

# data structure that keeps worker state, shared between  K8sEventListener and WorkerManager
# data members need mutex protection
class WorkerStates(object):
    def __init__(self):
        self.to_launch = []
        self.to_kill = []
        self.ready = []
        self._last_ready_time = 0
        self._last_killed_time = 0
        self._last_failed_launch_time = 0

    def num_tentative_workers(self):
        return len(self.to_launch) + len(self.ready) - len(self.to_kill)

# Listen on k8s events and update worker states.
class K8sEventListener(object):
    def __init__(self, event_filter, worker_states):
        # Check with the actual watch interface
        k8s_client.watch(event_filter, self.callback)
        self._worker_states = worker_states

    # Need to double check k8s event delivery guarantees
    def callback(self, event):
        if event.state == READY:
            self._worker_states._last_ready_time = time.Now()
            if event.pod in self._worker_states.to_launch:
                # Move pod from to_launch to ready
                self._worker_states.ready.append(event.pod)
                self._worker_states.to_launch.remove(event.pod)
            else:
                # Pod not launch by us. To be decided what to do

        elif event.state == KILLED:
            self._worker_states._last_killed_time = time.Now()
            self._worker_states.ready.remove(event.pod)
            if event.pod in self._worker_states.to_kill:
                self._worker_states.to_kill.remove(event.pod)
            else:
                # Pod killed externally by k8s. We could treat it differently from internal kill.

        elif event == LAUNCH_FAILED:
            self._worker_states._last_failed_launch_time = time.now()


# Monitor and manage worker numbers.
class WorkerManager(object):
    def __init__(self, num_workers, worker_spec, worker_states, master_addr):
        self._num_workers = num_workers
        self._master_addr = master_addr
        self._worker_states = worker_states

    # WorkerManager loop
    def run(self):
        while True:
            if self._num_workers > self._worker_states.num_tentative_workers():
                # If a worker just got killed or failed to launch, we should backoff.
                if ShouldLaunch(time.Now(), self._worker_states):
                    LaunchWorker(worker_spec, self._master_addr)
                    self._worker_states.to_launch.append(pod)
            elif self._num_workers > self._worker_states.num_tentative_workers():
                pod = FindWorkerToKill()
                KillWorker(pod)
                self._worker_states.to_kill.append(pod)
            time.sleep(30)

# Master service
message Task{
    string shard_name;  // Name of RecordIO shard file name 
    int32 start; // starting record number
    int32 end; // ending record number 
}

message Model{
    int32 model_iteration;
    map<string, double> parameters;
}

message Gradients{
    int32 model_iteration
    map<string, double> gradients;
}

rpc GetTask(Empty) returns (Task)
rpc GetModel(Empty) returns (stream Model)
rpc PushGradient(Gradients) returns (Empty)

class Master(object):
    def __init__(self, shard_names, num_record_per_shard, num_epoch, num_record_per_task, num_workers, worker_spec, num_gradients_to_wait):
        # Create TaskQueue
        self._task_q = TaskQueue(self, shard_names, num_record_per_shard, num_epoch, num_record_per_task):
        # Create a shared worker_states
        worker_states = WorkerStates()
        # Create event listener
        self._event_listener = K8sEventListener(MakeFilter(worker_spec), worker_states)
        # Create WorkerManager and start manager loop
        self._worker_manager = WorkerManager(num_workers, worker_spec, worker_states)
        threading.Thread(self._worker_mamager.run()).start()

        # Randomized or loaded from a saved model
        self._model = initial_model()
        self._model_iteration = 0

        self._num_gradients = 0
        self._accumulated_grad = {}
        # Condition variable to signal model update.
        self._model_cv = threading.Condition()

    # RPC method
    def GetTask(self):
        try:
            task = self._task_q.next()
        except StopIteration:
            # All tasks finished, set a flag to let worker exit.
            task = Task()
            task.shard_name = ""
        rpc.reply(task)


    # RPC method
    def PushGradient(self, gradients):
        if gradients.model_iteration < self._model_iteration:
            LOG("gradients dropped")
            return
        # Need mutex protection.
        self._accumulated_grad += gradients
        self._num_gradients += 1
        if self._num_gradients >= num_gradients_to_wait:
            self._model += self._accumulated_grad * learning_rate
            self._model_iteration += 1
            self._model_cv.signal()
            self._num_gradients = 0
    
    # RPC method
    def GetModel(self):
        while True:
            self._model_cv.wait()
            model = BuildModelProto(self._model_iteration, self._model_iteration)
            yield model
 
    # TODO: other methods we might need.
    def HTTPServer()
    def SaveModel()
    def LoadModel()


class Worker(object):
    def __init__(self, user_module, master_addr):
        self._master_client = MasterStub(master_addr)
        self._model = None
        self._model_iteration = None

    #  Main loop
    def main_loop(self):
        while True:
            task = self._master_client.GetTask()
            if not task.shard_name:
                # All tasks finished. make worker exit with success status. 
                sys.exit(0)
            # Read label and data tensor.
            label, data = ReadRecordIO(task.shard_name, task.start, task.end)
            gradients = ComputeGradients(data, label, model, user_module)
            pb = BuildGradientProto(gradients, self._model_iteration)
            self._master_client.PushGradient(pb)

    # Model Update Loop
    def model_loop(self):
        for m in self._master_client.GetModel():
            # Need mutex protection
            self._model, self._model_iteration = m
    
    def run(self):
        # Start model_loop in a thread and wait until there is a model
        t = Threading.thread(self.model_loop)
        t.daemon = True
        t.start()
        while not self._model:
            time.sleep(30)
        main_loop()
