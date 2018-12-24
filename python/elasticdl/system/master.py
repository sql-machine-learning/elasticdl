from binascii import crc32
from recordio import File
from contextlib import ExitStack
import queue
import threading
import time


__work_id = 1
__work_id_lock = threading.Lock()
def _new_work_id():
    """
    returns a unique work id. Thread safe. 
    """
    global __work_id
    global __work_id_lock
    with __work_id_lock:
        __work_id += 1
        return __work_id


def _crc32(*parts):
    # convert each integer part into a 4-byte array, join them and return
    # crc32
    return crc32(b"".join(p.to_bytes(4, "little") for p in parts))


class Work(object):
    """
    Represents a unit of work. All data members are read only.
    """

    def __init__(self, file_index, offset, epoch, trial=0):
        self.file_index = file_index
        self.offset = offset
        self.epoch = epoch
        self.trial = trial
        # Finish earlier epoch first, but randomize the order of chunks within
        # an epoch.
        self.id = _new_work_id()
        self.priority = (epoch, _crc32(epoch, file_index, offset))

    def next_trial(self):
        return Work(self.file_index, self.offset, self.epoch, self.trial + 1)

    def next_epoch(self):
        return Work(self.file_index, self.offset, self.epoch + 1)


class EvalWork(object):
    """
    Signals worker to evaluation
    """
    def __init__(self):
        self.id = _new_work_id()
        # assign a higher priority than gradient works
        self.priority = (-1, self.id)
        # TODO: add payload

class WorkQueue(object):
    def __init__(self, num_epoch, max_trial, files):
        self._files = list(files)
        self._num_epoch = num_epoch
        self._max_trial = max_trial
        self._q = queue.PriorityQueue()
        self._in_flight = {}
        self._lock = threading.Lock()

    def _put_work(self, work):
        with self._lock:
            self._q.put((work.priority, work))

    def put(self, file_index, offset):
        self._put_work(Work(file_index, offset, 0))
    
    def put_eval(self):
        self._put_work(EvalWork())

    def get_work(self, timeout=None):
        with self._lock:
            work = self._q.get(timeout=timeout)[1]
            # TODO: support worker timeout
            self._in_flight[work.id] = work
            if isinstance(work, EvalWork):
                return (work.id, "", -1)
            return (work.id, self._files[work.file_index], work.offset)

    def work_done(self, work_id, result):
        with self._lock, ExitStack() as stack:
            stack.callback(self._q.task_done)
            work = self._in_flight.pop(work_id)
            if isinstance(work, EvalWork):
                print("Eval id: %s, result: %s" % (work.id, result))
                return

            next_work = None
            if not result:
                if work.trial + 1 < self._max_trial:
                    next_work = work.next_trial()
                else:
                    print("work failed", work)
            elif work.epoch + 1 < self._num_epoch:
                next_work = work.next_epoch()
            else:
                print("work finished", work)
            if next_work:
                self._q.put((next_work.priority, next_work))

    def join(self):
        self._q.join()


class Master(object):
    def __init__(self, data_files, num_epoch, max_trial):
        assert num_epoch > 0
        assert max_trial > 0

        self._data_files = data_files
        self._work_queue = WorkQueue(num_epoch, max_trial, files=data_files)
        self._lock = threading.Lock()
        self._num_workers = 0
        self._runner = threading.Thread(target=self.run, name="master")

    def register_worker(self):
        with self._lock:
            self._num_workers += 1
            return self._work_queue

    def run(self):
        # TODO: use a thread pool to build index in parallel.
        with self._lock:
            # Technically, we don't need blocking for building indices, but
            # doing so makes the system cleaner as if there is a corrupted
            # file, etc., the master will crash and no worker will be started.
            start = time.time()
            for i, f in enumerate(self._data_files):
                with File(f, "r") as fd:
                    for chunk in fd.get_index():
                        self._work_queue.put(i, chunk.offset)
            # TODO: decide how to do periodic eval
            self._work_queue.put_eval()
            print(
                "Time spent on building index: %s seconds:"
                % (time.time() - start)
            )
        self._work_queue.join()

    def start(self):
        self._runner.start()

    def join(self):
        self._runner.join()
