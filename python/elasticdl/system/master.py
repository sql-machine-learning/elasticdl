from binascii import crc32
from ..recordio.file import File
import queue
import threading
import time


class Work(object):
    '''
    Represents a unit of work. All data members are read only.
    '''
    @staticmethod
    def _id(*parts):
        # convert each integer part into a 4-byte array, join them and return
        # crc32
        return crc32(b''.join(p.to_bytes(4, 'little') for p in parts))

    def __init__(self, file_index, offset, epoch, trial=0):
        self.file_index = file_index
        self.offset = offset
        self.epoch = epoch
        self.trial = trial
        # Finish earlier epoch first, but randomize the order of chunks within
        # an epoch.
        self.id = _id(epoch, file_index, offset)
        self.priority = (epoch, self.id)

    def next_trial(self):
        return Work(self.file_index, self.offset, self.epoch, self.trial + 1)

    def next_epoch(self):
        return Work(self.file_index, self.offset, self.epoch + 1)


class WorkQueue(object):
    def __init__(self, num_epoch, max_trial, files):
        self._files = files
        self._num_epoch = num_epoch
        self._max_trial = max_trial
        self._q = queue.PriorityQueue()
        self._in_flight = {}
        self._lock = threading.Lock()

    def put(self, file_index, offset):
        with self._lock:
            work = Work(file_index, offset, 0)
            self._q.put((work.priority, work))

    def get_work(self):
        with self._lock:
            work = self._q.get()
            # TODO: support worker timeout
            self._in_flight[work.id] = work
            return (self.id, self._files[work.file_index], work.offset)

    def work_done(self, id, succeed):
        with self._lock:
            work = self._in_flight.pop(id)
            next_work = None
            if not succeed:
                if work.trial + 1 < self._max_trial:
                    next_work = work.next_trial()
                else:
                    print('work failed', work)
            elif work.epoch + 1 < self._num_epoch:
                next_work = work.next_epoch()
            else:
                print('work finished', work)
            if next_work:
                self._q.put((next_work.priority, next_work))
            self._q.task_done()

    def join(self):
        self._q.join()


class Master(object):
    def __init__(self, data_files, num_epoch, max_trial):
        assert num_epoch > 0
        assert max_trial > 0

        self._data_files = data_files
        self._work_queue = WorkQueue(num_epoch, max_trial)
        self._lock = threading.Lock()
        self._num_workers = 0

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
                with File(f, 'r') as fd:
                    for c in range(fd.get_index().total_chunks()):
                        self._work_queue.put(i, fd.get_index().chunk_offset(c))
            print('Time spent on building index: %s seconds:' % time.time() - start)
        self._work_queue.join()
