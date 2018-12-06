import queue
import unittest
import threading
from unittest.mock import patch
from collections import Counter
from recordio.file_index import _ChunkData as C
from master import Master

# used to keep some shared data across workers.
class RunLog(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._failures = set()
        self._run = Counter()

    # Return False to signal a failure
    def add_run(self, name, offset):
        with self._lock:
            key = (name, offset)
            # make every work to fail once
            ret = self._run[key] != 0
            self._run[key] += 1
            return ret
    
    def get(self):
        return self._run

run_log = RunLog()

class MockWorkerThread(threading.Thread):
    def __init__(self, q):
        self._q = q
        self._exiting = False
        threading.Thread.__init__(self)
    
    def run(self):
        while not self._exiting:
            try:
                work = self._q.get_work(timeout=1.0)
                success = run_log.add_run(work[1], work[2])
                self._q.work_done(work[0], success)
            except queue.Empty:
                pass
            
    
    def exit(self):
        self._exiting = True

class MockRecordIoFile(object):
    def __init__(self, index):
        self._index = index
    def __enter__(self):
        return self
    def __exit__(self,exc_type, exc_val, exc_tb):
        return
    def get_index(self):
        return self._index


class MockMasterThread(threading.Thread):
    def __init__(self):
        self.recordio_data = {'f0': [C(0, 100, 2), C(200, 100, 3)], 'f1': [C(10, 200, 4), C(210, 200, 4)]}
        self.master = Master(self.recordio_data.keys(), num_epoch=3, max_trial=2)
        threading.Thread.__init__(self)
    
    def run(self):
        # patch Master's recordio calls to inject mock data
        with patch('master.File', autospec=True) as mock:
            mock.side_effect=[MockRecordIoFile(index) for index in self.recordio_data.values()]
            self.master.run()

class MasterTest(unittest.TestCase):
    def test_normal(self):
        m = MockMasterThread()
        w1 = MockWorkerThread(m.master.register_worker())
        w2 = MockWorkerThread(m.master.register_worker())

        m.start()
        w1.start()
        w2.start()

        m.join()
        w1.exit()
        w2.exit()

        w1.join()
        w2.join()

        print(run_log.get())
        # Every work got executed 4 time, including 3 epochs and 1 retry.
        self.assertCountEqual(run_log.get().elements(), [('f0', 0), ('f1', 210), ('f1', 10), ('f0', 200)] * 4)

if __name__ == '__main__':
    unittest.main()
