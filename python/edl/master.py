from binascii import crc32
import queue

class Work(object):
    '''
    Represents a unit of work. All data members are read only.
    '''
    @staticmethod
    def _id(*parts):
        # convert each integer part into a 4-byte array, join them and return crc32
        return crc32(b''.join(for p in parts p.to_bytes(4, 'little')))

    def __init__(self, file_index, offset, epoch, trail=0):
        self.file_index = file_index
        self.offset = offset
        self.epoch = epoch
        self.trail = trail
        # Finish earlier epoch first, but randomize the order of chunks within an epoch.
        self.id = _id(epoch, file_index, offset)
        self.priority = (epoch, self.id)

    def next_trail(self):
        return Work(self.file_index, self.offset, self.epoch, self.trail + 1)

    def next_epoch(self):
        return Work(self.file_index, self.offset, self.epoch + 1)

class WorkQueue(object):
    def __init__(self, num_epoch, max_trail, files):
        self._files = files
        self._num_epoch = num_epoch
        self._max_trail = max_trail
        self._q = queue.PriorityQueue()
        self._in_flight = {}
    
    def put(self, file_index, offset):
        work = Work(file_index, offset, 0)
        self._q.put((work.priority, work))

    def get_work(self):
        work = self._q.get()
        self._in_flight[work.id] = work
        return (self.id, self._files[work.file_index], work.offset)
        
    def work_done(self, id, succeed):
        work = self._in_flight[id]
        next_work = None
        if not succeed:
            if work.trail + 1 < self._max_trail:
                next_work = work.next_trail() 
            else:
                print('work failed', work)
        elif work.epoch + 1 < self._num_epoch:
            next_work = work.next_epoch()
        else:
            print('work finished', work)
        if work:
            self._q.put(work)
        self._q.task_done()

class Master(object):
    def __init__(self, data_files, num_epoch, max_trail):
        assert num_epoch > 0
        assert max_trail > 0

        self._data_files = data_files
        self._work_queue = WorkQueue(num_epoch, max_trail)
        self._num_workers = 0
    
    # get recordio chunks and push them to workqueue
    def _prepare(self):
        pass

    def register_worker(self):
        self._num_workers += 1
        return self._work_queue
    
    

    

        
        
