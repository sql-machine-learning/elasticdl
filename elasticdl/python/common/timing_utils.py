import time

TIMING_GROUP = [
    "task_process",
    "batch_process",
    "get_model",
    "report_gradient",
]


class Timing(object):
    def __init__(self, enable, logger=None):
        self.enable = enable
        self.logger = logger
        self.timings = {}
        self.start_time = {}
        self.reset()

    def reset(self):
        if not self.enable:
            return
        for timing_type in TIMING_GROUP:
            self.timings[timing_type] = 0

    def start_record_time(self, timing_type):
        if not self.enable:
            return
        self.start_time[timing_type] = time.time()

    def end_record_time(self, timing_type):
        if not self.enable:
            return
        self.timings[timing_type] += time.time() - self.start_time[timing_type]

    def report_timing(self, reset=False):
        if not self.enable:
            return
        for timing_type in TIMING_GROUP:
            self.logger.debug(
                "%s time is %.6g seconds"
                % (timing_type, self.timings[timing_type])
            )
        if reset:
            self.reset()
