"""In process master for unittests"""
from elasticdl.python.tests import test_call_back


class InProcessMaster(object):
    def __init__(self, master, callbacks=[]):
        self._m = master
        self._callbacks = callbacks

    def get_task(self, req):
        return self._m.get_task(req, None)

    """
    def ReportGradient(self, req):
        for callback in self._callbacks:
            if test_call_back.ON_REPORT_GRADIENT_BEGIN in callback.call_times:
                callback()
        return self._m.ReportGradient(req, None)
    """

    def report_evaluation_metrics(self, req):
        for callback in self._callbacks:
            if test_call_back.ON_REPORT_EVALUATION_METRICS_BEGIN in (
                callback.call_times
            ):
                callback()
        return self._m.report_evaluation_metrics(req, None)

    def report_task_result(self, req):
        return self._m.report_task_result(req, None)

    def report_version(self, req):
        return self._m.report_version(req, None)
