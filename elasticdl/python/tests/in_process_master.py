"""In process master for unittests"""
from elasticdl.python.tests import test_call_back


class InProcessMaster(object):
    def __init__(self, master, callbacks=[]):
        self._m = master
        self._callbacks = callbacks

    def GetTask(self, req):
        return self._m.GetTask(req, None)

    def GetModel(self, req):
        return self._m.GetModel(req, None)

    def ReportVariable(self, req):
        return self._m.ReportVariable(req, None)

    def ReportGradient(self, req):
        for callback in self._callbacks:
            if test_call_back.ON_REPORT_GRADIENT_BEGIN in callback.call_times:
                callback()
        return self._m.ReportGradient(req, None)

    def ReportEvaluationMetrics(self, req):
        for callback in self._callbacks:
            if test_call_back.ON_REPORT_GRADIENT_BEGIN in callback.call_times:
                callback()
        return self._m.ReportEvaluationMetrics(req, None)

    def ReportTaskResult(self, req):
        return self._m.ReportTaskResult(req, None)
