"""In process master for unittests"""


class InProcessMaster(object):
    def __init__(self, master):
        self._m = master

    def GetTask(self, req):
        return self._m.GetTask(req, None)

    def GetModel(self, req):
        return self._m.GetModel(req, None)

    def ReportGradient(self, req):
        return self._m.ReportGradient(req, None)

    def ReportEvaluationMetrics(self, req):
        return self._m.ReportEvaluationMetrics(req, None)

    def ReportTaskResult(self, req):
        return self._m.ReportTaskResult(req, None)
