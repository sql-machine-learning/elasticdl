import argparse
import importlib
import os
import sys
from contextlib import contextmanager
from swamp_ps import SwampParameterServer
from swamp_worker import SwampWorker

# TODO: use @dataclass
class Handle(object):
    def __init__(self, prog, ps, worker):
        self.prog = prog
        self.ps = ps
        self.worker = worker


class ThreadLauncher(object):
    @staticmethod
    def launch(
        prog,
        num_worker,
        data,
        num_epoch=1,
        pull_probability=1.0,
        evaluation_frequency=4,
    ):
        print(
            "worker=%d, epoch=%d, pull probability=%1.6g"
            % (num_worker, num_epoch, pull_probability)
        )
        # launch ps
        ps = SwampParameterServer()

        workers = [
            SwampWorker(
                name="worker-%d" % i,
                ps=ps,
                umd=prog,
                train_dir=data + "/train",
                test_dir=data + "/test",
                epoch=num_epoch,
                pull_model_probability=pull_probability,
                evaluation_frequency=evaluation_frequency,
            )
            for i in range(num_worker)
        ]
        ps.start()
        for worker in workers:
            print("launch worker")
            worker.start()

        return Handle(prog, ps, workers)

    @staticmethod
    def shutdown(handle):
        # shutdown worker
        for w in handle.worker:
            w.join()
        handle.ps.join()


@contextmanager
def add_to_path(p):
    import sys

    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path


def path_import(absolute_path):
    """
    implementation taken from
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    with add_to_path(os.path.dirname(absolute_path)):
        spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def main(argv):
    def get_runner(runner):
        if runner == "thread":
            return ThreadLauncher()
        else:
            raise ValueError("Unknown runner: %s" % runner)

    def pos_int(x):
        v = int(x)
        if v < 0:
            raise ValueError("Positive integer required")
        return v

    parser = argparse.ArgumentParser(description="ElasticDL launcher.")
    parser.add_argument("script", help="training script")
    parser.add_argument(
        "--class_name",
        required=True,
        help=("python class that holds the model definition and optimizer, " "etc."),
    )
    parser.add_argument(
        "--runner", type=get_runner, required=True, help="training process runner"
    )

    parser.add_argument(
        "--num_worker", type=pos_int, required=True, help="number of workers"
    )
    parser.add_argument(
        "--num_epoch", type=pos_int, required=False, default=1, help="number of epoch"
    )
    parser.add_argument(
        "--pull_probability",
        type=float,
        required=False,
        default=1.0,
        help="the probability of pull model",
    )
    parser.add_argument(
        "--evaluation_frequency",
        type=pos_int,
        required=False,
        default=4,
        help="the evaluation frequency in batch number",
    )
    parser.add_argument(
        "--log_image", required=False, default="", help="filename for log image"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="base path that contains the input data in RecordIo format",
    )
    # TODO(l.zou): add support for passing arguments to the script.

    args = parser.parse_args(argv)

    # run script to create prog object
    module = path_import(args.script)
    prog = getattr(module, args.class_name)()

    # launch
    handle = args.runner.launch(
        prog,
        args.num_worker,
        args.input,
        args.num_epoch,
        args.pull_probability,
        args.evaluation_frequency,
    )
    args.runner.shutdown(handle)
    accuracy = handle.ps.get_accuracy()
    info = "w=%d p=%1.2g" % (args.num_worker, args.pull_probability)
    if len(args.log_image):
        file_name = args.log_image
    else:
        file_name = "w%dp%1.2ga%1.6g.png" % (
            args.num_worker,
            args.pull_probability,
            accuracy,
        )
    handle.ps.save_log_to_image(file_name, info)


if __name__ == "__main__":
    main(sys.argv[1:])
