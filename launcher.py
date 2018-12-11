import argparse
import importlib
import os
from contextlib import contextmanager
from elasticdl.tflib import ParameterServer, no_partition, ParameterServerClient, Worker 
from elasticdl.system import Master 


# TODO: use @dataclass
class Handle(object):
    def __init__(self, prog, ps, worker, master):
        self.prog = prog
        self.ps = ps
        self.worker = worker
        self.master = master


class ThreadLauncher(object):
    @staticmethod
    def launch(prog, num_ps, num_worker, data):
        # launch ps
        vars = prog.vars()
        ps = [
            ParameterServer(prog.optimizer(), vars)
            for _ in range(num_ps)
        ]
        for p in ps:
            p.start()

        # launch master
        filenames = []
        for root, dirs, names in os.walk(data, False):
            for filename in names:
                filenames.append(os.path.join(root,filename))
        master = Master(
            filenames,
            num_epoch=1,
            max_trial=1)
        master.start()
 
        # launch worker
        ps_client = ParameterServerClient(ps_configs=ps, partition_func=no_partition) 
        workers = [
            Worker(
                ps_client=ps_client,
                work_queue = master.register_worker(),
                forward_func = prog.forward,
                loss_func = prog.loss,
                optimizer = prog.optimizer(),
            )
            for _ in range(num_worker)
        ]
        for worker in workers:
            worker.start()

        return Handle(prog, ps, None, master)

    @staticmethod
    def shutdown(handle):
        # TODO: shutdown master
        # TODO: shutdown worker
        # shutdown ps
        for p in handle.ps:
            p.join()


# TODO: move this to a separate lib.
# copied from https://stackoverflow.com/questions/41861427


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
        spec = importlib.util.spec_from_file_location(
            absolute_path, absolute_path
        )
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
        help=(
            "python class that holds the model definition and optimizer, "
            "etc."
        ),
    )
    parser.add_argument(
        "--runner",
        type=get_runner,
        required=True,
        help="training process runner",
    )
    parser.add_argument(
        "--num_ps",
        type=pos_int,
        required=True,
        help="number of parameter servers",
    )
    parser.add_argument(
        "--num_worker", type=pos_int, required=True, help="number of workers"
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
    handle = args.runner.launch(prog, args.num_ps, args.num_worker, args.input)
    args.runner.shutdown(handle)


if __name__ == "__main__":
    main(
        [
            "test_data/mnist.py",
            "--class_name",
            "MnistCNN",
            "--runner",
            "thread",
            "--num_ps",
            "1",
            "--num_worker",
            "1",
            "--input",
            "/home/chris/project/github/elasticdl/python/elasticdl/datasets/mnist/fasion_train",
        ]
    )
