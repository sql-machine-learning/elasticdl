import tempfile
from contextlib import closing

import numpy as np
import recordio
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import JobType
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker


class DatasetName(object):
    IMAGENET = "imagenet1"
    FRAPPE = "frappe1"
    TEST_MODULE = "test_module1"
    IMAGE_DEFAULT = "image_default1"


def create_recordio_file(size, dataset_name, shape, temp_dir=None):
    """Creates a temporary file containing data of `recordio` format.

    Args:
        size: The number of records in the temporary file.
        dataset_name: A dataset name from `DatasetName`.
        shape: The shape of records to be created.
        temp_dir: The storage path of the temporary file.

    Returns:
        A python string indicating the temporary file name.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            if dataset_name == DatasetName.IMAGENET:
                image = np.random.randint(255, size=shape, dtype=np.uint8)
                image = tf.image.encode_jpeg(tf.convert_to_tensor(value=image))
                image = image.numpy()
                label = np.ndarray([1], dtype=np.int64)
                label[0] = np.random.randint(1, 11)
                example_dict = {
                    "image": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image])
                    ),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])
                    ),
                }
            elif dataset_name == DatasetName.FRAPPE:
                feature = np.random.randint(5383, size=(shape,))
                label = np.random.randint(2, size=(1,))
                example_dict = {
                    "feature": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=feature)
                    ),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])
                    ),
                }
            elif dataset_name == DatasetName.TEST_MODULE:
                x = np.random.rand(shape).astype(np.float32)
                y = 2 * x + 1
                example_dict = {
                    "x": tf.train.Feature(
                        float_list=tf.train.FloatList(value=x)
                    ),
                    "y": tf.train.Feature(
                        float_list=tf.train.FloatList(value=y)
                    ),
                }
            elif dataset_name == DatasetName.IMAGE_DEFAULT:
                image = np.random.rand(np.prod(shape)).astype(np.float32)
                label = np.ndarray([1], dtype=np.int64)
                label[0] = np.random.randint(0, 10)
                example_dict = {
                    "image": tf.train.Feature(
                        float_list=tf.train.FloatList(value=image)
                    ),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])
                    ),
                }
            else:
                raise ValueError("Unknown dataset name %s." % dataset_name)

            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


def distributed_train_and_evaluate(
    feature_shape,
    model_zoo_path,
    model_def,
    model_params="",
    training=True,
    dataset_name=DatasetName.IMAGE_DEFAULT,
    callback_classes=[],
    use_async=False,
    get_model_steps=1,
):
    """Runs distributed training and evaluation with a local master. Grpc
    calls are mocked by local master call.

    Args:
        feature_shape: The shape of model input.
        model_zoo_path: The directory that contains user-defined model files
            or a specific model file.
        model_def: The import path to the model definition function/class in
            the model zoo, e.g.  "cifar10_subclass.CustomModel".
        model_params: The dictionary of model parameters in a string that will
            be used to instantiate the model, e.g. "param1=1,param2=2".
        training: True for job type `TRAIN_WITH_EVALUATION`, False for
            job type `EVALUATION`.
        dataset_name: A dataset name from `DatasetName`.
        callback_classes: A List of callbacks that will be called at given
            stages of the training procedure.
        use_async: A python bool. True if using asynchronous updates.
        get_model_steps: Worker will perform `get_model` from the parameter
            server every this many steps.

    Returns:
        An integer indicating the model version after the distributed training
        and evaluation.
    """
    job_type = (
        JobType.TRAINING_WITH_EVALUATION
        if training
        else JobType.EVALUATION_ONLY
    )
    batch_size = 8 if dataset_name == DatasetName.IMAGENET else 16
    worker = Worker(
        1,
        job_type,
        batch_size,
        model_zoo_path,
        model_def=model_def,
        model_params=model_params,
        channel=None,
        get_model_steps=get_model_steps,
    )

    if dataset_name in [DatasetName.IMAGENET, DatasetName.FRAPPE]:
        record_num = batch_size
    else:
        record_num = 128
    shards = {
        create_recordio_file(record_num, dataset_name, feature_shape): (
            0,
            record_num,
        )
    }
    if training:
        training_shards = shards
        evaluation_shards = shards
    else:
        training_shards = {}
        evaluation_shards = shards
    task_d = _TaskDispatcher(
        training_shards,
        evaluation_shards,
        {},
        records_per_task=64,
        num_epochs=1,
    )

    checkpoint_service = CheckpointService("", 0, 0, True)
    if training:
        evaluation_service = EvaluationService(
            checkpoint_service, None, task_d, 0, 0, 1, False
        )
    else:
        evaluation_service = EvaluationService(
            checkpoint_service, None, task_d, 0, 0, 0, True
        )
    task_d.set_evaluation_service(evaluation_service)
    grads_to_wait = 1 if use_async else 2
    master = MasterServicer(
        grads_to_wait,
        batch_size,
        worker._opt_fn(),
        task_d,
        init_var=[],
        checkpoint_filename_for_init="",
        checkpoint_service=checkpoint_service,
        evaluation_service=evaluation_service,
        use_async=use_async,
    )
    callbacks = [
        callback_class(master, worker) for callback_class in callback_classes
    ]
    worker._stub = InProcessMaster(master, callbacks)

    for var in worker._model.trainable_variables:
        master.set_model_var(var.name, var.numpy())

    worker.run()

    req = elasticdl_pb2.GetTaskRequest()
    req.worker_id = 1
    task = master.GetTask(req, None)
    # No more task.
    if task.shard_name:
        raise RuntimeError(
            "There are some tasks unfinished after worker exits."
        )
    return master._version
