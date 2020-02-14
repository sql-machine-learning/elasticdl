import csv
import os
import tempfile
from collections.__init__ import namedtuple
from contextlib import closing
from pathlib import Path

import grpc
import numpy as np
import recordio
import tensorflow as tf
from odps import ODPS

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import (
    DistributionStrategy,
    JobType,
    MaxComputeConfig,
)
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.data.recordio_gen.frappe_recordio_gen import (
    load_raw_data,
)
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.ps.parameter_server import Parameters, ParameterServer
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker


class PserverArgs(object):
    def __init__(
        self,
        grads_to_wait=8,
        lr_scheduler="learning_rate_scheduler",
        lr_staleness_modulation=0,
        sync_version_tolerance=0,
        use_async=False,
        model_zoo=None,
        model_def=None,
        optimizer="optimizer",
        port=9999,
        log_level="INFO",
        job_name="test_pserver",
        namespace="default",
        master_addr="test:1111",
        evaluation_steps=0,
        checkpoint_dir=None,
        checkpoint_steps=None,
        keep_checkpoint_max=0,
        ps_id=0,
        num_ps_pods=1,
        num_workers=2,
        checkpoint_dir_for_init=None,
    ):
        self.grads_to_wait = grads_to_wait
        self.learning_rate_scheduler = lr_scheduler
        self.lr_staleness_modulation = lr_staleness_modulation
        self.sync_version_tolerance = sync_version_tolerance
        self.use_async = use_async
        self.model_zoo = model_zoo
        self.model_def = model_def
        self.optimizer = optimizer
        self.port = port
        self.log_level = log_level
        self.job_name = job_name
        self.namespace = namespace
        self.master_addr = master_addr
        self.evaluation_steps = evaluation_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_steps = checkpoint_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.ps_id = ps_id
        self.num_ps_pods = num_ps_pods
        self.num_workers = num_workers
        self.checkpoint_dir_for_init = checkpoint_dir_for_init


class DatasetName(object):
    IMAGENET = "imagenet1"
    FRAPPE = "frappe1"
    TEST_MODULE = "test_module1"
    IMAGE_DEFAULT = "image_default1"
    CENSUS = "census1"


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
            elif dataset_name == DatasetName.CENSUS:
                example_dict = {
                    "workclass": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"Private"])
                    ),
                    "education": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"HS-grad"])
                    ),
                    "marital-status": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"Widowed"])
                    ),
                    "occupation": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[b"Exec-managerial"]
                        )
                    ),
                    "relationship": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"Not-in-family"])
                    ),
                    "race": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"White"])
                    ),
                    "sex": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"Female"])
                    ),
                    "native-country": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"United-States"])
                    ),
                    "age": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[np.random.randint(10, 100)]
                        )
                    ),
                    "capital-gain": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[np.random.randint(100, 4000)]
                        )
                    ),
                    "capital-loss": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[np.random.randint(2000, 7000)]
                        )
                    ),
                    "hours-per-week": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[np.random.randint(10, 70)]
                        )
                    ),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[np.random.randint(0, 2)]
                        )
                    ),
                }
            else:
                raise ValueError("Unknown dataset name %s." % dataset_name)

            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


def create_iris_csv_file(size, columns, temp_dir=None):
    """Creates a temporary CSV file.

    Args:
        size: The number of records in the CSV file.
        columns: The names of columns in the CSV file.
        temp_dir: The storage path of the CSV file.

    Returns:
        A python string indicating the temporary file name.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)

    features = np.random.random((size, 4))
    features = np.round(features, 4)
    labels = np.random.randint(0, 2, (size, 1))
    value_data = np.concatenate((features, labels), axis=1)

    csv_file_name = temp_file.name + ".csv"
    with open(csv_file_name, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)
        csv_writer.writerows(value_data)

    return csv_file_name


def create_pserver(
    model_zoo_path, model_def, grads_to_wait, use_async, num_ps_pods
):
    ports = [i + 12345 for i in range(num_ps_pods)]
    channels = []
    for port in ports:
        addr = "localhost:%d" % port
        channel = build_channel(addr)
        channels.append(channel)

    pservers = []
    for port in ports:
        args = PserverArgs(
            grads_to_wait=grads_to_wait,
            use_async=True,
            port=port,
            model_zoo=model_zoo_path,
            model_def=model_def,
        )
        pserver = ParameterServer(args)
        pserver.prepare()
        pservers.append(pserver)
    return ports, channels, pservers


def distributed_train_and_evaluate(
    feature_shape,
    model_zoo_path,
    model_def,
    model_params="",
    eval_metrics_fn="eval_metrics_fn",
    loss="loss",
    training=True,
    dataset_name=DatasetName.IMAGE_DEFAULT,
    callback_classes=[],
    use_async=False,
    get_model_steps=1,
    ps_channels=None,
    pservers=None,
    distribution_strategy=DistributionStrategy.PARAMETER_SERVER,
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
        eval_metrics_fn: The name of the evaluation metrics function defined
            in the model file.
        loss: The name of the loss function defined in the model file.
        training: True for job type `TRAIN_WITH_EVALUATION`, False for
            job type `EVALUATION`.
        dataset_name: A dataset name from `DatasetName`.
        callback_classes: A List of callbacks that will be called at given
            stages of the training procedure.
        use_async: A bool. True if using asynchronous updates.
        get_model_steps: Worker will perform `get_model` from the parameter
            server every this many steps.
        ps_channels: A channel list to all parameter server pods.
        pservers: A list of parameter server pods.
        distribution_strategy: The distribution startegy used by workers, e.g.
            DistributionStrategy.PARAMETER_SERVER or
            DistributionStrategy.AllreduceStrategy.

    Returns:
        An integer indicating the model version after the distributed training
        and evaluation.
    """
    job_type = (
        JobType.TRAINING_WITH_EVALUATION
        if training
        else JobType.EVALUATION_ONLY
    )
    evaluation_steps = 1 if job_type == JobType.TRAINING_WITH_EVALUATION else 0
    batch_size = 8 if dataset_name == DatasetName.IMAGENET else 16
    pservers = pservers or []
    ps_channels = ps_channels or []

    model_module = load_module(
        get_module_file_path(model_zoo_path, model_def)
    ).__dict__

    for channel in ps_channels:
        grpc.channel_ready_future(channel).result()
    worker_arguments = [
        "--worker_id",
        "1",
        "--job_type",
        job_type,
        "--minibatch_size",
        batch_size,
        "--model_zoo",
        model_zoo_path,
        "--model_def",
        model_def,
        "--model_params",
        model_params,
        "--loss",
        loss,
        "--get_model_steps",
        get_model_steps,
        "--distribution_strategy",
        distribution_strategy,
    ]
    args = parse_worker_args(worker_arguments)
    worker = Worker(args, ps_channels=ps_channels)

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

    if training:
        evaluation_service = EvaluationService(
            None,
            task_d,
            0,
            0,
            evaluation_steps,
            False,
            model_module[eval_metrics_fn],
        )
    else:
        evaluation_service = EvaluationService(
            None,
            task_d,
            0,
            0,
            evaluation_steps,
            True,
            model_module[eval_metrics_fn],
        )
    task_d.set_evaluation_service(evaluation_service)

    master = MasterServicer(
        batch_size, task_d, evaluation_service=evaluation_service,
    )
    callbacks = [
        callback_class(master, worker) for callback_class in callback_classes
    ]

    in_process_master = InProcessMaster(master, callbacks)
    worker._stub = in_process_master
    for pservicer in pservers:
        pservicer._master_stub = in_process_master

    worker.run()

    req = elasticdl_pb2.GetTaskRequest()
    req.worker_id = 1
    task = master.get_task(req, None)
    # No more task.
    if task.shard_name:
        raise RuntimeError(
            "There are some tasks unfinished after worker exits."
        )
    return master._version


IRIS_TABLE_COLUMN_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class",
]


def create_iris_odps_table(odps_client, project_name, table_name):
    sql_tmpl = """
    DROP TABLE IF EXISTS {PROJECT_NAME}.{TABLE_NAME};
    CREATE TABLE {PROJECT_NAME}.{TABLE_NAME} (
           sepal_length DOUBLE,
           sepal_width  DOUBLE,
           petal_length DOUBLE,
           petal_width  DOUBLE,
           class BIGINT);

    INSERT INTO {PROJECT_NAME}.{TABLE_NAME} VALUES
    (6.4,2.8,5.6,2.2,2),
    (5.0,2.3,3.3,1.0,1),
    (4.9,2.5,4.5,1.7,2),
    (4.9,3.1,1.5,0.1,0),
    (5.7,3.8,1.7,0.3,0),
    (4.4,3.2,1.3,0.2,0),
    (5.4,3.4,1.5,0.4,0),
    (6.9,3.1,5.1,2.3,2),
    (6.7,3.1,4.4,1.4,1),
    (5.1,3.7,1.5,0.4,0),
    (5.2,2.7,3.9,1.4,1),
    (6.9,3.1,4.9,1.5,1),
    (5.8,4.0,1.2,0.2,0),
    (5.4,3.9,1.7,0.4,0),
    (7.7,3.8,6.7,2.2,2),
    (6.3,3.3,4.7,1.6,1),
    (6.8,3.2,5.9,2.3,2),
    (7.6,3.0,6.6,2.1,2),
    (6.4,3.2,5.3,2.3,2),
    (5.7,4.4,1.5,0.4,0),
    (6.7,3.3,5.7,2.1,2),
    (6.4,2.8,5.6,2.1,2),
    (5.4,3.9,1.3,0.4,0),
    (6.1,2.6,5.6,1.4,2),
    (7.2,3.0,5.8,1.6,2),
    (5.2,3.5,1.5,0.2,0),
    (5.8,2.6,4.0,1.2,1),
    (5.9,3.0,5.1,1.8,2),
    (5.4,3.0,4.5,1.5,1),
    (6.7,3.0,5.0,1.7,1),
    (6.3,2.3,4.4,1.3,1),
    (5.1,2.5,3.0,1.1,1),
    (6.4,3.2,4.5,1.5,1),
    (6.8,3.0,5.5,2.1,2),
    (6.2,2.8,4.8,1.8,2),
    (6.9,3.2,5.7,2.3,2),
    (6.5,3.2,5.1,2.0,2),
    (5.8,2.8,5.1,2.4,2),
    (5.1,3.8,1.5,0.3,0),
    (4.8,3.0,1.4,0.3,0),
    (7.9,3.8,6.4,2.0,2),
    (5.8,2.7,5.1,1.9,2),
    (6.7,3.0,5.2,2.3,2),
    (5.1,3.8,1.9,0.4,0),
    (4.7,3.2,1.6,0.2,0),
    (6.0,2.2,5.0,1.5,2),
    (4.8,3.4,1.6,0.2,0),
    (7.7,2.6,6.9,2.3,2),
    (4.6,3.6,1.0,0.2,0),
    (7.2,3.2,6.0,1.8,2),
    (5.0,3.3,1.4,0.2,0),
    (6.6,3.0,4.4,1.4,1),
    (6.1,2.8,4.0,1.3,1),
    (5.0,3.2,1.2,0.2,0),
    (7.0,3.2,4.7,1.4,1),
    (6.0,3.0,4.8,1.8,2),
    (7.4,2.8,6.1,1.9,2),
    (5.8,2.7,5.1,1.9,2),
    (6.2,3.4,5.4,2.3,2),
    (5.0,2.0,3.5,1.0,1),
    (5.6,2.5,3.9,1.1,1),
    (6.7,3.1,5.6,2.4,2),
    (6.3,2.5,5.0,1.9,2),
    (6.4,3.1,5.5,1.8,2),
    (6.2,2.2,4.5,1.5,1),
    (7.3,2.9,6.3,1.8,2),
    (4.4,3.0,1.3,0.2,0),
    (7.2,3.6,6.1,2.5,2),
    (6.5,3.0,5.5,1.8,2),
    (5.0,3.4,1.5,0.2,0),
    (4.7,3.2,1.3,0.2,0),
    (6.6,2.9,4.6,1.3,1),
    (5.5,3.5,1.3,0.2,0),
    (7.7,3.0,6.1,2.3,2),
    (6.1,3.0,4.9,1.8,2),
    (4.9,3.1,1.5,0.1,0),
    (5.5,2.4,3.8,1.1,1),
    (5.7,2.9,4.2,1.3,1),
    (6.0,2.9,4.5,1.5,1),
    (6.4,2.7,5.3,1.9,2),
    (5.4,3.7,1.5,0.2,0),
    (6.1,2.9,4.7,1.4,1),
    (6.5,2.8,4.6,1.5,1),
    (5.6,2.7,4.2,1.3,1),
    (6.3,3.4,5.6,2.4,2),
    (4.9,3.1,1.5,0.1,0),
    (6.8,2.8,4.8,1.4,1),
    (5.7,2.8,4.5,1.3,1),
    (6.0,2.7,5.1,1.6,1),
    (5.0,3.5,1.3,0.3,0),
    (6.5,3.0,5.2,2.0,2),
    (6.1,2.8,4.7,1.2,1),
    (5.1,3.5,1.4,0.3,0),
    (4.6,3.1,1.5,0.2,0),
    (6.5,3.0,5.8,2.2,2),
    (4.6,3.4,1.4,0.3,0),
    (4.6,3.2,1.4,0.2,0),
    (7.7,2.8,6.7,2.0,2),
    (5.9,3.2,4.8,1.8,1),
    (5.1,3.8,1.6,0.2,0),
    (4.9,3.0,1.4,0.2,0),
    (4.9,2.4,3.3,1.0,1),
    (4.5,2.3,1.3,0.3,0),
    (5.8,2.7,4.1,1.0,1),
    (5.0,3.4,1.6,0.4,0),
    (5.2,3.4,1.4,0.2,0),
    (5.3,3.7,1.5,0.2,0),
    (5.0,3.6,1.4,0.2,0),
    (5.6,2.9,3.6,1.3,1),
    (4.8,3.1,1.6,0.2,0);
    """
    odps_client.execute_sql(
        sql_tmpl.format(PROJECT_NAME=project_name, TABLE_NAME=table_name),
        hints={"odps.sql.submit.mode": "script"},
    )


def get_odps_client_from_env():
    project = os.environ[MaxComputeConfig.PROJECT_NAME]
    access_id = os.environ[MaxComputeConfig.ACCESS_ID]
    access_key = os.environ[MaxComputeConfig.ACCESS_KEY]
    endpoint = os.environ.get(MaxComputeConfig.ENDPOINT)
    return ODPS(access_id, access_key, project, endpoint)


def create_iris_odps_table_from_env():
    project = os.environ[MaxComputeConfig.PROJECT_NAME]
    table_name = os.environ["MAXCOMPUTE_TABLE"]
    create_iris_odps_table(get_odps_client_from_env(), project, table_name)


def delete_iris_odps_table_from_env():
    project = os.environ[MaxComputeConfig.PROJECT_NAME]
    table_name = os.environ["MAXCOMPUTE_TABLE"]
    get_odps_client_from_env().delete_table(
        table_name, project, if_exists=True
    )


def get_random_batch(batch_size):
    shape = (28, 28)
    shape = (batch_size,) + shape
    num_classes = 10
    images = tf.random.uniform(shape)
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32
    )
    return images, labels


def get_mnist_dataset(batch_size):
    (
        (x_train, y_train),
        (x_test, y_test),
    ) = tf.keras.datasets.mnist.load_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.0
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db = db.batch(batch_size).repeat(2)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.batch(batch_size)

    return db, test_db


def get_frappe_dataset(batch_size):
    home = str(Path.home())
    Args = namedtuple("Args", ["data"])
    args = Args(data=os.path.join(home, ".keras/datasets"))
    x_train, y_train, x_val, y_val, x_test, y_test = load_raw_data(args)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.int64)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.int64)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

    db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db = db.batch(batch_size).repeat(2)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.batch(batch_size)
    return db, test_db


def save_checkpoint_without_embedding(model, checkpoint_dir, version=100):
    checkpoint_saver = CheckpointSaver(checkpoint_dir, 0, 0, False)
    params = Parameters()
    for var in model.trainable_variables:
        params.non_embedding_params[var.name] = var
    params.version = version
    model_pb = params.to_model_pb()
    checkpoint_saver.save(version, model_pb, False)
