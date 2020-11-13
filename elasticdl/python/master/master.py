# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading
import time
from concurrent import futures

import grpc
from kubernetes.client import V1EnvVar

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.args import wrap_go_args_with_string
from elasticdl.python.common.constants import (
    GRPC,
    InstanceManagerStatus,
    JobType,
)
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_utils import (
    get_dict_from_params_str,
    get_module_file_path,
    get_optimizer_info,
    load_callbacks_from_module,
    load_model_from_module,
    load_module,
    set_callback_parameters,
)
from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.data.reader.data_reader_factory import create_data_reader
from elasticdl.python.elasticdl.callbacks import MaxStepsStopping
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.k8s_instance_manager import InstanceManager
from elasticdl.python.master.rendezvous_server import HorovodRendezvousServer
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl_client.common.args import (
    build_arguments_from_parsed_result,
    parse_envs,
    wrap_python_args_with_string,
)
from elasticdl_client.common.constants import (
    BashCommandTemplate,
    ClusterSpecConfig,
    DistributionStrategy,
)


def _make_task_dispatcher(
    training_data,
    validation_data,
    prediction_data,
    records_per_task,
    num_epochs,
    data_reader_params,
    create_data_reader_fn,
    callbacks_list,
):
    def _maybe_create_shards(data_origin):
        kwargs = get_dict_from_params_str(data_reader_params)
        partition = kwargs.get("partition", None) if kwargs else None
        return (
            create_data_reader_fn(
                data_origin=data_origin,
                records_per_task=records_per_task,
                partition=partition,
            ).create_shards()
            if data_origin
            else {}
        )

    prediction_f_records = _maybe_create_shards(prediction_data)

    return _TaskDispatcher(
        _maybe_create_shards(training_data),
        _maybe_create_shards(validation_data),
        prediction_f_records,
        records_per_task,
        # Only generate prediction tasks for 1 epoch
        1 if prediction_f_records else num_epochs,
        callbacks_list,
    )


class Master(object):
    def __init__(self, args):
        self.logger = get_logger("master", level=args.log_level.upper())

        self.num_ps_pods = args.num_ps_pods
        self.checkpoint_output_path = args.checkpoint_dir
        self.distribution_strategy = args.distribution_strategy

        # Master addr
        master_ip = os.getenv("MY_POD_IP", "localhost")
        self.master_addr = "%s:%d" % (master_ip, args.port)
        self.job_type = Master._get_job_type(args)
        self.rendezvous_server = None
        if self.distribution_strategy == DistributionStrategy.ALLREDUCE:
            self.rendezvous_server = HorovodRendezvousServer(master_ip)

        # Initialize the components from the model definition
        model_module = load_module(
            get_module_file_path(args.model_zoo, args.model_def)
        ).__dict__
        if not args.custom_training_loop:
            model_inst = load_model_from_module(args.model_def, model_module)
            self._optimizer = model_module[args.optimizer]()

            # Initialize the callbacks
            self.callbacks_list = load_callbacks_from_module(
                args.callbacks, model_module
            )
            self.callbacks_list.set_model(model_inst)
            set_callback_parameters(
                self.callbacks_list,
                batch_size=args.minibatch_size,
                saved_model_path=args.output,
                checkpoint_path=args.checkpoint_dir,
            )
        else:
            self.callbacks_list = None
            self._optimizer = None

        self._set_completed_steps_by_checkpoint(args.checkpoint_dir_for_init)

        self._create_data_reader_fn = create_data_reader
        if args.custom_data_reader in model_module:
            self._create_data_reader_fn = model_module[args.custom_data_reader]

        # Start task queue
        records_per_task = args.minibatch_size * args.num_minibatches_per_task
        self.task_d = _make_task_dispatcher(
            args.training_data,
            args.validation_data,
            args.prediction_data,
            records_per_task,
            args.num_epochs,
            args.data_reader_params,
            self._create_data_reader_fn,
            self.callbacks_list,
        )

        self.task_d.add_deferred_callback_create_train_end_task()
        if args.eval_metrics_fn in model_module:
            self.evaluation_service = self._create_evaluation_service(
                model_module[args.eval_metrics_fn], args.evaluation_steps
            )
        else:
            self.evaluation_service = None

        # Initialize instance manager
        self.instance_manager = self._create_instance_manager(args)

        # Initialize master service
        self.master_servicer, self.server = self._create_master_service(args)

        self._should_stop = False
        self._exit_code = 0
        threading.Thread(
            target=self._check_timeout_tasks,
            name="check_timeout_tasks",
            daemon=True,
        ).start()

    def _set_completed_steps_by_checkpoint(self, checkpoint_dir_for_init):
        if not checkpoint_dir_for_init:
            return

        if not CheckpointSaver.check_checkpoint_valid(checkpoint_dir_for_init):
            raise ValueError(
                "Invalid checkpoint directory {}".format(
                    checkpoint_dir_for_init
                )
            )

        model_verion = CheckpointSaver.get_version_from_checkpoint(
            checkpoint_dir_for_init
        )
        for callback in self.callbacks_list.callbacks:
            if isinstance(callback, MaxStepsStopping):
                callback.set_completed_steps(model_verion)

    def request_stop(self, err_msg=None):
        """Request master to quit"""
        self._should_stop = True
        if err_msg:
            self.logger.error(err_msg)
            # TODO (chengfu.wcy) create meaningful status codes
            self._exit_code = -1

    def prepare(self):
        """
        Start the components one by one. Make sure that it is ready to run.
        """
        # Start the master GRPC server
        self.logger.info("Starting master RPC server")
        self.server.start()
        self.logger.info("Master RPC server started")

        # Start the worker manager if requested
        if self.instance_manager:
            self.instance_manager.update_status(InstanceManagerStatus.PENDING)
            if self.distribution_strategy == DistributionStrategy.ALLREDUCE:
                # Start rendezvous server for workers to initialize Horovod
                self.rendezvous_server.start()
            else:
                self.instance_manager.start_parameter_servers()
            self.instance_manager.start_workers()
            self.instance_manager.update_status(InstanceManagerStatus.RUNNING)

    def run(self):
        """
        The main loop of master.
        Dispatch the tasks to the workers until all the tasks are completed.
        """
        try:
            while True:
                if self.instance_manager.all_workers_failed:
                    raise Exception(
                        "All workers fail with unrecoverable errors"
                    )
                    break
                if self.task_d.finished():
                    if self.instance_manager:
                        self.instance_manager.update_status(
                            InstanceManagerStatus.FINISHED
                        )
                    break
                if self.instance_manager.all_workers_completed:
                    self.logger.info(
                        "All workers completed but there are unfinished tasks"
                    )
                if self._should_stop:
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            self.logger.warning("Server stopping")
        finally:
            self._stop()
        return self._exit_code

    def _stop(self):
        """
        Stop all the components.
        Make sure that the created services and components are shut down.
        """
        self.logger.info("Stopping master")

        self.logger.info("Stopping RPC server")
        self.server.stop(None)  # grace = None
        self.logger.info("RPC server stopped")
        self.logger.info("Master stopped")

    @staticmethod
    def _get_job_type(args):
        if all(
            (args.training_data, args.validation_data, args.evaluation_steps,)
        ):
            job_type = JobType.TRAINING_WITH_EVALUATION
        elif all(
            (
                args.validation_data,
                not args.training_data,
                not args.prediction_data,
            )
        ):
            job_type = JobType.EVALUATION_ONLY
        elif all(
            (
                args.prediction_data,
                not args.validation_data,
                not args.training_data,
            )
        ):
            job_type = JobType.PREDICTION_ONLY
        else:
            job_type = JobType.TRAINING_ONLY

        return job_type

    def _create_evaluation_service(self, eval_func, evaluation_steps):
        evaluation_service = None
        if (
            self.job_type == JobType.TRAINING_WITH_EVALUATION
            or self.job_type == JobType.EVALUATION_ONLY
        ):
            self.logger.info(
                "Creating evaluation service with " "evaluation steps %d",
                evaluation_steps,
            )
            evaluation_service = EvaluationService(
                self.task_d,
                evaluation_steps,
                self.job_type == JobType.EVALUATION_ONLY,
                eval_func,
            )
            self.task_d.set_evaluation_service(evaluation_service)

        return evaluation_service

    def _create_master_service(self, args):
        self.logger.info("Creating master service")
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=64),
            options=[
                ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
                (
                    "grpc.max_receive_message_length",
                    GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )
        master_servicer = MasterServicer(
            args.minibatch_size,
            evaluation_service=self.evaluation_service,
            master=self,
        )
        elasticdl_pb2_grpc.add_MasterServicer_to_server(
            master_servicer, server
        )
        server.add_insecure_port("[::]:{}".format(args.port))
        self.logger.info("The port of the master server is: %d", args.port)

        return master_servicer, server

    def _create_instance_manager(self, args):
        instance_manager = None

        container_command = ["/bin/bash"]
        if args.num_workers:
            assert args.worker_image, "Worker image cannot be empty"
            worker_args = self._create_worker_args(args)
            ps_args = self._create_ps_args(args)

            env_dict = parse_envs(args.envs)
            env = []
            for key in env_dict:
                env.append(V1EnvVar(name=key, value=env_dict[key]))

            kwargs = get_dict_from_params_str(args.aux_params)
            disable_relaunch = kwargs.get("disable_relaunch", False)
            cluster_spec = self._get_image_cluster_spec(args.cluster_spec)

            instance_manager = InstanceManager(
                self.task_d,
                rendezvous_server=self.rendezvous_server,
                job_name=args.job_name,
                image_name=args.worker_image,
                worker_command=container_command,
                worker_args=worker_args,
                namespace=args.namespace,
                num_workers=args.num_workers,
                worker_resource_request=args.worker_resource_request,
                worker_resource_limit=args.worker_resource_limit,
                worker_pod_priority=args.worker_pod_priority,
                num_ps=args.num_ps_pods,
                ps_command=container_command,
                ps_args=ps_args,
                ps_resource_request=args.ps_resource_request,
                ps_resource_limit=args.ps_resource_limit,
                ps_pod_priority=args.ps_pod_priority,
                volume=args.volume,
                image_pull_policy=args.image_pull_policy,
                restart_policy=args.restart_policy,
                cluster_spec=cluster_spec,
                cluster_spec_json=args.cluster_spec_json,
                envs=env,
                disable_relaunch=disable_relaunch,
                log_file_path=args.log_file_path,
            )

        return instance_manager

    def _create_worker_args(self, args):
        worker_client_command = (
            BashCommandTemplate.SET_PIPEFAIL
            + " python -m elasticdl.python.worker.main"
        )
        worker_args = [
            "--master_addr",
            self.master_addr,
            "--job_type",
            self.job_type,
        ]
        worker_args.extend(
            build_arguments_from_parsed_result(args, filter_args=["envs"])
        )
        worker_args = wrap_python_args_with_string(worker_args)
        worker_args.insert(0, worker_client_command)
        worker_args = ["-c", " ".join(worker_args)]
        return worker_args

    def _create_ps_args(self, args):
        if args.distribution_strategy == DistributionStrategy.PARAMETER_SERVER:
            opt_type, opt_args = get_optimizer_info(self._optimizer)
            ps_command = "elasticdl_ps"
            ps_command_args = [
                "-job_name=" + args.job_name,
                "-namespace=" + args.namespace,
                "-master_addr=" + self.master_addr,
                "-port=2222",
                "-use_async=" + ("true" if args.use_async else "false"),
                "-grads_to_wait=" + str(args.grads_to_wait),
                "-lr_staleness_modulation="
                + ("true" if args.lr_staleness_modulation else "false"),
                "-sync_version_tolerance=" + str(args.sync_version_tolerance),
                "-evaluation_steps=" + str(args.evaluation_steps),
                "-num_ps_pods=" + str(args.num_ps_pods),
                "-num_workers=" + str(args.num_workers),
                "-checkpoint_dir=" + str(args.checkpoint_dir),
                "-checkpoint_steps=" + str(args.checkpoint_steps),
                "-keep_checkpoint_max=" + str(args.keep_checkpoint_max),
                "-checkpoint_dir_for_init="
                + str(args.checkpoint_dir_for_init),
                "-opt_type=" + opt_type,
                "-opt_args=" + opt_args,
            ]
            ps_command_args = wrap_go_args_with_string(ps_command_args)
            # Execute source /root/.bashrc to add the file path
            # of `elasticdl_ps` into the PATH environment variable.
            ps_args = ["source", "/root/.bashrc_elasticdl", "&&", ps_command]
            ps_args.extend(ps_command_args)
            ps_args = ["-c", " ".join(ps_args)]
            return ps_args
        else:
            return []

    def _get_image_cluster_spec(self, cluster_spec):
        if cluster_spec:
            filename = os.path.basename(cluster_spec)
            image_cluster_spec = os.path.join(
                ClusterSpecConfig.CLUSTER_SPEC_DIR, filename
            )
            return image_cluster_spec
        return cluster_spec

    def _check_timeout_tasks(self):
        while True:
            doing_tasks = self.task_d._doing.copy()
            cur_time = time.time()
            avg_time = self.master_servicer.get_average_task_complete_time()
            for task_id, (worker_id, task, start_time) in doing_tasks.items():
                if task.type == elasticdl_pb2.TRAINING:
                    start_time = self.master_servicer.get_worker_liveness_time(
                        worker_id
                    )
                if task.type in [
                    elasticdl_pb2.TRAINING,
                    elasticdl_pb2.EVALUATION,
                ]:
                    if (cur_time - start_time) > 3 * avg_time[task.type]:
                        self.logger.info(
                            "worker %d timeout, relaunch it" % worker_id
                        )
                        self.task_d.recover_tasks(worker_id)
                        # TODO: save worker logs before remove it
                        self.instance_manager._remove_worker(worker_id)
                        break
            time.sleep(30)
