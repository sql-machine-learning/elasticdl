import logging
import os
import tempfile
import uuid
from urllib.parse import urlparse

import docker

from elasticdl.python.common.file_helper import copy_if_not_exists


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)-8s "
    "[%(filename)s:%(lineno)d] %(message)s"
)
logging.getLogger().setLevel("INFO")


def build_and_push_docker_image(
    model_zoo,
    docker_image_prefix,
    base_image="",
    extra_pypi="",
    cluster_spec="",
):
    """Build and push a Docker image containing ElasticDL and the model
zoo.  The parameter model_zoo could be a local directory or an URL.
In the later case, we do git clone.

    The basename of the Docker image is auto-generated and is globally
unique.  The full name is docker_image_prefix + "/" + basename.

    The fullname of the Docker image is docker_image_prefix + "/" +
basename.  Unless prefix is None or "", _push_docker_image is called
after _build_docker_image.

    Returns the full Docker image name.  So the caller can docker rmi
    fullname later.

    """
    with tempfile.TemporaryDirectory() as ctx_dir:

        # Copy ElasticDL Python source tree into the context directory.
        elasticdl = _find_elasticdl_root()
        edl_dest = os.path.join(ctx_dir, os.path.basename(elasticdl))
        copy_if_not_exists(elasticdl, edl_dest, is_dir=True)

        # Copy model zoo source tree into the context directory.
        model_zoo_dest = os.path.join(ctx_dir, os.path.basename(model_zoo))
        copy_if_not_exists(model_zoo, model_zoo_dest, is_dir=True)

        # Copy cluster specification file into the context directory.
        if cluster_spec:
            copy_if_not_exists(
                cluster_spec,
                os.path.join(ctx_dir, os.path.basename(cluster_spec)),
                is_dir=False,
            )

        # Create the Dockerfile.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as df:
            df.write(
                _create_dockerfile(
                    os.path.basename(elasticdl),
                    # Note that we need `abspath` here since `urlparse`
                    # does not handle directory names correctly sometimes
                    os.path.basename(os.path.abspath(model_zoo)),
                    os.path.basename(cluster_spec),
                    base_image,
                    extra_pypi,
                )
            )

        image_name = _generate_unique_image_name(docker_image_prefix)
        client = docker.APIClient(base_url="unix://var/run/docker.sock")
        _build_docker_image(client, ctx_dir, df.name, image_name)

        if docker_image_prefix:
            _push_docker_image(client, image_name)

    return image_name


def _find_elasticdl_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    )


def _create_dockerfile(
    elasticdl, model_zoo, cluster_spec="", base_image="", extra_pypi_index=""
):
    HEAD = """
FROM {BASE_IMAGE} as base
ENV PYTHONPATH=/:/model_zoo
"""
    if elasticdl:
        HEAD = """
%s
COPY %s/elasticdl /elasticdl
""" % (
            HEAD,
            elasticdl,
        )

    LOCAL_ZOO = """
RUN pip install -r /elasticdl/requirements.txt \
  --extra-index-url="${EXTRA_PYPI_INDEX}"
RUN make -f /elasticdl/Makefile
COPY {MODEL_ZOO} /model_zoo/{MODEL_ZOO}
"""
    REMOTE_ZOO = """
RUN pip install -r /elasticdl/requirements.txt \
  --extra-index-url="${EXTRA_PYPI_INDEX}"
RUN make -f /elasticdl/Makefile
RUN apt-get update && apt-get install -y git
RUN git clone --recursive {MODEL_ZOO} /model_zoo
"""
    pr = urlparse(model_zoo)
    if not pr.path:
        raise RuntimeError(
            "urlparse(model_zoo) {} has no path field".format(model_zoo)
        )
    if pr.scheme in ["file", ""]:
        tmpl = HEAD + LOCAL_ZOO
        model_zoo = pr.path  # Remove the "file://" prefix if any.
    else:
        tmpl = HEAD + REMOTE_ZOO

    if cluster_spec:
        tmpl = """
%s
COPY %s /cluster_spec/%s
""" % (
            tmpl,
            cluster_spec,
            cluster_spec,
        )

    tmpl = (
        """
%s
ARG REQS=/model_zoo/{MODEL_ZOO}/requirements.txt
RUN if [ -f $REQS ]; then \
      pip install -r $REQS --extra-index-url="${EXTRA_PYPI_INDEX}"; \
    fi
# Check whether TensorFlow has been installed in the base image
# so we can fail earlier if the specified base image is problematic
RUN python -c 'import sys, pkgutil; exit_code = 0 if \
    pkgutil.find_loader("tensorflow") else 1; \
    print("TensorFlow is installed") if exit_code == 0 \
    else print("TensorFlow must be installed in the base image"); \
    sys.exit(exit_code)'
"""
        % tmpl
    )

    return tmpl.format(
        BASE_IMAGE=base_image
        if base_image
        else "tensorflow/tensorflow:2.0.0b1-py3",
        ELASTIC_DL=elasticdl,
        MODEL_ZOO=model_zoo,
        EXTRA_PYPI_INDEX=extra_pypi_index,
    )


def _generate_unique_image_name(prefix):
    return os.path.join(
        prefix if prefix else "", "elasticdl:" + uuid.uuid4().hex
    )


def _print_docker_progress(line):
    error = line.get("error", None)
    if error:
        raise RuntimeError("Docker image build: " + error)
    stream = line.get("stream", None)
    if stream:
        logging.info(stream)
    else:
        logging.info(line)


def _build_docker_image(client, ctx_dir, dockerfile, image_name):
    logging.info("===== Building Docker Image =====")
    for line in client.build(
        dockerfile=dockerfile,
        path=ctx_dir,
        rm=True,
        tag=image_name,
        decode=True,
    ):
        _print_docker_progress(line)


def _push_docker_image(client, image_name):
    logging.info("===== Pushing Docker Image =====")
    for line in client.push(image_name, stream=True, decode=True):
        _print_docker_progress(line)
