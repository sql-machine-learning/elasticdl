#!/bin/bash
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

set -e

JOB_TYPE=$1
MAXCOMPUTE_TABLE="odps_integration_build_$TRAVIS_BUILD_NUMBER_$(date +%s)"

if [[ "$JOB_TYPE" == "odps" ]] && \
{ [[ "$ODPS_ACCESS_ID" == "" ]] || \
[[ "$ODPS_ACCESS_KEY" == "" ]]; }; then
    echo "Skipping ODPS related integration tests since \
    either ODPS_ACCESS_ID or ODPS_ACCESS_KEY is not set"
else
    echo "Running ElasticDL job: ${JOB_TYPE}"
    if [[ "$JOB_TYPE" == "odps" ]]; then
        export MAXCOMPUTE_TABLE
        bash scripts/travis/create_odps_table.sh
    fi
    PS_NUM=2
    WORKER_NUM=1
    docker run --rm -it --net=host \
        -e MAXCOMPUTE_TABLE="$MAXCOMPUTE_TABLE" \
        -e MAXCOMPUTE_PROJECT="$MAXCOMPUTE_PROJECT" \
        -e MAXCOMPUTE_AK="$ODPS_ACCESS_ID" \
        -e MAXCOMPUTE_SK="$ODPS_ACCESS_KEY" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$HOME"/.kube:/root/.kube \
        -v /home/"$USER"/.minikube/:/home/"$USER"/.minikube/ \
        -v "$PWD":/work \
        -w /work elasticdl:ci \
        bash -c "scripts/client_test.sh $JOB_TYPE $PS_NUM $WORKER_NUM"
    python3 scripts/validate_job_status.py "$JOB_TYPE" "$PS_NUM" "$WORKER_NUM"
    if [[ "$JOB_TYPE" == "odps" ]]; then
        bash scripts/travis/cleanup_odps_table.sh
    fi
fi
