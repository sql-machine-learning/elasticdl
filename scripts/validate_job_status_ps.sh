#!/usr/bin/env bash

# We intentionally `set +e` here since we want to allow certain kinds of known failures that can be ignored.
# For example, pods may not have been created yet so neither `kubectl get pod` nor `kubectl delete pod` would
# be successful at earlier stages of this script.
set +e

JOB_TYPE=$1
MASTER_POD_NAME=elasticdl-test-${JOB_TYPE}-master
PS_POD_NAME=elasticdl-test-${JOB_TYPE}-ps-0
WORKER_POD_NAME=elasticdl-test-${JOB_TYPE}-worker-0
CHECK_INTERVAL_SECS=10

function get_pod_status {
    local pod_status=$(kubectl get pod $1 -o jsonpath='{.status.phase}')
    echo ${pod_status}
}

# If TensorBoard service keeps running when the tasks are finished,
# the master pod status would always be running and thus cannot reflect the true status.
# This function finds the true status under master pod's `metadata.labels.status`
# when TensorBoard service is enabled.
function get_master_pod_label_status {
    local master_pod_status=$(kubectl get pod ${MASTER_POD_NAME} -o jsonpath='{.metadata.labels.status}')
    echo ${master_pod_status}
}

for i in {1..200}; do
    MASTER_POD_STATUS=$(get_pod_status ${MASTER_POD_NAME})
    MASTER_POD_LABEL_STATUS=$(get_master_pod_label_status)
    PS_POD_STATUS=$(get_pod_status ${PS_POD_NAME})
    WORKER_POD_STATUS=$(get_pod_status ${WORKER_POD_NAME})

    if [[ "$MASTER_POD_STATUS" == "Succeeded" ]] &&
     [[ "$PS_POD_STATUS" == "Succeeded" ]] &&
     [[ "$WORKER_POD_STATUS" == "Succeeded" ]]; then
      echo "ElasticDL job succeeded."
      kubectl delete pod ${MASTER_POD_NAME}
      exit 0
    elif [[ "$MASTER_POD_STATUS" == "Running" ]] &&
     [[ "$MASTER_POD_LABEL_STATUS" == "Finished" ]] &&
     [[ "$PS_POD_STATUS" == "Succeeded" ]] &&
     [[ "$WORKER_POD_STATUS" == "Succeeded" ]]; then
      echo "ElasticDL job succeeded (master pod keeps running for TensorBoard service)."
      kubectl delete pod ${MASTER_POD_NAME}
      exit 0
    elif [[ "$MASTER_POD_STATUS" == "Failed" ]] ||
       [[ "$PS_POD_STATUS" == "Failed" ]] ||
       [[ "$WORKER_POD_STATUS" == "Failed" ]]; then
      echo "ElasticDL job failed."
      kubectl describe pod ${MASTER_POD_NAME}
      echo "\nMaster log:\n"
      kubectl logs ${MASTER_POD_NAME} | tail
      echo "\nPS log:\n"
      kubectl logs ${PS_POD_NAME} | tail
      echo "\nWorker log:\n"
      kubectl logs ${WORKER_POD_NAME} | tail
      kubectl delete pod ${MASTER_POD_NAME}
      exit 1
    else
      echo "Master: ${MASTER_POD_STATUS}, PS: ${WORKER_POD_STATUS}, Worker: ${WORKER_POD_STATUS}. Continue checking..."
      sleep ${CHECK_INTERVAL_SECS}
    fi
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
done
echo "ElasticDL job timed out."

kubectl delete pod ${MASTER_POD_NAME}

exit 1
