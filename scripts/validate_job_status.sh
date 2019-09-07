set -x

JOB_TYPE=$1
MASTER_POD_NAME=elasticdl-test-mnist-${JOB_TYPE}-master
WORKER_0_POD_NAME=elasticdl-test-mnist-${JOB_TYPE}-worker-0
WORKER_1_POD_NAME=elasticdl-test-mnist-${JOB_TYPE}-worker-1
CHECK_INTERVAL_SECS=10

function get_pod_status {
    local pod_status=$(kubectl get pod $1 -o jsonpath='{.status.phase}')
    echo ${pod_status}
}

for i in {1..200}; do
    MASTER_POD_STATUS=$(get_pod_status ${MASTER_POD_NAME})
    WORKER_0_POD_STATUS=$(get_pod_status ${WORKER_0_POD_NAME})
    WORKER_1_POD_STATUS=$(get_pod_status ${WORKER_1_POD_NAME})

    if [[ "$MASTER_POD_STATUS" == "Succeeded" ]] &&
     [[ "$WORKER_0_POD_STATUS" == "Succeeded" ]] &&
     [[ "$WORKER_1_POD_STATUS" == "Succeeded" ]]; then
      echo "ElasticDL job succeeded."
      kubectl delete pod ${MASTER_POD_NAME}
      exit 0
    elif [[ "$MASTER_POD_STATUS" == "Failed" ]] ||
       [[ "$WORKER_0_POD_STATUS" == "Failed" ]] ||
       [[ "$WORKER_1_POD_STATUS" == "Failed" ]]; then
      echo "ElasticDL job failed."
      kubectl describe pod ${MASTER_POD_NAME}
      echo "\nMaster log:\n"
      kubectl logs ${MASTER_POD_NAME} | tail
      echo "\nWorker0 log:\n"
      kubectl logs ${WORKER_0_POD_NAME} | tail
      echo "\nWorker1 log:\n"
      kubectl logs ${WORKER_1_POD_NAME} | tail
      kubectl delete pod ${MASTER_POD_NAME}
      exit 1
    else
      echo "Master: ${MASTER_POD_STATUS}, Worker0: ${WORKER_0_POD_STATUS}, Worker1: ${WORKER_1_POD_STATUS}. Continue checking..."
      sleep ${CHECK_INTERVAL_SECS}
    fi
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
done
echo "ElasticDL job timed out."

kubectl delete pod ${MASTER_POD_NAME}

exit 1
