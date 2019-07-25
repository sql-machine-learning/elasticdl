set -x

MASTER_POD_NAME=elasticdl-test-mnist-master
WORKER_0_POD_NAME=elasticdl-test-mnist-worker-0
WORKER_1_POD_NAME=elasticdl-test-mnist-worker-1
CHECK_INTERVAL_SECS=10

function get_pod_status {
    local pod_status=$(kubectl get pod $1 -o jsonpath='{.status.phase}')
    echo ${pod_status}
}

for i in {1..200}; do
    MASTER_POD_STATUS=$(get_pod_status ${MASTER_POD_NAME})
    WORKER_0_POD_STATUS=$(get_pod_status ${WORKER_0_POD_NAME})
    WORKER_1_POD_STATUS=$(get_pod_status ${WORKER_1_POD_NAME})

    if [ "$MASTER_POD_STATUS" == "Succeeded" ] &&
     [ "$WORKER_0_POD_STATUS" == "Succeeded" ] &&
     [ "$WORKER_1_POD_STATUS" == "Succeeded" ]; then
      echo "ElasticDL job succeeded."
      exit 0
    elif [ "$MASTER_POD_STATUS" == "Failed" ] ||
       [ "$WORKER_0_POD_STATUS" == "Failed" ] ||
       [ "$WORKER_1_POD_STATUS" == "Failed" ]; then
      echo "ElasticDL job failed."
      kubectl describe pod $MASTER_POD_NAME
      kubectl logs $MASTER_POD_NAME | tail
      exit 1
    else
      echo "Master pod status: ${MASTER_POD_STATUS}. Continue checking..."
      sleep ${CHECK_INTERVAL_SECS}
    fi
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
done
echo "ElasticDL job timed out."
exit 1
