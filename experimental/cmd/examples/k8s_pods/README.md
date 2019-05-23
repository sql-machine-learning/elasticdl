# Build and Run

## Start K8s Metrics Server

[Metrics server](https://kubernetes.io/docs/tasks/debug-application-cluster/core-metrics-pipeline/#metrics-server) aggregates cluster-wide resource usage statistics.
The example demonstrates fetching resource usages for a pod from metrics server. For this purpose, the metrics server needs to be started in the test cluster (Docker internal k8s or minikube):

```
kubectl apply -f k8s/addons/metrics-server.yaml
```

Wait for ~30 seconds for the server to start. Run `kubectl top node` or `kubectl top pod`  to verify. For example on a Mac, `kubctl top node` should display something like:

```
NAME                 CPU(cores)   CPU%      MEMORY(bytes)   MEMORY%
docker-for-desktop   709m         23%       1790Mi          30%
```

## Build develop Docker image

Change to `swamp` directory and build dev Docker image:

```
docker build -t swamp:dev - < Dockerfile.dev
```

## Compile and run POC

### Start Docker container

```
docker run --net=host --rm -it \
    -v $HOME/go:/go \
    -v $HOME/.kube:/.kube \
    swamp:dev /bin/bash
```
Note that:
* The `elasticdl` git repo should be under your `$HOME/go/src` directory
* Mac's Docker App should have kubernetes enabled and make sure local `kubectl` works and your `$HOME/.kube` directory point to local cluster started by Docker app.
* If you use `minikube` instead of Docker internal k8s environment, you will also need to map `.minikube` directory into the container, i.e.:

   ```
   docker run --net=host --rm -it \
       -v $HOME/go:/go \
       -v $HOME/.kube:/.kube \
       -v $HOME/.minikube:$HOME/.minikube \
       swamp:dev /bin/bash
   ```

### Compile and run GO sample

In the container, first install k8s go client and metrics client.

```
go get github.com/kubernetes/client-go/...
go get github.com/kubernetes/metrics/...
```

Change to your `elasticdl` repo directory under `/go/src` and do:

```
go run cmd/examples/k8s_pods/main.go -kubeconfig=/.kube/config
```

The example app lists all pods with metrics, delete the pod with name 'pod-example' and restart it. Meanwhile, there is a goroutine listening and print out pod events.

### Run python sample
In the container, change to your `elasticdl` repo directory, go to `cmd/examples/k8s_pods` directory and do:

```
python main.py
```

The example app deletes the pod with name 'pod-example' in namespace 'swamp-samples' and restart it. After that, it lists all the pods and namespaces in kubernetes.
