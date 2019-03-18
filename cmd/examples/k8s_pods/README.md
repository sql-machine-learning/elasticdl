# Build and Run

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

### Compile and run

In the container, first install k8s go client

```
go get github.com/kubernetes/client-go/...
```

Change to your `elasticdl` repo directory under `/go/src` and do:

```
go run cmd/examples/k8s_pods/main.go -kubeconfig=/.kube/config
```

The POC app lists all PODs, delete the POD with name 'poc' and restart it. Meanwhile, there is a goroutine listening and print out POD events.
