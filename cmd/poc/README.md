# Build and Run

## Build develop Docker image

Change to `swamp` directory and build dev Docker image:

```
docker build -t swamp_dev - < Dockerfile.dev
```

## Compile and run POC

### Start Docker container
  
```
docker run --net=host --rm -it -v $HOME/go:/go -v $HOME/.kube:/.kube swamp_dev:latest /bin/bash
```
Note that:
* The `elasticdl` git repo should be under your `$HOME/go/src` directory
* Mac's Docker App should have kubernetes enabled and make sure local `kubectl` works and your `$HOME/.kube` directory point to local cluster started by Docker app.


### Compile and run

In the container, first install k8s go client

```
go get github.com/kubernetes/client-go/...
```

Change to your `elasticdl` repo directory under `/go/src` and do:

```
go build cmd/poc/main.go
```

Now you can run the POC application:

```
./main -kubeconfig=/.kube/config
```

The POC app lists all PODs, delete the POD with name 'poc' and restart it. Meanwhile, there is a goroutine listening and print out POD events.
