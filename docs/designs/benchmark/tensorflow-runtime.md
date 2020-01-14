# Design for TensorFlow-runtime-based Dsitributed Training Support

This document describes the design for supporting TensorFlow-runtime-based Dsitributed training

## Motication

While ElasticDL is a third-party framework for TensorFlow distributed training, TensorFlow also has its native distributed training surpport. Though ElasticDL provide additional features like Elastic Scheduling, Fault-Tolerance and Kubernetes-native, It's still interesting which performance is better. Here we provide an interface for convenient deployment of TensorFlow-runtime distributed training.

## Design Components

To minimize the influence on training scripts writing, we introduce two components: the local client and the master in Kubernetes cluster. Things that the users needs to do are:

* Speicify number of workers using os.environ['NUM_WORKERS'], example:

```python
num_workers = os.environ['NUM_WORKERS']
```
* Rename the script to be executed with "exec.py", and make sure the command "python exec.py" could start the script with all the parameter needed.

* Build the image and make sure "exec.py" is in the working directory.

### Local Client

The client will build another Docker image with a HTTP server upon the image passed to it. The new image will be used by the master.

After image build completion, the client will send a GET query to the master in the cluster, which include the image name, the training strategy, number of workers or parameter servers and etc. 

### Master in Cluster

The master is a HTTP server, it will handle the GET query from clients. Accoring to the parameter of the GET query, the master will create worker cluster or parameter server cluster of specific number by talking to Kubernetes Api server. Since the image is already with an HTTP server on it, master could talk to those pods with HTTP query.

After the cluster(s) is created and ready, the master will send a POST query to each pod to set the environment variables like "TF_CONFIG" and "NUM_WORKERS" inside the pods. Then the master will send a GET query to each pod to run the "exec.py" python script 

![master-client](/docs/images/tf-runtime-demo-design-doc.png)

## Usage

1. start the Minikube

​       ``` minikube start --vm-driver=hyperkit --memory="6144MB"```

2. set up the docker environment

   ```source env.sh```

3. build training image

   ```shell
   cd allreduce_keras_example
   sh build_image.sh
   ```

4. build master pod image and create master pod

   ```
   cd master
   sh build_master_image.sh
   ```

5. submit training task

   ```shell
   cd client
   client.py --master "http://192.168.64.5:31898" \
   				  --strategy AllReduce \
   					--ps-num 0 \
   					--worker-num 4 \
   					--image allreduce\
   					--task test
   ```

   ​        

   Notice: --master: should be the ip address of your minikube vm instance.

​                --stratege: could be AllReduce or ParameterServer

​                --ps-num: number of parameter server pods

​                --worker-num: number of worker pods

​                --image: the training image built at step.3

​                --task: the name you give to this task