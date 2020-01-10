This is a demo of tensorflow distribute training on K8s

Requirements:
    1. minikube, docker installed

Usage:
    1. start the Minikube
        minikube start --vm-driver=hyperkit --memory="6144MB"

    2. set up the docker environment
        source env.sh

    3. build training image
        cd allreduce
        sh build_image.sh

    4. build master pod image and create master pod
        cd master
        sh build_master_image.sh

    5. submit training task
        cd client
        python client.py --master "http://192.168.64.5:31898" \
                         --strategy AllReduce \
                         --ps-num 0 \
                         --worker-num 4 \
                         --image allreduce\
                         --task test
                         
        [!Notice!]: --master: should be the ip address of your minikube vm instance.
                --stratege: could be AllReduce or ParameterServer
                --ps-num: number of parameter server pods
                --worker-num: number of worker pods
                --image: the training image built at step.3
                --task: the name you give to this task