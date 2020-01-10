eval $(minikube docker-env)
export DOCKER_BASE_URL=${DOCKER_HOST}
export DOCKER_TLSCERT=${HOME}/.minikube/certs/cert.pem
export DOCKER_TLSKEY=${HOME}/.minikube/certs/key.pem
