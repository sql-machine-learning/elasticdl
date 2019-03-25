from kubernetes import client, config
from kubernetes.client.rest import ApiException
from pprint import pprint


def get_all_pods(v1api, print_result=False):
    try:
        ret = v1api.list_pod_for_all_namespaces(watch=False)
    except ApiException as e:
        print('Exception when calling CoreV1Api->list_pod_for_all_namespaces: %s\n' % e)
    if print_result:
        print('------------ PODS ------------')
        for i in ret.items:
            print('%s\t%s\t%s\t%s' % (i.metadata.name,
                                      i.metadata.namespace, i.status.pod_ip, i.status.phase))
    return ret


def get_all_namespaces(v1api, print_result=False):
    try:
        ret = v1api.list_namespace()
    except ApiException as e:
        print('Exception when calling CoreV1Api->list_namespace: %s\n' % e)
    if print_result:
        print('--------- NAMESPACES ---------')
        for i in ret.items:
            print('%s\t%s' % (i.metadata.name, i.status.phase))
    return ret


def delete_namespace(v1api, name):
    try:
        v1api.delete_namespace(name, grace_period_seconds=0)
    except ApiException as e:
        print('Exception when calling CoreV1Api->delete_namespace: %s\n' % e)


def delete_pod_if_exist(v1api, name, namespace='default'):
    pods = get_all_pods(v1api)
    for i in pods.items:
        if i.metadata.name == name and i.metadata.namespace == namespace:
            # Delete immediately
            body = client.V1DeleteOptions(grace_period_seconds=0)
            try:
                v1api.delete_namespaced_pod(name, namespace, body=body)
            except ApiException as e:
                print(
                    'Exception when calling CoreV1Api->delete_namespaced_pod: %s\n' % e)
            break


def create_namespace_if_not_existed(v1api, name):
    namespaces = get_all_namespaces(v1api)
    existed = False
    for i in namespaces.items:
        if i.metadata.name == name:
            existed = True
            break
    if not existed:
        v1metadata = client.models.V1ObjectMeta(name=name)
        namespace = client.models.V1Namespace(metadata=v1metadata)
        try:
            v1api.create_namespace(namespace)
        except ApiException as e:
            print('Exception when calling CoreV1Api->create_namespace: %s\n' % e)


def create_namespace_pod(v1api, pod, namespace_name='default'):
    try:
        api_response = v1api.create_namespaced_pod(namespace_name, pod)
    except ApiException as e:
        print('Exception when calling CoreV1Api->create_namespaced_pod: %s\n' % e)


def main():
    namespace_name = 'swamp-samples'
    pod_name = 'pod-example'

    config.load_kube_config()
    v1api = client.CoreV1Api()

    delete_pod_if_exist(v1api, pod_name, namespace_name)
    create_namespace_if_not_existed(v1api, namespace_name)

    ports = [client.models.V1ContainerPort(
        name='http', protocol='TCP', container_port=80)]
    containers = [client.models.V1Container(
        name='poc-container1', image='nginx:1.12', ports=ports)]
    v1metadata = client.models.V1ObjectMeta(name=pod_name)
    v1spec = client.models.V1PodSpec(containers=containers)
    v1pod = client.models.V1Pod(metadata=v1metadata, spec=v1spec)
    create_namespace_pod(v1api, v1pod, namespace_name)

    get_all_pods(v1api, print_result=True)
    get_all_namespaces(v1api, print_result=True)


if __name__ == '__main__':
    main()
