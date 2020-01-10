from flask import Flask, request
from kubernetes import client, config
import yaml, json, requests, re, time, socket

app = Flask(__name__)
config.load_incluster_config()

hostname = socket.gethostname()
hostip = socket.gethostbyname(hostname)

ps_set = {}
ar_set = {}
all_set = {}

set_incr_id = 0

with open("template.yaml") as fh:
    stateful_set_yaml = yaml.load(fh, Loader=yaml.FullLoader)


class SetMeta:
    def __init__(self, name="default"):
        global hostip
        self.name = name
        self.worker_stateful_set = ""
        self.worker_ip = []
        self.ps_stateful_set = ""
        self.ps_ip = []
        self.basic_json = {"MASTER":hostip, "SET_ID":0, "NUM_WORKERS":0, "TF_CONFIG":{"cluster":{"worker":[]}, "task":{"type": None, "index": -1}}}

    def set_id(self, idx):
        self.basic_json["SET_ID"] = idx


    def check_readiness(self):
        #TODO: check ready pods
        appsv1 = client.AppsV1Api()
        ready_pods = 0
        return len(self.ps_ip)+len(self.worker_ip)==ready_pods

    def apply_env(self):
        for i_worker_ip in self.worker_ip:
            self.basic_json["TF_CONFIG"]["cluster"]["worker"].append(i_worker_ip+":1234")
        self.basic_json["NUM_WORKERS"] = len(self.worker_ip)

        if len(self.ps_ip)>0:
            self.basic_json["TF_CONFIG"]["cluster"]["ps"] = []
            for i_ps_ip in self.worker_ip:
                self.basic_json["TF_CONFIG"]["cluster"]["ps"].append(i_ps_ip+":1234")

        for i, i_worker_ip in enumerate(self.worker_ip):
            cur_json = self.basic_json.copy()
            cur_json["TF_CONFIG"]["task"]["type"] = "worker"
            cur_json["TF_CONFIG"]["task"]["index"] = i
            requests.post("http://"+i_worker_ip+":7896/config", json=cur_json)

        for i, i_ps_ip in enumerate(self.ps_ip):
            cur_json = self.basic_json.copy()
            cur_json["TF_CONFIG"]["task"]["type"] = "ps"
            cur_json["TF_CONFIG"]["task"]["index"] = i
            requests.post("http://"+i_ps_ip+":7896/config", json=cur_json)

    def start(self):
        for i_ps_ip in self.worker_ip:
            requests.get("http://"+i_ps_ip+":7896/run")

        for i_worker_ip in self.worker_ip:
            requests.get("http://"+i_worker_ip+":7896/run")


def create_from_yaml(yaml_object):
    group, _, version = yaml_object["apiVersion"].partition("/")
    if version == "":
        version = group
        group = "core"

    group = "".join(group.split(".k8s.io,1"))
    func_to_call = "{0}{1}Api".format(group.capitalize(), version.capitalize())

    k8s_api = getattr(client, func_to_call)()

    kind = yaml_object["kind"]
    kind = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', kind)
    kind = re.sub('([a-z0-9])([A-Z])', r'\1_\2', kind).lower()

    if "namespace" in yaml_object["metadata"]:
        namespace = yaml_object["metadata"]["namespace"]
    else:
        namespace = "default"

    try:
        if hasattr(k8s_api, "create_namespaced_{0}".format(kind)):
            resp = getattr(k8s_api, "create_namespaced_{0}".format(kind))(
                body=yaml_object, namespace=namespace)
        else:
            resp = getattr(k8s_api, "create_{0}".format(kind))(
                body=yaml_object)
    except Exception as e:
        raise e

    print("{0} created. status='{1}'".format(kind, str(resp.status)))

    return k8s_api


def list_pods_ip(selector):
    v1 = client.CoreV1Api()
    ret = v1.list_pod_for_all_namespaces(watch=False, label_selector=selector)
    return [i.status.pod_ip for i in ret.items]


def create_from_yaml_file(file):
    with open(file) as fh:
        yaml_object = yaml.load(fh)
    return create_from_yaml(yaml_object)

def create_statefulset(num_pods: int, image_name: str, taskname: str, tasktype: str, set_id: int):
    global stateful_set_yaml
    current_yaml = stateful_set_yaml.copy()
    current_name = taskname + "-" + tasktype + "-" + str(set_id)
    current_yaml["metadata"]["name"] = current_name
    current_yaml["spec"]["serviceName"] = current_name
    current_yaml["spec"]["replicas"] = num_pods
    current_yaml["spec"]["selector"]["matchLabels"]["type"] = tasktype
    current_yaml["spec"]["selector"]["matchLabels"]["app"] = taskname+"-"+str(set_id)
    current_yaml["spec"]["template"]["metadata"]["labels"]["type"] = tasktype
    current_yaml["spec"]["template"]["metadata"]["labels"]["app"] = taskname+"-"+str(set_id)
    current_yaml["spec"]["template"]["spec"]["containers"][0]["name"] = current_name
    current_yaml["spec"]["template"]["spec"]["containers"][0]["image"] = image_name
    return create_from_yaml(current_yaml)


def apply_task(strategy, worker_num, ps_num, img, taskname):
    global set_incr_id, ar_set, ps_set
    current_id = set_incr_id
    set_incr_id += 1

    current_meta = SetMeta(taskname)
    current_meta.set_id(current_id)
    if strategy == "ParameterServer":
        ps_num = int(ps_num)

        ps_set[current_id] = current_meta

        # deal with parameter server
        create_statefulset(ps_num, img, taskname, "ps", current_id)
        #TODO: use readiness probe
        selector = "type=ps,app="+taskname+"-"+str(current_id)
        current_meta.ps_ip = list_pods_ip(selector)
        current_meta.ps_stateful_state = taskname+"-ps-"+str(current_id)
    else:
        ar_set[current_id] = current_meta

    all_set[current_id] = current_meta

    worker_num = int(worker_num)

    # deal with worker server
    create_statefulset(worker_num, img, taskname, "worker", current_id)
    #TODO: check is the statefulset ready instead of sleep
    time.sleep(15)
    selector = "type=worker,app="+taskname+"-"+str(current_id)
    current_meta.worker_ip = list_pods_ip(selector)
    current_meta.worker_stateful_state = taskname+"-worker-"+str(current_id)
    current_meta.apply_env()
    time.sleep(3)
    current_meta.start()
    return 1, current_id


@app.route('/create', methods=['GET'])
def accept_get():
    strategy = request.args.get("strategy")
    if strategy is None:
        return "Strategy has to be specified", 400
    worker_num = request.args.get("n_worker")
    if worker_num is None:
        return "Number of worker has to be specified", 400
    ps_num = request.args.get("n_ps")
    if ps_num is None and strategy=="ParameterServer":
        return "Number of Parameter server has to be specified", 400
    exec_img = request.args.get("img")
    taskname = request.args.get("name")
    if taskname is None:
        taskname = "default"
    status, task_id = apply_task(strategy, worker_num, ps_num, exec_img, taskname)
    if status==1:
        return "OK", 200
    else:
        return "Internal error", 500


app.run(host="0.0.0.0", port=7897)