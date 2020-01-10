import docker
import argparse
import requests

client = docker.from_env()

def build_pod_image(img_name, build_name):
    with open("./pod_img/Dockerfile", 'w') as fh:
        fh.write("FROM %s\n" % img_name)
        fh.write("RUN pip install requests\n")
        fh.write("RUN pip install flask\n")
        fh.write("ADD pod.py ./\n")
        fh.write('ENTRYPOINT ["python", "pod.py"]\n')
    client.images.build(path='./pod_img', tag=build_name)

parser = argparse.ArgumentParser()
parser.add_argument("--master", type=str)
parser.add_argument("--strategy", type=str, default="AllReduce")
parser.add_argument("--ps-num", type=int, default=0)
parser.add_argument("--worker-num", type=int)
parser.add_argument("--image", type=str)
parser.add_argument("--task", type=str, default="default")

args = parser.parse_args()

build_pod_image(args.image, args.image+'_pod')
params = {}
params["strategy"] = args.strategy
params["n_worker"] = args.worker_num
params["n_ps"] = args.ps_num
params["img"] = args.image+'_pod'
params["name"] = args.task
print(args.master)
requests.get(args.master+"/create", params=params)