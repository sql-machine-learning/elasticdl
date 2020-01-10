from flask import Flask, request
import json, requests, os, sys

app = Flask(__name__)

num_worker_json = None
tf_config_json = None

@app.route('/config', methods=['POST'])
def config_env():
    global num_worker_json, tf_config_json
    json_data = request.get_json()
    num_worker_json = json_data['NUM_WORKERS']
    tf_config_json = json_data['TF_CONFIG']
    return "OK", 200

@app.route('/run', methods=['GET'])
def run():
    global num_worker_json, tf_config_json
    os.environ["NUM_WORKERS"] = str(num_worker_json)
    os.environ["TF_CONFIG"] = json.dumps(tf_config_json)
    print(os.environ["NUM_WORKERS"], file=sys.stdout)
    print(os.environ["TF_CONFIG"], file=sys.stdout)
    os.popen("python exec.py > outputs.log")
    return "OK", 200

app.run(host="0.0.0.0", port=7896)