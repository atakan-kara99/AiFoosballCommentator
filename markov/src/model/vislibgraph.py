import requests
import json


SERVER = "http://backend:3000"

headers = {
    "Content-Type": "application/json"
}

nodes = []

NO_SERVER = False


def add_node(id):
    if NO_SERVER:
        print("No server for graph available.")
        return False
    try:
        url = SERVER + '/api/vertices'
        data = {
            "id": str(id),
            "value": str(id)
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        nodes.append(str(id))
        return res.status_code
    except:
        return False


def add_link(source, target):  # Use source and target node id which was used in add_node.
    if NO_SERVER:
        print("No server for graph available.")
        return False
    try:
        if str(source) not in nodes:
            add_node(source)

        if str(target) not in nodes:
            add_node(target)

        url = SERVER + '/api/links'
        data = {
            "source": str(source),
            "target": str(target)
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        return res.status_code
    except:
        return False

def reset():
    try:
        url = SERVER + '/api/reset'
        data = {
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        print('Visual server detected. Sending updates for training steps.')
        return res.status_code
    except:
        NO_SERVER = True
        print('No visual server detected. Running training without.')
        return False
