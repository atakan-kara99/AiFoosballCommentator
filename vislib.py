import requests
import json

SERVER = "http://backend:3000"
headers = {"Content-Type": "application/json"}
NO_SERVER = False

def kpi(name, value):
    """
    Send a KPI to server or update it.
    name: name of the kpi
    value: value of kpi
    """
    global NO_SERVER
    if NO_SERVER:
        print("No server available to send", name, value)
        return False
    try:
        url = SERVER + '/kpi/push'
        data = { "id": str(name), "value": str(value) }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        return res.status_code
    except Exception as e:
        print("Couldn't send to server", e)
        NO_SERVER = True
        return False

def pushText(text):
    """
    Send commentary text to frontend.
    text: text snipped to be sent.
    """
    global NO_SERVER
    if NO_SERVER:
        print("No server available to send", text)
        return False
    try:
        url = SERVER + '/pushText'
        data = { "text": str(text) }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        return res.status_code
    except Exception as e:
        print("Couldn't send to server", e)
        NO_SERVER = True
        return False
