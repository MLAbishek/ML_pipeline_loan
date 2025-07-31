import requests

BASE_URL = "http://127.0.0.1:10000"

features = [2850, 1250, 85, 10, 1100, 4100, "Sunny", 0, 0]
label = 123


def print_response(resp):
    print("Status code:", resp.status_code)
    try:
        print("JSON:", resp.json())
    except Exception:
        print("Raw response:", resp.text)


print("Testing /submit ...")
resp = requests.post(f"{BASE_URL}/submit", json={"features": features, "label": label})
print_response(resp)

print("\nTesting /predict ...")
resp = requests.post(f"{BASE_URL}/predict", json={"features": features})
print_response(resp)

print("\nTesting /retrain ...")
resp = requests.post(f"{BASE_URL}/retrain")
print_response(resp)
