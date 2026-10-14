"""Send sample traffic to the API so the monitoring dashboards have data."""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient(rng):
    return {
        "age": rng.randint(30, 77), "sex": rng.randint(0, 1), "cp": rng.randint(1, 4),
        "trestbps": rng.randint(95, 200), "chol": rng.randint(125, 400),
        "fbs": rng.randint(0, 1), "restecg": rng.randint(0, 2),
        "thalach": rng.randint(80, 200), "exang": rng.randint(0, 1),
        "oldpeak": round(rng.uniform(0, 5), 1), "slope": rng.randint(1, 3),
        "ca": rng.choice([0, 1, 2, 3, None]), "thal": rng.choice([3, 6, 7, None]),
    }


def post(url, payload):
    request = urllib.request.Request(
        url + "/predict", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(7)
    counts = {}
    for i in range(args.n):
        payload = random_patient(rng)
        if i % 10 == 9:  # every 10th request is deliberately invalid
            payload["age"] = -1
        status = post(args.url, payload)
        counts[status] = counts.get(status, 0) + 1
        time.sleep(args.delay)
    print("status counts:", counts)


if __name__ == "__main__":
    main()
