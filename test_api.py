"""
SelfHealingNN — API Test Suite
Run: python test_api.py
Requires the server to be running at http://localhost:8000
"""
import sys
import json
import time
import urllib.request
import urllib.error
import urllib.parse

BASE = "http://localhost:8000"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []

def check(name, passed, detail=""):
    status = PASS if passed else FAIL
    print(f"  {status} {name}", f"  ({detail})" if detail else "")
    results.append((name, passed))

def get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    res = urllib.request.urlopen(req, timeout=10)
    return json.loads(res.read())

def post_multipart(path, fields, file_path=None):
    import io, os, random, string
    boundary = ''.join(random.choices(string.ascii_letters, k=28))
    body = []
    for k, v in fields.items():
        body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    if file_path:
        fname = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            data = f.read()
        body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{fname}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
        body.append(data)
        body.append(f'\r\n'.encode())
    body.append(f'--{boundary}--\r\n'.encode())
    raw = b''.join(body)
    req = urllib.request.Request(
        f"{BASE}{path}", data=raw,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )
    try:
        res = urllib.request.urlopen(req, timeout=120)
        return json.loads(res.read()), res.getcode()
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code

# ── Test 1: Health ──────────────────────────────────────────────────
print("\n[1] Health Check")
try:
    data = get("/api/health")
    check("Status ok",                data.get('status') == 'ok')
    check("Baseline checkpoint found", data.get('baseline_checkpoint') is True)
    check("Device field present",      'device' in data)
    check("AE field present",          'autoencoders_loaded' in data)
except Exception as e:
    check("Health endpoint reachable", False, str(e))

# ── Test 2: Methods ─────────────────────────────────────────────────
print("\n[2] Methods Endpoint")
try:
    data = get("/api/methods")
    methods = {m['id'] for m in data.get('methods', [])}
    expected = {'gradient','ae_dual','hybrid','regrowth','regrowth_gradient','rerouting','adaptive'}
    check("All 7 methods returned", methods == expected, str(methods))
    check("autoencoders_available field present", 'autoencoders_available' in data)
except Exception as e:
    check("Methods endpoint reachable", False, str(e))

# ── Test 3: Invalid Inputs ─────────────────────────────────────────
print("\n[3] Input Validation")
IMG = "test_images/OIP.jpeg"
try:
    data, code = post_multipart("/api/fault", {"fault_ratio": "99"}, IMG)
    check("Invalid fault_ratio rejected (99)", code == 422)
except Exception as e:
    check("Invalid fault_ratio validation", False, str(e))

try:
    data, code = post_multipart("/api/heal", {"fault_ratio": "20", "method": "fake_method"}, IMG)
    check("Invalid method rejected", code == 422)
except Exception as e:
    check("Invalid method validation", False, str(e))

# ── Test 4: Baseline ────────────────────────────────────────────────
print("\n[4] Baseline Endpoint")
try:
    data, code = post_multipart("/api/baseline", {}, IMG)
    check("HTTP 200 OK",         code == 200, f"got {code}")
    check(
        "accuracy returned or unavailable",
        'accuracy' in data and (
            data['accuracy'] is None or isinstance(data['accuracy'], (int, float))
        ),
    )
    check("prediction returned", 'prediction' in data)
    check("confidence returned", 'confidence' in data)
    accuracy = data.get('accuracy')
    accuracy_display = f"{accuracy}%" if accuracy is not None else "unavailable"
    print(f"     Baseline accuracy: {accuracy_display}  |  Prediction: {data.get('prediction')}")
except Exception as e:
    check("Baseline endpoint", False, str(e))

# ── Test 5: Fault ───────────────────────────────────────────────────
print("\n[5] Fault Injection Endpoint")
for ratio in [10, 20, 30, 40]:
    try:
        data, code = post_multipart("/api/fault", {"fault_ratio": str(ratio)}, IMG)
        check(f"fault_ratio={ratio}%  HTTP 200", code == 200)
        if code == 200:
            print(f"     fault={ratio}%  acc={data.get('accuracy')}%  pred={data.get('prediction')}")
    except Exception as e:
        check(f"fault_ratio={ratio}", False, str(e))

# ── Test 6: End-to-end Heal (adaptive, 20%) ─────────────────────────
print("\n[6] End-to-End Heal (adaptive, fault=20%)")
print("     NOTE: this takes 30–90s on CPU …")
try:
    t0 = time.time()
    data, code = post_multipart("/api/heal", {"fault_ratio": "20", "method": "adaptive"}, IMG)
    elapsed = round(time.time()-t0, 1)
    check("HTTP 200 OK",           code == 200, f"got {code}")
    if code == 200:
        check("baseline_acc present", 'baseline_acc' in data)
        check("faulty_acc present",   'faulty_acc' in data)
        check("healed_acc present",   'healed_acc' in data)
        check("recovery present",     'recovery' in data)
        check("method echoed back",   data.get('method') == 'adaptive')
        b = data.get('baseline_acc', 0)
        f = data.get('faulty_acc', 0)
        h = data.get('healed_acc', 0)
        r = data.get('recovery', 0)
        check("faulty_acc < baseline_acc", f < b, f"{f:.2f} < {b:.2f}")
        check("healed_acc > faulty_acc",   h > f,  f"{h:.2f} > {f:.2f}")
        print(f"     baseline={b}%  faulty={f}%  healed={h}%  recovery={r}%  time={elapsed}s")
    else:
        print(f"     Error response: {data}")
except Exception as e:
    check("Heal endpoint", False, str(e))

# ── Test 7: Concurrent Lock ─────────────────────────────────────────
print("\n[7] Concurrent Request Rejection")
import threading
from unittest.mock import patch
from fastapi.testclient import TestClient
import api

first_request_entered = threading.Event()
release_first_request = threading.Event()
first_response = []
first_errors = []
with open(IMG, "rb") as image_file:
    image_bytes = image_file.read()

def _pause_after_lock_acquired(*args, **kwargs):
    first_request_entered.set()
    if not release_first_request.wait(timeout=30):
        raise TimeoutError("Timed out waiting to release the first heal request")
    raise api.HTTPException(status_code=503, detail="Stopped by concurrency test")

with (
    patch.object(api, "_start_baseline_evaluation"),
    TestClient(api.app) as first_client,
    TestClient(api.app) as second_client,
):
    def _first_heal():
        try:
            response = first_client.post(
                "/api/heal",
                data={"fault_ratio": "10", "method": "gradient"},
                files={"image": ("OIP.jpeg", image_bytes, "image/jpeg")},
            )
            first_response.append(response.status_code)
        except Exception as exc:
            first_errors.append(exc)

    first_thread = threading.Thread(target=_first_heal)
    with patch.object(api, "_get_data_loaders", _pause_after_lock_acquired):
        try:
            first_thread.start()
            if not first_request_entered.wait(timeout=10):
                raise TimeoutError(
                    "First heal request did not reach the post-lock data-loader helper"
                )

            second_response = second_client.post(
                "/api/heal",
                data={"fault_ratio": "10", "method": "gradient"},
                files={"image": ("OIP.jpeg", image_bytes, "image/jpeg")},
            )
            check(
                "Second request rejected while first holds lock",
                second_response.status_code == 429,
                f"got {second_response.status_code}",
            )
        except Exception as exc:
            check("Second request rejected while first holds lock", False, str(exc))
        finally:
            release_first_request.set()
            if first_thread.ident is not None:
                first_thread.join(timeout=15)

    check(
        "First request released the experiment lock",
        not first_thread.is_alive() and first_response == [503] and not first_errors,
        f"responses={first_response}, errors={first_errors}, alive={first_thread.is_alive()}",
    )

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "="*50)
passed = sum(1 for _, p in results if p)
total  = len(results)
print(f"Results: {passed}/{total} passed")
if passed == total:
    print(f"{PASS} All tests passed!")
else:
    print(f"{FAIL} {total-passed} test(s) failed.")
sys.exit(0 if passed == total else 1)
