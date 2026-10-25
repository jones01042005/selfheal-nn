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
    check("accuracy returned",   'accuracy' in data and float(data['accuracy']) > 0)
    check("prediction returned", 'prediction' in data)
    check("confidence returned", 'confidence' in data)
    print(f"     Baseline accuracy: {data.get('accuracy')}%  |  Prediction: {data.get('prediction')}")
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
lock_results = []
def _heal():
    try:
        d, c = post_multipart("/api/heal", {"fault_ratio": "10", "method": "gradient"}, IMG)
        lock_results.append(c)
    except:
        lock_results.append(0)

t1 = threading.Thread(target=_heal)
t2 = threading.Thread(target=_heal)
t1.start(); time.sleep(0.5); t2.start()
t1.join(); t2.join()
check("At least one 429 returned when concurrent", 429 in lock_results, str(lock_results))

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
