import urllib.request, urllib.error, json, os, time

def post_multipart(url, fields, file_path=None):
    boundary = 'aBcDeFgH1234567890'
    body = []
    for k, v in fields.items():
        body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    if file_path:
        fname = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            data = f.read()
        body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{fname}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
        body.append(data)
        body.append(b'\r\n')
    body.append(f'--{boundary}--\r\n'.encode())
    raw = b''.join(body)
    req = urllib.request.Request(
        url, data=raw,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )
    try:
        res = urllib.request.urlopen(req, timeout=180)
        return json.loads(res.read()), res.getcode()
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code

IMG = 'test_images/OIP.jpeg'
BASE = 'http://localhost:8000'

print('=== End-to-End Heal Test (gradient, fault=10%) ===')
print('This takes ~30-90s on CPU...')
t0 = time.time()
d, c = post_multipart(BASE+'/api/heal', {'fault_ratio': '10', 'method': 'gradient'}, IMG)
elapsed = round(time.time()-t0, 1)
print(f'HTTP: {c}')
print(f'Time: {elapsed}s')
if c == 200:
    print(f'baseline_acc: {d.get("baseline_acc")}%')
    print(f'faulty_acc:   {d.get("faulty_acc")}%')
    print(f'healed_acc:   {d.get("healed_acc")}%')
    print(f'recovery:     {d.get("recovery")}%')
    print(f'method:       {d.get("method")}')
    print(f'prediction:   {d.get("prediction")}')
    b = float(d.get('baseline_acc', 0))
    f_ = float(d.get('faulty_acc', 0))
    h = float(d.get('healed_acc', 0))
    assert f_ < b, f"faulty {f_} should be < baseline {b}"
    assert h > f_, f"healed {h} should be > faulty {f_}"
    print('PASS: All healing assertions passed!')
else:
    print(f'FAIL: {d}')
