import urllib.request, urllib.error, json, os

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
        res = urllib.request.urlopen(req, timeout=90)
        return json.loads(res.read()), res.getcode()
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code

IMG = 'test_images/OIP.jpeg'
BASE = 'http://localhost:8000'
PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
errors = 0

def check(name, cond, detail=''):
    global errors
    if cond:
        print(f'  {PASS} {name}' + (f'  [{detail}]' if detail else ''))
    else:
        print(f'  {FAIL} {name}' + (f'  [{detail}]' if detail else ''))
        errors += 1

print('\n=== Test 1: Health ===')
d, c = post_multipart.__module__ and (lambda: (json.loads(urllib.request.urlopen(BASE+'/api/health').read()), 200))()
check('status ok', d.get('status') == 'ok', str(d))
check('baseline_checkpoint true', d.get('baseline_checkpoint') is True)
check('autoencoders_loaded true', d.get('autoencoders_loaded') is True)

print('\n=== Test 2: Methods ===')
d, _ = (json.loads(urllib.request.urlopen(BASE+'/api/methods').read()), 200)
ids = {m['id'] for m in d['methods']}
check('7 methods', ids == {'gradient','ae_dual','hybrid','regrowth','regrowth_gradient','rerouting','adaptive'}, str(ids))
check('ae available', d['autoencoders_available'] is True)

print('\n=== Test 3: Baseline ===')
d, c = post_multipart(BASE+'/api/baseline', {}, IMG)
check('HTTP 200', c == 200, str(c))
check('accuracy > 0', float(d.get('accuracy', 0)) > 0, str(d.get('accuracy')))
check('prediction present', 'prediction' in d)
print(f'    accuracy={d.get("accuracy")}%  pred={d.get("prediction")}  conf={d.get("confidence")}')

print('\n=== Test 4: Invalid Inputs ===')
d, c = post_multipart(BASE+'/api/fault', {'fault_ratio': '99'}, IMG)
check('fault_ratio=99 rejected 422', c == 422, str(c))

d, c = post_multipart(BASE+'/api/fault', {'fault_ratio': '0'}, IMG)
check('fault_ratio=0 rejected 422', c == 422, str(c))

d, c = post_multipart(BASE+'/api/heal', {'fault_ratio': '20', 'method': 'garbage'}, IMG)
check('method=garbage rejected 422', c == 422, str(c))

print('\n=== Test 5: Fault ===')
d, c = post_multipart(BASE+'/api/fault', {'fault_ratio': '20'}, IMG)
check('HTTP 200', c == 200, str(c))
check('accuracy present', 'accuracy' in d)
print(f'    fault=20%  accuracy={d.get("accuracy")}%  pred={d.get("prediction")}')

print(f'\n{"="*40}')
print(f'Tests done. Errors: {errors}')
if errors == 0:
    print('All tests PASSED!')
