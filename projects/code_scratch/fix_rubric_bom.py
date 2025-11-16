import os
import json

base = os.path.join(os.environ.get('TEMP','C:\\Windows\\Temp'), 'paperbench_demo', 'papers')
if not os.path.isdir(base):
    print('BASE_NOTFOUND', base)
    raise SystemExit(1)

fixed = []
errors = []
for name in sorted(os.listdir(base)):
    pdir = os.path.join(base, name)
    if not os.path.isdir(pdir):
        continue
    rfile = os.path.join(pdir, 'rubric.json')
    if not os.path.isfile(rfile):
        errors.append((name, 'missing'))
        continue
    try:
        # read using utf-8-sig to tolerate BOM, then write back as utf-8 (no BOM)
        with open(rfile, 'r', encoding='utf-8-sig') as f:
            s = f.read()
        # validate json
        try:
            json.loads(s)
        except Exception as je:
            errors.append((name, 'invalid', str(je)))
            continue
        # rewrite without BOM
        with open(rfile, 'w', encoding='utf-8') as f:
            f.write(s)
        fixed.append(name)
    except Exception as e:
        errors.append((name, 'error', str(e)))

print('BASE_DIR:', base)
print('FIXED_COUNT:', len(fixed))
for n in fixed:
    print(' FIXED:', n)
print('ERRORS_COUNT:', len(errors))
for e in errors:
    print(' ERROR:', e)
