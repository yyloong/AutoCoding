import os, json, sys
base = os.path.join(os.environ.get('TEMP','C:\\Windows\\Temp'), 'paperbench_demo', 'papers')
if not os.path.isdir(base):
    print('NOTFOUND', base)
    sys.exit(0)
problems = []
papers = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))])
for name in papers:
    pdir = os.path.join(base, name)
    r = os.path.join(pdir, 'rubric.json')
    if not os.path.isfile(r):
        problems.append((name, 'missing'))
    else:
        try:
            s = open(r, 'r', encoding='utf-8').read()
            if not s or s.strip() == '':
                problems.append((name, 'empty'))
            else:
                json.loads(s)
        except Exception as e:
            problems.append((name, 'invalid', str(e)))
print('BASE_DIR:', base)
print('TOTAL_PAPERS:', len(papers))
print('PROBLEMS_COUNT:', len(problems))
for p in problems:
    print(' -', p)
