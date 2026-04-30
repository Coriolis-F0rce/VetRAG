import json, hashlib
from pathlib import Path

base = Path(r'D:\Backup\PythonProject2\data_process\s4_augmented_output')
files = [base / 's4_augmented_all.json', base / 's4_addon.json']
out = base / 's4_merged.json'

seen = set()
results = []
for fp in files:
    if not fp.exists():
        print(f'Skipping {fp.name}')
        continue
    data = json.load(open(fp, encoding='utf-8'))
    print(f'{fp.name}: {len(data)} entries')
    for r in data:
        h = hashlib.md5((r.get('instruction','')+'|'+r.get('output','')).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            results.append(r)

json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'Total unique S4: {len(results)} -> {out}')
