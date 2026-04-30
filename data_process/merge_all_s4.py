import json, hashlib
from pathlib import Path

base = Path(r'D:\Backup\PythonProject2\data_process\s4_augmented_output')
# All S4 outputs: old slow run (11000) + new optimized (2009) + addon (269)
files = [
    base / 'progress.json',   # old slow run: 11000 results
    base / 's4_augmented_all.json',  # optimized new run: 1740
    base / 's4_addon.json',  # addon: 269
]

seen = set()
results = []
for fp in files:
    if not fp.exists():
        continue
    print(f'Processing {fp.name}...')
    if fp.name.endswith('.json') and not fp.name.startswith('s4'):
        # progress.json - special structure
        data = json.load(open(fp, encoding='utf-8')).get('results', [])
    else:
        data = json.load(open(fp, encoding='utf-8'))
    if not isinstance(data, list):
        print(f'  Skipped: not a list')
        continue
    count = 0
    for r in data:
        h = hashlib.md5((r.get('instruction','')+'|'+r.get('output','')).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            results.append(r)
            count += 1
    print(f'  {len(data)} -> {count} new unique')

out = base / 's4_final.json'
json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'Total unique S4 entries: {len(results)} -> {out}')
