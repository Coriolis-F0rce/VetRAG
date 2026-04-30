import json, hashlib, shutil
src = r'D:\Backup\PythonProject2\data_process\s4_augmented_output\progress.json'
dst = r'D:\Backup\PythonProject2\data_process\s4_augmented_output\s4_augmented_all.json'
print('Loading progress.json...')
d = json.load(open(src, encoding='utf-8'))
results = d.get('results', [])
print(f'Results: {len(results)}')
# Remove duplicates by hash
seen = set()
unique = []
for r in results:
    h = hashlib.md5((r.get('instruction','') + '|' + r.get('output','')).encode()).hexdigest()
    if h not in seen:
        seen.add(h)
        unique.append(r)
print(f'Unique: {len(unique)}')
json.dump(unique, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'Saved to {dst}')
