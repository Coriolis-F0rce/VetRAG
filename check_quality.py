import json, random
d = json.load(open(r'D:\Backup\PythonProject2\data_process\final_output\final_training_data.json', encoding='utf-8'))
print(f'Total: {len(d)}')
print()
samples = random.sample(d, 5)
for i, it in enumerate(samples):
    meta = it.get('metadata', {})
    src = meta.get('_source', '?')
    cat = meta.get('category', '?')
    print(f'[{i}] source={src}, cat={cat}')
    print(f'    Q: {it["instruction"][:60]}')
    print(f'    A: {it["output"][:80]}')
    print()
