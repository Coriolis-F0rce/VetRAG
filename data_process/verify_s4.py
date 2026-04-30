import json, hashlib
from pathlib import Path

base = Path(r'D:\Backup\PythonProject2\data_process\s4_augmented_output')

# Verify file contents first
for fname in ['progress.json', 's4_augmented_all.json', 's4_addon.json']:
    fp = base / fname
    if not fp.exists():
        print(f'{fname}: NOT FOUND')
        continue
    data = json.load(open(fp, encoding='utf-8'))
    if fname == 'progress.json':
        entries = data.get('results', [])
    elif isinstance(data, list):
        entries = data
    else:
        entries = []
    print(f'{fname}: {len(entries)} entries, size={fp.stat().st_size/1024:.0f}KB')
