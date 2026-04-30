import json
with open(r'D:\Backup\PythonProject2\data_process\s4_augmented_output\progress.json', encoding='utf-8') as f:
    d = json.load(f)
print(f'Processed: {d.get("processed", 0)}')
print(f'Results so far: {len(d.get("results", []))}')
