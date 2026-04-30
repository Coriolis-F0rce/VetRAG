import json
p = r'D:\Backup\PythonProject2\data_process\s4_augmented_output\progress.json'
try:
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    print(f'processed: {d.get("processed",0)}, results: {len(d.get("results",[]))}')
except Exception as e:
    print(f'Error: {e}')
