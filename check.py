import json
p = r'D:\Backup\PythonProject2\data_process\s4_augmented_output\progress.json'
d = json.load(open(p, encoding='utf-8'))
print(f'processed={d.get("processed",0)}, results={len(d.get("results",[]))}')
