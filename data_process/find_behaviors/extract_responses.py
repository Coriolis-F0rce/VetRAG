import json

# 读取原始文件
with open('api_responses.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有 content 字段并解析为 JSON
extracted_contents = []
for item in data['results']:
    content_str = item['response']['choices'][0]['message']['content']
    content_json = json.loads(content_str)
    extracted_contents.append(content_json)

# 保存到新文件
with open('dog_behaviors_professional.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_contents, f, ensure_ascii=False, indent=2)

print(f"已提取 {len(extracted_contents)} 个行为描述到 dog_behaviors_professional.json")