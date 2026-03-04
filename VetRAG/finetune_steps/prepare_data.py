import json
import os
import random
from glob import glob


INPUT_DIR = "faq_json"
OUTPUT_DIR = "datas"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


CHAT_TEMPLATE = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"


def load_json_files(file_pattern):
    samples = []
    for file_path in glob(file_pattern):
        print(f"读取文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
                samples.extend(data)
            else:
                for line in content.splitlines():
                    if line:
                        samples.append(json.loads(line))
    return samples

def convert_to_chatml(instruction, output):
    return CHAT_TEMPLATE.format(user=instruction, assistant=output)


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    all_samples = load_json_files(os.path.join(INPUT_DIR, "faq*.json"))
    print(f"共加载 {len(all_samples)} 条原始数据")


    chatml_texts = []
    for sample in all_samples:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        if instruction and output:
            text = convert_to_chatml(instruction, output)
            chatml_texts.append({"text": text})


    random.shuffle(chatml_texts)
    n = len(chatml_texts)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_data = chatml_texts[:train_end]
    val_data = chatml_texts[train_end:val_end]
    test_data = chatml_texts[val_end:]


    def save_jsonl(data, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"已保存 {len(data)} 条到 {path}")

    save_jsonl(train_data, "train.jsonl")
    save_jsonl(val_data, "val.jsonl")
    save_jsonl(test_data, "test.jsonl")

    print("数据准备完成！")

if __name__ == "__main__":
    main()