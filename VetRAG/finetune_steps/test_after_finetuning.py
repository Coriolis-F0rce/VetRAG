import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
import numpy as np
import re
import time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer as BgeTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer as QwenTokenizer
from peft import PeftModel

# ---------- 路径配置 ----------
MODEL_DIR = Path(r"D:/Backup/PythonProject2/VetRAG/models_finetuned/qwen3-finetuned1")
TEST_DATA = Path(__file__).parent / "datas" / "test.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 加载微调后的 Qwen 模型 ----------
print("加载微调模型...")
base_model_path = r"D:\Backup\PythonProject2\VetRAG\models\Qwen3-0.6B\qwen\Qwen3-0___6B"
tokenizer = QwenTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, MODEL_DIR)
model.eval()

# ---------- 加载 BGE 模型 ----------
print("加载 BGE 模型...")
bge_model_name = "BAAI/bge-large-zh-v1.5"
bge_tokenizer = BgeTokenizer.from_pretrained(bge_model_name)
bge_model = AutoModel.from_pretrained(bge_model_name)
bge_model.to(DEVICE).eval()

def get_bge_embedding(text):
    inputs = bge_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = bge_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings[0]

def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)

def parse_chatml(text):
    pattern = r"<\|im_start\|>user\s+(.*?)<\|im_end\|>.*?<\|im_start\|>assistant\s+(.*?)<\|im_end\|>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

# ---------- 读取测试集 ----------
print("读取测试集...")
with open(TEST_DATA, "r", encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f]

print(f"共加载 {len(test_samples)} 条测试样本")
print("\n测试集预览（前3条）：")
for i, sample in enumerate(test_samples[:3]):
    text = sample.get("text", "")
    print(f"样本 {i+1} text 前200字符：")
    print(repr(text[:200]))
    print("-" * 50)

# ---------- 评估循环 ----------
similarities = []
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

success = 0
fail_parse = 0
fail_empty = 0
fail_generate = 0

start_time = time.time()

for idx, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
    if idx % 100 == 0:
        print(f"\n处理第 {idx} 个样本...")

    text = sample.get("text", "")
    if not text:
        fail_empty += 1
        continue

    instruction, reference = parse_chatml(text)
    if instruction is None or reference is None:
        fail_parse += 1
        if fail_parse <= 5:
            print(f"\n解析失败样本 {idx}: {repr(text[:150])}")
        continue

    if not instruction or not reference:
        fail_empty += 1
        continue

    prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)

    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    except Exception as e:
        print(f"\n生成失败，样本 {idx}: {e}")
        fail_generate += 1
        continue

    try:
        gen_vec = get_bge_embedding(generated)
        ref_vec = get_bge_embedding(reference)
        sim = cosine_similarity(gen_vec, ref_vec)
        similarities.append(sim)
        success += 1
    except Exception as e:
        print(f"\n相似度计算失败，样本 {idx}: {e}")
        fail_generate += 1
        continue

    # 可选：清空缓存，防止显存碎片
    if idx % 50 == 0:
        torch.cuda.empty_cache()

# ---------- 统计结果 ----------
total_time = time.time() - start_time
print(f"\n总耗时: {total_time:.2f} 秒")
print(f"解析统计：成功 {success} 条，解析失败 {fail_parse} 条，空字段 {fail_empty} 条，生成失败 {fail_generate} 条")

if len(similarities) == 0:
    print("错误：没有成功生成任何样本，请检查上述日志。")
else:
    similarities = np.array(similarities)
    print(f"评估完成，共 {len(similarities)} 个样本")
    print(f"平均语义相似度: {similarities.mean():.4f}")
    print(f"标准差: {similarities.std():.4f}")
    print(f"中位数: {np.median(similarities):.4f}")
    print(f"四分位数: {np.percentile(similarities, [25, 75])}")

    with open("semantic_scores.jsonl", "w", encoding="utf-8") as f_out:
        for sim in similarities:
            f_out.write(json.dumps({"similarity": float(sim)}, ensure_ascii=False) + "\n")