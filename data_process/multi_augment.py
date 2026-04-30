#!/usr/bin/env python3
"""
S4: Multi-Augment (Optimized)
Apply 6 augmentation methods to a randomly sampled 500-entry subset.
- 5 API-based methods + 3 rule-based methods
- Processing only sampled entries for cost efficiency
Estimated API cost: ~150-200 API calls (500 samples * ~0.4 API methods per sample)
"""

import os
import json
import time
import random
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# ==================== Config ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2eedab5b21954b6bb26f7461706642f2")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

BASE_DIR = Path(r"D:\Backup\PythonProject2\data_process")
INPUT_FILE = BASE_DIR / "find_faq" / "merged_output" / "s1_merged_all.json"
OUTPUT_DIR = BASE_DIR / "s4_augmented_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 500
API_METHODS = ["synonym_rewrite", "sentence_transform", "perspective_shift", "emotion_variation"]
RULE_METHODS = ["noise_inject", "scenario_expand"]


# ==================== API Caller ====================

def call_api(prompt: str, max_tokens: int = 1024) -> Optional[str]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(DEEPSEEK_API_URL, data=payload)
            req.add_header("Authorization", f"Bearer {DEEPSEEK_API_KEY}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(2 ** attempt)
    return None


def parse_json(text: str) -> List[Dict]:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        if isinstance(data, list):
            return [i for i in data if isinstance(i, dict) and "instruction" in i and "output" in i]
        elif isinstance(data, dict) and "instruction" in data:
            return [data]
    except Exception:
        pass
    return []


# ==================== API Methods ====================

def synonym_rewrite(entry: Dict) -> List[Dict]:
    prompt = f"""请对以下问答对中的问题进行同义改写，生成一个不同表述但核心意图不变的新问题。
回答内容可以保持不变，也可以稍作调整以适应新问题。
以JSON格式返回：{{"instruction": "新问题", "output": "对应回答"}}

原问答：
问题：{entry['instruction']}
回答：{entry['output']}"""
    result = call_api(prompt)
    if result:
        return parse_json(result)
    return []


def sentence_transform(entry: Dict) -> List[Dict]:
    prompt = f"""请对以下问答对进行句式变换，将问题改为不同句式（如反问、设问等），保持核心含义不变。
生成1个新问答对。
以JSON格式返回：{{"instruction": "新问题", "output": "对应回答"}}

原问答：
问题：{entry['instruction']}
回答：{entry['output']}"""
    result = call_api(prompt)
    if result:
        return parse_json(result)
    return []


def perspective_shift(entry: Dict) -> List[Dict]:
    prompt = f"""请将以下问答对转换为"从宠物主人视角"表述的新问答对，改变人称和表述习惯。
以JSON格式返回：{{"instruction": "新问题", "output": "对应回答"}}

原问答：
问题：{entry['instruction']}
回答：{entry['output']}"""
    result = call_api(prompt)
    if result:
        return parse_json(result)
    return []


def emotion_variation(entry: Dict) -> List[Dict]:
    prompt = f"""请将以下问答对中的提问者情绪改为"焦急担忧"，生成一个新问答对。
回答应保持专业但适当安抚情绪。
以JSON格式返回：{{"instruction": "新问题", "output": "对应回答"}}

原问答：
问题：{entry['instruction']}
回答：{entry['output']}"""
    result = call_api(prompt)
    if result:
        return parse_json(result)
    return []


# ==================== Rule Methods ====================

def noise_inject(entry: Dict) -> List[Dict]:
    noises = ["啊", "呢", "嘛", "哈", "哦", "呀", "呗"]
    question = entry["instruction"]
    chars = list(question)
    for _ in range(random.randint(1, 2)):
        idx = random.randint(1, max(1, len(chars) - 1))
        chars.insert(idx, random.choice(noises))
    return [{"instruction": "".join(chars), "output": entry["output"], "metadata": entry.get("metadata", {})}]

def scenario_expand(entry: Dict) -> List[Dict]:
    scenarios = [
        "（我家狗突然）", "（今天带狗出门时）", "（狗子这两天）",
        "（第一次养狗，请问）", "（紧急求助！）", "（周末在家）",
    ]
    return [{
        "instruction": f"{random.choice(scenarios)}{entry['instruction']}",
        "output": entry["output"],
        "metadata": entry.get("metadata", {})
    }]


# ==================== Dispatcher ====================

def augment(entry: Dict, method: str) -> List[Dict]:
    fn_map = {
        "synonym_rewrite": synonym_rewrite,
        "sentence_transform": sentence_transform,
        "perspective_shift": perspective_shift,
        "emotion_variation": emotion_variation,
        "noise_inject": noise_inject,
        "scenario_expand": scenario_expand,
    }
    fn = fn_map.get(method)
    if fn:
        try:
            return fn(entry)
        except Exception:
            pass
    return []


# ==================== Main ====================

def main():
    print("=" * 60)
    print("S4: Multi-Augment (Optimized)")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_entries = json.load(f)
    print(f"\n[*] Total entries: {len(all_entries)}")
    print(f"[*] Sampling {MAX_SAMPLES} for augmentation")

    random.seed(42)
    sampled = random.sample(all_entries, min(MAX_SAMPLES, len(all_entries)))

    # Rule-based on ALL sampled
    print("\n[*] Running rule-based augmentations...")
    results: List[Dict] = []
    seen = set()
    rule_stats = defaultdict(int)

    for entry in sampled:
        for method in RULE_METHODS:
            qa_list = augment(entry, method)
            for qa in qa_list:
                h = hashlib.md5((qa.get("instruction", "") + "|" + qa.get("output", "")).encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    results.append(qa)
                    rule_stats[method] += 1

    for m, cnt in rule_stats.items():
        print(f"   {m}: {cnt}")

    # API-based on sampled (random subset of 200 for speed)
    api_subset = random.sample(sampled, min(200, len(sampled)))
    print(f"\n[*] Running API augmentations on {len(api_subset)} samples...")
    api_stats = defaultdict(int)

    for idx, entry in enumerate(api_subset):
        for method in API_METHODS:
            qa_list = augment(entry, method)
            for qa in qa_list:
                h = hashlib.md5((qa.get("instruction", "") + "|" + qa.get("output", "")).encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    results.append(qa)
                    api_stats[method] += 1
        if (idx + 1) % 20 == 0:
            print(f"   Progress: {idx + 1}/{len(api_subset)}, total results: {len(results)}")
        time.sleep(0.3)

    for m, cnt in api_stats.items():
        print(f"   {m}: {cnt}")

    # Save
    out_path = OUTPUT_DIR / "s4_augmented_all.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] S4 Done. Generated {len(results)} augmented samples -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
