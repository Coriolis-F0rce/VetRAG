#!/usr/bin/env python3
"""
S4 (Quick Add-on): Generate additional high-quality augmented samples.
100 samples * 6 methods = 600 more entries. Fast rule-based + targeted API.
"""

import os, json, time, random, hashlib, urllib.request
from pathlib import Path

KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2eedab5b21954b6bb26f7461706642f2")
URL = "https://api.deepseek.com/v1/chat/completions"
BASE = Path(r"D:\Backup\PythonProject2\data_process")
INPUT = BASE / "find_faq" / "merged_output" / "s1_merged_all.json"
OUT = BASE / "s4_augmented_output" / "s4_addon.json"
OUT_FINAL = BASE / "s4_augmented_output" / "s4_augmented_all.json"

def api(prompt, tokens=1024):
    payload = json.dumps({"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": tokens}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=payload)
            req.add_header("Authorization", f"Bearer {KEY}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(2 ** attempt)
    return None

def parse(text):
    try:
        for seg in text.split("```json")[1:]:
            d = json.loads(seg.split("```")[0].strip())
            if isinstance(d, dict) and "instruction" in d:
                return [d]
    except: pass
    return []

def main():
    print("S4 Add-on: Quick augment")
    with open(INPUT, encoding="utf-8") as f:
        all_entries = json.load(f)
    random.seed(99)
    sampled = random.sample(all_entries, min(100, len(all_entries)))
    results = []
    seen = set()

    # Rule-based
    for e in sampled:
        for method, fn in [("noise_inject", lambda: [{"instruction": "".join(random.choice([c, c]) if i % 3 else c for i, c in enumerate(list(e["instruction"]))), "output": e["output"], "metadata": e.get("metadata", {})}]),
                           ("scenario", lambda: [{"instruction": random.choice(["（我家狗突然）", "（今天带狗出门时）", "（狗子这两天）", "（第一次养狗，请问）"])+e["instruction"], "output": e["output"], "metadata": e.get("metadata", {})}])]:
            for qa in fn():
                h = hashlib.md5((qa.get("instruction","")+"|"+qa.get("output","")).encode()).hexdigest()
                if h not in seen:
                    seen.add(h); results.append(qa)

    # API-based (5 methods * 100 samples)
    methods = [
        ("同义改写", "请对以下问答对中的问题进行同义改写，生成一个不同表述但核心意图不变的新问题。回答保持不变。以JSON格式返回：{{\"instruction\": \"新问题\", \"output\": \"对应回答\"}}\n\n原问答：\n问题：{i}\n回答：{o}"),
        ("句式变换", "请将以下问答对的问题改为反问或设问句式，保持核心含义不变。以JSON格式返回：{{\"instruction\": \"新问题\", \"output\": \"对应回答\"}}\n\n原问答：\n问题：{i}\n回答：{o}"),
        ("视角转换", "请将以下问答对转换为从宠物主人视角表述的新问答对。以JSON格式返回：{{\"instruction\": \"新问题\", \"output\": \"对应回答\"}}\n\n原问答：\n问题：{i}\n回答：{o}"),
        ("增加细节", "请将以下问答对的问题增加更多具体细节（如年龄、体型等），生成一个新问题。以JSON格式返回：{{\"instruction\": \"新问题\", \"output\": \"对应回答\"}}\n\n原问答：\n问题：{i}\n回答：{o}"),
    ]

    for m_idx, (m_name, m_prompt) in enumerate(methods):
        for idx, e in enumerate(sampled):
            prompt = m_prompt.format(i=e["instruction"], o=e["output"])
            r = api(prompt)
            if r:
                for qa in parse(r):
                    h = hashlib.md5((qa.get("instruction","")+"|"+qa.get("output","")).encode()).hexdigest()
                    if h not in seen:
                        seen.add(h); results.append(qa)
            if (idx+1) % 25 == 0:
                print(f"  {m_name}: {idx+1}/100, total: {len(results)}")
            time.sleep(0.4)

    print(f"Generated {len(results)} add-on samples")
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved -> {OUT}")

    # Merge with existing s4_augmented_all.json
    existing = []
    if OUT_FINAL.exists():
        existing = json.load(open(OUT_FINAL, encoding="utf-8"))
    merged = existing + results
    # Dedup
    seen2, unique = set(), []
    for r in merged:
        h = hashlib.md5((r.get("instruction","")+"|"+r.get("output","")).encode()).hexdigest()
        if h not in seen2:
            seen2.add(h); unique.append(r)
    json.dump(unique, open(OUT_FINAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Merged: existing={len(existing)}, new={len(results)}, total unique={len(unique)}")

if __name__ == "__main__":
    main()
