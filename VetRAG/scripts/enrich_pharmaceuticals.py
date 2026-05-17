"""
用 DeepSeek API 补全药物 v1 字段。
用法：python scripts/enrich_pharmaceuticals.py [--workers 10]
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

INPUT = _project_root / "data" / "pharmaceuticals.json"
OUTPUT = _project_root / "data" / "pharmaceuticals_enriched.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

SYSTEM_PROMPT = """你是一位兽医药物专家。请根据输入的药物信息，补全以下字段。只输出 JSON，不要解释。"""

USER_PROMPT_TEMPLATE = """已知药物信息：
- 药物名称: {drug_name}
- 药物类别: {drug_class}
- 适应症: {indications}
- 参考剂量: {dosages}
- 给药途径: {routes}
- 给药频率: {frequencies}

请生成以下字段（输出标准 JSON）：

{{
  "drug_name_en": "英文通用名/拉丁名",
  "mechanism": "药理作用机制，50字以内",
  "contraindications": ["禁忌症1", "禁忌症2", ...],
  "side_effects": [
    {{"effect": "不良反应描述", "frequency": "常见/偶见/罕见"}}
  ],
  "drug_interactions": ["相互作用1", "相互作用2", ...],
  "monitoring": "用药监测建议，30字以内"
}}

要求：
1. drug_name_en 使用国际通用名（INN/USAN），不含品牌名
2. contraindications 列出2-4条关键禁忌
3. side_effects 列出2-4条，需标注频率
4. drug_interactions 列出2-3条临床重要相互作用
5. 所有字段用中文，drug_name_en 用英文"""


def enrich_one(drug: dict, model: str = "deepseek-chat") -> dict:
    """调用 DeepSeek 补全一个药物的字段。"""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        drug_name=drug.get("drug_name", ""),
        drug_class="、".join(drug.get("drug_class", [])) or "未知",
        indications="、".join(drug.get("indications", [])[:3]) or "未知",
        dosages="、".join(drug.get("dosages", [])[:2]) or "未知",
        routes="、".join(drug.get("routes", [])) or "未知",
        frequencies="、".join(drug.get("frequencies", [])) or "未知",
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(DEEPSEEK_URL, json=payload, headers=headers)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                # 提取 JSON
                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    return json.loads(match.group())
                return {"_error": "JSON parse failed", "_raw": content[:200]}
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"_error": str(e)}
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            return {"_error": str(e)}

    return {"_error": "max retries exceeded"}


def enrich_batch(drugs: list[dict], max_workers: int = 15) -> list[dict]:
    """并行补全一批药物。"""
    enriched = [None] * len(drugs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, drug in enumerate(drugs):
            # 跳过已有 v1 字段的药物
            if drug.get("mechanism") and drug.get("contraindications"):
                enriched[i] = drug
                continue
            future = executor.submit(enrich_one, drug)
            future_to_idx[future] = i

        if not future_to_idx:
            return drugs

        pbar = tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Enriching drugs",
            unit="d",
        )
        for future in pbar:
            idx = future_to_idx[future]
            result = future.result()
            drug = drugs[idx].copy()
            if "_error" not in result:
                drug["drug_name_en"] = result.get("drug_name_en", "")
                drug["mechanism"] = result.get("mechanism", "")
                drug["contraindications"] = result.get("contraindications", [])
                drug["side_effects"] = result.get("side_effects", [])
                drug["drug_interactions"] = result.get("drug_interactions", [])
                drug["monitoring"] = result.get("monitoring", "")
            else:
                drug["_enrich_error"] = result["_error"]
            enriched[idx] = drug
            pbar.set_postfix_str(drug["drug_name"][:20])

    return enriched


def main():
    import argparse

    parser = argparse.ArgumentParser(description="补全药物 v1 字段")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true", help="只统计不执行")
    args = parser.parse_args()

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "drugs" in data:
        drugs = data["drugs"]
    elif isinstance(data, list):
        drugs = data
    else:
        print("错误：无法识别的数据格式")
        return

    # 统计需要补全的
    need_enrich = sum(1 for d in drugs if not d.get("mechanism"))
    already = len(drugs) - need_enrich
    print(f"共 {len(drugs)} 种药物，已有 v1 字段: {already}，需补全: {need_enrich}")

    if args.dry_run:
        return

    if need_enrich == 0:
        print("无需补全。")
        return

    enriched = enrich_batch(drugs, max_workers=args.workers)

    # 统计结果
    ok = sum(1 for d in enriched if not d.get("_enrich_error"))
    err = sum(1 for d in enriched if d.get("_enrich_error"))
    print(f"\n补全完成: 成功 {ok}，失败 {err}")

    output_data = {"schema_version": "1.0", "drugs": enriched}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
