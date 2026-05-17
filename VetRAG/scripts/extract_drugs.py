"""
从 diseases.json 提取药物列表并输出 pharmaceuticals_v0.json。
用法：python scripts/extract_drugs.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

DATA_DIR = _project_root / "data"
DISEASES_FILE = DATA_DIR / "diseases.json"
OUTPUT_FILE = DATA_DIR / "pharmaceuticals_v0.json"


def split_drug_names(drug_field: str) -> list[str]:
    """拆分 drug 字段中的多药名，如 '阿莫西林-克拉维酸/恩诺沙星/头孢噻呋'"""
    if not drug_field:
        return []
    # 按 /、/ 、或 、 拆分
    parts = re.split(r"\s*[/、，,]\s*", drug_field)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def normalize_drug_name(name: str) -> str:
    """规范化药物名称，处理常见变体。"""
    name = name.strip()
    # 去掉括号内的说明，如"平衡电解质溶液（如乳酸林格氏液）"
    name = re.sub(r"[（(][^)）]*[)）]", "", name).strip()
    # 统一分隔符
    name = name.replace("／", "/").replace("、", "/")
    # 只取第一个有效药物名（复合名如"阿莫西林克拉维酸钾"保留）
    return name


def main():
    with open(DISEASES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    diseases = data if isinstance(data, list) else data.get("diseases", [])

    drug_map: dict[str, dict] = defaultdict(lambda: {
        "drug_name": "",
        "categories": set(),
        "indications": [],
        "dosages": [],
        "routes": set(),
        "frequencies": set(),
        "durations": set(),
        "treatment_names": [],
    })

    for disease in diseases:
        d_name = disease.get("disease_name", "")
        for t in disease.get("treatment", []):
            drug_field = t.get("drug", "")
            if not drug_field:
                continue
            drug_names = split_drug_names(drug_field)
            dosage = t.get("dosage", "")
            route = t.get("route", "")
            freq = t.get("frequency", "")
            duration = t.get("duration", "")
            category = t.get("category", "")
            treatment_name = t.get("name", "")

            for dn in drug_names:
                dn = normalize_drug_name(dn)
                if len(dn) < 2:
                    continue
                entry = drug_map[dn]
                entry["drug_name"] = dn
                if category:
                    entry["categories"].add(category)
                if d_name and d_name not in entry["indications"]:
                    entry["indications"].append(d_name)
                if dosage and dosage not in entry["dosages"]:
                    entry["dosages"].append(dosage)
                if route:
                    entry["routes"].add(route)
                if freq:
                    entry["frequencies"].add(freq)
                if duration:
                    entry["durations"].add(duration)
                if treatment_name and treatment_name not in entry["treatment_names"]:
                    entry["treatment_names"].append(treatment_name)

    # 转换为列表输出
    output = []
    for name in sorted(drug_map.keys()):
        entry = drug_map[name]
        output.append({
            "drug_name": entry["drug_name"],
            "drug_class": list(entry["categories"]),
            "indications": entry["indications"],
            "dosages": entry["dosages"],
            "routes": list(entry["routes"]),
            "frequencies": list(entry["frequencies"]),
            "durations": list(entry["durations"]),
            "treatment_names": entry["treatment_names"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"提取完成: {len(output)} 种药物 → {OUTPUT_FILE}")

    # 统计
    with_dosage = sum(1 for d in output if d["dosages"])
    multi_indication = sum(1 for d in output if len(d["indications"]) > 1)
    print(f"  含剂量信息: {with_dosage}")
    print(f"  多种适应症: {multi_indication}")
    print(f"  平均适应症数: {sum(len(d['indications']) for d in output) / len(output):.1f}")


if __name__ == "__main__":
    main()
