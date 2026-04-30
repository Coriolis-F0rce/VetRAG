#!/usr/bin/env python3
"""
S2: Disease Knowledge Base -> QA Pairs Converter
Convert structured disease knowledge into diverse question-answer pairs.
No API needed - all generation is rule-based template.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any


BASE_DIR = Path(r"D:\Backup\PythonProject2\data_process\find_diseases")
KB_FILE = BASE_DIR / "dog_diseases_knowledge_base.json"
OUTPUT_FILE = BASE_DIR / "s2_disease_qa.json"


# ---- QA Template Functions ----

def t_disease_what(d: Dict) -> Dict:
    """Template 1: What is X disease?"""
    return {
        "instruction": f"什么是{d['disease_name']}？",
        "output": f"{d['disease_name']}（{d.get('disease_type', '疾病类型未知')}）是一种{d['disease_type']}。"
                   f"该病主要影响{d['affected_species'][0] if d.get('affected_species') else '犬'}。"
                   f"主要症状包括：{'、'.join(d.get('key_symptoms', [])[:5])}。"
                   f"如需了解更多，请咨询兽医。",
        "metadata": {"category": "disease_knowledge", "type": "what_is", "disease": d["disease_name"]}
    }


def t_symptoms(d: Dict) -> Dict:
    """Template 2: What are symptoms of X?"""
    symptoms = d.get("key_symptoms", [])
    if not symptoms:
        return None
    return {
        "instruction": f"狗得了{d['disease_name']}会有什么症状？",
        "output": f"{d['disease_name']}的典型症状包括：{'、'.join(symptoms)}。"
                   f"如发现以上症状，请尽快带犬就医确诊。",
        "metadata": {"category": "disease_knowledge", "type": "symptoms", "disease": d["disease_name"]}
    }


def t_treatment(d: Dict) -> Dict:
    """Template 3: How to treat X?"""
    treatments = d.get("treatment", [])
    if not treatments:
        return None
    return {
        "instruction": f"狗得了{d['disease_name']}怎么治疗？",
        "output": f"针对{d['disease_name']}，常用治疗方案包括：{'；'.join(treatments[:4])}。"
                   f"具体用药方案请遵医嘱，切勿自行用药。",
        "metadata": {"category": "disease_knowledge", "type": "treatment", "disease": d["disease_name"]}
    }


def t_prevention(d: Dict) -> Dict:
    """Template 4: How to prevent X?"""
    prev = d.get("prevention", [])
    if not prev:
        return None
    return {
        "instruction": f"怎么预防{d['disease_name']}？",
        "output": f"预防{d['disease_name']}的措施包括：{'；'.join(prev[:3])}。"
                   f"如需接种疫苗或采取其他专业预防措施，请咨询兽医。",
        "metadata": {"category": "disease_knowledge", "type": "prevention", "disease": d["disease_name"]}
    }


def t_zoonotic(d: Dict) -> Dict:
    """Template 5: Is X zoonotic?"""
    z = d.get("zoonotic", "未知")
    species = d.get("affected_species", [])
    return {
        "instruction": f"{d['disease_name']}会传染给人吗？",
        "output": f"{d['disease_name']}的人畜共患性为：{z}。"
                   f"易感物种包括：{', '.join(species[:5])}。"
                   f"日常接触患病动物时请注意防护，详情请咨询兽医。",
        "metadata": {"category": "disease_knowledge", "type": "zoonotic", "disease": d["disease_name"]}
    }


def t_infectious(d: Dict) -> Dict:
    """Template 6: Is X infectious?"""
    inf = d.get("infectiousness_details", "传染性未知。")
    return {
        "instruction": f"{d['disease_name']}会传染给其他狗吗？",
        "output": f"{d['disease_name']}的传染性信息：{inf}。"
                   f"如家中有其他犬只，请做好隔离防护措施。",
        "metadata": {"category": "disease_knowledge", "type": "infectious", "disease": d["disease_name"]}
    }


def t_diagnosis(d: Dict) -> Dict:
    """Template 7: How to diagnose X?"""
    diag = d.get("diagnosis", [])
    if not diag:
        return None
    return {
        "instruction": f"狗得了{d['disease_name']}怎么诊断？",
        "output": f"{d['disease_name']}的诊断方法包括：{'；'.join(diag[:4])}。"
                   f"具体诊断方案由兽医根据实际情况决定。",
        "metadata": {"category": "disease_knowledge", "type": "diagnosis", "disease": d["disease_name"]}
    }


def t_affected_species(d: Dict) -> Dict:
    """Template 8: What species are affected?"""
    species = d.get("affected_species", [])
    if len(species) <= 1:
        return None
    return {
        "instruction": f"哪些动物会得{d['disease_name']}？",
        "output": f"{d['disease_name']}的易感动物包括：{', '.join(species)}。"
                   f"如家中有多物种宠物，需注意隔离防护。",
        "metadata": {"category": "disease_knowledge", "type": "affected_species", "disease": d["disease_name"]}
    }


# ---- Templates that generate MULTIPLE QAs from one disease ----

TEMPLATES = [
    t_disease_what,
    t_symptoms,
    t_treatment,
    t_prevention,
    t_zoonotic,
    t_infectious,
    t_diagnosis,
    t_affected_species,
]


def generate_all(disease: Dict) -> List[Dict]:
    """Generate all QA pairs from one disease entry."""
    results = []
    for template_fn in TEMPLATES:
        try:
            qa = template_fn(disease)
            if qa:
                results.append(qa)
        except Exception:
            pass
    return results


def main():
    print("=" * 60)
    print("S2: Disease KB -> QA Pairs")
    print("=" * 60)

    # Load knowledge base
    with open(KB_FILE, "r", encoding="utf-8") as f:
        diseases = json.load(f)
    print(f"\n[*] Loaded {len(diseases)} diseases from knowledge base")

    # Generate QA pairs
    all_qa = []
    type_stats: Dict[str, int] = {}

    for d in diseases:
        qa_list = generate_all(d)
        for qa in qa_list:
            all_qa.append(qa)
            t = qa["metadata"]["type"]
            type_stats[t] = type_stats.get(t, 0) + 1

    print(f"\n[*] Generated {len(all_qa)} QA pairs")

    # Stats
    print(f"\n[*] By template type:")
    for t, cnt in sorted(type_stats.items(), key=lambda x: -x[1]):
        print(f"   {t}: {cnt}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved {len(all_qa)} QA pairs -> {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
