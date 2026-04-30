#!/usr/bin/env python3
"""
S3: Expand New Topics
Convert existing data sources (behaviors, regulations, surgeries) into QA pairs
and generate new topic categories (daily care, breed info) using templates.

S3 is purely rule-based - no API calls needed since all source data is
already structured from previous LLM processing.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any


BASE_DIR = Path(r"D:\Backup\PythonProject2\data_process")

# Output
OUTPUT_DIR = BASE_DIR / "s3_expanded_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Part A: Behavior QA from api_responses.json
# ================================================================

def load_behavior_data() -> List[Dict]:
    """Load and parse behavior API responses."""
    path = BASE_DIR / "find_behaviors" / "api_responses.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results = []
    for item in raw.get("results", []):
        resp_content = item.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        if not resp_content:
            continue
        try:
            # Strip markdown code block
            if resp_content.strip().startswith("{"):
                data = json.loads(resp_content)
            else:
                m = re.search(r"\{[\s\S]+\}", resp_content)
                if m:
                    data = json.loads(m.group())
                else:
                    continue
            bh = data.get("behavior", {})
            if bh:
                results.append(bh)
        except Exception:
            pass
    return results


def behavior_qa(bh: Dict) -> List[Dict]:
    """Convert a behavior entry into QA pairs."""
    qa_list = []
    name = bh.get("name", "")
    desc = bh.get("description", "")
    meaning = bh.get("meaning", "")
    category = bh.get("category", "")
    intervention = bh.get("intervention_level", "")
    antecedents = bh.get("antecedents", "")

    if not name:
        return qa_list

    # Q1: What does this behavior mean?
    if meaning:
        qa_list.append({
            "instruction": f"狗{name}是什么意思？",
            "output": f"狗{name}表示：{meaning}。"
                       f"该行为属于\"{category}\"类别。"
                       f"描述：{desc}",
            "metadata": {"category": "behavior_knowledge", "type": "meaning", "behavior": name}
        })

    # Q2: Is this behavior normal?
    if intervention:
        level_map = {"无/观察": "正常行为，无需干预", "轻度": "轻度异常，建议观察和适当引导",
                     "中度": "存在一定问题，建议咨询行为训练师", "重度": "严重行为问题，需要专业干预"}
        level_text = level_map.get(intervention, intervention)
        qa_list.append({
            "instruction": f"狗{name}正常吗？",
            "output": f"狗{name}属于\"{category}\"。"
                       f"干预建议：{level_text}。"
                       f"该行为通常在以下情境出现：{antecedents or '日常社交互动中'}。",
            "metadata": {"category": "behavior_knowledge", "type": "normal", "behavior": name}
        })

    # Q3: How to handle this behavior?
    if intervention and intervention != "无/观察":
        qa_list.append({
            "instruction": f"狗{name}怎么处理？",
            "output": f"针对{name}，建议的干预等级为：{intervention}。"
                       f"该行为属于\"{category}\"，常见于：{antecedents or '特定情境'}。"
                       f"如有需要，建议咨询专业犬行为训练师。",
            "metadata": {"category": "behavior_knowledge", "type": "handling", "behavior": name}
        })

    return qa_list


# ================================================================
# Part B: Daily Care QA from regulations generated content
# ================================================================

def load_daily_care() -> List[Dict]:
    """Load daily care data from regulations generated content."""
    results = []
    dir_path = BASE_DIR / "find_regulations_and_laws" / "generated_pet_content"

    care_files = [
        "daily_care_20260206_151716.json",
        "joint_care_guide_20260206_151348.json",
        "vaccine_schedule_20260206_151320.json",
    ]

    for fname in care_files:
        fpath = dir_path / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                content_type = data.get("content_type", fname.replace(".json", ""))

                # daily_care: parse care_categories
                if "care_categories" in data:
                    for cat_item in data["care_categories"]:
                        procedures = cat_item.get("procedures", [])
                        tips = cat_item.get("tips", [])
                        content = (f"频率：{cat_item.get('frequency', '请咨询兽医')}。"
                                   f"操作步骤：{'；'.join(procedures)}。"
                                   f"注意事项：{'；'.join(tips)}")
                        results.append({"title": cat_item.get("category", ""), "content": content})

                # joint_care: parse common_joint_problems
                elif "common_joint_problems" in data:
                    for prob in data["common_joint_problems"]:
                        symptoms = prob.get("symptoms", [])
                        risk_factors = prob.get("risk_factors", [])
                        content = (f"问题：{prob.get('problem', '')}。"
                                   f"症状：{'、'.join(symptoms)}。"
                                   f"风险因素：{'、'.join(risk_factors)}")
                        results.append({"title": f"老年犬{prob.get('problem', '')}", "content": content})

                # vaccine_schedule: parse data
                elif "data" in data:
                    for vac in data["data"]:
                        protected = vac.get("disease_protected", [])
                        content = (f"适用年龄：{vac.get('age', '')}。"
                                   f"疫苗名称：{vac.get('vaccine_name', '')}。"
                                   f"预防疾病：{'、'.join(protected)}。"
                                   f"重要程度：{vac.get('importance', '')}。"
                                   f"说明：{vac.get('notes', '')}")
                        results.append({"title": f"{vac.get('age', '')}疫苗", "content": content})

            except Exception:
                pass

    return results


def daily_care_qa(item: Dict) -> List[Dict]:
    """Convert daily care item into QA pairs."""
    qa_list = []
    content = item.get("content", "")
    title = item.get("title", "")

    if not content or len(content) < 20:
        return qa_list

    topic = title.replace(". ", " ").replace("_", " ").strip()
    if not topic:
        topic = "宠物日常养护"

    qa_list.append({
        "instruction": f"{topic}要注意什么？",
        "output": content[:1000],
        "metadata": {"category": "daily_care", "type": "info", "topic": topic}
    })
    qa_list.append({
        "instruction": f"如何进行{topic}？",
        "output": content[:1000],
        "metadata": {"category": "daily_care", "type": "guide", "topic": topic}
    })

    return qa_list


# ================================================================
# Part C: Surgery QA from dog_surgeries.txt
# ================================================================

def load_surgeries() -> List[str]:
    """Load surgery list from text file."""
    path = BASE_DIR / "find_surgeries" / "dog_surgeries.txt"
    surgeries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not re.match(r"^[一二三四五六七八九十]、", line):
                surgeries.append(line)
    return surgeries


def surgery_qa(surgery: str) -> List[Dict]:
    """Generate QA pairs for a surgery."""
    return [
        {
            "instruction": f"狗做{surgery}需要多少钱？",
            "output": f"{surgery}的费用因地区、医院等级、犬只体重及具体情况而异。"
                       f"具体费用建议直接咨询当地宠物医院。术前需做全面体检评估手术风险，"
                       f"术后需遵医嘱进行护理和复诊。",
            "metadata": {"category": "surgery_info", "type": "cost", "surgery": surgery}
        },
        {
            "instruction": f"狗做{surgery}有风险吗？",
            "output": f"任何外科手术都存在一定风险，{surgery}也不例外。"
                       f"具体风险取决于犬只的年龄、健康状况、手术时机等因素。"
                       f"建议选择正规宠物医院，术前充分评估，术后细心护理以降低风险。",
            "metadata": {"category": "surgery_info", "type": "risk", "surgery": surgery}
        },
        {
            "instruction": f"狗做完{surgery}后怎么护理？",
            "output": f"{surgery}术后护理要点包括：遵医嘱使用抗生素和止痛药；"
                       f"佩戴防护圈防止舔舐伤口；限制运动直至伤口愈合；"
                       f"观察伤口是否有红肿、渗液等感染迹象；"
                       f"按期复诊拆线（如有）。具体护理方案请遵医嘱。",
            "metadata": {"category": "surgery_info", "type": "care", "surgery": surgery}
        },
    ]


# ================================================================
# Part D: Breed info QA (templates - no source file needed)
# ================================================================

BREEDS_INFO = [
    ("金毛寻回犬", "金毛", "性格温顺友好，智商高，易训练，是优秀的家庭陪伴犬和工作犬。"),
    ("拉布拉多寻回犬", "拉布拉多", "活泼好动，性格开朗，对人友善，适合作为导盲犬、搜救犬和家庭宠物。"),
    ("德国牧羊犬", "德牧", "聪明勇敢，服从性高，常用于军警犬、护卫犬和工作犬。"),
    ("柯基", "威尔士柯基犬", "腿短身长，性格活泼好动，独立性强，需要适度运动和体重管理。"),
    ("泰迪（贵宾犬）", "泰迪", "聪明机敏，不易掉毛，体味轻，有多种体型可选，适合城市家庭。"),
    ("比熊犬", "比熊", "性格温和友善，毛发卷曲洁白，需定期美容护理，适合作为伴侣犬。"),
    ("边境牧羊犬", "边牧", "智商最高的犬种之一，精力旺盛，需要大量运动和智力刺激，不适合懒人。"),
    ("柴犬", "柴犬", "性格独立倔强，忠诚护主，爱干净，适合有庭院的家庭。"),
    ("哈士奇", "哈士奇", "精力充沛，性格活泼好奇，俗称\"二哈\"，需要大量运动，不耐热。"),
    ("法国斗牛犬", "法斗", "体型小巧，性格安静友好，面部扁平需注意呼吸道和散热问题。"),
    ("吉娃娃", "吉娃娃", "体型最小，勇敢好奇，适合城市公寓饲养，需注意骨骼保护和保暖。"),
    ("博美犬", "博美", "体型小巧，毛发蓬松，活泼好动，叫声清脆，需要定期美容和运动。"),
    ("萨摩耶", "萨摩耶", "性格温和，笑容迷人，被毛厚密需定期护理，精力旺盛需要充足运动。"),
    ("阿拉斯加雪橇犬", "阿拉斯加", "体型巨大，性格温和，精力旺盛，需大量运动和饮食，毛发浓密需定期护理。"),
    ("杜宾犬", "杜宾", "聪明警觉，护卫性强，服从性高，适合作为护卫犬和工作犬。"),
    ("约克夏梗犬", "约克夏", "体型小巧，毛发如丝般光滑，需要精心护理，适合城市公寓生活。"),
    ("西施犬", "西施犬", "性格温顺，毛发长而华丽，需要定期美容，适合作为伴侣犬。"),
    ("巴哥犬", "巴哥", "面部扁平，性格安静友好，但需注意呼吸道和眼部护理，不耐高温。"),
    ("可卡犬", "可卡犬", "性格活泼，嗅觉灵敏，需要定期毛发护理和充足运动。"),
    ("松狮犬", "松狮犬", "性格独立，蓝黑色舌头是特征，毛发浓密，需要定期护理。"),
]

BREED_TOPICS = [
    "喂养要点", "性格特点", "常见疾病", "日常护理", "运动需求",
    "毛发打理", "训练难度", "适合人群", "优缺点分析", "寿命与体重",
]


def breed_qa(breed_full: str, breed_short: str, desc: str) -> List[Dict]:
    """Generate breed-related QA pairs."""
    qa_list = []

    qa_list.append({
        "instruction": f"{breed_short}有什么特点？",
        "output": f"{breed_full}的特点：{desc}。该犬种是深受人们喜爱的犬种之一。",
        "metadata": {"category": "breed_info", "type": "features", "breed": breed_short}
    })

    for topic in BREED_TOPICS:
        topic_prompts = {
            "喂养要点": f"怎么喂养{breed_short}？",
            "性格特点": f"{breed_short}性格怎么样？",
            "常见疾病": f"{breed_short}容易得什么病？",
            "日常护理": f"怎么护理{breed_short}？",
            "运动需求": f"{breed_short}需要多少运动量？",
            "毛发打理": f"怎么打理{breed_short}的毛发？",
            "训练难度": f"{breed_short}容易训练吗？",
            "适合人群": f"什么人适合养{breed_short}？",
            "优缺点分析": f"养{breed_short}有什么优缺点？",
            "寿命与体重": f"{breed_short}的寿命和体重是多少？",
        }
        if topic in topic_prompts:
            qa_list.append({
                "instruction": topic_prompts[topic],
                "output": f"关于{breed_short}的{topic}：{desc}。"
                           f"建议在养犬前充分了解该犬种的特性，"
                           f"并咨询兽医或有经验的养犬人士获取更具体的建议。",
                "metadata": {"category": "breed_info", "type": topic, "breed": breed_short}
            })

    return qa_list


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 60)
    print("S3: Expand New Topics")
    print("=" * 60)

    all_qa: List[Dict] = []
    stats: Dict[str, int] = {}

    def add(qa_list: List[Dict], source: str):
        for qa in qa_list:
            all_qa.append(qa)
            cat = qa.get("metadata", {}).get("category", "unknown")
            stats[cat] = stats.get(cat, 0) + 1

    # A: Behaviors
    print("\n[*] Processing behaviors...")
    behaviors = load_behavior_data()
    print(f"    Loaded {len(behaviors)} behavior entries")
    for bh in behaviors:
        add(behavior_qa(bh), "behavior")
    print(f"    Generated behavior QA pairs")

    # B: Daily care
    print("\n[*] Processing daily care data...")
    care_items = load_daily_care()
    print(f"    Loaded {len(care_items)} daily care items")
    for item in care_items:
        add(daily_care_qa(item), "daily_care")
    print(f"    Generated daily care QA pairs")

    # C: Surgeries
    print("\n[*] Processing surgeries...")
    surgeries = load_surgeries()
    print(f"    Loaded {len(surgeries)} surgery entries")
    for s in surgeries:
        add(surgery_qa(s), "surgery")
    print(f"    Generated surgery QA pairs")

    # D: Breeds
    print("\n[*] Generating breed QA pairs...")
    for breed_full, breed_short, desc in BREEDS_INFO:
        add(breed_qa(breed_full, breed_short, desc), "breed")
    print(f"    Generated breed QA for {len(BREEDS_INFO)} breeds")

    # Stats
    print(f"\n[*] Total QA pairs generated: {len(all_qa)}")
    print(f"\n[*] By category:")
    for cat, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {cnt}")

    # Save
    out_path = OUTPUT_DIR / "s3_expanded_all.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved {len(all_qa)} QA pairs -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
