#!/usr/bin/env python3
"""
S5: Safety & Format QA Generator
Generate general-purpose safety and format QA pairs to prevent catastrophic
forgetting and ensure the model follows proper response guidelines.

These are generated using templates + a small amount of API calls for variety.
Cost estimate: ~500-1000 entries, ~¥10 API cost.
"""

import os
import json
import time
import random
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# ==================== Config ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2eedab5b21954b6bb26f7461706642f2")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

BASE_DIR = Path(r"D:\Backup\PythonProject2\data_process")
OUTPUT_DIR = BASE_DIR / "s5_safety_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Hardcoded Safety QA Templates ====================
# These are generated without API calls

SAFETY_QA_TEMPLATES: List[Dict] = [
    # --- Disclaimer templates ---
    {
        "instruction": "你是什么？",
        "output": "我是一个宠物健康知识助手，可以回答关于犬只养护、疾病预防、行为训练等方面的问题。请注意，我的回答仅供参考，不能替代专业兽医的诊断和治疗。如有紧急情况，请立即就医。",
        "metadata": {"category": "safety", "type": "self_intro"}
    },
    {
        "instruction": "你能不能帮我开药？",
        "output": "我不能开药或提供处方。用药方案需要由执业兽医根据犬只的具体情况（体重、年龄、既往病史、过敏史等）来决定。请带犬只就医，由兽医开具合适的处方。",
        "metadata": {"category": "safety", "type": "no_prescription"}
    },
    {
        "instruction": "我的狗吃了毒药，快死了怎么办？",
        "output": "请立即拨打当地宠物急救电话，或前往最近的24小时宠物医院。同时可以尝试催吐（但某些情况绝对禁止催吐），保留食物或毒药包装以便兽医诊断。切勿自行用药。",
        "metadata": {"category": "safety", "type": "emergency"}
    },
    {
        "instruction": "狗不吃东西，可以用人用药吗？",
        "output": "绝对不可以。犬只的生理构造与人类不同，许多人用药对犬只有毒（如布洛芬、对乙酰氨基酚等）。请带犬只就医，让兽医选择安全有效的动物专用药物。",
        "metadata": {"category": "safety", "type": "no_human_meds"}
    },
    {
        "instruction": "网上买的疫苗可以自己给狗打吗？",
        "output": "强烈不建议自行网购疫苗并给犬只接种。疫苗的运输和储存需要严格的冷链管理，网购疫苗来源不明，质量无法保证。正确做法是带犬只到正规宠物医院，由兽医接种。",
        "metadata": {"category": "safety", "type": "no_self_vaccine"}
    },
    {
        "instruction": "狗感冒了，可以喂999感冒灵吗？",
        "output": "不可以。999感冒灵等人类感冒药含有对犬只有毒的成分（如对乙酰氨基酚）。犬只感冒应就医，由兽医开具动物专用药物。",
        "metadata": {"category": "safety", "type": "no_human_meds"}
    },
    {
        "instruction": "狗呕吐了，要禁食多久？",
        "output": "一般建议禁食6-12小时观察，但具体时长取决于犬只年龄、健康状况及呕吐原因。建议先咨询兽医，根据专业建议进行护理。如果持续呕吐、伴有血或精神萎靡，应立即就医。",
        "metadata": {"category": "safety", "type": "vet_consult"}
    },
    {
        "instruction": "我的狗需要做手术吗？",
        "output": "是否需要手术需要由兽医根据犬只的具体病情、体检结果和影像学检查来综合判断。我可以介绍常见手术的一般信息，但最终治疗方案请遵医嘱。",
        "metadata": {"category": "safety", "type": "vet_consult"}
    },
    {
        "instruction": "狗体外驱虫用什么药好？",
        "output": "常用体外驱虫药包括尼可信、超可信、拜宠爽、大宠爱等品牌。选择时应考虑犬只体重、年龄及生活环境。不同药物针对的寄生虫种类不同，建议咨询兽医选择最适合的产品。",
        "metadata": {"category": "safety", "type": "deworming"}
    },
    {
        "instruction": "狗打疫苗后精神不好，正常吗？",
        "output": "疫苗接种后可能出现轻微精神沉郁、食欲减退、注射部位轻微肿胀等正常反应，通常1-2天内自行恢复。但如果出现高热、呕吐、呼吸困难、面部肿胀等严重反应，应立即就医。",
        "metadata": {"category": "safety", "type": "vaccine_reaction"}
    },
    {
        "instruction": "刚买回来的狗可以打疫苗吗？",
        "output": "建议新犬到家后先观察7-14天，确认健康状况良好、无异常后再进行免疫。其间不要给犬只洗澡或带出门，减少应激。驱虫和疫苗的顺序应遵医嘱。",
        "metadata": {"category": "safety", "type": "vaccine_timing"}
    },
    {
        "instruction": "母狗发情期能洗澡吗？",
        "output": "发情期间可以洗澡，但要注意水温适中，避免着凉。如果母狗正在流血，建议使用温水清洁外阴部即可，不要用刺激性清洁剂。如有异常分泌物的犬只应就医。",
        "metadata": {"category": "safety", "type": "care_during_heat"}
    },
    {
        "instruction": "狗可以吃人的驱虫药吗？",
        "output": "不可以。人和犬只的寄生虫种类和用药剂量完全不同，错误用药可能导致中毒或治疗无效。请使用正规宠物驱虫药，具体用药方案请咨询兽医。",
        "metadata": {"category": "safety", "type": "no_human_meds"}
    },
    {
        "instruction": "狗吃了巧克力怎么处理？",
        "output": "巧克力中的可可碱对犬只有毒。发现误食后应立即联系兽医，告知摄入量和犬只体重。兽医可能会进行催吐或活性炭处理。症状严重时需要住院支持治疗。",
        "metadata": {"category": "safety", "type": "emergency_poisoning"}
    },
    {
        "instruction": "狗骨折了怎么急救？",
        "output": "发现犬只骨折时，不要自行移动或复位骨折部位。用硬板（如木板、杂志）轻轻固定伤处，避免过度用力。尽快送医，期间保持犬只安静，减少挣扎。",
        "metadata": {"category": "safety", "type": "emergency_injury"}
    },
    {
        "instruction": "狗发烧了怎么办？",
        "output": "犬只正常体温为38-39.2摄氏度，超过39.5摄氏度为发热。不要自行给犬只服用退烧药（人类退烧药对犬只有毒），应带犬只就医查找发热原因，对症治疗。",
        "metadata": {"category": "safety", "type": "fever"}
    },
    {
        "instruction": "狗皮肤溃烂怎么处理？",
        "output": "发现犬只皮肤溃烂时，不要自行用药或包扎，以免加重感染。应带犬只就医，兽医会进行清创、涂片检查病原体，并开具适当的抗生素或外用药。",
        "metadata": {"category": "safety", "type": "skin_problem"}
    },
    {
        "instruction": "狗眼睛发红睁不开是怎么回事？",
        "output": "犬只眼睛发红可能由外伤、感染、过敏、青光眼等多种原因引起。应尽快就医，避免犬只抓挠眼睛（可佩戴防护圈）。不要自行使用人用眼药水。",
        "metadata": {"category": "safety", "type": "eye_problem"}
    },
    {
        "instruction": "幼犬多大可以出门遛？",
        "output": "幼犬通常在完成基础免疫疫苗（约16周龄）后才能安全出门遛弯。在此之前可以在家中活动，或在安全的室内环境进行社会化训练。出门时要避免接触未知犬只和污染环境。",
        "metadata": {"category": "safety", "type": "puppy_care"}
    },
    {
        "instruction": "狗有跳蚤怎么办？",
        "output": "发现跳蚤后应使用正规宠物驱虫药（如尼可信、大宠爱等）进行治疗，并同时进行环境清洁（吸尘、清洗犬只用品）。跳蚤可能传播绦虫，如发现犬只有蹭屁股行为应同时驱虫。",
        "metadata": {"category": "safety", "type": "parasite"}
    },
    {
        "instruction": "狗抽搐是怎么回事？",
        "output": "犬只抽搐可能由中毒、脑部疾病、低血糖、癫痫等多种原因引起。发作时应保持环境安静安全，不要把手伸入犬只口中。待抽搐停止后立即就医，进行全面检查。",
        "metadata": {"category": "safety", "type": "neurological"}
    },
    {
        "instruction": "老年犬要注意什么？",
        "output": "老年犬（通常7岁以上）应增加体检频率（每半年一次），关注关节、牙齿、心脏、肾脏等器官的健康。建议喂食老年犬专用粮，控制体重，适量运动，保持口腔卫生。",
        "metadata": {"category": "safety", "type": "senior_care"}
    },
    {
        "instruction": "狗绝育好不好？",
        "output": "绝育可以预防子宫蓄脓、乳腺肿瘤、前列腺疾病等，降低某些癌症风险，也能避免无序繁殖。但手术有风险，且可能增加肥胖和某些疾病的概率。建议与兽医讨论，根据犬只品种、年龄和健康状况做决定。",
        "metadata": {"category": "safety", "type": "sterilization"}
    },
    {
        "instruction": "狗吃多了拉稀怎么办？",
        "output": "轻度消化不良可先禁食6-12小时，提供清洁饮水，之后喂少量易消化食物（如煮鸡胸肉+米饭）。如拉稀持续超过24小时、伴有血便或精神萎靡，应就医。",
        "metadata": {"category": "safety", "type": "digestive"}
    },
    {
        "instruction": "可以给狗喝牛奶吗？",
        "output": "大多数成年犬有乳糖不耐症，牛奶可能导致腹泻。如想补充营养可选择羊奶或宠物专用奶。幼犬应使用宠物专用奶粉，不要用牛奶替代。",
        "metadata": {"category": "safety", "type": "diet"}
    },
    {
        "instruction": "狗可以吃葡萄吗？",
        "output": "绝对不可以。葡萄和葡萄干对犬只有毒，即使少量也可能导致急性肾衰竭。如发现误食，应立即就医。日常请将葡萄制品放在犬只无法触及的地方。",
        "metadata": {"category": "safety", "type": "food_poisoning"}
    },
    {
        "instruction": "狗晕车怎么办？",
        "output": "幼犬可提前进行适应性训练。出发前4小时禁食，途中保持通风，可在兽医指导下使用晕车药。不要在车内给犬只喂食，保持平稳驾驶。",
        "metadata": {"category": "safety", "type": "motion_sickness"}
    },
    {
        "instruction": "被陌生的狗咬了怎么办？",
        "output": "被陌生犬只咬伤后，立即用肥皂水清洗伤口至少15分钟，碘伏消毒，然后尽快就医。根据伤口情况可能需要缝合、打破伤风针和狂犬疫苗。不要轻视犬咬伤，即使伤口看起来不严重。",
        "metadata": {"category": "safety", "type": "bite_injury"}
    },
    {
        "instruction": "狗流鼻血是什么原因？",
        "output": "犬只流鼻血可能由外伤、鼻部感染、凝血功能障碍、肿瘤等多种原因引起。如为单侧少量鼻血且很快停止，可观察；如为双侧、反复或大量出血，应立即就医排查病因。",
        "metadata": {"category": "safety", "type": "nose_bleeding"}
    },
    {
        "instruction": "狗牙结石怎么清理？",
        "output": "轻微牙结石可使用犬用牙刷和牙膏在家清洁。如结石严重，需在宠物医院进行麻醉超声波洁牙。日常应坚持刷牙，使用洁齿骨等辅助清洁产品，预防牙结石生成。",
        "metadata": {"category": "safety", "type": "dental_care"}
    },
]

# ==================== API-based Variety QA ====================

def call_api(prompt: str) -> Optional[str]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 800,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(DEEPSEEK_API_URL, data=payload)
        req.add_header("Authorization", f"Bearer {DEEPSEEK_API_KEY}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception:
        pass
    return None


def generate_variety_qa(seed_topic: str) -> Optional[Dict]:
    """Generate a variety QA for a given topic using API."""
    prompt = f"""请为犬只医疗健康领域生成一个问答对，主题是"{seed_topic}"。
要求：
- 问题符合真实宠物主人的提问习惯，口语化
- 回答专业、实用，包含必要的免责声明（"请咨询兽医"等）
- 如果涉及禁忌行为（如自行用药），回答中必须强调不可行并说明原因

以JSON格式返回：
{{"instruction": "问题", "output": "回答", "metadata": {{"category": "safety", "type": "{seed_topic}"}}}}"""
    result = call_api(prompt)
    if not result:
        return None
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        data = json.loads(result)
        if isinstance(data, dict) and "instruction" in data and "output" in data:
            return data
    except Exception:
        pass
    return None


# ==================== Format QA Templates ====================

FORMAT_QA: List[Dict] = [
    {
        "instruction": "请问贵宾犬和泰迪是什么关系？",
        "output": "贵宾犬和泰迪其实是同一犬种。\"泰迪\"是指贵宾犬的一种美容造型（将毛发修剪成泰迪熊的样子）。由于这个造型非常受欢迎，很多人误以为泰迪是另一个品种。",
        "metadata": {"category": "format", "type": "terminology"}
    },
    {
        "instruction": "犬瘟和狗瘟是同一种病吗？",
        "output": "是的，犬瘟（犬瘟热）和狗瘟是同一种疾病的不同叫法。犬瘟热是正式学名，由犬瘟热病毒（CDV）引起，是一种高传染性的病毒性疾病。",
        "metadata": {"category": "format", "type": "terminology"}
    },
    {
        "instruction": "狂犬疫苗每年都要打吗？",
        "output": "在中国，狂犬疫苗通常要求每年接种一次（法律强制）。但具体方案也需结合疫苗品牌和当地法规。在一些国家，部分狂犬疫苗已改为三年一针。建议咨询当地兽医了解具体规定。",
        "metadata": {"category": "format", "type": "vaccine_schedule"}
    },
    {
        "instruction": "小狗和大狗吃的狗粮一样吗？",
        "output": "不一样。幼犬（通常1岁以下）应喂幼犬专用粮，其热量和营养配比更适合生长发育。成年犬喂成犬粮。老年犬（通常7岁以上）建议喂老年犬专用粮。不同生命阶段对营养的需求不同。",
        "metadata": {"category": "format", "type": "nutrition"}
    },
    {
        "instruction": "什么是处方粮？和普通狗粮有什么区别？",
        "output": "处方粮是专门为患有特定疾病（如肾病、心脏病、肥胖等）的犬只设计的治疗性食品，其营养成分经过特殊配比以辅助疾病管理。需要由兽医诊断后推荐使用，不能作为正常犬只的主食。",
        "metadata": {"category": "format", "type": "prescription_food"}
    },
    {
        "instruction": "犬瘟热治愈率高吗？",
        "output": "犬瘟热的治愈率取决于发现时机和犬只体质。早期发现并积极治疗，治愈率约为50-80%。但出现神经系统症状后，治愈率显著降低，部分犬只即使存活也会留有后遗症。",
        "metadata": {"category": "format", "type": "disease_prognosis"}
    },
]


# ==================== Main ====================

def main():
    print("=" * 60)
    print("S5: Safety & Format QA Generator")
    print("=" * 60)

    all_qa: List[Dict] = []

    # 1. Add hardcoded safety QA
    print(f"\n[*] Hardcoded safety QA: {len(SAFETY_QA_TEMPLATES)}")
    all_qa.extend(SAFETY_QA_TEMPLATES)

    # 2. Add hardcoded format QA
    print(f"[*] Hardcoded format QA: {len(FORMAT_QA)}")
    all_qa.extend(FORMAT_QA)

    # 3. Generate variety QA via API
    variety_topics = [
        "疫苗接种注意事项", "寄生虫防治", "皮肤过敏处理", "消化道问题",
        "呼吸系统疾病", "骨骼关节问题", "牙齿护理", "行为异常识别",
        "中毒急救", "术后护理", "老年犬养护", "孕期犬护理",
        "幼犬喂养", "季节性养护", "室内安全", "外出安全",
        "多宠家庭注意事项", "特殊品种养护", "体重管理", "营养补充",
    ]

    print(f"\n[*] Generating {len(variety_topics)} variety QAs via API...")
    api_qa_count = 0
    for topic in variety_topics:
        qa = generate_variety_qa(topic)
        if qa:
            all_qa.append(qa)
            api_qa_count += 1
        else:
            print(f"    [!] Failed: {topic}")
        time.sleep(0.5)  # Rate limit

    print(f"[*] API variety QA: {api_qa_count}")

    # Stats
    categories = defaultdict(int)
    for qa in all_qa:
        categories[qa.get("metadata", {}).get("category", "unknown")] += 1

    print(f"\n[*] Total QA pairs: {len(all_qa)}")
    print(f"[*] By category:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {cnt}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "s5_safety_qa.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved {len(all_qa)} safety/format QA pairs -> {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
