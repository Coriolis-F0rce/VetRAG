#!/usr/bin/env python3
"""
犬急诊急救FAQ生成器
用于处理紧急场景下的快速、准确医学指导
语气：专业、严肃、紧迫，以保障犬只生命安全为首要目标
"""

import json
import re
import time
import requests
from typing import List, Dict, Optional
from tqdm import tqdm

# ================ 配置区 ================
DEEPSEEK_API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"  # 请替换或使用环境变量
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
OUTPUT_FILE = "faq5.json"

# ================ 紧急问题列表 ================
# 直接从用户输入中提取（已手动整理为列表）
EMERGENCY_QUESTIONS = [
    "误食巧克力应如何紧急处理？",
    "犬中暑的急救措施有哪些？",
    "被车撞伤后现场该做什么？",
    "犬突然抽搐发作时如何应对？",
    "误食老鼠药的第一步是什么？",
    "严重外伤出血怎样有效止血？",
    "呼吸困难、舌头发紫怎么办？",
    "误食葡萄或葡萄干需催吐吗？",
    "异物卡喉窒息如何实施海姆立克法？",
    "腿部骨折如何就地固定？",
    "眼部受伤或异物入眼如何冲洗？",
    "触电后昏迷无呼吸怎样急救？",
    "溺水后怎样排出呼吸道积水？",
    "蜂蜇伤后出现面部肿胀如何处理？",
    "被毒蛇咬伤能否切开吸出毒液？",
    "烫伤、烧伤第一时间冲水多久？",
    "频繁呕吐且无法进食该禁食吗？",
    "严重腹泻导致虚弱脱水怎么补水？",
    "超过两天未排便、痛苦呻吟怎么办？",
    "腹部急速膨胀、干呕怀疑胃扭转怎么送医？",
    "尿闭超过12小时、频繁蹲厕无尿怎么办？",
    "母犬分娩时努责1小时仍无胎儿出生如何处理？",
    "产后母犬突然抽搐、步态僵硬是低血钙吗？",
    "癫痫持续发作超过5分钟如何用药急救？",
    "意识丧失、无呼吸心跳是否实施心肺复苏？",
    "从高处跳下或坠落需警惕哪些内脏损伤？",
    "皮肤撕裂伤、可见肌肉应冲洗还是包扎？",
    "吞食尖锐异物（针、骨片）能喂食促排吗？",
    "误吞纽扣电池需立即催吐吗？",
    "误食木糖醇口香糖后血糖骤降如何补糖？",
    "洋葱中毒导致溶血、尿血如何家庭干预？",
    "大麻中毒共济失调、瞳孔放大需镇静吗？",
    "酒精中毒昏迷怎么防止误吸？",
    "对乙酰氨基酚中毒高铁血红蛋白血症可用什么解毒剂？",
    "误食防冻液乙二醇中毒的黄金解毒时间窗？",
    "冻伤、体温过低如何缓慢复温？",
    "突发全身荨麻疹、面部眼睑肿胀首选药物？",
    "过敏性休克倒地、牙龈苍白肾上腺素如何使用？",
    "被大型犬咬伤穿透腹腔应湿敷还是加压？",
    "头部撞击后昏迷、瞳孔不等大提示颅压升高如何搬运？",
    "眼睑被撕裂、眼球完整应怎样保护创面？",
    "角膜划伤、犬频繁眯眼能否用人类眼药水？",
    "急性青光眼眼球坚硬如石、疼痛嚎叫需立即降眼压吗？",
    "耳廓血肿快速膨大能否自行穿刺抽吸？",
    "鼻出血不止、凝血障碍怎样压迫止血？",
    "咳出鲜血、泡沫样痰需保持安静还是头低脚高？",
    "呕吐物含鲜血或咖啡渣样物应禁食禁水吗？",
    "排柏油样黑便、同时牙龈苍白提示消化道大出血如何处置？",
    "肉眼血尿伴排尿痛苦是否可热敷膀胱？",
    "黄疸短期内急剧加深、精神萎靡需警惕急性肝衰竭？",
    "腹胀如鼓、叩诊鼓音、呼吸急促怀疑胃扩张能插管排气吗？",
    "剧烈疼痛、哀嚎、不让触碰腹部可能原因及临时止痛禁忌？",
    "后肢突然瘫痪、本体反射消失疑似椎间盘突出怎么保定？",
    "短暂晕厥后意识恢复需排查心源性还是神经源性？",
    "夜间突发端坐呼吸、湿性咳嗽怀疑左心衰能否给予利尿剂？",
    "哮喘发作、呼气困难可参考人类吸入药物吗？",
    "肺水肿、口鼻涌出粉红色泡沫液如何体位引流？",
    "被蜱虫叮咬后突发后肢无力、共济失调是蜱瘫吗？",
    "进食变质肉类后出现吞咽困难、流涎、进行性麻痹疑似肉毒杆菌中毒？",
    "完全性肠梗阻、已停止排便排气能否灌油？",
    "误食锌币后数小时呕吐、精神沉郁需立刻摄片吗？",
    "铁中毒（误服补铁剂）出现呕血、黑便的特效拮抗剂？",
    "香烟尼古丁中毒初期兴奋后抑制，需洗胃还是活性炭？",
    "咖啡渣摄入后心动过速、躁动不安的镇静选择？",
    "误食发霉食物后震颤、癫痫发作疑似青霉毒素中毒？",
    "生面团在胃内发酵膨胀导致腹痛、醉酒状态如何催吐？",
    "误食含木糖醇的花生酱后30分钟内是否必须送医？",
    "蛇咬伤头部导致喉头水肿窒息需紧急气管切开吗？",
    "棕隐士蜘蛛咬伤致皮肤坏死、全身溶血的局部处理？",
    "蝎子蛰伤后局部剧痛、流涎能否冰敷并就医？",
    "犬舔食蟾蜍后大量流涎、口吐泡沫只需冲洗口腔吗？",
    "铅中毒（啃咬旧漆）贫血、神经症状的驱铅药物？",
    "锌中毒（吞食金属螺母）溶血危象需输血指征？",
    "有机磷农药中毒流涎、肌颤、瞳孔缩小的解毒顺序？",
    "拟除虫菊酯中毒持续全身震颤可用肌松剂吗？",
    "被猫抓伤深部穿刺伤需考虑巴斯德菌感染如何清创？",
    "草籽进入耳道引起剧烈甩头、耳血肿怎样安全取出？",
    "爪垫玻璃割伤、动脉喷射状出血的近心端加压点？",
    "趾甲完全撕脱、甲床暴露如何消毒包扎？",
    "肛门腺破裂流脓、疼痛排便困难能否在家冲洗？",
    "直肠脱出呈圆柱状、黏膜充血水肿怎样还纳？",
    "子宫脱出体外、黏膜干燥应急送医前如何保护？",
    "阴茎脱出嵌顿、龟头充血无法回纳需冷敷吗？",
    "阴囊外伤血肿迅速增大、触痛明显提示睾丸破裂？",
    "灰尘、沙砾进入眼内能否用生理盐水冲洗出？",
    "误服布洛芬等NSAIDs出现呕吐、黑便的胃黏膜保护措施？",
    "误服抗抑郁药（5-羟色胺再摄取抑制剂）出现血清素综合征表现？",
    "误服降压药（β阻滞剂）心动过缓、低血压可否家用阿托品？",
    "误服维生素D3灭鼠剂高钙血症危象的急救原则？",
    "误服减肥药（麻黄碱）兴奋、高热、高血压如何物理降温？"
]

# ================ API调用类 ================
class EmergencyFAQGenerator:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _call_api(self, prompt: str, max_tokens: int = 1500) -> str:
        """带重试机制的API调用"""
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,          # 低温度保证输出稳定、严谨
                    "top_p": 0.9,
                    "max_tokens": max_tokens
                }
                resp = requests.post(
                    DEEPSEEK_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=90
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                else:
                    time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
        return "【错误】多次尝试后仍无法获取回答，请检查网络或API密钥。"

    def _build_prompt(self, question: str) -> str:
        """构建急诊急救专用Prompt——专业、严肃、紧迫"""
        return f"""你是一名持有执业兽医资格证的小动物急诊与重症医学专家，有10年急诊临床经验。
现在宠物主人遇到了紧急情况，向你咨询。请你以专业、冷静、果断的语气，直接给出可立即执行的急救指导。

问题：{question}

你必须严格遵守以下要求：

1. **内容结构**（用中文自然段落，不要使用Markdown标记）：
   - 第一句：用【紧急度】开头，标明“立即就医”/“紧急处理”/“警惕观察”等。
   - 随后分步骤说明：①现场处理、②就医准备、③途中监护、④绝对禁忌。
   - 如果涉及药物，必须明确给出**禁止联合使用**的食物或药物（例如：头孢类与酒精）。
   - 结尾以“⚠️ 免责声明”结束，内容为：“本建议仅为急诊指导，不能替代兽医当面诊疗。若宠物情况持续恶化，请立即前往动物医院。”

2. **语言风格**：
   - 使用短句，信息密度高，不加任何表情符号。
   - 禁止使用“可能”“也许”“大概”等模糊词汇；必须使用“必须”“立即”“严禁”等确定性措辞。
   - 涉及剂量时，如果无法给出精确值，写“需由兽医根据体重计算”，不得自行编造剂量。

3. **专业底线**：
   - 所有信息必须符合WSAVA、ACVIM等国际临床共识，不得提供已被循证医学否定的错误急救方法（如酒精擦身降温、催吐双氧水等）。
   - 涉及毒物时，尽量给出解毒剂名称及使用前提。

请直接输出急救指导，不要输出JSON，不要额外解释。
"""

    def generate(self, question: str) -> Dict:
        """生成单个急诊FAQ条目"""
        prompt = self._build_prompt(question)
        answer = self._call_api(prompt, max_tokens=1500)
        return {
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "type": "emergency_faq",
                "category": "急诊急救",
                "question": question
            }
        }

# ================ 主流程 ================
def main(limit: Optional[int] = None):
    """
    limit: 限制生成数量，用于快速测试
    """
    print("🚑 犬急诊急救FAQ生成器（专业严肃版）")
    print(f"📋 待生成问题总数：{len(EMERGENCY_QUESTIONS) if limit is None else limit}")

    questions = EMERGENCY_QUESTIONS[:limit] if limit else EMERGENCY_QUESTIONS
    generator = EmergencyFAQGenerator(DEEPSEEK_API_KEY)
    all_results = []

    print("\n📌 开始生成急诊答案……")
    for q in tqdm(questions, desc="急诊FAQ"):
        result = generator.generate(q)
        all_results.append(result)
        time.sleep(0.5)  # 控制请求频率，避免限流

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 生成完成！共 {len(all_results)} 个条目")
    print(f"💾 已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        print("🧪 测试模式：仅生成前3个问题")
        main(limit=3)
    elif "--full" in sys.argv:
        print("🚨 全量生成模式：将生成全部急救FAQ条目")
        confirm = input("确认生成全部 {} 个问题？(y/n): ".format(len(EMERGENCY_QUESTIONS)))
        if confirm.lower() == 'y':
            main()
        else:
            print("已取消")
    else:
        print("请指定运行模式：")
        print("  python emergency_faq.py --test   # 快速测试（仅3条）")
        print("  python emergency_faq.py --full   # 全量生成（全部问题）")
        print("\n默认使用测试模式（--test）")
        main(limit=3)