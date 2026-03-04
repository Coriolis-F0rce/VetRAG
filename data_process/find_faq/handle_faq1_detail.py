import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests

# ================ 配置部分 ================
DEEPSEEK_API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


# 安全提示：实际项目中建议将API密钥存储在环境变量中
# import os
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ================ 数据结构定义 ================
@dataclass
class TabooTopic:
    id: int
    name: str
    category: str


@dataclass
class ChainTopic:
    id: int
    name: str
    rounds: List[str]  # 每轮的主题描述


# ================ 数据加载 ================
def load_topics_from_file(file_path: str) -> tuple[List[TabooTopic], List[ChainTopic]]:
    """从文件加载主题数据"""
    taboo_topics = []
    chain_topics = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    current_category = None
    taboo_id = 1

    # 解析禁忌主题
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('🔒'):
            continue
        elif line.startswith('饮食安全') or line.startswith('用药安全') or line.startswith('急救误区') or \
                line.startswith('日常护理') or line.startswith('行为与环境') or line.startswith('医疗干预'):
            current_category = line
        elif line.startswith('🔗'):
            break  # 开始解析追问链部分
        elif line and not line.startswith('→') and current_category:
            # 移除序号
            topic_name = line.split('. ')[-1] if '. ' in line else line
            taboo_topics.append(TabooTopic(id=taboo_id, name=topic_name, category=current_category))
            taboo_id += 1

    # 解析追问链主题
    parsing_chains = False
    chain_id = 1

    for line in lines:
        line = line.strip()
        if line.startswith('🔗'):
            parsing_chains = True
            continue

        if parsing_chains and line:
            if '→' not in line:  # 主主题
                chain_name = line
                chain_rounds = []
            elif '→' in line:  # 子主题
                rounds = [r.strip() for r in line.split('→')[1:]]
                chain_rounds.extend(rounds)
                if len(rounds) >= 2:  # 至少有2轮
                    chain_topics.append(ChainTopic(id=chain_id, name=chain_name, rounds=chain_rounds))
                    chain_id += 1

    return taboo_topics, chain_topics


# ================ API调用核心 ================
class DeepSeekFAQGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_single_taboo(self, topic: str) -> Dict:
        """生成单个禁忌主题的FAQ"""
        prompt = self._build_single_taboo_prompt(topic)
        response = self._call_api(prompt)

        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # 如果返回的不是纯JSON，尝试提取JSON部分
            print(f"JSON解析错误，原始响应: {response[:200]}...")
            return {"instruction": f"关于{topic}的提问", "input": "", "output": response}

    def generate_chain_round(self, chain_id: int, round_num: int,
                             current_topic: str, context: str = "") -> Dict:
        """生成追问链的一轮"""
        prompt = self._build_chain_round_prompt(chain_id, round_num, current_topic, context)
        response = self._call_api(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"JSON解析错误，原始响应: {response[:200]}...")
            return {"instruction": f"（追问链{chain_id}-{round_num}）{current_topic}",
                    "input": "", "output": response}

    def _build_single_taboo_prompt(self, topic: str) -> str:
        """构建单禁忌主题的prompt"""
        return f"""# 角色
你是一名持有执业兽医资格证的小动物急诊与预防医学专家，专注犬类临床指南与主人教育。

# 任务
根据参数生成犬医RAG专用FAQ条目，内容需符合WSAVA/ACVIM临床共识，语言精准无歧义。

# 参数
[TYPE] = single_taboo
[TOPIC] = {topic}

# 生成规则
## instruction字段
- single_taboo：狗主人真实口吻提问，例："能给狗狗喂巧克力吗？"

## output字段（强制结构）
【结论】首句定性（"绝对禁止"/"严禁"/"不建议"）  
【机制】1句医学原理（含术语：如"N-丙基二硫化物致红细胞氧化损伤"）  
【风险】1句后果（关联品种/体重差异，例"吉娃娃5g洋葱即可致溶血"）  
【行动】分号分隔：①紧急处理（含禁忌）；②就医指征；③预防要点  
→ 全文120-180字，禁用"可能""一般"，用"必须""立即"等确定性措辞

## 格式铁律
- 仅输出纯净JSON：{{"instruction": "...", "input": "", "output": "..."}}
- output内用中文分号/句号分段，禁用Markdown、编号、括号注释
- 专业校验：涉及法律时写"依据《动物防疫法》及地方养犬条例"，不编造法条号

# 示例
[TOPIC]=喂食木糖醇
→ {{"instruction": "狗狗误食木糖醇口香糖需要重视吗？", "input": "", "output": "极度危险！木糖醇致犬胰岛素骤升，15分钟内引发低血糖、急性肝坏死。小型犬致死量仅0.1g/kg。行动：勿催吐；立即送医并携带包装；日常严格收纳无糖食品。"}}

现在请为以下主题生成内容：
[TOPIC]={topic}"""

    def _build_chain_round_prompt(self, chain_id: int, round_num: int,
                                  current_topic: str, context: str) -> str:
        """构建追问链的prompt"""
        context_part = f"[CONTEXT] = {context}" if context else "[CONTEXT] = {}"

        return f"""# 角色
你是一名持有执业兽医资格证的小动物急诊与预防医学专家，专注犬类临床指南与主人教育。

# 任务
根据参数生成犬医RAG专用FAQ条目，内容需符合WSAVA/ACVIM临床共识，语言精准无歧义。

# 参数
[TYPE] = chain_round
[CHAIN_ID] = {chain_id}
[ROUND] = {round_num}
{context_part}

# 生成规则
## instruction字段
- chain_round：严格格式"（追问链{chain_id}-{round_num}）{current_topic}"
狗主人真实口吻提问，例："我的狗狗骨折了，能给他手动复位吗？"

## output字段（强制结构）
【结论】首句定性（"绝对禁止"/"严禁"/"不建议"）  
【机制】1句医学原理（含术语：如"N-丙基二硫化物致红细胞氧化损伤"）  
【风险】1句后果（关联品种/体重差异，例"吉娃娃5g洋葱即可致溶血"）  
【行动】分号分隔：①紧急处理（含禁忌）；②就医指征；③预防要点  
→ 全文120-180字，禁用"可能""一般"，用"必须""立即"等确定性措辞

## 格式铁律
- 仅输出纯净JSON：{{"instruction": "...", "input": "", "output": "..."}}
- output内用中文分号/句号分段，禁用Markdown、编号、括号注释
- 专业校验：涉及法律时写"依据《动物防疫法》及地方养犬条例"，不编造法条号

# 示例
[CHAIN_ID]=3, [ROUND]=2, [CONTEXT]=前轮说明骨折禁止复位
→ {{"instruction": "（追问链3-2）送医前需要给狗狗喂水吗？", "input": "", "output": "禁止喂食喂水。创伤犬需紧急麻醉检查，胃内容物增加误吸风险。行动：保持侧卧清理口鼻；监测牙龈颜色（苍白提示休克）；专注安全转运并告知兽医禁食时间。"}}

现在请生成内容："""

    def _call_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # 低温度确保一致性
            "max_tokens": 500
        }

        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API调用失败: {response.status_code}")
                print(f"响应: {response.text}")
                return '{"error": "API调用失败"}'

        except Exception as e:
            print(f"API调用异常: {e}")
            return '{"error": "连接异常"}'


# ================ 主程序 ================
def main():
    # 1. 加载数据
    print("加载主题数据...")
    taboo_topics, chain_topics = load_topics_from_file("faq1.txt")

    print(f"加载到 {len(taboo_topics)} 个禁忌主题")
    print(f"加载到 {len(chain_topics)} 个追问链")

    # 2. 初始化生成器
    generator = DeepSeekFAQGenerator(DEEPSEEK_API_KEY)

    # 3. 生成单禁忌FAQ
    print("\n开始生成单禁忌FAQ...")
    single_results = []

    for topic in taboo_topics[:5]:  # 先测试前5个
        print(f"生成: {topic.name}")
        result = generator.generate_single_taboo(topic.name)
        result["metadata"] = {
            "id": topic.id,
            "category": topic.category,
            "type": "single_taboo"
        }
        single_results.append(result)
        time.sleep(1)  # 避免速率限制

    # 4. 生成追问链FAQ
    print("\n开始生成追问链FAQ...")
    chain_results = []

    for chain in chain_topics[:2]:  # 先测试前2个追问链
        print(f"\n生成追问链 {chain.id}: {chain.name}")
        context = ""

        for i, round_topic in enumerate(chain.rounds, 1):
            print(f"  第{i}轮: {round_topic}")
            result = generator.generate_chain_round(
                chain_id=chain.id,
                round_num=i,
                current_topic=round_topic,
                context=context
            )
            result["metadata"] = {
                "chain_id": chain.id,
                "round": i,
                "total_rounds": len(chain.rounds),
                "type": "chain_round"
            }
            chain_results.append(result)

            # 更新上下文（使用output的前50字）
            if "output" in result:
                context = result["output"][:50] + "..."

            time.sleep(1)

    # 5. 保存结果
    print("\n保存结果...")

    # 保存单禁忌结果
    with open("single_faqs.json", "w", encoding="utf-8") as f:
        json.dump(single_results, f, ensure_ascii=False, indent=2)

    # 保存追问链结果
    with open("chain_faqs.json", "w", encoding="utf-8") as f:
        json.dump(chain_results, f, ensure_ascii=False, indent=2)

    # 保存合并结果（用于RAG）
    all_results = single_results + chain_results
    with open("all_faqs.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！生成 {len(single_results)} 个单禁忌FAQ，{len(chain_results)} 个追问链条目")
    print("结果已保存到: single_faqs.json, chain_faqs.json, all_faqs.json")



# ================ 批处理增强版 ================
def batch_generate_all():
    """批量生成所有内容（带错误重试）"""
    # 加载数据
    taboo_topics, chain_topics = load_topics_from_file("faq1.txt")
    generator = DeepSeekFAQGenerator(DEEPSEEK_API_KEY)

    all_results = []

    # 生成所有单禁忌主题
    for topic in taboo_topics:
        print(f"生成单禁忌 {topic.id}/40: {topic.name}")

        for attempt in range(3):  # 最多重试3次
            try:
                result = generator.generate_single_taboo(topic.name)
                if "error" not in str(result):
                    result["metadata"] = {
                        "id": topic.id,
                        "category": topic.category,
                        "type": "single_taboo"
                    }
                    all_results.append(result)
                    break
            except Exception as e:
                print(f"  尝试 {attempt + 1} 失败: {e}")
                time.sleep(2)

        time.sleep(0.5)  # 请求间隔

    # 生成所有追问链
    for chain in chain_topics:
        print(f"\n生成追问链 {chain.id}: {chain.name}")
        context = ""

        for i, round_topic in enumerate(chain.rounds, 1):
            print(f"  第{i}轮: {round_topic}")

            for attempt in range(3):
                try:
                    result = generator.generate_chain_round(
                        chain_id=chain.id,
                        round_num=i,
                        current_topic=round_topic,
                        context=context
                    )
                    if "error" not in str(result):
                        result["metadata"] = {
                            "chain_id": chain.id,
                            "round": i,
                            "total_rounds": len(chain.rounds),
                            "type": "chain_round"
                        }
                        all_results.append(result)

                        # 更新上下文
                        if "output" in result:
                            context = result["output"][:100] + "..."
                        break
                except Exception as e:
                    print(f"    尝试 {attempt + 1} 失败: {e}")
                    time.sleep(2)

            time.sleep(0.5)

    # 保存结果
    with open("faq1.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共生成 {len(all_results)} 个条目")
    return all_results


# ================ 运行选项 ================
if __name__ == "__main__":
    batch_generate_all()
