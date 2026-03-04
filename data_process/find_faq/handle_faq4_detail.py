#!/usr/bin/env python3
"""
预防保健篇 · 犬医FAQ生成脚本
用于从 faq4.txt 生成 180 个单问答 + 20 条追问链（共约250个条目）
调用 DeepSeek API，风格专业、亲切、实用
"""

import json
import re
import time
import requests
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# ================ 配置区 ================
DEEPSEEK_API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"  # 请替换或使用环境变量
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
OUTPUT_FILE = "faq4.json"


# ================ 解析 faq4.txt ================
def parse_preventive_faq(text: str) -> Tuple[List[str], List[Dict]]:
    """
    返回:
        single_topics: 180个单问答标题，如 "幼犬首次疫苗接种计划"
        chain_list: 列表，每个元素为 {"name": 链名, "rounds": [轮次问题]}
    """
    lines = text.strip().split('\n')
    single_topics = []
    chain_list = []

    # 状态
    in_single = True
    in_chains = False
    current_chain = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 切换到追问链部分
        if line.startswith("一、20个预防养护追问链设计"):
            in_single = False
            in_chains = True
            continue

        # 解析单问答
        if in_single:
            # 匹配 "1. 幼犬首次疫苗接种计划" 这样的行
            match = re.match(r'^\d+\.\s+(.+)', line)
            if match:
                topic = match.group(1).strip()
                single_topics.append(topic)

        # 解析追问链
        if in_chains:
            # 链名称，如 "1. 幼犬首次疫苗接种计划"
            if re.match(r'^\d+\.\s+.+', line) and "第" not in line and "轮" not in line:
                if current_chain:
                    chain_list.append(current_chain)
                chain_name = re.sub(r'^\d+\.\s+', '', line).strip()
                current_chain = {"name": chain_name, "rounds": []}
            # 轮次问题，如 "第1轮：幼犬第一年需要接种哪些核心疫苗？"
            elif "第" in line and "轮：" in line and current_chain:
                round_q = line.split("：", 1)[1].strip()
                current_chain["rounds"].append(round_q)

    # 添加最后一个链
    if current_chain and current_chain not in chain_list:
        chain_list.append(current_chain)

    return single_topics, chain_list


# ================ 标题转口语提问 ================
def topic_to_instruction(topic: str) -> str:
    """
    将标题（如"幼犬首次疫苗接种计划"）转化为主人真实口吻的问题
    """
    # 常见模式映射
    mapping = [
        (r'^幼犬(.*)', r'我家幼犬\1，该怎么操作？'),
        (r'^成年犬(.*)', r'我家成年犬\1，需要注意什么？'),
        (r'^(.*)必要性评估$', r'有必要给狗狗做\1吗？'),
        (r'^(.*)适用情况$', r'什么情况下狗狗需要\1？'),
        (r'^(.*)选择策略$', r'如何为狗狗选择\1？'),
        (r'^(.*)识别与处理$', r'怎么识别和处理狗狗的\1？'),
        (r'^(.*)替代方案$', r'除了\1还有别的办法吗？'),
        (r'^(.*)接种安全性$', r'狗狗\1安全吗？'),
        (r'^(.*)疫苗接种注意事项$', r'狗狗打\1疫苗要注意什么？'),
        (r'^(.*)频率建议$', r'狗狗\1多久做一次？'),
        (r'^(.*)重点$', r'给狗狗做\1主要查什么？'),
        (r'^(.*)筛查项目$', r'狗狗的\1需要查哪些？'),
        (r'^(.*)的意义与周期$', r'狗狗的\1有什么用？多久做一次？'),
        (r'^(.*)应用$', r'狗狗的\1怎么用？'),
        (r'^(.*)选择指南$', r'怎么给狗狗选\1？'),
        (r'^(.*)护理$', r'狗狗\1怎么护理？'),
        (r'^(.*)管理$', r'狗狗\1该怎么管理？'),
        (r'^(.*)预防$', r'怎么预防狗狗\1？'),
        (r'^(.*)处理$', r'狗狗\1了怎么办？'),
        (r'^(.*)风险$', r'给狗狗\1有风险吗？'),
        (r'^(.*)利弊$', r'给狗狗\1有什么好处和坏处？'),
        (r'^(.*)替代方案$', r'除了\1还有什么选择？'),
    ]
    for pattern, repl in mapping:
        if re.search(pattern, topic):
            return re.sub(pattern, repl, topic)

    # 默认：直接变成疑问句
    if topic.endswith('方法') or topic.endswith('技巧'):
        return f"狗狗{topic}有哪些具体方法？"
    if topic.endswith('清单'):
        return f"狗狗{topic}包括哪些内容？"
    if topic.endswith('标准'):
        return f"狗狗{topic}是什么？"
    if topic.startswith('预防'):
        return f"如何{topic}？"

    return f"狗狗{topic}，主人应该知道什么？"


# ================ API调用类 ================
class PreventiveCareFAQGenerator:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _call_api(self, prompt: str, max_tokens: int = 1200) -> str:
        """带重试的API调用"""
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
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
                    wait = 2 ** attempt
                    time.sleep(wait)
            except Exception:
                wait = 2 ** attempt
                time.sleep(wait)
        return "【生成失败】多次重试后仍无法获取回答。"

    def _build_single_prompt(self, topic: str) -> str:
        """单问答 prompt：专业但亲切的预防保健知识"""
        instruction = topic_to_instruction(topic)
        return f"""你是一位资深宠物医生，擅长犬的预防保健和健康管理。请用专业但亲切的语气，回答宠物主人关于狗狗健康的问题。

问题：{instruction}

写作要求：
- 开头可用“铲屎官你好～”等自然口语。
- 内容应包括：原因/背景、具体可操作的建议、注意事项、温馨小贴士。
- 不用列出编号，用自然段落
- 控制在300~500字之间。
- 必须是原创、准确、基于兽医学共识的回答。

请直接输出回答，不要输出JSON。"""

    def _build_chain_prompt(self, chain_name: str, round_q: str,
                            round_num: int, context: Optional[str] = None) -> str:
        """追问链轮次 prompt"""
        context_part = f"\n上文回顾：{context}\n" if context else ""
        return f"""你是一位资深宠物医生，正在解答一个连续追问的狗狗预防保健问题。

追问主题：{chain_name}
当前轮次：第{round_num}轮
当前问题：{round_q}
{context_part}
要求：
- 只回答当前这一轮的问题，不要重复之前内容。
- 语言亲切自然，像在微信上耐心解答朋友的问题。
- 200~400字，信息密度高，建议具体。
- 可以适当使用表情符号，但不要过度。

直接输出回答，不要输出JSON。"""

    def generate_single(self, topic: str) -> Dict:
        """生成一个单问答条目"""
        prompt = self._build_single_prompt(topic)
        answer = self._call_api(prompt, max_tokens=1200)
        instruction = topic_to_instruction(topic)
        return {
            "instruction": instruction,
            "input": "",
            "output": answer,
            "metadata": {
                "type": "single_qa",
                "topic": topic,
                "category": "preventive_care"
            }
        }

    def generate_chain(self, chain: Dict, chain_index: int) -> List[Dict]:
        """生成整个追问链的所有轮次"""
        results = []
        context = None
        for idx, round_q in enumerate(chain["rounds"], start=1):
            prompt = self._build_chain_prompt(
                chain_name=chain["name"],
                round_q=round_q,
                round_num=idx,
                context=context
            )
            answer = self._call_api(prompt, max_tokens=800)
            instruction = f"（预防保健追问{chain_index}-{idx}）{round_q}"
            results.append({
                "instruction": instruction,
                "input": "",
                "output": answer,
                "metadata": {
                    "type": "chain_round",
                    "chain_name": chain["name"],
                    "chain_index": chain_index,
                    "round": idx,
                    "total_rounds": len(chain["rounds"]),
                    "question": round_q
                }
            })
            # 更新上下文（取回答前50字）
            if answer:
                context = answer[:60] + "..."
            time.sleep(0.3)  # 礼貌间隔
        return results


# ================ 主流程 ================
def main(limit_single: Optional[int] = None, limit_chain: Optional[int] = None):
    """
    limit_single: 限制生成单问答数量（用于测试）
    limit_chain: 限制生成追问链数量（用于测试）
    """
    print("🐕‍🦺 犬医 · 预防保健FAQ生成器")

    # 读取文件
    with open("faq4.txt", "r", encoding="utf-8") as f:
        content = f.read()

    single_topics, chain_list = parse_preventive_faq(content)
    print(f"📋 解析到 {len(single_topics)} 个单问答主题")
    print(f"🔗 解析到 {len(chain_list)} 个追问链（共{sum(len(c['rounds']) for c in chain_list)}轮）")

    # 限制数量（用于快速测试）
    if limit_single is not None:
        single_topics = single_topics[:limit_single]
    if limit_chain is not None:
        chain_list = chain_list[:limit_chain]

    generator = PreventiveCareFAQGenerator(DEEPSEEK_API_KEY)
    all_results = []

    # 生成单问答
    print("\n📌 开始生成单问答...")
    for topic in tqdm(single_topics, desc="单问答"):
        result = generator.generate_single(topic)
        all_results.append(result)
        time.sleep(0.5)  # 避免触发限流

    # 生成追问链
    print("\n🔗 开始生成追问链...")
    for idx, chain in enumerate(tqdm(chain_list, desc="追问链"), start=1):
        chain_results = generator.generate_chain(chain, idx)
        all_results.extend(chain_results)
        time.sleep(0.5)

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 生成完成！共 {len(all_results)} 个条目")
    print(f"💾 已保存至 {OUTPUT_FILE}")
    print("📊 统计：{} 单问答, {} 追问轮次".format(
        sum(1 for r in all_results if r["metadata"]["type"] == "single_qa"),
        sum(1 for r in all_results if r["metadata"]["type"] == "chain_round")
    ))


if __name__ == "__main__":
    import sys

    # 命令行支持 --test 参数用于快速测试
    if "--test" in sys.argv:
        print("🧪 测试模式：仅生成前2个单问答和前1个追问链")
        main(limit_single=2, limit_chain=1)
    elif "--full" in sys.argv:
        print("🚀 全量生成模式：将消耗API额度，预计2-3元")
        confirm = input("确认全量生成全部180+20条？(y/n): ")
        if confirm.lower() == 'y':
            main()
        else:
            print("已取消")
    else:
        print("请指定运行模式：")
        print("  python script.py --test   # 快速测试（仅少量）")
        print("  python script.py --full   # 全量生成（完整180+20）")
        print("\n默认使用测试模式（--test）")
        main(limit_single=2, limit_chain=1)