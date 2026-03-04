#!/usr/bin/env python3
"""
犬伦理与法律FAQ生成器
用于处理安乐死决策、动物福利法规、兽医职业伦理等专业议题
语气：专业、严谨、中立、共情，面向宠物主人及从业者
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
OUTPUT_FILE = "faq6.json"


# ================ 解析 faq6.txt ================
def parse_ethics_topics(file_path: str) -> List[str]:
    """从 faq6.txt 读取伦理法律主题列表（每行一个主题）"""
    topics = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # 忽略注释行
                topics.append(line)
    return topics


# ================ 伦理法律FAQ生成器 ================
class EthicsLawFAQGenerator:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _call_api(self, prompt: str, max_tokens: int = 1800) -> str:
        """带重试机制的API调用"""
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,  # 低温度保证事实准确性
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

    def _build_prompt(self, topic: str) -> str:
        """构建伦理法律专题FAQ的Prompt"""
        return f"""你是一位专门研究动物伦理与法律的小动物临床兽医师，同时拥有兽医法学硕士学位，擅长处理宠物医疗伦理困境与动物福利法规咨询。

请围绕以下主题，撰写一份面向宠物主人及兽医从业者的专业伦理/法律FAQ：

主题：{topic}

写作要求：
1. **结构完整**（使用中文自然段落，不采用Markdown标记，不使用列表编号）：
   - 核心问题：用一段话简要重述主题，界定讨论范围。
   - 医学/法律背景：相关法规、伦理原则（如动物福利五项自由、知情同意、兽医职业伦理、现行法律条款等）。
   - 决策框架/行动指南：如果涉及决策，提供清晰的步骤或考量因素；如果不涉及决策，则说明当前共识与争议。
   - 伦理困境分析：客观呈现不同立场（如宠物主人、兽医、社会公众、动物本身）的合理观点。
   - 对主人的建议：具体、可操作的建议，包括如何与兽医沟通、寻求第二意见、法律援助、心理支持等。
   - 总结：简明扼要的结语。

2. **语言风格**：
   - 专业、冷静、共情，但不煽情、不偏激。
   - 使用“您”尊称读者。
   - 禁止使用任何表情符号。
   - 字数严格控制在450-650字之间。

3. **内容底线**：
   - 不得提供具体的医疗建议（除非是普遍接受的伦理规范）。
   - 不得鼓励任何违法行为。
   - 不得贬低、污名化任何一方（主人、兽医、执法者等）。
   - 涉及法律条文时，以“《动物防疫法》《民法典》等法规”指代，不编造具体法条号。

请直接输出FAQ正文，不要输出JSON包裹，不要额外解释。
"""

    def generate(self, topic: str) -> Dict:
        """生成单个伦理法律FAQ条目"""
        prompt = self._build_prompt(topic)
        answer = self._call_api(prompt, max_tokens=1800)
        return {
            "instruction": topic,  # 直接使用主题作为主人提问
            "input": "",
            "output": answer,
            "metadata": {
                "type": "ethics_law_faq",
                "category": "伦理与法律",
                "topic": topic
            }
        }


# ================ 主流程 ================
def main(limit: Optional[int] = None):
    print("⚖️ 犬伦理与法律FAQ生成器（专业严肃版）")

    # 解析主题文件
    try:
        topics = parse_ethics_topics("faq6.txt")
    except FileNotFoundError:
        print("❌ 未找到 faq6.txt 文件，请确保文件在当前目录下。")
        return

    if not topics:
        print("❌ faq6.txt 为空或解析失败")
        return

    print(f"📋 解析到 {len(topics)} 个伦理法律主题")

    if limit:
        topics = topics[:limit]
        print(f"🧪 测试模式：仅生成前 {limit} 个主题")

    generator = EthicsLawFAQGenerator(DEEPSEEK_API_KEY)
    all_results = []

    print("\n📌 开始生成伦理法律FAQ……")
    for topic in tqdm(topics, desc="伦理FAQ"):
        result = generator.generate(topic)
        all_results.append(result)
        time.sleep(0.5)  # 控制请求频率

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 生成完成！共 {len(all_results)} 个条目")
    print(f"💾 已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        main(limit=3)
    elif "--full" in sys.argv:
        print("🚨 全量生成模式：将生成全部伦理法律FAQ条目")
        confirm = input("确认生成全部主题？(y/n): ")
        if confirm.lower() == 'y':
            main()
        else:
            print("已取消")
    else:
        print("请指定运行模式：")
        print("  python ethics_law_faq.py --test   # 快速测试（仅3条）")
        print("  python ethics_law_faq.py --full   # 全量生成（全部主题）")
        print("\n默认使用测试模式（--test）")
        main(limit=3)