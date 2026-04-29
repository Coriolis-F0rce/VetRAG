#!/usr/bin/env python3
"""
宠物主人情感支持FAQ生成器
用于处理宠物临终、安乐死决策、哀伤疗愈、照顾压力等情感支持议题
语气：温暖、共情、专业、实用，面向宠物主人
"""

import json
import time
import requests
from typing import List, Dict, Optional
from tqdm import tqdm

# ================ 配置区 ================
DEEPSEEK_API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"  # 请替换为你的DeepSeek API密钥，或从环境变量读取
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
OUTPUT_FILE = "faq7.json"
INPUT_FILE = "faq7.txt"  # 存放情感支持问题列表的文件（每行一个问题）


# ================ 解析问题列表 ================
def parse_questions(file_path: str) -> List[str]:
    """从 faq_emotion.txt 读取情感支持问题列表（每行一个宠物主人提问）"""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # 忽略空行和注释行
                questions.append(line)
    return questions


# ================ 情感支持FAQ生成器 ================
class EmotionFAQGenerator:
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
                    "temperature": 0.7,          # 稍高温度，使回答更温暖自然
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
        """构建情感支持FAQ的Prompt"""
        return f"""你是一位经验丰富、富有同理心的宠物心理咨询师，同时也具备小动物临床兽医学知识。你擅长倾听宠物主人的情感困扰，并给予温暖、专业、实用的建议。

请针对以下宠物主人的提问，写一段温暖共情的回答。

问题：{question}

写作要求：
1. **情感共情**：首先表达对主人情感的理解和认同，肯定主人的爱与付出。
2. **专业支撑**：适当融入兽医常识、宠物行为学或心理健康知识，但不要使用冷冰冰的术语。
3. **实用建议**：提供具体、可操作的建议或决策参考（如如何评估宠物生活质量、如何与兽医沟通、如何照顾自己等）。
4. **语言风格**：温暖、自然、流畅，使用“您”尊称读者，避免说教，避免使用表情符号和Markdown标记。
5. **结构自然**：采用自然段落，不编号，不分点，让回答读起来像朋友间的贴心交谈。
6. **字数控制**：300-500字之间。
7. **内容底线**：不提供具体的医疗方案（除非是公认的护理常识），不鼓励违法行为，不评判主人的选择。

请直接输出回答内容，不要输出JSON包裹，不要额外解释。"""

    def generate(self, question: str) -> Dict:
        """生成单个情感支持FAQ条目"""
        prompt = self._build_prompt(question)
        answer = self._call_api(prompt, max_tokens=1500)
        return {
            "instruction": question,   # 主人提问作为instruction
            "input": "",
            "output": answer,
            "metadata": {
                "type": "emotion_faq",
                "category": "情感支持",
                "topic": question[:20] + "..."  # 简单摘要
            }
        }


# ================ 主流程 ================
def main(limit: Optional[int] = None):
    print("🐾 宠物主人情感支持FAQ生成器（温暖共情版）")

    # 解析问题文件
    try:
        questions = parse_questions(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ 未找到 {INPUT_FILE} 文件，请确保文件在当前目录下。")
        return

    if not questions:
        print(f"❌ {INPUT_FILE} 为空或解析失败")
        return

    print(f"📋 解析到 {len(questions)} 个情感支持问题")

    if limit:
        questions = questions[:limit]
        print(f"🧪 测试模式：仅生成前 {limit} 个问题")

    generator = EmotionFAQGenerator(DEEPSEEK_API_KEY)
    all_results = []

    print("\n📌 开始生成情感支持FAQ……")
    for q in tqdm(questions, desc="情感FAQ"):
        result = generator.generate(q)
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
        print("🚨 全量生成模式：将生成全部情感支持FAQ条目")
        confirm = input("确认生成全部问题？(y/n): ")
        if confirm.lower() == 'y':
            main()
        else:
            print("已取消")
    else:
        print("请指定运行模式：")
        print("  python emotion_faq.py --test   # 快速测试（仅3条）")
        print("  python emotion_faq.py --full   # 全量生成（全部问题）")
        print("\n默认使用测试模式（--test）")
        main(limit=3)

