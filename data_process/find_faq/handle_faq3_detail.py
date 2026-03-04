import json
import re
import time
import requests
from typing import List, Dict, Tuple, Optional

# ================ 配置区 ================
DEEPSEEK_API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"  # 请替换为真实密钥或使用环境变量
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


# ================ 主题解析器 ================
def parse_daily_faq(text: str) -> Tuple[List[str], List[Dict]]:
    """
    解析日常问题FAQ文本，返回：
    - single_questions: 180个问题字符串列表
    - chain_topics: 追问链列表，每个元素为 {"name": 链名, "rounds": [轮次问题列表]}
    """
    lines = text.strip().split('\n')
    single_questions = []
    chain_topics = []

    # 状态标记
    in_single = True
    in_chains = False
    current_chain = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 判断是否进入追问链部分
        if line.startswith("# 二十个日常问题追问链主题"):
            in_single = False
            in_chains = True
            continue

        # 解析单问答
        if in_single:
            # 匹配形如 "1. 狗狗挑食不吃饭怎么办？"
            match = re.match(r'^\d+\.\s+(.+)', line)
            if match:
                question = match.group(1).strip()
                single_questions.append(question)

        # 解析追问链
        if in_chains:
            # 链标题，如 "追问链1：狗狗拆家怎么办？"
            if line.startswith("追问链"):
                if current_chain:
                    chain_topics.append(current_chain)
                chain_name = line.split("：")[0] if "：" in line else line
                current_chain = {"name": chain_name, "rounds": []}
            # 轮次，如 "第1轮：拆家是因为分离焦虑还是精力过剩？"
            elif line.startswith("第") and "轮：" in line and current_chain:
                round_text = line.split("：", 1)[1].strip()
                current_chain["rounds"].append(round_text)

    # 添加最后一个追问链
    if current_chain and current_chain not in chain_topics:
        chain_topics.append(current_chain)

    return single_questions, chain_topics


# ================ API调用核心 ================
class DailyFAQGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _call_api(self, prompt: str, max_tokens: int = 1200) -> str:
        """调用DeepSeek API"""
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=self.headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                return f"API错误: {resp.status_code}"
        except Exception as e:
            return f"调用异常: {str(e)}"

    def _build_single_prompt(self, question: str) -> str:
        """生成单问答的prompt"""
        return f"""你是一位资深的宠物行为顾问和犬类训练专家，请为宠物主人解答以下日常养狗问题。

问题：{question}

要求：
1. 答案要专业、准确、可操作，同时通俗易懂，适合普通宠物主人阅读。
2. 结构清晰，建议包含以下部分（用中文自然段落，不要使用Markdown标记）：
   - 问题定性/原因分析：简要说明为什么狗狗会出现这种行为。
   - 解决方案：分步骤给出具体、可执行的纠正或改善方法。
   - 预防建议：如何避免问题复发。
   - 常见误区：指出主人容易犯的错误。
3. 语气亲切、有同理心，避免恐吓式说教。
4. 字数控制在300-600字之间。

请直接输出答案，不要输出JSON包裹，不要额外解释。"""

    def _build_chain_prompt(self, chain_name: str, round_text: str,
                            round_num: int, context: Optional[str] = None) -> str:
        """生成追问链轮次的prompt"""
        context_part = f"\n上文回顾：{context}\n" if context else ""
        return f"""你是一位资深的宠物行为顾问和犬类训练专家，现在正在解答一个连续追问的狗狗日常问题。

追问链主题：{chain_name}
当前轮次：第{round_num}轮
当前问题：{round_text}
{context_part}
要求：
1. 直接回答当前轮次的具体问题，不要重复回答之前已经覆盖的内容。
2. 答案需深入、聚焦，避免泛泛而谈。
3. 如果涉及训练步骤，请给出清晰的操作细节。
4. 字数控制在200-400字之间。
5. 语言通俗易懂，适合宠物主人阅读。

请直接输出答案，不要输出JSON包裹，不要额外解释。"""

    def generate_single(self, question: str) -> Dict:
        """生成单问答条目"""
        prompt = self._build_single_prompt(question)
        answer = self._call_api(prompt, max_tokens=1200)

        return {
            "instruction": question,
            "input": "",
            "output": answer,
            "metadata": {
                "type": "single_qa",
                "question": question
            }
        }

    def generate_chain(self, chain: Dict, chain_index: int) -> List[Dict]:
        """生成单个追问链的所有轮次"""
        results = []
        context = None

        for idx, round_text in enumerate(chain["rounds"], start=1):
            prompt = self._build_chain_prompt(
                chain_name=chain["name"],
                round_text=round_text,
                round_num=idx,
                context=context
            )
            answer = self._call_api(prompt, max_tokens=800)

            instruction = f"（{chain['name']}-第{idx}轮）{round_text}"

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
                    "question": round_text
                }
            })

            # 更新上下文（取答案前50字作为下一轮的参考）
            if answer:
                context = answer[:50] + "..."
            time.sleep(0.3)  # 控制请求间隔

        return results


# ================ 批量生成主流程 ================
def main():
    print("🐕 狗狗日常问题FAQ生成器")
    print("正在解析主题数据...")


    with open("faq3.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
    single_questions, chain_topics = parse_daily_faq(file_content)
    print(
        f"✅ 解析完成：{len(single_questions)} 个单问答，{len(chain_topics)} 个追问链（共{sum(len(c['rounds']) for c in chain_topics)}轮）")

    # 初始化生成器
    generator = DailyFAQGenerator(DEEPSEEK_API_KEY)

    all_results = []

    # --- 生成单问答（前10个用于测试，生产时可移除切片）---
    for i, q in enumerate(single_questions, 1):
        print(f"  正在生成 {i}/10: {q[:20]}...")
        result = generator.generate_single(q)
        all_results.append(result)
        time.sleep(0.5)

    # --- 生成追问链（前2个链用于测试）---
    for i, chain in enumerate(chain_topics, 1):
        print(f"  正在生成追问链 {i}/{2}: {chain['name']}")
        chain_results = generator.generate_chain(chain, i)
        all_results.extend(chain_results)
        time.sleep(0.5)

    # --- 保存结果 ---
    output_file = "faq3.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存至 {output_file}")
    print(
        f"📊 共生成 {len(all_results)} 个FAQ条目（{sum(1 for r in all_results if r['metadata']['type'] == 'single_qa')} 单问答，{sum(1 for r in all_results if r['metadata']['type'] == 'chain_round')} 追问轮次）")


if __name__ == "__main__":
    main()