#!/usr/bin/env python3
"""
FAQ数据清洗脚本（新版）
对已生成的FAQ问答对进行二次优化，使提问更符合人类口吻，
情感类问题增强情感支持和实用建议，理性类问题保持专业性。
清洗后直接返回 instruction 和 output，无需 need_modify 标记。
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import requests

# ================ 配置区 ================
DEEPSEEK_API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"  # 请替换为您的DeepSeek API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
INPUT_DIR = r"D:\Backup\PythonProject2\data_process\find_faq\augmented_output"
OUTPUT_DIR = r"D:\Backup\PythonProject2\data_process\find_faq\new_augmented_output"
BATCH_DELAY = 0.5  # 每次API调用后的延迟（秒）
MAX_RETRIES = 3  # API调用失败重试次数


# ================ 清洗器类 ================
class FAQCleaner:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _call_api(self, prompt: str, max_tokens: int = 2000) -> str:
        """调用DeepSeek API，带重试机制"""
        for attempt in range(MAX_RETRIES):
            try:
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,  # 较低温度，保证稳定性
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
            except Exception as e:
                print(f"API调用异常: {e}")
                time.sleep(2 ** attempt)
        return ""

    def _build_clean_prompt(self, instruction: str, output: str) -> str:
        """构建清洗Prompt（精细版：智能处理宠物主体）"""
        return f"""你是一位专业的FAQ数据优化专家。请根据以下要求，对给定的问答对进行审核和优化。

    原始提问：{instruction}
    原始回答：{output}

    **任务**：
    1. **判断**：该问答对是否已经“及格”（即提问符合人类口吻，回答合理、有用）。对于情感类问题（如安乐死、宠物走失、生病照顾等），提问应体现情感，回答应包含情感支持和具体建议；对于理性分析问题，回答应保持专业性。
    2. **优化**：如果问答对不及格，请生成修改后的提问和回答。如果已经及格，则无需修改。

    **具体要求**：
    - 提问必须符合人类宠物主人的真实口吻，自然、口语化。
    - 回答需温暖、共情（情感类）或专业、清晰（理性类），并提供可行的建议或参考。
    - 如果原问答已经合理（可以接受多语言表达、情景描述、少量噪声），则返回原回答。
    - **关于宠物主体的处理**（请严格遵循以下规则）：
      - 如果原始提问中明确提到“狗”、“狗狗”、“犬”、“小狗”、“我家狗”等与狗直接相关的词语，或使用泛指词如“宠物”、“我的宠物”、“猫狗”等，则**保留原主体**，无需强制改为“狗”。
      - 如果原始提问中主体是其他特定动物（如“猫”、“鸟”、“兔子”、“仓鼠”等），则需将主体**改写为“狗”**，并相应调整回答中的动物名称（例如将“猫抓伤”改为“狗抓伤”），但保持问题的逻辑正确和情境合理。
      - 如果原始提问中主体是人类自身（例如“我被猫抓伤了”），则保留人的主体，但将涉及的动物改为“狗”（例如“我被狗抓伤了”），并确保回答中提及的动物也相应调整。
      - 如果原始提问中完全没有提及任何动物，请根据上下文推断，若明显是宠物相关则默认使用“狗”；若与宠物无关（如人类医学问题），则无需修改。
    - 禁止使用Markdown标记或表情符号。

    **输出格式**（必须返回严格的JSON，不要附加任何其他内容）：
    {{
        "instruction": "及格的原提问或修改后的提问",
        "output": "及格的原回答或修改后的回答"
    }}

    请直接输出JSON，不要添加任何额外说明。"""

    def _build_clean_prompt(self, instruction: str, output: str) -> str:
        """构建清洗Prompt（用户自定义版本）"""
        return f"""你是一位专业的FAQ数据优化专家。请根据以下要求，对给定的问答对进行审核和优化。

    原始提问：{instruction}
    原始回答：{output}

    **任务**：
    1. **判断**：该问答对是否已经“及格”（即提问符合人类口吻，回答合理、有用）。对于情感类问题（如安乐死、宠物走失、生病照顾等），提问应体现情感，回答应包含情感支持和具体建议；对于理性分析问题，回答应保持专业性。
    2. **优化**：如果问答对不及格，请生成修改后的提问和回答。如果已经及格，则无需修改。

    **具体要求**：
    - 提问必须符合人类宠物主人的真实口吻，自然、口语化。
    - 回答需温暖、共情（情感类）或专业、清晰（理性类），并提供可行的建议或参考。
    - 如果原问答已经合理（可以接受多语言表达、情景描述、少量噪声），则返回原回答。
    - 禁止使用Markdown标记或表情符号。

    **输出格式**（必须返回严格的JSON，不要附加任何其他内容）：
    {{
        "instruction": "及格的原提问或修改后的提问",
        "output": "及格的原回答或修改后的回答"
    }}

    请直接输出JSON，不要添加任何额外说明。"""

    def clean_one(self, instruction: str, output: str) -> Dict[str, str]:
        """
        清洗单个问答对，返回包含 instruction 和 output 的字典。
        如果API调用失败，返回原内容。
        """
        prompt = self._build_clean_prompt(instruction, output)
        response = self._call_api(prompt, max_tokens=2500)
        if not response:
            # API调用失败，返回原内容
            return {"instruction": instruction, "output": output}

        # 尝试解析JSON
        try:
            # 有时候模型会输出额外的文字，尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            # 确保返回的字段存在
            new_instruction = result.get("instruction", instruction)
            new_output = result.get("output", output)
            return {"instruction": new_instruction, "output": new_output}
        except json.JSONDecodeError:
            print(f"JSON解析失败，原始响应：{response[:200]}")
            return {"instruction": instruction, "output": output}


# ================ 文件处理函数 ================
def process_file(file_path: str, cleaner: FAQCleaner, output_dir: str) -> int:
    """处理单个JSON文件，返回清洗后的条目数"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"文件 {file_path} 不是列表格式，跳过")
        return 0

    cleaned_data = []
    for item in tqdm(data, desc=f"清洗 {os.path.basename(file_path)}"):
        # 检测条目中使用的提问字段名和回答字段名
        # 常见字段名：instruction / question / input 等
        q_field = None
        a_field = None
        for field in ["instruction", "question", "input"]:
            if field in item:
                q_field = field
                break
        for field in ["output", "answer"]:
            if field in item:
                a_field = field
                break

        if q_field is None or a_field is None:
            print("警告：条目缺少提问或回答字段，跳过")
            cleaned_data.append(item)
            continue

        instruction = item[q_field]
        output = item[a_field]

        # 清洗
        result = cleaner.clean_one(instruction, output)
        new_instruction = result["instruction"]
        new_output = result["output"]

        # 创建新条目，保留所有原始字段，仅更新提问和回答
        new_item = item.copy()
        new_item[q_field] = new_instruction
        new_item[a_field] = new_output
        # 可选：添加清洗标记
        if "metadata" not in new_item:
            new_item["metadata"] = {}
        if isinstance(new_item["metadata"], dict):
            new_item["metadata"]["cleaned"] = True
        else:
            # 如果metadata不是字典，转换为字典
            new_item["metadata"] = {"cleaned": True, "original": new_item.get("metadata", "")}

        cleaned_data.append(new_item)
        time.sleep(BATCH_DELAY)  # 控制请求频率

    # 保存到输出目录，保持原文件名
    out_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    return len(cleaned_data)


# ================ 主流程 ================
def main():
    print("🧹 FAQ数据清洗脚本（新版）")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有JSON文件
    json_files = []
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(".json"):
            json_files.append(os.path.join(INPUT_DIR, file))

    if not json_files:
        print("❌ 输入目录中没有找到JSON文件")
        return

    print(f"📂 发现 {len(json_files)} 个JSON文件")

    # 初始化清洗器
    cleaner = FAQCleaner(DEEPSEEK_API_KEY)

    total_cleaned = 0
    for file_path in json_files:
        print(f"\n🔧 处理文件: {os.path.basename(file_path)}")
        try:
            count = process_file(file_path, cleaner, OUTPUT_DIR)
            total_cleaned += count
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    print(f"\n✅ 清洗完成！共处理 {total_cleaned} 个条目，结果保存在 {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # 测试模式：只处理第一个文件的前3条
        test_output_dir = os.path.join(OUTPUT_DIR, "test")
        os.makedirs(test_output_dir, exist_ok=True)
        json_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
        if json_files:
            with open(json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)[:3]
            test_file = os.path.join(test_output_dir, "test_sample.json")
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("测试模式：处理第一个文件的前3条")
            # 临时修改输入目录为测试文件所在目录
            original_input = INPUT_DIR
            INPUT_DIR = test_output_dir
            main()
            INPUT_DIR = original_input
        else:
            print("无文件可测试")
    else:
        main()