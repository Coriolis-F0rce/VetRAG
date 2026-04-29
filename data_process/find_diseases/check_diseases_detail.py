import json
import re
import requests
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback
import os

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
API_KEY = "sk-2eedab5b21954b6bb26f7461706642f2"

# 文件路径配置（相对于当前脚本的路径）
BASE_DIR = Path(__file__).parent.parent if Path(__file__).parent.name == "find_diseases" else Path(__file__).parent
INPUT_JSON = str(BASE_DIR / "find_diseases" / "dog_diseases_knowledge_base.json")
OUTPUT_JSON = str(BASE_DIR / "find_diseases" / "dog_diseases_professional.json")
PROGRESS_FILE = str(BASE_DIR / "find_diseases" / "processing_progress.json")
DEBUG_DIR = str(BASE_DIR / "find_diseases" / "debug_logs")

# 处理配置
BATCH_SIZE = 5  # 每批处理数量（较小以避免API限制）
REQUEST_DELAY = 3  # 请求延迟（秒）
MAX_RETRIES = 3  # 最大重试次数
TIMEOUT = 90  # 超时时间（秒）

def extract_first_json_object(response_text: str) -> Dict[str, Any]:
    if not response_text:
        raise ValueError("响应文本为空")

    # 找到第一个 {
    start = response_text.find('{')
    if start == -1:
        raise ValueError("未找到JSON开始标记'}'")

    # 使用栈来匹配大括号
    stack = []
    in_string = False
    escape_next = False
    string_char = None

    for i in range(start, len(response_text)):
        char = response_text[i]

        # 处理转义字符
        if escape_next:
            escape_next = False
            continue

        # 处理字符串开始/结束
        if char in ('"', "'") and not in_string:
            in_string = True
            string_char = char
        elif char == string_char and in_string:
            in_string = False
            string_char = None
        elif char == '\\' and in_string:
            escape_next = True
            continue

        # 只有在不在字符串中时才处理括号
        if not in_string:
            if char == '{':
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                else:
                    # 不匹配的右括号
                    break

                # 如果栈为空，说明找到了完整的JSON
                if not stack:
                    end = i + 1
                    json_str = response_text[start:end]

                    # 尝试解析
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"JSON解析失败: {e}")

    raise ValueError("未找到完整的JSON对象")


def repair_json_string(json_str: str) -> str:
    if not json_str:
        return json_str

    # 移除代码块标记
    json_str = json_str.strip()
    if "```json" in json_str:
        json_str = json_str.replace("```json", "").replace("```", "").strip()
    elif "```" in json_str:
        json_str = json_str.replace("```", "").strip()

    # 移除JSON外的文本（保留第一个JSON对象）
    lines = json_str.split('\n')
    json_lines = []
    in_json = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            in_json = True
        if in_json:
            json_lines.append(line)
        if stripped.endswith('}') or stripped.endswith(']'):
            in_json = False

    json_str = '\n'.join(json_lines).strip()

    # 确保以 { 开头，以 } 结尾
    start = json_str.find('{')
    end = json_str.rfind('}') + 1

    if start != -1 and end > start:
        json_str = json_str[start:end]

    # 计算括号数量并补全
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')

    # 补全大括号
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)

    # 补全中括号
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)

    # 移除多余的逗号（数组和对象末尾）
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    return json_str.strip()


def safe_parse_json(response_text: str, extract_first: bool = True) -> Dict[str, Any]:
    if not response_text:
        raise ValueError("输入文本为空")

    # 清理文本
    cleaned_text = response_text.strip()

    # 尝试的解析方法
    attempts = []

    if extract_first:
        attempts.append(lambda: extract_first_json_object(cleaned_text))

    attempts.extend([
        # 尝试直接解析
        lambda: json.loads(cleaned_text),
        # 尝试清理后解析
        lambda: json.loads(repair_json_string(cleaned_text)),
        # 尝试提取JSON部分
        lambda: json.loads(re.search(r'\{.*\}', cleaned_text, re.DOTALL).group()),
    ])

    for i, attempt in enumerate(attempts):
        try:
            result = attempt()
            print(f"第{i + 1}种解析方法成功")
            return result
        except (json.JSONDecodeError, ValueError, IndexError, AttributeError) as e:
            if i == len(attempts) - 1:
                print(f"所有解析方法均失败: {e}")
                raise
            continue

    raise ValueError("无法解析JSON")


# ========== API调用函数 ==========

def call_deepseek_api_safe(
        prompt: str,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT
) -> Dict[str, Any]:

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 强化的系统提示
    system_prompt = """你是专业的兽医AI助手。请严格按照以下要求输出：
    1. 只输出一个完整的JSON对象，不要有任何额外文本
    2. 不要添加任何解释、注释、说明
    3. 不要使用代码块标记（如```json或```）
    4. 确保JSON格式完全正确，可以直接被解析
    5. 确保所有字符串都使用双引号
    6. 确保没有尾随逗号

    输出格式必须是有效的JSON，并且只包含要求的字段。"""

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # 低随机性确保格式稳定
        "max_tokens": 4000,  # 确保足够的输出长度
        "top_p": 0.95,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            print(f"API调用尝试 {attempt + 1}/{max_retries}")

            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                timeout=timeout
            )

            if response.status_code != 200:
                print(f"API错误: 状态码 {response.status_code}")
                print(f"响应: {response.text[:200]}")

                if response.status_code == 429:  # 频率限制
                    wait_time = (attempt + 1) * 10
                    print(f"频率限制，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue

                raise Exception(f"API调用失败，状态码: {response.status_code}")

            # 解析API响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content:
                raise ValueError("API返回内容为空")

            print(f"API响应长度: {len(content)} 字符")

            # 尝试安全解析JSON
            try:
                parsed_data = safe_parse_json(content, extract_first=True)
                print("JSON解析成功")
                return parsed_data

            except Exception as parse_error:
                print(f"JSON解析失败: {parse_error}")

                # 保存原始响应用于调试
                if not os.path.exists(DEBUG_DIR):
                    os.makedirs(DEBUG_DIR, exist_ok=True)

                debug_filename = f"{DEBUG_DIR}/debug_response_{int(time.time())}.txt"
                with open(debug_filename, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"原始响应已保存到: {debug_filename}")

                # 尝试更激进的修复
                repaired = repair_json_string(content)
                try:
                    parsed_data = json.loads(repaired)
                    print("修复后解析成功")
                    return parsed_data
                except:
                    pass

                # 等待后重试
                if attempt < max_retries - 1:
                    wait_time = 3 * (attempt + 1)
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"JSON解析失败: {parse_error}")

        except requests.exceptions.Timeout:
            print(f"请求超时，尝试 {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            raise Exception("请求超时")

        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise Exception(f"API请求失败: {e}")

    raise Exception("所有重试尝试均失败")


# ========== 数据处理函数 ==========

def load_existing_data(input_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]  # 如果是单个对象，转换为列表
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"输入文件格式不正确，应为列表或对象，实际为: {type(data)}")


def load_progress(progress_file: str) -> Dict[str, Any]:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载进度文件失败: {e}")

    return {
        "processed_count": 0,
        "failed_items": [],
        "skipped_items": [],
        "success_count": 0,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def save_progress(progress_file: str, progress_data: Dict[str, Any]):
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)


def save_partial_results(output_file: str, results: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output_data = {
        "metadata": metadata or {
            "total_count": len(results),
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "partial_save": True
        },
        "diseases": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


# ========== Prompt模板 ==========

SIMPLIFIED_PROMPT_TEMPLATE = """任务：优化和标准化犬病档案数据

请根据提供的原始疾病数据，生成标准化的临床决策级犬病档案。确保所有字段都按照以下JSON结构填写：

{{
  "disease_name": "疾病标准名称（又被称为：别名）",
  "disease_type": "疾病类型，如病毒性传染病",
  "standard_codes": {{"icd11_vet": "编码或'未收录'", "snomed_ct": "编码或'未收录'"}},
  "disease_category": "病毒性传染病/细菌性/寄生虫/行为问题/中毒/内科/外科等",
  "affected_species": ["犬", "其他易感动物"],
  "zoonotic": true/false,
  "symptoms": {{"primary": ["核心症状"], "secondary": ["次要症状"]}},
  "symptom_weights": {{"症状": 权重值}},
  "differential_symptoms": ["关键鉴别特征"],
  "urgency_level": 1/2/3,
  "contagious_level": 0.0-1.0,
  "incidence_level": 0.0-1.0,
  "severity_level": 0.0-1.0,
  "prevalence_by_age": {{"puppy": 值, "adult": 值, "senior": 值}},
  "onset_pattern": "急性/亚急性/慢性",
  "seasonality": "季节特征",
  "common_triggers": ["诱因"],
  "behavioral_flag": true/false,
  "critical_keywords": ["关键词"],
  "emergency_threshold": "就医红线",
  "misdiagnosis_risks": ["易混淆疾病"],
  "diagnosis": ["诊断方法"],
  "treatment": [{{"category": "类别", "name": "方案", "drug": "药物", "dosage": "剂量", "route": "途径", "frequency": "频率"}}],
  "prevention": ["预防措施"],
  "faq": [{{"question": "问题", "answer": "答案"}}],
  "emergency_guidelines": ["指南"],
  "prognosis": "预后描述",
  "cost_estimation": "费用估算",
  "source_refs": ["参考文献"]
}}

原始数据：
{现有JSON数据}

重要：请只输出一个完整的JSON对象，不要添加任何额外文本。确保JSON格式完全正确。"""


def process_disease(
        disease_data: Dict[str, Any],
        index: int,
        total_count: int
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        disease_name = disease_data.get("disease_name", f"疾病_{index + 1}")
        print(f"\n处理第 {index + 1}/{total_count} 条: {disease_name}")

        # 构建prompt
        prompt = SIMPLIFIED_PROMPT_TEMPLATE.format(
            现有JSON数据=json.dumps(disease_data, ensure_ascii=False, indent=2)
        )

        # 调用API
        print("正在调用API...")
        start_time = time.time()
        processed_data = call_deepseek_api_safe(prompt)
        elapsed_time = time.time() - start_time
        print(f"API调用完成，耗时: {elapsed_time:.2f}秒")

        # 验证必要字段
        required_fields = ["disease_name", "disease_type", "symptoms", "urgency_level"]
        missing_fields = []
        for field in required_fields:
            if field not in processed_data:
                missing_fields.append(field)

        if missing_fields:
            print(f"警告: 缺少必要字段 {missing_fields}")
            # 尝试补充缺失字段
            for field in missing_fields:
                if field in disease_data:
                    processed_data[field] = disease_data[field]
                elif field == "disease_name":
                    processed_data[field] = disease_name

        return processed_data, "成功"

    except Exception as e:
        error_msg = str(e)
        print(f"处理疾病失败: {error_msg}")
        if "API密钥" in error_msg or "API_KEY" in error_msg:
            print("请检查API密钥设置")
        return None, error_msg


def main():
    print("=" * 60)
    print("犬病知识库专业化处理工具")
    print("=" * 60)

    # 检查API密钥
    if not API_KEY:
        print("错误: API密钥未设置!")
        print("请设置环境变量 DEEPSEEK_API_KEY")
        print("或在当前目录创建 .env 文件，内容为: DEEPSEEK_API_KEY=your_api_key")
        return

    # 检查输入文件
    if not os.path.exists(INPUT_JSON):
        print(f"错误: 输入文件不存在: {INPUT_JSON}")
        return

    # 确保调试目录存在
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 加载数据
    print("\n正在加载数据...")
    try:
        diseases = load_existing_data(INPUT_JSON)
        total_count = len(diseases)
        print(f"✓ 加载成功，共 {total_count} 条疾病数据")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return

    # 加载进度
    progress = load_progress(PROGRESS_FILE)
    processed_count = progress.get("processed_count", 0)
    failed_items = progress.get("failed_items", [])
    skipped_items = progress.get("skipped_items", [])
    success_count = progress.get("success_count", 0)

    # 计算需要处理的数量
    start_index = processed_count
    remaining_count = total_count - start_index

    if remaining_count <= 0:
        print("\n所有数据已处理完成!")
        print(f"总处理数: {total_count}, 成功: {success_count}, 失败: {len(failed_items)}")
        return

    print(f"\n进度: 已处理 {processed_count} 条，跳过 {len(skipped_items)} 条")
    print(f"剩余 {remaining_count} 条待处理")

    # 确认是否继续
    if start_index > 0:
        response = input(f"是否从第 {start_index + 1} 条开始继续处理？(y/n): ")
        if response.lower() != 'y':
            print("已取消处理")
            return

    # 处理数据
    results = []
    batch_start_time = time.time()

    for i in range(start_index, total_count):
        # 检查是否应该跳过
        should_skip = False
        for skipped in skipped_items:
            if isinstance(skipped, dict) and skipped.get("index") == i:
                print(f"\n跳过已标记为跳过的项目: {skipped.get('name', f'索引{i}')}")
                should_skip = True
                break
            elif isinstance(skipped, int) and skipped == i:
                print(f"\n跳过已标记为跳过的项目: 索引{i}")
                should_skip = True
                break

        if should_skip:
            processed_count += 1
            continue

        # 处理疾病
        disease = diseases[i]
        processed, status = process_disease(disease, i, total_count)

        if processed:
            results.append(processed)
            success_count += 1
            print(f"✓ 处理成功")
        else:
            # 记录失败
            disease_name = disease.get("disease_name", f"疾病_{i + 1}")
            failed_items.append({
                "index": i,
                "name": disease_name,
                "error": status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            print(f"✗ 处理失败: {status}")

        processed_count += 1

        # 更新进度
        progress.update({
            "processed_count": processed_count,
            "success_count": success_count,
            "failed_items": failed_items,
            "skipped_items": skipped_items,
            "last_processed": disease.get("disease_name", f"疾病_{i + 1}"),
            "last_processed_index": i,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # 每处理BATCH_SIZE条保存一次结果
        if (i + 1) % BATCH_SIZE == 0 or i == total_count - 1:
            # 保存进度
            save_progress(PROGRESS_FILE, progress)
            print(f"进度已保存: {PROGRESS_FILE}")

            # 保存部分结果
            if results:
                metadata = {
                    "total_processed": processed_count,
                    "success_count": success_count,
                    "failed_count": len(failed_items),
                    "skipped_count": len(skipped_items),
                    "batch_save": True,
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                save_partial_results(OUTPUT_JSON, results, metadata)
                print(f"已保存 {len(results)} 条结果到: {OUTPUT_JSON}")

            # 计算统计信息
            batch_time = time.time() - batch_start_time
            avg_time = batch_time / min(BATCH_SIZE, i - start_index + 1) if i > start_index else batch_time
            print(f"批次处理耗时: {batch_time:.2f}秒，平均每条: {avg_time:.2f}秒")

            # 避免API频率限制
            if i < total_count - 1:
                print(f"等待 {REQUEST_DELAY} 秒后继续...")
                time.sleep(REQUEST_DELAY)
                batch_start_time = time.time()

    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"{'=' * 60}")

    final_stats = {
        "总数据量": total_count,
        "成功处理": success_count,
        "处理失败": len(failed_items),
        "跳过项目": len(skipped_items),
        "开始时间": progress.get("start_time", "未知"),
        "完成时间": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    for key, value in final_stats.items():
        print(f"{key}: {value}")

    # 保存最终结果
    if results:
        final_output = {
            "metadata": {
                "total_count": total_count,
                "success_count": success_count,
                "failed_count": len(failed_items),
                "skipped_count": len(skipped_items),
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_time": progress.get("start_time", "未知"),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_stats": final_stats
            },
            "diseases": results
        }

        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"\n最终结果已保存到: {OUTPUT_JSON}")

        if failed_items:
            failed_log = str(BASE_DIR / "find_diseases" / "failed_items.json")
            with open(failed_log, 'w', encoding='utf-8') as f:
                json.dump({
                    "count": len(failed_items),
                    "failed_items": failed_items,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
            print(f"失败项目日志: {failed_log}")

    print(f"\n处理完成! 请检查输出文件: {OUTPUT_JSON}")


def print_usage():
    """打印使用说明"""
    print("使用说明:")
    print("  1. 确保已安装所需依赖: pip install requests python-dotenv")
    print("  2. 设置DeepSeek API密钥:")
    print("     - 创建 .env 文件，内容: DEEPSEEK_API_KEY=your_api_key")
    print("     - 或设置环境变量: export DEEPSEEK_API_KEY=your_api_key")
    print("  3. 确保输入文件存在: find_diseases/dog_diseases_knowledge_base.json")
    print("  4. 运行脚本: python check_diseases_detail.py")
    print("\n可选参数:")
    print("  --reset: 重置处理进度，从头开始")
    print("  --skip=N: 跳过前N条记录")
    print("  --help: 显示此帮助信息")


if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) > 1:
        if "--help" in sys.argv or "-h" in sys.argv:
            print_usage()
            sys.exit(0)
        elif "--reset" in sys.argv:
            print("重置处理进度...")
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
                print("进度文件已删除")
        elif "--skip=" in ''.join(sys.argv):
            for arg in sys.argv:
                if arg.startswith("--skip="):
                    try:
                        skip_count = int(arg.split("=")[1])
                        progress = load_progress(PROGRESS_FILE)
                        progress["processed_count"] = skip_count
                        save_progress(PROGRESS_FILE, progress)
                        print(f"已设置为跳过前 {skip_count} 条记录")
                    except ValueError:
                        print("错误: skip参数必须是数字")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，已保存当前进度")
        print(f"下次运行将从当前进度继续")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        traceback.print_exc()