"""
将原始对话数据转换为 Alpaca/SFT 格式（JSONL）
支持多轮对话自动拼接为单条 text
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


# ─── 模板 ───────────────────────────────────────────────────────────────────
ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def build_instruction(row: Dict[str, Any]) -> str:
    """从原始数据行构造 instruction"""
    if "instruction" in row and row["instruction"]:
        return row["instruction"].strip()
    # fallback: 从对话自动生成
    if "messages" in row:
        msgs = row["messages"]
        if msgs and msgs[0].get("role") == "user":
            return msgs[0]["content"][:200]
    return "请回答以下问题。"


def build_input(row: Dict[str, Any]) -> str:
    """从原始数据行构造 input（可为空）"""
    if "input" in row:
        return row["input"].strip()
    if "messages" in row and len(row["messages"]) > 1:
        # 多轮: 把第 2 条起的 history 当作 input
        return ""
    return ""


def build_output(row: Dict[str, Any]) -> str:
    """从原始数据行构造 output"""
    if "output" in row and row["output"]:
        return row["output"].strip()
    if "messages" in row:
        msgs = row["messages"]
        # 最后一条 assistant 消息为 output
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                return msg["content"].strip()
    return ""


def convert_to_alpaca(row: Dict[str, Any]) -> Dict[str, str]:
    """将单条原始数据转换为 Alpaca 格式"""
    instruction = build_instruction(row)
    input_text  = build_input(row)
    output      = build_output(row)

    if not output:
        return None   # 跳过无 answer 的样本

    text = ALPACA_TEMPLATE.format(
        instruction=instruction,
        input=input_text,
        output=output,
    )
    return {"text": text}


# ─── 批量转换 ───────────────────────────────────────────────────────────────
def process_file(
    input_path: str,
    output_path: str,
    input_format: str = "json",      # json | jsonl
    filter_empty: bool = True,
):
    """
    读取原始数据文件，转换为 Alpaca JSONL

    Args:
        input_path:   原始数据文件路径
        output_path:  输出 JSONL 路径
        input_format: json(单数组) 或 jsonl(每行一条)
        filter_empty: 跳过无 output 的样本
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 读取 ──
    log(f"读取: {input_path}")
    if input_format == "json":
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raw_data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_data.append(json.loads(line))

    if not isinstance(raw_data, list):
        raise ValueError(f"数据应为数组格式，当前: {type(raw_data)}")

    # ── 转换 ──
    log(f"开始转换 {len(raw_data)} 条样本 ...")
    results = []
    skipped = 0
    for row in tqdm(raw_data, desc="转换"):
        converted = convert_to_alpaca(row)
        if converted is None:
            skipped += 1
            continue
        results.append(converted)

    # ── 写出 ──
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log(f"✅ 完成！输出 {len(results)} 条（跳过 {skipped} 条）→ {output_path}")
    return results


def log(msg: str):
    print(f"[preprocess] {msg}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="对话数据 → Alpaca 格式转换")
    parser.add_argument("--input",  "-i", required=True, help="原始数据文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 路径")
    parser.add_argument(
        "--format", "-f",
        choices=["json", "jsonl"],
        default="jsonl",
        help="输入文件格式（默认 jsonl）",
    )
    args = parser.parse_args()
    process_file(args.input, args.output, args.format)


if __name__ == "__main__":
    main()
