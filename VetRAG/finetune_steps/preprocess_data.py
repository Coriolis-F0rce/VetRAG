"""
将 VetRAG 原始训练数据转换为 Alpaca/SFT 格式 JSONL
支持两种输入格式：JSON 数组 或 JSONL
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm


# ── Alpaca 模板 ──────────────────────────────────────────────────────────────
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def _safe(text: str) -> str:
    return (text or "").strip()


def _build_instruction(row: dict) -> str:
    """从原始行提取 instruction"""
    if row.get("instruction"):
        return _safe(row["instruction"])
    if row.get("messages"):
        msgs = row["messages"]
        for m in msgs:
            if m.get("role") == "user":
                return _safe(m["content"])[:300]
    return "请回答以下问题。"


def _build_input(row: dict) -> str:
    """从原始行提取 input（可选，可为空）"""
    if "input" in row:
        return _safe(row["input"])
    if row.get("messages") and len(row["messages"]) > 1:
        return ""
    return ""


def _build_output(row: dict) -> str:
    """从原始行提取 output（assistant 回复）"""
    if row.get("output"):
        return _safe(row["output"])
    if row.get("messages"):
        for m in reversed(row["messages"]):
            if m.get("role") == "assistant":
                return _safe(m["content"])
    return ""


def convert_row(row: dict) -> dict | None:
    """将一条原始数据转换为 Alpaca 格式；无 output 则返回 None"""
    output = _build_output(row)
    if not output:
        return None
    return {"text": ALPACA_TEMPLATE.format(
        instruction=_build_instruction(row),
        input=_build_input(row),
        output=output,
    )}


def convert_file(
    input_path: str,
    output_path: str,
    input_format: str = "json",
    filter_empty: bool = True,
):
    """
    读取原始数据 → 输出 Alpaca JSONL

    Args:
        input_path:   原始数据路径
        output_path:  目标 JSONL 路径
        input_format: "json"（JSON 数组）或 "jsonl"（逐行 JSON）
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 读取 ──
    print(f"[preprocess] 读取: {input_path}")
    if input_format == "json":
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raw_data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

    if not isinstance(raw_data, list):
        raise ValueError(f"数据根节点应为数组，当前类型: {type(raw_data)}")

    # ── 转换 ──
    print(f"[preprocess] 转换 {len(raw_data)} 条样本 ...")
    results = []
    skipped = 0
    for row in tqdm(raw_data, desc="转换为 Alpaca"):
        converted = convert_row(row)
        if converted is None:
            skipped += 1
            continue
        results.append(converted)

    # ── 写出 ──
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[preprocess] ✅ 完成！输出 {len(results)} 条（跳过 {skipped} 条）→ {output_path}")
    return results


def split_train_val(
    jsonl_path: str,
    output_dir: str,
    train_ratio: float = 0.95,
    seed: int = 42,
):
    """将单个 JSONL 文件按比例划分为训练集 / 验证集"""
    import random
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    random.seed(seed)
    random.shuffle(lines)
    n_train = int(len(lines) * train_ratio)
    train_lines, val_lines = lines[:n_train], lines[n_train:]

    train_out = output_dir / "train.jsonl"
    val_out   = output_dir / "val.jsonl"

    for path, data in [(train_out, train_lines), (val_out, val_lines)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[preprocess] 划分完成：训练 {len(train_lines)} / 验证 {len(val_lines)}")
    print(f"  train → {train_out}")
    print(f"  val   → {val_out}")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原始数据 → Alpaca JSONL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_convert = sub.add_parser("convert", help="转换为 Alpaca 格式")
    p_convert.add_argument("--input",  "-i", required=True, help="原始数据路径")
    p_convert.add_argument("--output", "-o", required=True, help="输出 JSONL 路径")
    p_convert.add_argument("--format", "-f", choices=["json", "jsonl"], default="jsonl")

    p_split = sub.add_parser("split", help="按比例划分训练/验证集")
    p_split.add_argument("--input",  "-i", required=True, help="JSONL 文件路径")
    p_split.add_argument("--output", "-o", required=True, help="输出目录")
    p_split.add_argument("--ratio",  "-r", type=float, default=0.95)
    p_split.add_argument("--seed",   "-s", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "convert":
        convert_file(args.input, args.output, args.format)
    elif args.cmd == "split":
        split_train_val(args.input, args.output, args.ratio, args.seed)
