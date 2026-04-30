#!/usr/bin/env python3
"""
S1: 合并去重脚本
合并所有增强数据源，进行规则过滤和质量统计。
"""

import os
import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Any


BASE_DIR = Path(r"D:\Backup\PythonProject2\data_process\find_faq")
OUTPUT_DIR = BASE_DIR / "merged_output"
INPUT_DIRS = [
    BASE_DIR / "augmented_output",
    BASE_DIR / "new_augmented_output",
]

# 规则配置
MIN_INSTRUCTION_LEN = 5
MAX_INSTRUCTION_LEN = 300
MIN_OUTPUT_LEN = 10
MAX_OUTPUT_LEN = 2000
MIN_WORDS = 2  # 最少词数


def compute_hash(item: Dict) -> str:
    """基于 instruction + output 计算哈希，用于去重"""
    text = (item.get("instruction", "") + "|" + item.get("output", "")).strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def is_valid_entry(item: Dict) -> bool:
    """规则过滤：检查条目是否有效"""
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()

    # 长度过滤
    if len(instruction) < MIN_INSTRUCTION_LEN or len(instruction) > MAX_INSTRUCTION_LEN:
        return False
    if len(output) < MIN_OUTPUT_LEN or len(output) > MAX_OUTPUT_LEN:
        return False

    # 词数过滤
    if len(instruction) < MIN_WORDS:
        return False

    # 过滤纯空白/无意义内容
    if not instruction or not output:
        return False

    # 过滤纯标点符号
    if not re.search(r"[\u4e00-\u9fff]|[a-zA-Z]", instruction):
        return False

    # 过滤明显重复字符（超过50%相同字符）
    if len(instruction) > 5:
        char_counts = defaultdict(int)
        for c in instruction:
            char_counts[c] += 1
        max_ratio = max(char_counts.values()) / len(instruction)
        if max_ratio > 0.7:
            return False

    return True


def normalize(item: Dict) -> Dict:
    """标准化条目格式，保留 metadata"""
    return {
        "instruction": item.get("instruction", "").strip(),
        "input": item.get("input", "").strip(),
        "output": item.get("output", "").strip(),
        "metadata": item.get("metadata", {})
    }


def load_json_file(file_path: Path) -> List[Dict]:
    """加载单个 JSON 文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except Exception as e:
        print(f"  [!] Load failed {file_path.name}: {e}")
        return []


def collect_all_entries() -> List[Dict]:
    """从所有输入目录收集条目"""
    all_entries = []
    for input_dir in INPUT_DIRS:
        if not input_dir.exists():
            print(f"  [!] Dir not found: {input_dir}")
            continue
        json_files = list(input_dir.glob("*.json"))
        print(f"\n[*] {input_dir.name}:")
        print(f"   Found {len(json_files)} JSON files")
        for jf in sorted(json_files):
            entries = load_json_file(jf)
            normalized = [normalize(e) for e in entries]
            valid_cnt = sum(1 for e in normalized if is_valid_entry(e))
            print(f"   {jf.name}: {len(entries)} entries -> {valid_cnt} valid")
            all_entries.extend(normalized)
    return all_entries


def dedup_by_hash(entries: List[Dict]) -> List[Dict]:
    """基于哈希去重"""
    seen: Set[str] = set()
    unique: List[Dict] = []
    dup_count = 0

    for entry in entries:
        h = compute_hash(entry)
        if h not in seen:
            seen.add(h)
            unique.append(entry)
        else:
            dup_count += 1

    return unique, dup_count


def dedup_by_instruction(entries: List[Dict]) -> List[Dict]:
    """基于 instruction 相似度去重（简化版：完全相同则去重）"""
    seen_instructions: Set[str] = set()
    unique: List[Dict] = []
    dup_count = 0

    for entry in entries:
        inst = entry["instruction"].strip()
        if inst not in seen_instructions:
            seen_instructions.add(inst)
            unique.append(entry)
        else:
            dup_count += 1

    return unique, dup_count


def statistics(entries: List[Dict], title: str = "") -> None:
    """Output statistics"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"  Total entries: {len(entries)}")

    if not entries:
        return

    # 长度分布
    inst_lens = [len(e["instruction"]) for e in entries]
    out_lens = [len(e["output"]) for e in entries]
    print(f"\n  Question length (instruction):")
    print(f"    Avg: {sum(inst_lens)/len(inst_lens):.1f}, "
          f"Min: {min(inst_lens)}, Max: {max(inst_lens)}")
    print(f"  Answer length (output):")
    print(f"    Avg: {sum(out_lens)/len(out_lens):.1f}, "
          f"Min: {min(out_lens)}, Max: {max(out_lens)}")

    # 分类统计
    categories = defaultdict(int)
    for e in entries:
        meta = e.get("metadata", {})
        cat = meta.get("category", "unknown")
        categories[cat] += 1

    print(f"\n  Category distribution:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        pct = cnt / len(entries) * 100
        print(f"    {cat}: {cnt} ({pct:.1f}%)")


def save_output(entries: List[Dict], output_path: Path, name: str) -> None:
    """保存输出文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] Saved {name}: {len(entries)} entries -> {output_path}")


def main():
    print("=" * 60)
    print("S1: Merge and Deduplicate Script")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 收集所有条目
    print("\n[Step 1] Collecting all augmented data...")
    all_entries = collect_all_entries()
    statistics(all_entries, "Step 1 - Raw Data")

    # 2. 规则过滤
    print("\n[Step 2] Rule filtering...")
    valid_entries = [e for e in all_entries if is_valid_entry(e)]
    filtered_count = len(all_entries) - len(valid_entries)
    print(f"  Filtered out {filtered_count} invalid entries")
    print(f"  Remaining {len(valid_entries)} entries")
    statistics(valid_entries, "Step 2 - After Filter")

    # 3. 哈希去重
    print("\n[Step 3] Hash dedup (exact instruction|output)...")
    unique_by_hash, dup_hash = dedup_by_hash(valid_entries)
    print(f"  Removed {dup_hash} exact duplicates")
    print(f"  Remaining {len(unique_by_hash)} entries")
    statistics(unique_by_hash, "Step 3 - After Hash Dedup")

    # 4. instruction 去重
    print("\n[Step 4] Instruction dedup...")
    unique_by_inst, dup_inst = dedup_by_instruction(unique_by_hash)
    print(f"  Removed {dup_inst} instruction duplicates")
    print(f"  Remaining {len(unique_by_inst)} entries")
    statistics(unique_by_inst, "Step 4 - After Instruction Dedup")

    # 5. 保存结果
    print("\n[Step 5] Saving results...")

    # Full version
    save_output(unique_by_inst, OUTPUT_DIR / "s1_merged_all.json", "full")

    # 按分类拆分
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for entry in unique_by_inst:
        cat = entry.get("metadata", {}).get("category", "unknown")
        by_category[cat].append(entry)

    for cat, items in sorted(by_category.items()):
        cat_file = f"s1_merged_{cat}.json"
        save_output(items, OUTPUT_DIR / cat_file, cat)

    # 统计摘要
    summary = {
        "step": "S1 - 合并去重",
        "raw_total": len(all_entries),
        "after_filter": len(valid_entries),
        "after_hash_dedup": len(unique_by_hash),
        "after_inst_dedup": len(unique_by_inst),
        "removed_by_filter": filtered_count,
        "removed_by_hash_dup": dup_hash,
        "removed_by_inst_dup": dup_inst,
        "by_category": {cat: len(items) for cat, items in sorted(by_category.items())}
    }
    with open(OUTPUT_DIR / "s1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[OK] S1 Done. Total valid entries: {len(unique_by_inst)}")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"   Main output: s1_merged_all.json ({len(unique_by_inst)} entries)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
