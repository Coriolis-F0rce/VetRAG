"""
LoRA 合并脚本 — 将 LoRA adapter 与基础模型合并，输出完整 HF 格式权重。

用法：python scripts/merge_lora.py

输出到 models_merged/ 目录：
  - qwen3-0.6b-vet-finetuned      (基于 Qwen3-0.6B, r=8)
  - qwen3-0.6b-vet-finetuned1     (基于 Qwen3-0.6B, r=8)
  - qwen3-1.7b-vet-finetuned      (基于 Qwen3-1.7B, r=16)
"""

import os
import sys
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FINETUNED_DIR = PROJECT_ROOT / "models_finetuned"
MERGED_DIR = PROJECT_ROOT / "models_merged"

# 需要合并的模型：{输出名: (adapter路径, 基础模型路径)}
MODELS_TO_MERGE = {
    "qwen3-0.6b-vet-finetuned": (
        FINETUNED_DIR / "qwen3-finetuned",
        MODELS_DIR / "Qwen3-0.6B" / "qwen" / "Qwen3-0___6B",
    ),
    "qwen3-0.6b-vet-finetuned1": (
        FINETUNED_DIR / "qwen3-finetuned1",
        MODELS_DIR / "Qwen3-0.6B" / "qwen" / "Qwen3-0___6B",
    ),
    "qwen3-1.7b-vet-finetuned": (
        FINETUNED_DIR / "qwen3-1.7b-vet-finetuned",
        MODELS_DIR / "Qwen3-1.7B",
    ),
}


def merge_single(adapter_path: Path, base_model_path: Path, output_path: Path):
    """合并单个 LoRA adapter 到基础模型并保存"""

    print(f"\n{'='*60}")
    print(f"合并: {adapter_path.name}")
    print(f"  基础模型: {base_model_path}")
    print(f"  输出路径: {output_path}")

    # 1. 加载基础模型
    print("  加载基础模型...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",  # 在 CPU 上合并，避免 GPU OOM
    )

    # 2. 加载 LoRA adapter 并合并
    print("  加载 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    print("  合并权重 (merge_and_unload)...")
    merged = model.merge_and_unload()

    # 3. 保存合并后的模型
    print("  保存合并后模型...")
    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path), safe_serialization=True)

    # 4. 复制 tokenizer 文件
    print("  复制 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_path))

    # 5. 写入标记文件（表示这是合并后的完整模型）
    (output_path / "MERGE_INFO.txt").write_text(
        f"Merged LoRA adapter\n"
        f"  Adapter: {adapter_path}\n"
        f"  Base:    {base_model_path}\n"
    )

    print(f"  [OK] 完成，输出: {output_path}")
    del base_model, model, merged
    torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("LoRA → 完整权重 合并工具")
    print(f"输出目录: {MERGED_DIR}")
    print("=" * 60)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    for name, (adapter_path, base_model_path) in MODELS_TO_MERGE.items():
        output_path = MERGED_DIR / name

        # 检查 adapter 是否存在
        if not adapter_path.exists():
            print(f"\n[WARN]  跳过 {name}: adapter 路径不存在 {adapter_path}")
            continue

        # 检查是否为 LoRA adapter
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.exists():
            print(f"\n[WARN]  跳过 {name}: 未找到 adapter_config.json，可能已是完整模型")
            continue

        # 检查基础模型是否存在
        if not base_model_path.exists():
            print(f"\n[ERR] 跳过 {name}: 基础模型路径不存在 {base_model_path}")
            continue

        merge_single(adapter_path, base_model_path, output_path)

    print(f"\n{'='*60}")
    print("全部合并完成。输出文件：")
    for p in sorted(MERGED_DIR.glob("*")):
        if p.is_dir():
            print(f"  {p}/")


if __name__ == "__main__":
    main()
