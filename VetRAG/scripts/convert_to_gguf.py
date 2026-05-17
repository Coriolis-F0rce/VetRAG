"""
GGUF 转换脚本 — 将 HF 格式模型转换为 Ollama 可用的 GGUF 格式。

依赖：需要 llama.cpp 仓库（自动浅克隆到项目根目录）

用法：
  python scripts/convert_to_gguf.py              # 转换所有模型
  python scripts/convert_to_gguf.py --quantize   # 转换 + q4_K_M 量化
  python scripts/convert_to_gguf.py --model qwen3-0.6b-vet-finetuned  # 只转换指定模型

流程：
  1. 使用 llama.cpp/convert_hf_to_gguf.py 将 HF → FP16 GGUF
  2. (可选) 使用 llama.cpp/build/bin/llama-quantize 量化为 q4_K_M
  3. 输出到 models_gguf/ 目录
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
MERGED_DIR = PROJECT_ROOT / "models_merged"
GGUF_DIR = PROJECT_ROOT / "models_gguf"
MODELS_DIR = PROJECT_ROOT / "models"

# 需要转换的模型：{输出名: HF 模型路径}
MODELS_TO_CONVERT = {
    # 基础模型（可直接 ollama pull，这里提供 GGUF 备选方案）
    "qwen3-0.6b-base": MODELS_DIR / "Qwen3-0.6B" / "qwen" / "Qwen3-0___6B",
    "qwen3-1.7b-base": MODELS_DIR / "Qwen3-1.7B",
    # 微调模型（来自合并步骤）
    "qwen3-0.6b-vet-finetuned": MERGED_DIR / "qwen3-0.6b-vet-finetuned",
    "qwen3-0.6b-vet-finetuned1": MERGED_DIR / "qwen3-0.6b-vet-finetuned1",
    "qwen3-1.7b-vet-finetuned": MERGED_DIR / "qwen3-1.7b-vet-finetuned",
}


def ensure_llama_cpp():
    """确保 llama.cpp 仓库可用（浅克隆）"""
    if LLAMA_CPP_DIR.exists():
        print(f"[OK] llama.cpp 已存在: {LLAMA_CPP_DIR}")
        return True

    print("[DL] 克隆 llama.cpp（浅克隆，仅最新 commit）...")
    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/ggerganov/llama.cpp.git",
                str(LLAMA_CPP_DIR),
            ],
            check=True,
        )
        print("[OK] llama.cpp 克隆完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERR] 克隆失败: {e}")
        print("   请手动克隆: git clone --depth 1 https://github.com/ggerganov/llama.cpp.git")
        return False
    except FileNotFoundError:
        print("[ERR] 未找到 git，请先安装 git 或手动克隆 llama.cpp")
        return False


def find_convert_script():
    """查找 convert_hf_to_gguf.py 路径"""
    script_path = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if script_path.exists():
        return script_path
    return None


def find_quantize_binary():
    """查找 llama-quantize 可执行文件"""
    # 常见的构建目录
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",  # Windows
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def convert_single(name: str, hf_path: Path, quantize: bool = False):
    """转换单个模型 HF → GGUF"""
    print(f"\n{'='*60}")
    print(f"转换: {name}")
    print(f"  源路径: {hf_path}")

    if not hf_path.exists():
        print("  [WARN]  跳过: 路径不存在")
        return False

    convert_script = find_convert_script()
    if not convert_script:
        print("  [ERR] 未找到 convert_hf_to_gguf.py")
        return False

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    outfile = GGUF_DIR / f"{name}.gguf"
    tmp_outfile = GGUF_DIR / f"{name}.fp16.gguf"

    # FP16 转换（中间产物，量化后会被替换）
    actual_outfile = tmp_outfile if quantize else outfile

    print("  执行转换 (FP16)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LLAMA_CPP_DIR)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    try:
        result = subprocess.run(
            [
                sys.executable, str(convert_script),
                str(hf_path),
                "--outfile", str(actual_outfile),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分钟超时
        )
        if result.returncode != 0:
            print(f"  [ERR] 转换失败:\n{result.stderr[-500:]}")
            return False
        print(f"  [OK] FP16 GGUF 已生成: {actual_outfile}")
    except subprocess.TimeoutExpired:
        print("  [ERR] 转换超时")
        return False

    # 量化步骤
    if quantize:
        quantize_bin = find_quantize_binary()
        if not quantize_bin:
            print("  [WARN]  未找到 llama-quantize，跳过量化，保留 FP16 版本")
            # 重命名 FP16 文件为最终文件
            tmp_outfile.rename(outfile)
            return True

        quant_type = "q4_K_M"
        print(f"  量化中 ({quant_type})...")
        try:
            result = subprocess.run(
                [str(quantize_bin), str(tmp_outfile), str(outfile), quant_type],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode == 0:
                print(f"  [OK] 量化完成: {outfile}")
                tmp_outfile.unlink()  # 删除中间 FP16 文件
                return True
            else:
                print(f"  [WARN]  量化失败，保留 FP16 版本\n{result.stderr[-300:]}")
                tmp_outfile.rename(outfile)
                return True  # FP16 也算成功
        except subprocess.TimeoutExpired:
            print("  [WARN]  量化超时，保留 FP16 版本")
            tmp_outfile.rename(outfile)
            return True

    return True


def main():
    parser = argparse.ArgumentParser(description="HF 模型 → GGUF 转换工具")
    parser.add_argument(
        "--quantize", action="store_true",
        help="转换为 q4_K_M 量化 GGUF（需要编译 llama.cpp）"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="只转换指定模型（名称来自 MODELS_TO_CONVERT 的 key）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HF → GGUF 转换工具")
    print(f"输出目录: {GGUF_DIR}")
    if args.quantize:
        print("量化: q4_K_M")
    print("=" * 60)

    # 确保 llama.cpp 可用
    if not ensure_llama_cpp():
        sys.exit(1)

    # 筛选要转换的模型
    if args.model:
        if args.model in MODELS_TO_CONVERT:
            models = {args.model: MODELS_TO_CONVERT[args.model]}
        else:
            print(f"[ERR] 未知模型: {args.model}")
            print(f"   可用: {', '.join(MODELS_TO_CONVERT.keys())}")
            sys.exit(1)
    else:
        models = MODELS_TO_CONVERT

    # 转换
    success = 0
    failed = 0
    for name, hf_path in models.items():
        if convert_single(name, hf_path, quantize=args.quantize):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"转换完成: {success} 成功, {failed} 失败")
    if success > 0:
        print("输出文件：")
        for gguf_file in sorted(GGUF_DIR.glob("*.gguf")):
            size_mb = gguf_file.stat().st_size / (1024 * 1024)
            print(f"  {gguf_file.name}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
