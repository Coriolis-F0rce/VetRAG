"""
Ollama 模型导入脚本 — 将 Modelfiles 导入到本地 Ollama。

用法：
  python scripts/setup_ollama.py              # 导入所有模型
  python scripts/setup_ollama.py --pull-base  # 先从 Ollama 官方库拉取基础模型，再导入微调模型

前提条件：Ollama 已安装并运行中。
         基础模型可通过 --pull-base 拉取，或在 Ollama 中已存在。
"""

import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELFILES_DIR = PROJECT_ROOT / "scripts" / "modelfiles"

# Ollama 模型定义：{ollama 模型名: Modelfile 路径}
OLLAMA_MODELS = {
    "vetrag-qwen3-0.6b-base": MODELFILES_DIR / "Modelfile.qwen3-0.6b-base",
    "vetrag-qwen3-1.7b-base": MODELFILES_DIR / "Modelfile.qwen3-1.7b-base",
    "vetrag-qwen3-0.6b-vet": MODELFILES_DIR / "Modelfile.qwen3-0.6b-vet",
    "vetrag-qwen3-0.6b-vet1": MODELFILES_DIR / "Modelfile.qwen3-0.6b-vet1",
    "vetrag-qwen3-1.7b-vet": MODELFILES_DIR / "Modelfile.qwen3-1.7b-vet",
}

# 需要从 Ollama 官方拉取的基础模型（如果本地没有）
BASE_MODELS = ["qwen3:0.6b", "qwen3:1.7b"]


def run_ollama(*args, check=True):
    """运行 ollama 命令"""
    cmd = ["ollama"] + list(args)
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def list_ollama_models():
    """列出已安装的 Ollama 模型"""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    models = []
    for line in result.stdout.strip().split("\n")[1:]:  # 跳过表头
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def main():
    parser = argparse.ArgumentParser(description="Ollama 模型导入工具")
    parser.add_argument(
        "--pull-base", action="store_true",
        help="从 Ollama 官方库拉取 Qwen3 基础模型"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="只导入指定模型"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要执行的命令，不实际执行"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VetRAG Ollama 模型导入")
    print("=" * 60)

    existing = list_ollama_models()
    print(f"已安装的 Ollama 模型: {existing}")

    # 拉取基础模型
    if args.pull_base:
        for base_model in BASE_MODELS:
            if base_model not in existing:
                print(f"\n[DL] 拉取基础模型: {base_model}")
                if not args.dry_run:
                    try:
                        run_ollama("pull", base_model)
                    except subprocess.CalledProcessError as e:
                        print(f"  [ERR] 拉取失败: {e.stderr}")
            else:
                print(f"  [OK] 已存在: {base_model}")

    # 选择要导入的模型
    if args.model:
        if args.model in OLLAMA_MODELS:
            models_to_create = {args.model: OLLAMA_MODELS[args.model]}
        else:
            print(f"[ERR] 未知模型: {args.model}")
            print(f"   可用: {', '.join(OLLAMA_MODELS.keys())}")
            sys.exit(1)
    else:
        models_to_create = OLLAMA_MODELS

    # 创建模型
    for model_name, modelfile_path in models_to_create.items():
        print(f"\n[PKG] 创建模型: {model_name}")
        print(f"   Modelfile: {modelfile_path}")

        if not modelfile_path.exists():
            print(f"  [WARN]  Modelfile 不存在，跳过")
            continue

        # 对于 GGUF 类型模型，检查 GGUF 文件是否存在
        modelfile_content = modelfile_path.read_text()
        if ".gguf" in modelfile_content and "FROM qwen3:" not in modelfile_content:
            for line in modelfile_content.split("\n"):
                if line.startswith("FROM") and line.endswith(".gguf"):
                    gguf_path = Path(line.replace("FROM ", "").strip())
                    if str(gguf_path).startswith(".."):
                        gguf_path = (modelfile_path.parent / gguf_path).resolve()
                    if not gguf_path.exists():
                        print(f"  [WARN]  GGUF 文件不存在: {gguf_path}")
                        print(f"     请先运行 scripts/convert_to_gguf.py")
                        continue

        if args.dry_run:
            print(f"  [DRY-RUN] ollama create {model_name} -f {modelfile_path}")
            continue

        try:
            run_ollama("create", model_name, "-f", str(modelfile_path))
            print(f"  [OK] 模型创建成功: {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERR] 创建失败: {e.stderr}")

    # 显示最终模型列表
    print(f"\n{'='*60}")
    print("当前 Ollama 模型列表：")
    if not args.dry_run:
        subprocess.run(["ollama", "list"])


if __name__ == "__main__":
    main()
