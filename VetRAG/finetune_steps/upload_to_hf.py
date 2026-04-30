"""
模型上传脚本 — 合并 LoRA adapter 后上传至 HuggingFace Hub
上传完成后可本地下载：git clone https://huggingface.co/<你的用户名>/Qwen3-1.7B-VetRAG

用法：
  # 1. 先在 autodl 上登录 HF（只需一次）
  pip install huggingface_hub
  huggingface-cli login
  # 输入你的 HF token（https://huggingface.co/settings/tokens 创建）

  # 2. 训练结束后执行上传
  python upload_to_hf.py

  # 3. 本地下载（PowerShell / cmd）
  git clone https://huggingface.co/<你的用户名>/Qwen3-1.7B-VetRAG
  # 或
  huggingface-cli download <你的用户名>/Qwen3-1.7B-VetRAG
"""
import os
import torch
from pathlib import Path
from huggingface_hub import HfApi, login

# ─────────────────────────────────────────────────────────────────────────────
# ★★★  必改配置区  ★★★
# ─────────────────────────────────────────────────────────────────────────────

# 合并后的模型保存路径（与 finetune_qlora.py 的 OUTPUT_DIR 对应）
MERGED_MODEL_PATH = "/root/autodl-tmp/output/Qwen3-1.7B-vetrag-merged"

# 基础模型路径（用于合并）
BASE_MODEL_PATH = "/root/autodl-tmp/huggingface/models/Qwen3-1.7B"

# LoRA adapter 路径
ADAPTER_PATH = "/root/autodl-tmp/huggingface/models/qwen3-1.7b-vet-finetuned"

# HuggingFace 仓库名（改成你自己的用户名）
# 格式：<你的HF用户名>/Qwen3-1.7B-VetRAG
HF_REPO_ID = "MrK-means/Qwen3-1.7B-VetRAG"

# HF token（建议通过 huggingface-cli login 登录，或直接设置环境变量 HF_TOKEN）
# 不填则读取 HF_TOKEN 环境变量
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─────────────────────────────────────────────────────────────────────────────
# 检查
# ─────────────────────────────────────────────────────────────────────────────

def check_paths():
    errors = []
    if not Path(BASE_MODEL_PATH).exists():
        errors.append(f"❌ 基础模型路径不存在: {BASE_MODEL_PATH}")
    if not Path(ADAPTER_PATH).exists():
        errors.append(f"❌ LoRA adapter 路径不存在: {ADAPTER_PATH}")
    if HF_REPO_ID == "YOUR_HF_USERNAME/Qwen3-1.7B-VetRAG":
        errors.append("❌ 请先修改 HF_REPO_ID 为你的 HF 用户名，例如：yuqi58249/Qwen3-1.7B-VetRAG")
    if not HF_TOKEN and not os.environ.get("HF_TOKEN"):
        errors.append("⚠️  未检测到 HF_TOKEN，请先运行 `huggingface-cli login` 或设置环境变量")
    if errors:
        for e in errors:
            print(e)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 合并 LoRA → 完整模型
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_upload():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("=" * 60)
    print("  HuggingFace 模型上传流程")
    print("=" * 60)

    if not check_paths():
        print("\n请修复上述错误后重试。")
        return

    # 登录 HF
    if HF_TOKEN:
        print(f"\n[登录] HuggingFace ...")
        login(token=HF_TOKEN)
    else:
        print("[登录] 使用缓存的 HF 凭证 ...")

    # 1. 加载基础模型
    print(f"\n[1/4] 加载基础模型 ← {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 合并在 CPU 上做，避免显存爆炸
        trust_remote_code=True,
    )

    # 2. 加载并合并 LoRA
    print(f"\n[2/4] 合并 LoRA adapter ← {ADAPTER_PATH}")
    merged_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("  合并中（.merge_and_unload()）...")
    merged_model = merged_model.merge_and_unload()
    print("  ✅ 合并完成")

    # 3. 保存合并后模型
    print(f"\n[3/4] 保存合并模型 → {MERGED_MODEL_PATH}")
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    AutoTokenizer.from_pretrained(BASE_MODEL_PATH).save_pretrained(MERGED_MODEL_PATH)
    print("  ✅ 模型保存完成")

    # 4. 上传至 HF
    print(f"\n[4/4] 上传至 HuggingFace → {HF_REPO_ID}")
    api = HfApi()
    api.upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=MERGED_MODEL_PATH,
        repo_type="model",
        commit_message="Upload merged Qwen3-1.7B VetRAG fine-tuned model",
    )
    print(f"\n{'=' * 60}")
    print(f"  ✅ 上传完成！")
    print(f"  仓库地址：https://huggingface.co/{HF_REPO_ID}")
    print(f"{'=' * 60}")
    print(f"\n本地下载命令：")
    print(f"  git clone https://huggingface.co/{HF_REPO_ID}")
    print(f"\n或使用 huggingface-cli：")
    print(f"  huggingface-cli download {HF_REPO_ID}")


if __name__ == "__main__":
    merge_and_upload()
