"""
QLoRA 微调脚本 — Qwen3-1.7B
平台：AutoDL（ModelScope 下载源）

流程：
  1. 下载模型（若未下载）
  2. 加载 tokenizer + 4bit 量化模型
  3. 应用 LoRA adapter
  4. 在 alpaca 格式数据集上微调
  5. 保存 adapter 权重
"""
import os
import sys
import time
import torch
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# ★★★  必改配置区  ★★★
# ─────────────────────────────────────────────────────────────────────────────

# 模型来源：AutoDL ModelScope 缓存目录
# （本地调试时改为本地路径，如 r"D:\Backup\PythonProject2\VetRAG\models\Qwen3-1.7B"）
MODEL_PATH = "/root/autodl-tmp/huggingface/models/Qwen3-1.7B"

# 输出目录
OUTPUT_DIR = "/root/autodl-tmp/huggingface/models/qwen3-1.7b-vet-finetuned"

# 训练数据（Alpaca JSONL 格式）
DATA_DIR = Path(__file__).parent.resolve() / "datas"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE   = DATA_DIR / "val.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# ★★★  训练后自动化配置  ★★★
# ─────────────────────────────────────────────────────────────────────────────

# 训练结束后是否自动上传 HuggingFace 并关机
AUTO_UPLOAD_AFTER_TRAIN = False  # 设为 True 则训练结束后自动合并+上传+关机
HF_REPO_ID = "MrK-means/Qwen3-1.7B-VetRAG"   # HF 仓库名（修改为你的用户名）
MERGED_OUTPUT_DIR = "/root/autodl-tmp/output/Qwen3-1.7B-vetrag-merged"
AUTO_SHUTDOWN_AFTER_UPLOAD = False  # 设为 True 则上传后自动关机（默认 False，防止误操作）

# ─────────────────────────────────────────────────────────────────────────────
# ★★★  超参数  ★★★
# ─────────────────────────────────────────────────────────────────────────────

# 训练
EPOCHS            = 3
BATCH_SIZE         = 8          # 32GB vGPU: 8 / RTX 3090-24G: 4
GRAD_ACC          = 4           # 有效 batch = BATCH_SIZE * GRAD_ACC = 32
LEARNING_RATE     = 2e-4
WARMUP_RATIO      = 0.03
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY      = 0.01

# LoRA
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT  = 0.05

# 数据
MAX_SEQ_LEN   = 2048           # Qwen3-1.7B 上下文窗口
USE_PADDING   = True           # Qwen 系列需要显式 pad

# 日志 / 保存
LOGGING_STEPS = 10
SAVE_STEPS    = 200
EVAL_STEPS    = 200
SAVE_TOTAL_LIMIT = 2

# ─────────────────────────────────────────────────────────────────────────────
# 量化配置
# ─────────────────────────────────────────────────────────────────────────────

def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # nf4 精度优于 fp4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 加载模型 & Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_model_tokenizer(model_path: str):
    print(f"[加载] tokenizer ← {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        use_fast=False,
    )
    # Qwen 系列 EOS 即 PAD
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("[Tokenizer] pad_token 已设为 eos_token")

    print(f"[加载] 模型（4bit NF4）← {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        quantization_config=build_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 应用 LoRA
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 预处理：构建 instruction-tuning labels
# 只对 assistant 回复部分计算 loss，用户部分 mask 为 -100
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_qwen3(examples, tokenizer, max_length: int = MAX_SEQ_LEN):
    """
    将 Alpaca text 转换为 causal LM 格式。
    assistant 之前的 token labels = -100（不贡献梯度）。
    """
    ASSISTANT_MARKER = "\n### Response:\n"

    all_input_ids   = []
    all_labels     = []

    for text in examples["text"]:
        if ASSISTANT_MARKER not in text:
            # 无 assistant 标记，全部计算 loss
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            all_input_ids.append(tokens["input_ids"])
            all_labels.append(tokens["input_ids"])
        else:
            user_part  = text.split(ASSISTANT_MARKER)[0] + ASSISTANT_MARKER
            asst_part  = text.split(ASSISTANT_MARKER)[1]

            user_ids  = tokenizer(user_part,  add_special_tokens=True, truncation=False)["input_ids"]
            asst_ids  = tokenizer(asst_part, add_special_tokens=False, truncation=False)["input_ids"]

            total = len(user_ids) + len(asst_ids)
            if total > max_length:
                excess = total - max_length
                if excess < len(asst_ids):
                    asst_ids = asst_ids[:-excess]
                else:
                    user_ids = user_ids[:max_length // 2]
                    asst_ids = asst_ids[:max_length - len(user_ids)]

            input_ids = user_ids + asst_ids
            labels    = [-100] * len(user_ids) + asst_ids

            all_input_ids.append(input_ids)
            all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "labels":    all_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 训练后自动化：合并 LoRA + 上传 HF + 关机
# ─────────────────────────────────────────────────────────────────────────────

def _post_train_automation(tokenizer):
    from peft import PeftModel

    print("\n" + "=" * 60)
    print("  训练后自动化流程开始")
    print("=" * 60)

    # 1. 合并 LoRA → 完整模型
    print(f"\n[1/4] 加载基础模型 ← {MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"[2/4] 合并 LoRA adapter ← {OUTPUT_DIR}")
    merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    merged_model = merged_model.merge_and_unload()
    print("  ✅ 合并完成")

    # 2. 保存合并模型
    print(f"\n[3/4] 保存合并模型 → {MERGED_OUTPUT_DIR}")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    print("  ✅ 保存完成")

    # 3. 上传至 HF
    print(f"\n[4/4] 上传至 HuggingFace → {HF_REPO_ID}")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            repo_id=HF_REPO_ID,
            folder_path=MERGED_OUTPUT_DIR,
            repo_type="model",
            commit_message="Upload merged Qwen3-1.7B VetRAG fine-tuned model",
        )
        print(f"  ✅ 上传完成！https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"  ⚠️  上传失败: {e}")
        print("  模型已保存在本地，上传失败不影响后续。请手动上传：")
        print(f"  python upload_to_hf.py")
        return  # 上传失败不关机，给手动补救机会

    # 4. 关机
    if AUTO_SHUTDOWN_AFTER_UPLOAD:
        print("\n🛑 上传完成，5 秒后关机 ...")
        time.sleep(5)
        subprocess.run(["poweroff"], check=True)
    else:
        print("\n🟢 上传完成，实例保持运行。手动关机：poweroff")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 模型 & tokenizer
    model, tokenizer = load_model_tokenizer(MODEL_PATH)

    # 2. LoRA
    model = apply_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)

    # 3. 数据集
    print(f"[数据] 加载训练集 ← {TRAIN_FILE}")
    raw_train = load_dataset("json", data_files=str(TRAIN_FILE), split="train")

    print(f"[数据] 加载验证集 ← {VAL_FILE}")
    raw_val = load_dataset("json", data_files=str(VAL_FILE), split="train")

    def _preprocess(examples):
        return preprocess_qwen3(examples, tokenizer)

    train_dataset = raw_train.map(
        _preprocess,
        batched=True,
        remove_columns=raw_train.column_names,
        desc="预处理训练集",
    )
    val_dataset = raw_val.map(
        _preprocess,
        batched=True,
        remove_columns=raw_val.column_names,
        desc="预处理验证集",
    )

    # 4. 训练参数
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        tf32=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="paged_adamw_32bit",
        max_length=MAX_SEQ_LEN,
        packing=False,
        gradient_checkpointing=True,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("=" * 50)
    print(f"  显卡型号   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  有效 batch : {BATCH_SIZE} × {GRAD_ACC} = {BATCH_SIZE * GRAD_ACC}")
    print(f"  总训练步数 : {len(train_dataset) * EPOCHS // (BATCH_SIZE * GRAD_ACC)}")
    print("=" * 50)

    print("\n🚀 开始 QLoRA 微调 ...")
    trainer.train()

    # 6. 保存
    print(f"\n💾 保存模型 ← {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ 微调完成！")

    # ── 7. 训练后自动化：合并 LoRA + 上传 HF + 关机 ──────────────────────────
    if AUTO_UPLOAD_AFTER_TRAIN:
        _post_train_automation(tokenizer)


if __name__ == "__main__":
    main()
