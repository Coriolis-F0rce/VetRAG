"""
QLoRA 微调脚本 - 基于 trl + peft + transformers
适配 Llama-3.2-1B-Instruct 模型（GGUF Q4 量化版本）
"""
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ─── 日志配置 ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─── 配置类 ─────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_path: str = "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"   # ★ 模型路径（GGUF）
    tokenizer_path: Optional[str] = None                            # ★ 可单独指定 tokenizer 路径
    model_revision: str = "main"                                    # GGUF 文件的 revision
    trust_remote_code: bool = True


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"   # bfloat16 / float16 / float32
    bnb_4bit_quant_type: str = "nf4"           # nf4 / fp4
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraConfig_:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"          # none / all / lora_only
    task_type: str = "CAUSAL_LM"
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class DataConfig:
    data_path: str = "data/alpaca_formatted.jsonl"   # ★ 训练数据路径
    dataset_text_field: str = "text"
    max_seq_length: int = 2048


# ─── 量化配置 ───────────────────────────────────────────────────────────────
def build_quantization_config(cfg: QuantizationConfig) -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


# ─── 加载模型 & Tokenizer ───────────────────────────────────────────────────
def load_model_and_tokenizer(
    model_cfg: ModelConfig,
    quant_cfg: QuantizationConfig,
):
    log.info(f"正在加载 Tokenizer: {model_cfg.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_path,
        trust_remote_code=model_cfg.trust_remote_code,
        use_fast=False,
    )
    # Llama 系列强制添加 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("已设置 pad_token = eos_token")

    log.info(f"正在加载模型（4bit 量化）: {model_cfg.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_path,
        quantization_config=build_quantization_config(quant_cfg),
        device_map="auto",
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # QLoRA 训练前准备
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ─── 构建 PEFT 模型 ─────────────────────────────────────────────────────────
def build_peft_model(model, lora_cfg: LoraConfig_):
    log.info("应用 LoRA adapter ...")
    lora_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=lora_cfg.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─── 加载数据集 ──────────────────────────────────────────────────────────────
def load_dataset_(data_cfg: DataConfig):
    log.info(f"加载数据集: {data_cfg.data_path}")
    dataset = load_dataset("json", data_files=data_cfg.data_path, split="train")
    log.info(f"样本数量: {len(dataset)}")
    return dataset


# ─── 训练 ───────────────────────────────────────────────────────────────────
def main():
    # ★★★ 运行时配置（按需修改） ★★★
    model_cfg    = ModelConfig()
    quant_cfg    = QuantizationConfig()
    lora_cfg     = LoraConfig_()
    data_cfg     = DataConfig()

    # 训练超参数（TrainingArguments）
    output_dir = "finetune_steps/checkpoints"
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    num_train_epochs = 3
    learning_rate = 2e-4
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    logging_steps = 10
    save_steps = 100
    save_total_limit = 2

    # 加载
    model, tokenizer = load_model_and_tokenizer(model_cfg, quant_cfg)
    model = build_peft_model(model, lora_cfg)
    dataset = load_dataset_(data_cfg)

    # SFTTrainer 自动处理多轮对话格式
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_dir="finetune_steps/logs",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=True,
        tf32=True,
        optim="paged_adamw_32bit",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field=data_cfg.dataset_text_field,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        dataset_text_field=data_cfg.dataset_text_field,
        packing=False,       # 短样本packing可开 True 加速，但会丢失样本级日志
    )

    log.info("开始 QLoRA 微调 ...")
    trainer.train()

    # 保存 adapter
    log.info(f"保存 adapter → {output_dir}/final_adapter")
    trainer.save_model(f"{output_dir}/final_adapter")
    tokenizer.save_pretrained(f"{output_dir}/final_adapter")
    log.info("✅ 微调完成！")


if __name__ == "__main__":
    main()
