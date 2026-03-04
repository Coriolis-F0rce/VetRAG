import torch
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ---------- 路径定义 ----------
MODEL_PATH = Path(r"D:\Backup\PythonProject2\VetRAG\models\Qwen3-0.6B\qwen\Qwen3-0___6B").resolve()
OUTPUT_DIR = Path(r"D:/Backup/PythonProject2/VetRAG/models_finetuned/qwen3-finetuned1").resolve()

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "datas"

# ---------- 超参数 ----------
EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACC = 8
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.15
MAX_SEQ_LEN = 1024
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 50
warmup_ratio = 0.1
weight_decay = 0.02

# ---------- 量化配置 ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---------- 加载模型和 tokenizer ----------
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    config=config,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# ---------- LoRA 配置 ----------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------- 数据集预处理 ----------
def preprocess_function(examples):
    assistant_start = "<|im_start|>assistant\n"

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for text in examples["text"]:
        asst_pos = text.find(assistant_start)
        if asst_pos == -1:
            tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            labels = input_ids.copy()
        else:
            user_part = text[:asst_pos]
            assistant_part = text[asst_pos:]

            user_tokens = tokenizer(user_part, add_special_tokens=False, truncation=False)["input_ids"]
            assistant_tokens = tokenizer(assistant_part, add_special_tokens=False, truncation=False)["input_ids"]

            total_len = len(user_tokens) + len(assistant_tokens)
            if total_len > MAX_SEQ_LEN:
                excess = total_len - MAX_SEQ_LEN
                assistant_tokens = assistant_tokens[:len(assistant_tokens) - excess]

            input_ids = user_tokens + assistant_tokens
            attention_mask = [1] * len(input_ids)
            labels = ([-100] * len(user_tokens)) + assistant_tokens

            if len(input_ids) > MAX_SEQ_LEN:
                input_ids = input_ids[:MAX_SEQ_LEN]
                labels = labels[:MAX_SEQ_LEN]
                attention_mask = attention_mask[:MAX_SEQ_LEN]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    }


raw_train_dataset = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")
raw_eval_dataset = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"), split="train")

train_dataset = raw_train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_train_dataset.column_names
)
eval_dataset = raw_eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_eval_dataset.column_names
)

# ---------- 训练参数配置 ----------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_length=MAX_SEQ_LEN,
    packing=False,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    weight_decay=0.02,
    save_total_limit=2,
)

# ---------- 创建 Trainer ----------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)