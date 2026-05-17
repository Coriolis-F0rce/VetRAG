---
name: merge-lora
description: 将 LoRA adapter 合并到基础模型，输出完整 HF 权重
disable-model-invocation: true
---

合并位于 `models_finetuned/` 的 LoRA adapter 到基础模型。

用法: 用户说"合并 LoRA"时触发。

执行:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/merge_lora.py
```

输出到 `models_merged/` 目录。
