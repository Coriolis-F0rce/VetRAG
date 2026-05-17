---
name: convert-gguf
description: 将 HF 格式模型转为 Ollama 可用的 GGUF 格式
disable-model-invocation: true
---

将 `models_merged/` 中的完整 HF 模型转为 GGUF。

用法: 用户说"转换为 GGUF"时触发。

全体转换:
```bash
python scripts/convert_to_gguf.py
```

单模型转换:
```bash
python scripts/convert_to_gguf.py --model <model-name>
```

需要先安装 sentencepiece 和克隆 llama.cpp（脚本会自动处理）。
