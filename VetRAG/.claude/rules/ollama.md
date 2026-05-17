---
paths:
  - "src/rag_interface.py"
  - "src/core/domain_guard.py"
  - "scripts/web_api.py"
  - "scripts/setup_ollama.py"
  - "eval/**"
---

# Ollama 模型管理

- **命名规范**: `vetrag-{base_model}-{variant}`
  - 基础模型: `vetrag-qwen3-0.6b-base`, `vetrag-qwen3-1.7b-base`
  - 微调模型: `vetrag-qwen3-0.6b-vet`, `vetrag-qwen3-0.6b-vet1`, `vetrag-qwen3-1.7b-vet`
- **生成参数**: `temperature=0.0`（greedy），`num_predict=512`，`repeat_penalty=1.2`
- **配置来源**: `src/core/config.py` 中的 `OLLAMA_GENERATOR_MODEL` 和 `OLLAMA_GUARD_MODEL`
- **微调模型上架流程**: LoRA merge → GGUF convert → `ollama create`（不可跳过任何一步）
- **Ollama 不可用时**: DomainGuard 会保守放行所有请求（不回退到 transformers）
