# VetRAG — 兽医 RAG 问答系统

> 开发工作流详见 [workflow.md](workflow.md)

## 技术栈
- **语言**: Python 3.11+
- **Web**: FastAPI + uvicorn (端口 5002)
- **LLM 推理**: Ollama（模型名 `vetrag-qwen3-*`）
- **向量库**: ChromaDB (持久化到 `chroma_db/`)
- **Embedding**: BAAI/bge-large-zh-v1.5 (sentence-transformers)
- **检索**: 混合检索 Dense + BM25 (RRF 融合)
- **领域守卫**: Ollama 零样本分类（基础模型）

## 常用命令

```bash
# 启动服务
python scripts/web_api.py

# 测试
python -m pytest tests/ -v

# 单独测试某文件
python -m pytest tests/api/test_web_api.py -v

# Lint (仅检查，不修改)
ruff check .

# 模型管理
python scripts/merge_lora.py          # LoRA adapter → 完整权重
python scripts/convert_to_gguf.py     # HF → GGUF
python scripts/setup_ollama.py        # 导入模型到 Ollama
```

## 目录结构

```
VetRAG/
├── src/                    # 核心源码
│   ├── rag_interface.py    # QwenGenerator + RAGInterface
│   └── core/
│       ├── config.py       # 集中配置（环境变量 + 默认值）
│       └── domain_guard.py # 领域守卫
├── scripts/                # 入口脚本 & 工具
│   ├── web_api.py          # FastAPI 服务
│   ├── merge_lora.py       # LoRA 合并
│   ├── convert_to_gguf.py  # GGUF 转换
│   └── setup_ollama.py     # Ollama 模型导入
├── eval/                   # A/B 实验 & 评分
├── models/                 # HF 基础模型（不提交 Git）
├── models_finetuned/       # LoRA adapter（不提交 Git）
├── models_merged/          # 合并后完整权重（不提交 Git）
├── models_gguf/            # GGUF 文件（不提交 Git）
├── static/                 # 前端 UI
└── data/                   # 知识库 JSON
```

## 架构约束

- **QwenGenerator** (src/rag_interface.py) 是唯一的 LLM 调用入口，通过 Ollama API
- **DomainGuard** (src/core/domain_guard.py) 独立调用 Ollama，不依赖 QwenGenerator
- **配置** 统一从 `src/core/config.py` 读取，支持环境变量覆盖
- **Embedding 模型** 不走 Ollama，直接使用 sentence-transformers 加载
- **微调模型** 必须先 `merge_lora → convert_gguf → ollama create` 才能在 Ollama 中使用

## 红线

- 不可修改 `data/` 下的 JSON 知识库文件（除非明确要求添加数据）
- 不可提交 `.env`、`models*/`、`chroma_db/`、`logs/`
- 不可在答案生成的清洗逻辑中移除领域相关判断
- Ollama 模型命名必须遵循 `vetrag-{base}-{variant}` 格式
- 修改 `src/core/config.py` 后必须同步更新对应的 `.env.example` 说明

## Git 工作流

- **Commit 格式**: `<type>: <简短描述>`（如 `fix:`, `feat:`, `refactor:`, `chore:`）
- **分支**: `main` 稳定分支，功能分支命名 `feat/<描述>`
- **提交前**: 确保 `ruff check .` 无报错
