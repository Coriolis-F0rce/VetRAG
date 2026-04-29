# VetRAG - 兽医领域 RAG 问答系统

基于本地 Qwen3 模型与 ChromaDB 向量检索的兽医知识问答系统，支持流式输出。

## 项目简介

VetRAG 是一个专为兽医领域设计的检索增强生成（RAG）系统，旨在帮助宠物主人和兽医从业者快速获取准确的宠物医疗和护理知识。

**核心功能：**
- 知识库问答：基于本地向量数据库，精准检索相关文档并生成回答
- 流式输出：SSE 实时流式响应，体验流畅
- 本地推理：无需云端 API，所有处理在本地完成
- 支持模型：Qwen3-0.6B（支持微调版本）、BGE 中文向量化模型

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         启动方式                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python web_api.py     →  Web 服务（http://localhost:8000）      │
│  python run.py         →  命令行问答工具                        │
│  python build_index.py  →  仅构建向量索引                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

                           知识入库流程
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  data/*.json ──→ VetRAGDataLoader ──→ 语义分块 ──→ BGE 向量化   │
│                              │                                   │
│                         内容类型派发                               │
│  behaviors ── breeds ── cares ── diseases ── surgeries         │
│                              │                                   │
│                              ▼                                   │
│                      ChromaDB 持久化存储                          │
│                         (chroma_db/)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

                           查询流程
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  用户问题 ──→ ChromaDB 语义检索 ──→ 相关文档 + 相似度             │
│                              │                                   │
│                              ▼                                   │
│                    上下文组装 + 系统提示词                         │
│                              │                                   │
│                              ▼                                   │
│                    Qwen3 LLM 本地推理 ──→ 流式输出 (SSE)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.10 | 3.11 / 3.12 |
| 内存 | 8 GB | 16 GB+ |
| 显存 | - | NVIDIA GPU（支持 CUDA）8GB+ |
| 磁盘 | 5 GB | 20 GB（包含模型） |
| 网络 | 下载模型用 | - |

> 推荐使用 conda 或 venv 创建独立环境，参考 `conda-environment.yml`。

## 快速开始

### 1. 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 conda
conda env create -f conda-environment.yml
conda activate vetrag
```

### 2. 下载模型（如需要）

```bash
# Qwen3-0.6B 基础模型
# 参考 HuggingFace: Qwen/Qwen3-0.6B
# 下载至 models/Qwen3-0.6B/

# BGE 向量化模型（首次运行时自动下载）
# 模型: BAAI/bge-large-zh-v1.5
```

### 3. 配置环境变量

```bash
cp envs/.env.example .env
# 编辑 .env 填入实际路径
```

主要配置项说明：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `Qwen3_MODEL_PATH` | Qwen3 基础模型路径 | `models/Qwen3-0.6B/` |
| `QWEN3_FINETUNED_PATH` | 微调模型路径 | `models_finetuned/qwen3-finetuned` |
| `CHROMA_PERSIST_DIR` | 向量数据库目录 | `./chroma_db` |
| `BGE_MODEL_NAME` | 向量化模型名称 | `BAAI/bge-large-zh-v1.5` |
| `API_PORT` | Web 服务端口 | `8000` |

### 4. 构建向量索引

```bash
cd VetRAG
python build_index.py
```

知识库数据（5 个 JSON 文件）将自动加载并向量化：

| 文件 | 内容 | 记录数 |
|------|------|--------|
| `behaviors.json` | 犬类行为分析 | 50+ |
| `breeds.json` | 犬种信息 | 100 |
| `cares.json` | 护理指南（疫苗、关节护理、养犬法规） | 各类 |
| `diseases.json` | 兽医疾病库 | 123 |
| `surgeries.json` | 兽医手术指南 | 99 |

### 5. 启动服务

```bash
# Web 服务（推荐）
python web_api.py
# 访问 http://localhost:8000 使用网页界面
# 或访问 http://localhost:8000/docs 查看 API 文档

# 命令行工具
python run.py
```

## 项目结构

```
VetRAG/
├── web_api.py              # FastAPI Web 服务入口
├── build_index.py          # 向量索引构建脚本
├── run.py                  # 命令行问答工具
├── pyproject.toml          # 项目配置
├── requirements.txt        # pip 依赖清单
├── conda-environment.yml   # conda 环境配置
├── Dockerfile              # Docker 镜像构建
├── docker-compose.yml      # Docker Compose 编排
│
├── src/                    # 核心源码
│   ├── core/
│   │   ├── config.py      # 集中配置（环境变量 + 路径）
│   │   └── logging.py     # loguru 统一日志
│   ├── json_loader.py     # JSON 数据加载与语义分块
│   ├── vector_store_chroma.py  # ChromaDB 向量存储
│   ├── rag_pipeline.py    # RAG Pipeline 编排
│   ├── rag_interface.py   # RAG 接口 + Qwen 生成器封装
│   ├── increment_manager.py   # 增量更新管理
│   └── clean_up.py        # 清理工具
│
├── data/                   # 知识库 JSON 数据
├── static/
│   └── index.html         # Web UI
│
├── finetune_steps/         # Qwen3 微调流程（QLORA）
│   ├── qlora_finetune.py  # 训练脚本
│   ├── prepare_data.py    # 数据准备
│   ├── test_before_finetuning.py   # 微调前评估
│   └── test_after_finetuning.py    # 微调后评估
│
├── docs/                   # 技术文档
│   ├── api.md             # API 参考
│   ├── rag_pipeline.md    # Pipeline 技术文档
│   └── deployment.md      # 部署与效果评价
│
└── tests/                  # 单元测试（79 个用例，全部通过）
    ├── api/                # FastAPI 接口测试
    ├── core/               # 配置与日志测试
    └── rag/                # JSON / 向量 / Pipeline 测试
```

## 模块说明

### `src/json_loader.py` - 数据加载与分块

VetRAG 数据加载器，支持多种 JSON 结构：

| 解析方法 | 内容类型 | 分块策略 |
|---------|---------|---------|
| `_parse_behaviors()` | `behaviors` | 每个行为 1 个 chunk |
| `_parse_breeds()` | `breeds` | 每个犬种 1 个 chunk |
| `_parse_cleaned_dog_care()` | `cares` | 按护理类别分块 |
| `_parse_diseases()` | `diseases_professional` | 每病 4 语义块（概述、治疗、流行病学、FAQ） |
| `_parse_surgeries()` | `surgeries` | 每手术 2 块（概述、术后护理） |

### `src/vector_store_chroma.py` - 向量存储

ChromaDB 持久化向量存储封装：

- 自动降级：优先使用 `BAAI/bge-large-zh-v1.5`，失败则回退到 `paraphrase-multilingual-MiniLM-L12-v2`
- 去重机制：基于内容 hash 跳过已处理的 chunk
- 增量索引：仅处理新增或变化的文档

### `src/rag_interface.py` - RAG 接口

封装检索与生成的核心接口：

- **查询**：检索相关文档 → 组装上下文 → 调用 LLM → 流式返回
- **文档清洗**：自动移除 JSON 代码块和 Markdown 标题，保证生成质量
- **系统提示词**：默认使用 `src/core/config.py` 中的 `SYSTEM_PROMPT_VET`
- **流式推理**：支持 SSE 实时流式输出

### `src/core/` - 核心配置

- `config.py`：所有环境变量统一管理，业务代码禁止直接读取 `os.getenv()`
- `logging.py`：loguru 全局日志配置，控制台（彩色）+ 文件（轮转 10MB/7天保留）

## API 接口

服务启动后访问 `http://localhost:8000/docs`（Swagger UI）可查看完整文档。

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 网页 UI |
| `GET` | `/stats` | 系统状态（向量库文档数、模型加载状态） |
| `POST` | `/query` | 非流式问答，返回完整 JSON |
| `GET` | `/stream` | 流式问答（SSE） |

### 请求示例

**非流式查询：**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "金毛有哪些常见健康问题？", "top_k": 3}'
```

**流式查询：**

```bash
curl -N "http://localhost:8000/stream?question=金毛有哪些常见健康问题&top_k=3"
```

## 环境变量说明

| 变量 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Qwen3_MODEL_PATH` | 路径 | `models/Qwen3-0.6B/` | Qwen3 基础模型目录 |
| `QWEN3_FINETUNED_PATH` | 路径 | `models_finetuned/qwen3-finetuned` | 微调模型目录 |
| `QWEN3_FINETUNED_PATH_V1` | 路径 | `models_finetuned/qwen3-finetuned1` | 微调模型 v1 |
| `CHROMA_PERSIST_DIR` | 字符串 | `./chroma_db` | 向量数据库路径 |
| `CHROMA_COLLECTION_NAME` | 字符串 | `veterinary_rag` | ChromaDB 集合名 |
| `BGE_MODEL_NAME` | 字符串 | `BAAI/bge-large-zh-v1.5` | 向量化模型 |
| `BGE_MODEL_FALLBACK` | 字符串 | `paraphrase-multilingual-MiniLM-L12-v2` | 回退模型 |
| `HF_ENDPOINT` | URL | - | HuggingFace 镜像（国内推荐） |
| `API_HOST` | 字符串 | `0.0.0.0` | 服务监听地址 |
| `API_PORT` | 整数 | `8000` | 服务端口 |
| `LOG_LEVEL` | 字符串 | `INFO` | 日志级别 |

## 常见问题

**Q: 启动时报 `ModuleNotFoundError: No module named 'chromadb'`？**

```bash
pip install chromadb sentence-transformers
```

**Q: 向量化时报内存不足？**

减小批处理大小，或将 `BGE_MODEL_NAME` 切换为轻量模型：
```python
# 环境变量
BGE_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
```

**Q: 模型下载慢或失败？**

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 如何添加新的知识库文件？**

1. 放置 JSON 文件到 `data/` 目录
2. 运行 `python build_index.py` 重新索引

## 部署方式

| 方式 | 适用场景 | 文档 |
|------|---------|------|
| pip / conda | 有 GPU 的开发环境 | 见下方快速开始 |
| Docker Compose | 生产部署、一键启动 | 详见 [docs/deployment.md](docs/deployment.md) |

### Docker 快速部署

```bash
cd VetRAG

# 启动 API 服务
docker-compose up -d vetrag-api

# 验证
curl http://localhost:8000/stats
```

详细部署步骤（GPU 配置、模型挂载、增量更新）请参考 [docs/deployment.md](docs/deployment.md)。

## 效果评价

| 评价维度 | 方法 | 脚本 |
|---------|------|------|
| 微调效果（基础 vs 微调） | 1027 条测试集，语义余弦相似度 | `finetune_steps/test_before_finetuning.py` / `test_after_finetuning.py` |
| 检索质量验证 | ChromaDB 查询相似度 | `python -c "from src.vector_store_chroma import ..."` |
| 端到端问答质量 | 真实问题验证 + 流式响应检查 | `curl http://localhost:8000/stream` |
| 生产监控 | A/B 测试、延迟 P50/P95/P99 | 详见 [docs/deployment.md](docs/deployment.md) |

**量化结果**（1027 条测试集，BGE-large-zh-v1.5 嵌入）：

| 配置 | 均值 | 中位数 | 标准差 |
|------|------|-------|--------|
| 原始 Qwen3-0.6B | 0.8868 | 0.8940 | 0.048 |
| **QLoRA 微调后** | **0.9344** | **0.9410** | **0.034** |

## 贡献指南

1. Fork 本仓库，创建功能分支
2. 编写测试用例（目标 79+）
3. 运行 `ruff check src/ tests/` 确保无 lint 错误
4. 提交 Pull Request

## 许可证

MIT License
