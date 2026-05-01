# Changelog

所有重要的版本变更都会记录在此文件中。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范。

## [Unreleased] - 2026-05-02

> 本版本正在开发中。

### 添加
- **领域边界守卫**（Domain Guard）：LLM 零样本分类过滤非宠物狗问题
  - 新增 `src/core/domain_guard.py`（DomainGuard 类）
  - 新增 `USE_DOMAIN_GUARD` 配置项（默认 true）
  - `query()` 和 `query_stream()` 集成 Guard 预检查
  - 新增 21 个测试用例（test_domain_guard.py）
- **RAG 混合检索**（Hybrid Search）：Dense HNSW + BM25 + RRF 融合
  - 新增 `src/retrievers/bm25_index.py`（jieba 中文分词 + BM25Okapi）
  - 新增 `src/retrievers/hybrid_retriever.py`（RRF 融合）
  - 重构 `src/vector_store_chroma.py`：集成混合检索，向后兼容
  - 新增 `USE_HYBRID_SEARCH` 等 5 个配置项
- 文档结构：`README.md`、`docs/api.md`、`docs/rag_pipeline.md`、`CHANGELOG.md`
- 项目依赖管理：`requirements.txt`、`pyproject.toml`、`conda-environment.yml`
- 集中配置模块：`src/core/config.py`
- 统一日志模块：`src/core/logging.py`
- 完整测试套件：100+ 个测试用例（pytest 8.3.4，Python 3.13）
- GitHub Actions CI 工作流：lint（ruff）+ 测试 + 覆盖率报告
- Docker 部署配置：`Dockerfile`（Python 3.11-slim）+ `docker-compose.yml`
- 本地忽略规则：`VetRAG/.gitignore`

### 修复
- 测试路径问题：修正 `conftest.py` 和所有测试文件的 `parents[N]` 路径解析（Windows 下行为）
- CI workflow 路径问题：ruff 检查路径、pytest 工作目录、coverage 上报路径
- vector_store Integration tests：`pytest.importorskip()` 替代 `patch()` 避免未安装时报错
- API 无效 JSON 测试：改用 `pytest.raises` 正确捕获异常
- BM25Ok API 兼容性问题（`okapi BM25` 参数差异）
- HybridRetriever RRF 融合字段名 bug（`doc_id` → `id`）

### 重构
- 导入规范化：`rag_interface.py` 移入 `src/`，所有模块统一 `from src.xxx` 相对导入
- 配置集中化：`web_api.py` 改用 `CHROMA_PERSIST_DIR`/`QWEN3_FINETUNED_PATH`
- `src/rag_pipeline.py`：修复懒加载导入，改用相对导入
- `build_index.py`：重写为直接组件调用，修复不存在的方法
- `SYSTEM_PROMPT_VET`：改为宠物狗专属提示词

---

## [0.1.0] - 2026-04-28

### 首次发布

初始功能版本，包含完整的 RAG 问答系统核心功能。

### 添加
- **FastAPI Web 服务**（`web_api.py`）：提供 `/`、`/stats`、`/query`、`/stream` 四个端点，支持流式 SSE 输出
- **RAG 接口封装**（`rag_interface.py`）：集成 ChromaDB 检索 + Qwen3 生成，支持流式/非流式两种模式
- **命令行工具**（`run.py`）：交互式问答，支持向量库构建和实时检索
- **索引构建脚本**（`build_index.py`）：基于 `src/rag_pipeline.py` 的 Pipeline 编排
- **JSON 数据加载器**（`src/json_loader.py`）：支持 5 种 JSON 格式的语义分块
- **ChromaDB 向量存储**（`src/vector_store_chroma.py`）：BGE 向量化 + 持久化存储
- **增量更新管理器**（`src/increment_manager.py`）：文件变更检测与增量入库
- **清理工具**（`src/clean_up.py`）：向量库清理与状态重置
- **Qwen3 微调流程**（`finetune_steps/`）：QLORA 训练脚本与数据准备
- **知识库数据**（`data/`）：5 个 JSON 文件，包含行为、犬种、护理、疾病、手术数据

### 技术选型
| 组件 | 技术 |
|------|------|
| 前端 | 原生 HTML + JavaScript（SSE） |
| 后端 | FastAPI + Uvicorn |
| LLM | Qwen3-0.6B（本地推理） |
| Embedding | BAAI/bge-large-zh-v1.5 |
| 向量数据库 | ChromaDB |
| 微调框架 | PEFT (QLORA) + TRL + BitsAndBytes |
| 日志 | loguru |
