# Changelog

所有重要的版本变更都会记录在此文件中。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 规范。

## [Unreleased] - 2026-05-17

> 本版本正在开发中。

### 添加
- **Ollama 推理迁移**：LLM 推理从 transformers 本地加载迁移到 Ollama API
  - QwenGenerator 改为通过 Ollama REST API 调用
  - DomainGuard 独立使用 Ollama 基础模型（零样本分类）
  - 模型从 Qwen3-0.6B 升级到 Qwen3-1.7B
  - 新增 `OLLAMA_GENERATOR_MODEL`、`OLLAMA_GUARD_MODEL`、`OLLAMA_BASE_URL` 配置项
  - 模型命名规范 `vetrag-{base}-{variant}`
- **犬科药学知识库**：331 种药物独立 chunk 到 ChromaDB（1189 chunks 总计）
  - 新增 `scripts/extract_drugs.py`：从 diseases.json 治疗数据提取药物列表
  - 新增 `scripts/enrich_pharmaceuticals.py`：通过 DeepSeek API 补全药理字段（机制、禁忌、副作用、相互作用、监测）
  - 新增 `data/pharmaceuticals.json`：331 种药物，14 字段全 v1 格式
  - `src/json_loader.py` 新增 `_parse_pharmaceuticals()` 方法，兼容 v0/v1 格式
  - 药物 chunk metadata 包含 drug_name、drug_class、indication_count、dosage_available
- **DeepSeek LLM-as-Judge**：匿名 4 答案对比评分体系
  - 5 维度打分（accuracy、relevance、completeness、format、safety）+ 推理 + 对比 + 胜者
  - ThreadPoolExecutor 并行评分（默认 15 workers），50 题从 ~4 分钟降至 ~14 秒
  - `eval/scoring/deepseek_judge.py`：`score_all_parallel()` 方法
  - `eval/scripts/run_deepseek_judge.py`：`--workers` / `--sequential` / `--testset` 参数
- **A/B 实验框架**：4 组对比（微调+/-RAG，基础+/-RAG）
  - `eval/datasets/testset_50.json`：50 条犬科药学测试用例
  - `eval/scripts/run_ab_experiment.py`：A/B 实验驱动
  - `eval/results/`：原始答案 + 评分 + 汇总报告
- **答案清洗管线**：`_clean_text` 13 步后处理（移除 think 标签、代码块、JSON、Markdown、emoji、免责声明、去重等）
- **全局配置**：Claude Code 权限配置全局化到 `~/.claude/settings.json`，新项目无需重新配置
- **workflow.md**：开发工作流参考文档

### 修复
- FastAPI SSE 流式输出阻塞问题：`queue.Queue` → `asyncio.Queue` + `async for`
- DeepSeek Judge 只评分 10 题而非 50 题：默认 testset 参数修正

### 重构
- 项目 `.claude/settings.json` 精简为仅项目特定配置（ruff hook + data/ 保护）
- 文件规范化：`pharmaceuticals_v0.json` → `pharmaceuticals.json`（删除冗余模板）

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
