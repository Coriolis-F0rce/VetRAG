# VetRAG 项目任务清单

> 最后更新：2026-04-29

## 配置管理 ✅ 已完成

| 任务 | 状态 | 完成日期 |
|------|------|----------|
| 扫描项目依赖，创建 requirements.txt | ✅ 完成 | 2026-04-28 |
| 创建 pyproject.toml | ✅ 完成 | 2026-04-28 |
| 创建 .env.example | ✅ 完成 | 2026-04-28 |
| 创建 conda-environment.yml | ✅ 完成 | 2026-04-28 |
| 创建 src/core/config.py 集中配置 | ✅ 完成 | 2026-04-28 |
| 创建 src/core/logging.py 统一日志 | ✅ 完成 | 2026-04-28 |
| 更新 .gitignore | ✅ 完成 | 2026-04-28 |

---

## 测试文件 ✅ 已完成

> 最后更新：2026-04-28（含修复）
> 负责人：Agent

---

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 创建 tests/ 目录结构 | ✅ 完成 | 2026-04-28 | |
| 编写 JSON 数据解析测试 | ✅ 完成 | 2026-04-28 | 28 个测试用例，含真实文件解析 |
| 编写 Embedding 服务测试 | ✅ 完成 | 2026-04-28 | 14 个测试用例，含 chromadb 集成 |
| 编写 RAG Pipeline 测试 | ✅ 完成 | 2026-04-28 | 11 个测试用例，含方法存在性检查 |
| 编写 FastAPI 接口测试 | ✅ 完成 | 2026-04-28 | 11 个测试用例，含 Mock RAGInterface |
| 配置 pytest.ini | ✅ 完成 | 2026-04-28 | 含 markers / filterwarnings / coverage |
| 配置 GitHub Actions CI | ✅ 完成 | 2026-04-28 | lint + pytest + coverage（含路径修复） |

**测试运行结果：79/79 全部通过**（pytest 8.3.4 / Python 3.13）

---

## 配置文档（README） ✅ 已完成

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 创建 README.md | ✅ 完成 | 2026-04-29 | 完整项目文档：简介/架构/快速开始/模块说明/FAQ |
| 创建 docs/ 目录技术文档 | ✅ 完成 | 2026-04-29 | `docs/api.md` + `docs/rag_pipeline.md` |
| 创建 CHANGELOG.md | ✅ 完成 | 2026-04-29 | 遵循 Keep a Changelog 规范 |

---

## 项目结构调整 ✅ 已完成

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 规范化导入路径 | ✅ 完成 | 2026-04-29 | 统一 `src.` 前缀，移除 `sys.path` hack |
| Docker 配置完善 | ✅ 完成 | 2026-04-29 | `Dockerfile` + `docker-compose.yml` |
| 目录清理 | ✅ 完成 | 2026-04-29 | 删除测试残留、补充 `.gitignore` |

---

## 执行记录

### 2026-04-29（下午）
- 完成项目结构调整全部 3 项任务
- **规范化导入路径**：`rag_interface.py` 移入 `src/`，所有模块改用 `from src.xxx` 相对导入，移除裸导入；`build_index.py` 改用直接组件调用；`web_api.py` 改用 `CHROMA_PERSIST_DIR`/`QWEN3_FINETUNED_PATH`/`SYSTEM_PROMPT_VET`
- **Docker 配置**：创建 `Dockerfile`（Python 3.11-slim，预下载 BGE 模型）+ `docker-compose.yml`（含 API + Jupyter Notebook）+ `.dockerignore`
- **目录清理**：删除测试残留 `temp_test_chroma/`，创建 `VetRAG/.gitignore`
- 全部 79 个测试通过

### 2026-04-29
- 完成配置文档阶段全部 3 项任务
- 创建 `VetRAG/README.md`（完整项目文档）
- 创建 `VetRAG/docs/api.md`（API 参考文档）
- 创建 `VetRAG/docs/rag_pipeline.md`（Pipeline 技术文档）
- 创建 `VetRAG/CHANGELOG.md`（遵循 Keep a Changelog 规范）

### 2026-04-28（下午）
- 完成测试文件阶段全部 7 项任务
- 修复所有测试文件的 `parents[N]` 路径问题（conftest + 4 个测试文件）
- 修复 CI workflow 路径（ruff 检查 + pytest 运行 + coverage 上报）
- 修复 vector_store Integration tests：`pytest.importorskip("chromadb")` 替代 `patch`
- 修复 API 无效 JSON 测试：`pytest.raises` 捕获实际异常
- 运行结果：79/79 全部通过（pytest 8.3.4 / Python 3.13.5）

### 2026-04-28（上午）
- 完成配置管理全部 7 项任务
- 创建 `VetRAG/requirements.txt`
- 创建 `VetRAG/pyproject.toml`（含 pytest / ruff / coverage 配置）
- 创建 `VetRAG/conda-environment.yml`
- 创建 `VetRAG/envs/.env.example`
- 创建 `VetRAG/src/core/config.py`（集中配置模块）
- 创建 `VetRAG/src/core/logging.py`（loguru 统一日志）
- 更新 `.gitignore`（补充新增文件、日志、缓存等）
- 更新 `workflow.md`（同步完成项，添加 TODO.md 引用）
