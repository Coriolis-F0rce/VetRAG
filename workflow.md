# 工作规范

> **任务追踪** — 所有当前执行中的任务和待办事项见 [TODO.md](TODO.md)，每次开始工作前同步更新该项目。

## 每次开始工作前

1. **拉取最新代码** — 执行 `git pull`，确保本地代码与远程同步
2. **检查 git 状态** — 执行 `git status`，确认当前工作区状态
3. **同步 TODO.md** — 查看 [TODO.md](TODO.md) 中的任务状态，确认本次工作内容和优先级
4. **临时文件规范** — 所有临时文件（测试脚本、日志、备份等）统一放入 `temp/` 目录，文件名需标注用途（如 `temp_read_docx.py`、`temp_debug.log`），禁止散落在项目根目录

## 项目目标

### 一、配置管理

- [x] **创建 `requirements.txt`** — 扫描所有 Python 文件的 `import` 语句，整理出所有第三方依赖及其版本号
- [x] **创建 `pyproject.toml`** — 使用现代 Python 打包标准，定义项目元数据、依赖范围（生产依赖 vs 开发依赖）
- [x] **创建 `.env.example`** — 整理所有环境变量模板，存入 `envs/` 目录
- [x] **创建 `conda-environment.yml`** — 补充 conda 环境配置文件，兼容 uv / conda / pip 多环境管理
- [x] **集中配置模块** — 创建 `src/core/config.py`，统一管理所有配置读取
- [x] **统一日志配置** — 创建 `src/core/logging.py`，集中配置 loguru 日志格式和级别

### 二、测试文件

- [x] **创建 `tests/` 目录结构** — 按模块划分测试目录（`tests/core/`、`tests/rag/`、`tests/api/`）
- [x] **编写 JSON 数据解析测试** — 针对 `src/json_loader.py` 的各解析方法编写单元测试，覆盖正常文件、异常文件（格式错误、空数据）场景
- [x] **编写 Embedding 服务测试** — 测试 ChromaDB 连接、文本向量化、向量搜索功能
- [x] **编写 RAG Pipeline 测试** — 测试检索→生成的完整链路，覆盖无结果、少结果、多结果等场景
- [x] **编写 FastAPI 接口测试** — 使用 `pytest` + `TestClient` 测试所有 API 端点
- [x] **配置 `pytest.ini`** — 定义测试发现路径、过滤规则、覆盖率报告配置
- [x] **配置 GitHub Actions CI** — 创建 `.github/workflows/ci.yml`，包含 lint（ruff）+ 测试 + 覆盖率报告

> 测试结果：79/79 全部通过（pytest 8.3.4 / Python 3.13）

### 三、配置文档（README）

- [x] **创建 `README.md`** — 项目说明文档，包含以下章节：
  - 项目简介（是什么、解决什么问题）
  - 系统架构图（文字版）
  - 环境要求（Python 版本、RAM、GPU）
  - 快速开始（安装依赖 → 启动服务 → 上传文档 → 提问）
  - 各模块说明（backend / frontend / rag）
  - 环境变量说明表
  - 常见问题（FAQ）
  - 贡献指南
- [x] **创建 `docs/` 目录** — 按模块存放详细技术文档（如 `docs/api.md`、`docs/rag_pipeline.md`）
- [x] **创建 `CHANGELOG.md`** — 记录版本变更历史

### 四、项目结构调整

- [x] **创建 `src/core/` 目录** — 将核心业务代码（配置和日志）统一收口为 `src/core/` 包结构
- [x] **规范化导入路径** — 统一使用 `from src.core.config` 或 `from src.json_loader` 的绝对导入方式，`rag_interface.py` 移入 `src/`，移除所有裸导入和 `sys.path` hack
- [x] **分离配置层** — 在 `src/core/config.py` 中集中管理所有配置加载逻辑，禁止在业务代码中直接读取环境变量
- [x] **Docker 配置完善** — 创建 `Dockerfile`（Python 3.11-slim）+ `docker-compose.yml`（API + Jupyter）+ `.dockerignore`
- [x] **添加 `.gitignore`** — 确认包含所有必要的忽略规则
- [x] **目录清理** — 删除测试残留目录，规范化本地 `.gitignore`

## 代码修改完成后

1. **先运行验证** — 确认修改后项目能正常启动或测试通过
2. **清理临时文件** — 将 `temp/` 中本次产生的临时文件确认无误后删除（或保留说明用途）
3. **提交 git** — 执行 `git add` → `git commit` → `git push`，commit 信息需清晰描述本次做了什么（如 `feat: 添加 conda 环境配置` / `fix: 修复 WebSocket 连接未清理资源的问题`）

---

## 当前项目技术栈参考

| 组件 | 技术 | 位置 |
|------|------|------|
| 前端 | 原生 HTML + SSE | `static/index.html` + `web_api.py` |
| 后端 | FastAPI + Uvicorn | `web_api.py` |
| LLM | Qwen3-0.6B（本地） | `rag_interface.py` |
| Embedding | BAAI/bge-large-zh-v1.5 | `src/vector_store_chroma.py` |
| 向量数据库 | ChromaDB | `src/vector_store_chroma.py` |
| 数据解析 | JSON | `src/json_loader.py` |
| 微调框架 | PEFT + TRL + BitsAndBytes | `finetune_steps/qlora_finetune.py` |
| 日志 | loguru | `src/core/logging.py` |
| 环境管理 | uv / conda | `requirements.txt` / `conda-environment.yml` |

## 项目文件清单

```
VetRAG/
├── src/
│   ├── core/          # 配置和日志
│   │   ├── __init__.py
│   │   ├── config.py  # 集中配置
│   │   └── logging.py # 统一日志
│   ├── json_loader.py    # JSON 数据加载与解析
│   ├── vector_store_chroma.py  # ChromaDB 向量存储
│   ├── rag_pipeline.py    # RAG Pipeline
│   ├── rag_interface.py   # RAG 对话接口（原顶级移入）
│   ├── increment_manager.py  # 增量更新管理
│   └── clean_up.py        # 清理工具
├── tests/              # 测试套件（79 个用例）
│   ├── conftest.py
│   ├── api/
│   ├── core/
│   └── rag/
├── finetune_steps/        # Qwen3 微调流程
├── static/index.html      # Web UI
├── docs/                  # 技术文档
│   ├── api.md
│   └── rag_pipeline.md
├── data/                   # 知识库 JSON 数据
├── web_api.py            # FastAPI 服务
├── run.py                 # 交互式命令行入口
├── build_index.py         # 构建向量索引
├── README.md              # 项目文档
├── CHANGELOG.md          # 版本记录
├── Dockerfile            # Docker 镜像构建
├── docker-compose.yml     # Docker Compose 编排
├── .dockerignore         # Docker 构建排除
├── requirements.txt       # pip 依赖
├── pyproject.toml         # 项目配置
├── conda-environment.yml  # Conda 环境
├── pytest.ini             # 测试配置
├── .env.example          # 环境变量模板
├── .gitignore           # 本地忽略规则
└── envs/.env.example     # 同上
