# 工作规范

> **进度追踪**
> - 进行中的任务和待办事项见 [TODO.md](TODO.md)
> - 已完成的工作成果记录见 [accomplishments.md](accomplishments.md)

## 每次开始工作前

1. **拉取最新代码** — 执行 `git pull`，确保本地代码与远程同步
2. **检查 git 状态** — 执行 `git status`，确认当前工作区状态
3. **同步 TODO.md** — 查看 [TODO.md](TODO.md) 中的任务状态，确认本次工作内容和优先级
4. **临时文件规范** — 所有临时文件（测试脚本、日志、备份等）统一放入 `temp/` 目录，文件名需标注用途（如 `temp_read_docx.py`、`temp_debug.log`），禁止散落在项目根目录

## 项目进度总览

| 模块 | 状态 | 详情 |
|------|------|------|
| 配置管理 | ✅ 完成 | 见 [accomplishments.md](accomplishments.md#一-配置管理) |
| 测试文件 | ✅ 完成 | 79/79 测试通过，见 [accomplishments.md](accomplishments.md#二-测试文件) |
| 配置文档（README） | ✅ 完成 | 见 [accomplishments.md](accomplishments.md#三-配置文档readme) |
| 项目结构调整 | ✅ 完成 | 见 [accomplishments.md](accomplishments.md#四-项目结构调整) |
| 训练数据扩充 | ✅ 完成 | 31,410 条，见 [accomplishments.md](accomplishments.md#五-训练数据扩充) |
| QLoRA 微调流水线 | ✅ 完成 | eval_acc 71.6%，见 [accomplishments.md](accomplishments.md#六-qloRA-微调流水线--qwen3-17b) |
| 本地模型部署 | ✅ 完成 | 见 [accomplishments.md](accomplishments.md#七-本地模型部署) |
| RAG 优化 | 🔄 进行中 | 见 [TODO.md](TODO.md#当前进行中) |
| 多轮对话优化 | ⏳ 待启动 | 见 [TODO.md](TODO.md#1-多轮对话优化) |
| 模型压缩与加速 | ⏳ 待启动 | 见 [TODO.md](TODO.md#2-模型压缩与加速) |
| 上线准备 | ⏳ 待启动 | 见 [TODO.md](TODO.md#3-上线准备) |

## 代码修改完成后

1. **先运行验证** — 确认修改后项目能正常启动或测试通过
2. **清理临时文件** — 将 `temp/` 中本次产生的临时文件确认无误后删除（或保留说明用途）
3. **提交 git** — 执行 `git add` → `git commit` → `git push`，commit 信息需清晰描述本次做了什么（如 `feat: 添加 conda 环境配置` / `fix: 修复 WebSocket 连接未清理资源的问题`）

---

## 远程服务器连接信息

> **AutoDL 服务器（SeetaCloud）**

| 项目 | 值 |
|------|-----|
| SSH 主机 | `connect.westb.seetacloud.com` |
| SSH 端口 | `31783` |
| 用户名 | `root` |
| 密码 | `YE++CVGoWBve` |

> **AutoDL 本地模型路径**：`/root/autodl-tmp/huggingface/models/Qwen3-1.7B`
>
> **训练数据路径**（上传后）：`/root/data/final_training_data_alpaca.jsonl`
>
> **微调输出路径**：`/root/autodl-tmp/huggingface/models/qwen3-1.7b-vet-finetuned`
>
> **合并模型输出路径**：`/root/autodl-tmp/output/Qwen3-1.7B-vetrag-merged`
>
> **HuggingFace Hub 仓库**：`MrK-means/Qwen3-1.7B-VetRAG`

---

## 当前项目技术栈参考

| 组件 | 技术 | 位置 |
|------|------|------|
| 前端 | 原生 HTML + SSE | `static/index.html` + `web_api.py` |
| 后端 | FastAPI + Uvicorn | `web_api.py` |
| LLM | Qwen3-1.7B（AutoDL，本地微调后加载） | `rag_interface.py` |
| Embedding | BAAI/bge-large-zh-v1.5 | `src/vector_store_chroma.py` |
| 向量数据库 | ChromaDB | `src/vector_store_chroma.py` |
| 数据解析 | JSON | `src/json_loader.py` |
| 微调框架 | PEFT + TRL + BitsAndBytes（QLoRA） | `VetRAG/finetune_steps/` |
| 微调模型 | Qwen3-1.7B（AutoDL ModelScope） | `download_qwen3_1b7.py` |
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
data_process/              # 训练数据生成脚本
├── merge_and_dedup.py     # S1：合并去重
├── qa_from_diseases.py    # S2：疾病→问答对
├── expand_topics.py       # S3：新话题扩充
├── multi_augment.py       # S4：指令多样化（运行中）
├── safety_qa.py           # S5：安全/格式 QA
├── final_merge.py         # S6：最终合并导出
└── final_output/          # 最终训练集输出
    ├── final_training_data.json
    ├── final_training_data_alpaca.jsonl
    └── final_summary.json