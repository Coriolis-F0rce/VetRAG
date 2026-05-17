# VetRAG 开发工作流

> 本文档描述 VetRAG 项目的日常开发与维护操作流程。供开发者和 Claude Code 会话参考。

---

## 一、知识库数据管理

### 1.1 添加新数据源

1. 将 JSON 文件放入 `data/` 目录
2. 在 `src/json_loader.py` 中新增对应的 `_parse_*()` 方法
3. 在 `_parse_file_based_on_type()` 中注册文件名匹配规则
4. 在 `scripts/build_index.py` 的 `file_paths` 列表中添加新文件
5. 在 `src/core/config.py` 的 `DATA_FILES` 列表中添加文件名

### 1.2 重建向量索引

```bash
# 全量重建（清空 ChromaDB 后重新入库）
python scripts/build_index.py

# 增量更新（仅处理新增/变更文件）
# build_index.py 内部会检查 processed_ids.pkl
```

### 1.3 验证索引

```bash
python -c "
from src.vector_store_chroma import ChromaVectorStore
from src.core.config import CHROMA_PERSIST_DIR
v = ChromaVectorStore(collection_name='veterinary_rag', persist_directory=CHROMA_PERSIST_DIR, model_name='BAAI/bge-large-zh-v1.5')
print(f'Total chunks: {v.collection.count()}')
"
```

### 1.4 药品数据维护

药品数据源分两阶段生成：

1. **提取**：`python scripts/extract_drugs.py` — 从 diseases.json 的治疗方案中提取药物列表
2. **补全**：`python scripts/enrich_pharmaceuticals.py` — 通过 DeepSeek API 补全药理字段（机制、禁忌、副作用、相互作用、监测）

最终数据：`data/pharmaceuticals.json`（331 种药物，14 字段，所有药物已补全至 v1）。

---

## 二、模型管理

### 2.1 模型命名规范

`vetrag-{base_model}-{variant}`
- 基础模型：`vetrag-qwen3-0.6b-base`、`vetrag-qwen3-1.7b-base`
- 微调模型：`vetrag-qwen3-1.7b-vet`

### 2.2 微调模型上架流程

不能跳过任何一步：

```
LoRA adapter (models_finetuned/)
    ↓ merge_lora.py
完整权重 (models_merged/)
    ↓ convert_to_gguf.py
GGUF (models_gguf/)
    ↓ setup_ollama.py
Ollama 模型注册
```

```bash
python scripts/merge_lora.py       # 1. 合并 LoRA
python scripts/convert_to_gguf.py  # 2. 转 GGUF
python scripts/setup_ollama.py     # 3. 导入 Ollama
```

### 2.3 生成参数

- `temperature=0.0`（greedy decoding，确保可复现）
- `num_predict=512`
- `repeat_penalty=1.2`

---

## 三、评估体系

### 3.1 A/B 实验

```bash
# 运行 4 组 A/B 实验（微调+/-RAG，基础+/-RAG）
python eval/scripts/run_ab_experiment.py

# 结果输出到 eval/results/
```

### 3.2 DeepSeek Judge 评分

```bash
# 并行评分（默认 10 workers）
python eval/scripts/run_deepseek_judge.py \
  --testset eval/datasets/testset_50.json \
  --results eval/results/raw_ab_50q_20260517.json

# 顺序评分（调试用）
python eval/scripts/run_deepseek_judge.py \
  --testset eval/datasets/testset_50.json \
  --results eval/results/raw_ab_50q_20260517.json \
  --sequential

# 自定义并发数
python eval/scripts/run_deepseek_judge.py ... --workers 20
```

### 3.3 评分维度

| 维度 | 说明 | 分值范围 |
|------|------|---------|
| accuracy | 医学事实准确性 | 1-5 |
| relevance | 回答相关性 | 1-5 |
| completeness | 信息完整性 | 1-5 |
| format | 格式规范性（无 emoji/代码块/免责声明） | 1-5 |
| safety | 安全性（是否包含危险建议） | 1-5 |

### 3.4 结果解读

- `eval/results/judge_scores_*.json` — 各维度原始打分 + 推理 + 对比 + 胜者
- `eval/results/summary_*.json` — 各组的维度均分和排名汇总
- 匿名 4 答案对比（A/B/C/D），评分者不知道答案来源

---

## 四、日常开发流程

### 4.1 快捷命令（Makefile）

```bash
make help          # 列出所有命令
make run-api       # 启动 FastAPI 服务（http://localhost:5002）
make build-index   # 构建向量索引
make test          # 运行测试 + 覆盖率
make eval-ab       # 运行 A/B 实验
make eval-deepseek # DeepSeek Judge 评分
make clean         # 清理临时文件
```

### 4.2 启动服务

```bash
# Web API（生产/测试用）
make run-api
# 或: python scripts/web_api.py
# 默认 http://localhost:5002

# 命令行交互
python scripts/run.py
```

### 4.3 运行测试

```bash
# 全量测试 + 覆盖率
make test
# 或: python -m pytest tests/ -v --cov=src --cov-report=term

# 单个模块
python -m pytest tests/api/test_web_api.py -v
```

### 4.4 代码检查

```bash
make lint
# 或: ruff check .
```

### 4.5 CI/CD

GitHub Actions 自动运行（`.github/workflows/ci.yml`）：
- 触发：push/PR 到 main 分支
- 步骤：ruff check → pytest + coverage → 上传覆盖率报告

---

## 五、文档更新清单

以下文档需要在重要变更后同步更新：

| 文档 | 何时更新 |
|------|---------|
| `README.md` | 架构/技术栈变化、新增核心功能 |
| `CHANGELOG.md` | 每个版本发布 |
| `knowledge.md` | 重要技术决策、评估结果 |
| `TODO.md` | 任务优先级变化、新发现的问题 |
| `docs/api.md` | API 端点/参数变更 |
| `docs/rag_pipeline.md` | Pipeline 流程变更 |
| `docs/deployment.md` | 部署方式/配置变更 |
| `workflow.md`（本文件） | 开发流程变更 |

---

## 六、环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_GENERATOR_MODEL` | `vetrag-qwen3-1.7b-vet` | Ollama 生成模型名 |
| `OLLAMA_GUARD_MODEL` | `qwen3:1.7b` | 领域守卫模型名 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | 向量库路径 |
| `BGE_MODEL_NAME` | `BAAI/bge-large-zh-v1.5` | Embedding 模型 |
| `USE_HYBRID_SEARCH` | `false` | 启用混合检索 |
| `API_PORT` | `8000` | Web 服务端口（web_api.py 当前硬编码 5002，待统一） |
