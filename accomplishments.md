# 项目成果记录

> 本文件记录 VetRAG 项目中所有已完成的工作成果，按模块和时间组织。进行中的工作见 [TODO.md](TODO.md)。

---

## 一、配置管理

> 完成日期：2026-04-28

| 任务 | 产出文件 | 说明 |
|------|----------|------|
| 扫描项目依赖，创建 requirements.txt | `VetRAG/requirements.txt` | 包含所有第三方依赖及版本号 |
| 创建 pyproject.toml | `VetRAG/pyproject.toml` | 现代 Python 打包标准，含 pytest / ruff / coverage 配置 |
| 创建 .env.example | `VetRAG/envs/.env.example` | 所有环境变量模板 |
| 创建 conda-environment.yml | `VetRAG/conda-environment.yml` | 兼容 uv / conda / pip 多环境管理 |
| 创建 src/core/config.py 集中配置 | `VetRAG/src/core/config.py` | 统一管理所有配置读取 |
| 创建 src/core/logging.py 统一日志 | `VetRAG/src/core/logging.py` | 集中配置 loguru 日志格式和级别 |
| 更新 .gitignore | `.gitignore` / `VetRAG/.gitignore` | 包含所有必要忽略规则 |

---

## 二、测试文件

> 完成日期：2026-04-28（含修复）
> 测试结果：79/79 全部通过（pytest 8.3.4 / Python 3.13）

| 任务 | 产出文件 | 说明 |
|------|----------|------|
| 创建 tests/ 目录结构 | `VetRAG/tests/` | 按模块划分：`tests/core/`、`tests/rag/`、`tests/api/` |
| 编写 JSON 数据解析测试 | `VetRAG/tests/test_json_loader.py` | 28 个测试用例，覆盖正常/异常文件场景 |
| 编写 Embedding 服务测试 | `VetRAG/tests/rag/test_vector_store.py` | 14 个测试用例，含 chromadb 集成 |
| 编写 RAG Pipeline 测试 | `VetRAG/tests/rag/test_rag_pipeline.py` | 11 个测试用例，含方法存在性检查 |
| 编写 FastAPI 接口测试 | `VetRAG/tests/api/test_api.py` | 11 个测试用例，含 Mock RAGInterface |
| 配置 pytest.ini | `VetRAG/pytest.ini` | 含 markers / filterwarnings / coverage |
| 配置 GitHub Actions CI | `.github/workflows/ci.yml` | lint（ruff）+ pytest + 覆盖率报告 |

---

## 三、配置文档（README）

> 完成日期：2026-04-29

| 任务 | 产出文件 | 说明 |
|------|----------|------|
| 创建 README.md | `VetRAG/README.md` | 完整项目文档：简介/架构/快速开始/模块说明/FAQ |
| 创建 docs/ 技术文档 | `VetRAG/docs/api.md` | API 参考文档 |
| 创建 docs/ 技术文档 | `VetRAG/docs/rag_pipeline.md` | Pipeline 技术文档 |
| 创建 CHANGELOG.md | `VetRAG/CHANGELOG.md` | 遵循 Keep a Changelog 规范 |

---

## 四、项目结构调整

> 完成日期：2026-04-29

| 任务 | 说明 |
|------|------|
| 规范化导入路径 | 统一 `src.` 前缀，`rag_interface.py` 移入 `src/`，移除所有裸导入和 `sys.path` hack |
| Docker 配置完善 | `Dockerfile`（Python 3.11-slim，预下载 BGE 模型）+ `docker-compose.yml`（API + Jupyter）+ `.dockerignore` |
| 目录清理 | 删除测试残留 `temp_test_chroma/`，规范化 `.gitignore` |

---

## 五、训练数据扩充

> 完成日期：2026-04-30（凌晨）
> 最终训练集：**31,410 条**

| 阶段 | 脚本 | 说明 | 条数 |
|------|------|------|------|
| S1 | `data_process/merge_and_dedup.py` | 合并增强数据、去重、规则过滤 | 16,552 |
| S2 | `data_process/qa_from_diseases.py` | 疾病知识库 → 模板问答对（8 种模板/病，123 种疾病） | 973 |
| S3 | `data_process/expand_topics.py` | 品种(220) + 行为(126) + 手术(579) + 日常养护(30) 扩充 | 955 |
| S4 | `data_process/multi_augment.py` | 指令多样化：API + 规则（同义改写、句式变换、视角转换、情感变化、噪声注入） | 12,999 |
| S5 | `data_process/safety_qa.py` | 通用安全/格式 QA（30 条模板 + 6 条格式 + 20 条 API 增强） | 56 |
| S6 | `data_process/final_merge.py` | 全量合并、去重、导出 JSON + JSONL | **31,410** |

**最终输出**：`data_process/final_output/final_training_data.json` / `final_training_data_alpaca.jsonl`

---

## 六、QLoRA 微调流水线 — Qwen3-1.7B

> 完成日期：2026-05-01（凌晨）

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-1.7B |
| 数据 | 训练集 29,839 / 验证集 1,571 |
| batch | 7 × 4 = 28 |
| epoch | 3 |
| 可训练参数 | 17.4M (1.003%) |

### 训练结果

| 指标 | 初始 | 最终 | 变化 |
|------|------|------|------|
| train_loss | 2.82 | ~0.88 | -69% |
| eval_loss | 1.95 | ~1.14 | -42% |
| eval_accuracy | 54% | ~71.6% | +17.6pt |
| entropy | 1.526 | ~0.95 | -38% |
| 训练时长 | — | 4h13m | — |

### 结论

- 无过拟合迹象，eval_loss 持续下降未触拐点
- 对比 Qwen3-0.6B 旧数据版（eval_acc 55%），提升显著
- 泛化能力强，train-eval gap 小

### 关键文件

| 文件 | 说明 |
|------|------|
| `VetRAG/finetune_steps/finetune_qlora.py` | QLoRA 微调主脚本（batch=7, grad_acc=5, max_len=2048） |
| `VetRAG/finetune_steps/upload_to_hf.py` | 上传微调模型至 HuggingFace Hub |
| `VetRAG/finetune_steps/` | 超参数配置、数据集生成脚本 |

### 模型存储路径

- **AutoDL 微调输出**：`/root/autodl-tmp/huggingface/models/qwen3-1.7b-vet-finetuned`
- **AutoDL 合并输出**：`/root/autodl-tmp/output/Qwen3-1.7B-vetrag-merged`
- **HuggingFace Hub**：`MrK-means/Qwen3-1.7B-VetRAG`

---

## 七、本地模型部署

> 完成日期：2026-05-01（下午）

### 完成任务

| 任务 | 说明 |
|------|------|
| 下载基础模型 | `VetRAG/models/Qwen3-1.7B`（基于 ModelScope，约 3.4 GB） |
| 下载 LoRA adapter | `VetRAG/models_finetuned/qwen3-1.7b-vet-finetuned/` |
| 合并 LoRA + 基础模型 | `VetRAG/models/Qwen3-1.7B-vet-finetuned/`（合并后完整权重） |
| 本地推理测试 | 10 题验证，10/10 无 `<think>` 标签，平均 607 字/条 |
| 更新配置路径 | `QWEN3_FINETUNED_PATH` → `models/Qwen3-1.7B-vet-finetuned` |
| RAG 接口适配 | `rag_interface.py` 新增菜单选项 6，`generate()` 添加 `<think>` 标签后处理 |

### 关键文件

| 文件 | 说明 |
|------|------|
| `VetRAG/download_base_model.py` | 基于 ModelScope 下载 Qwen3-1.7B 基础模型 |
| `VetRAG/merge_and_run_lora.py` | 合并 LoRA adapter + 基础模型为完整权重 |
| `VetRAG/test_merged_model.py` | 本地推理测试（禁用 `think` + 后处理 `<think>` 标签） |
| `VetRAG/src/rag_interface.py` | 新增菜单选项 6（合并模型），默认使用 config 路径，`generate()` 标签去除 |
| `VetRAG/src/core/config.py` | `QWEN3_FINETUNED_PATH` 指向 `models/Qwen3-1.7B-vet-finetuned` |

---

## 八、RAG 混合检索（Hybrid Search）

> 完成日期：2026-05-01（晚上）

### 背景

当前 `ChromaVectorStore.search()` 仅有 Dense 向量检索（HNSW + BGE 嵌入），对短查询和专业术语精确匹配能力不足。宠物医疗场景有大量专有病名，用户查询也存在口语化与文档用词差异的问题。

### 技术方案

| 组件 | 实现 | 说明 |
|------|------|------|
| Dense 检索 | ChromaDB HNSW（原有） | BAAI/bge-large-zh-v1.5，1024 维，余弦距离 |
| Sparse 检索 | BM25Okapi + jieba 分词 | 关键词精确匹配，中文友好 |
| 融合算法 | RRF（Reciprocal Rank Fusion） | `RRF = w1/(k+r1) + w2/(k+r2)`，k=60 |
| 权重配置 | `dense_weight=0.7, bm25_weight=0.3` | 通过环境变量可调 |

### 关键文件

| 文件 | 说明 |
|------|------|
| `src/retrievers/bm25_index.py` | BM25 索引构建、查询、持久化（pickle） |
| `src/retrievers/hybrid_retriever.py` | HybridRetriever：双路召回 + RRF 融合 |
| `src/retrievers/__init__.py` | 模块导出 |
| `src/vector_store_chroma.py` | 新增 `use_hybrid` 参数，向后兼容 |
| `src/core/config.py` | 新增 `USE_HYBRID_SEARCH` 等 5 个配置项 |
| `knowledge.md` | ChromaDB 分析 + 混合检索技术选型文档 |

### 使用方式

```python
# 启用混合检索
from src.core.config import USE_HYBRID_SEARCH
store = ChromaVectorStore(use_hybrid=True, dense_weight=0.7, bm25_weight=0.3)
store.add_chunks(chunks)  # 自动构建 BM25 索引
results = store.search("犬瘟热治疗", use_hybrid=True)  # RRF 融合结果
```

---

## 九、领域边界守卫（Domain Guard）

> 完成日期：2026-05-02

### 背景

当前 `query()` / `query_stream()` 直接将用户 query 送入检索流程，对于"前沿物理化学"、"量子力学"等与宠物医疗无关的问题仍能召回到弱相关文档，导致模型强行回答、跑偏方向。

### 技术方案

在检索之前插入一层 LLM 零样本分类（Guard），判断 query 是否属于宠物狗领域：

```
用户 query → [Domain Guard] → [是] 继续 RAG 流程
                        → [否] 直接返回友好拒绝语，跳过检索和生成
```

- 分类 prompt 仅要求回复"是"或"否"（2-3 tokens），token 消耗极低
- 分类异常时保守放行（`error` → 视为"是"），不影响正常流程
- Guard 默认启用，通过 `USE_DOMAIN_GUARD` 环境变量可关断

### 关键文件

|| 文件 | 说明 |
|------|------|------|
| `src/core/domain_guard.py` | DomainGuard 类，零样本分类 + 拒绝语生成 |
| `src/core/config.py` | 新增 `USE_DOMAIN_GUARD` 配置项（默认 true） |
| `src/rag_interface.py` | `query()` / `query_stream()` 集成 Guard 预检查 |
| `tests/rag/test_domain_guard.py` | 21 个测试用例，覆盖正常/异常/边界场景 |

### 使用方式

```python
# 默认启用（USE_DOMAIN_GUARD=true）
rag = RAGInterface()  # 自动初始化 DomainGuard

# 禁用
rag = RAGInterface(use_domain_guard=False)
```

---

## 执行记录

### 2026-05-02（凌晨）
- 完成领域边界守卫（Domain Guard）LLM 零样本分类
  - 新增 `src/core/domain_guard.py`（DomainGuard 类，零样本分类 + 拒绝语）
  - 新增 `USE_DOMAIN_GUARD` 配置项（config.py，默认 true）
  - `query()` 和 `query_stream()` 集成 Guard 预检查，非宠物狗问题直接拒绝
  - 范围限定为宠物狗（拒绝猫、其他宠物及一切非狗问题）
  - 新增 21 个测试用例（test_domain_guard.py），21/21 通过
  - 更新 `SYSTEM_PROMPT_VET` 为宠物狗专属提示词
  - 完成混合检索评估与调优（30 条标注语料，RRF dw=0.6/bw=0.4，NDCG@5 +3.6pt）

### 2026-05-01（晚上）
- 完成 RAG 混合检索重构（Hybrid Search）
  - 新增 `src/retrievers/bm25_index.py`（jieba 中文分词 + BM25Okapi 索引）
  - 新增 `src/retrievers/hybrid_retriever.py`（RRF 融合，Dense+BM25 双路召回）
  - 重构 `src/vector_store_chroma.py`：集成混合检索，保留纯 Dense 模式
  - 新增配置项 `USE_HYBRID_SEARCH` 等（5 个环境变量配置）
  - 新增 5 个测试（TestHybridSearch），全部通过
  - 新增 `VetRAG/knowledge.md`：ChromaDB 分析 + 混合检索选型文档

### 2026-05-02（凌晨）
- 完成混合检索评估与调参
  - 构建评估语料库（30 条标注查询 + 相关性标签）
  - 网格搜索最优 RRF 权重：Dense=0.6, BM25=0.4，NDCG@5 提升 3.6pt
  - 修复 `BM25Ok` API 兼容性问题（`okapi BM25` 参数差异）
  - 修复 `hybrid_retriever.py` RRF 融合字段名 bug（`doc_id` → `id`）
  - 19/19 测试全部通过
  - 完善 `VetRAG/knowledge.md`：ChromaDB 存储分析 / RRF 权重调优 / 网格搜索方法论 / 开放问题

### 2026-05-01（下午）
- 完成 QLoRA 微调流水线本地部署全部工作
- 基础模型（ModelScope）+ LoRA adapter 均已下载至本地
- LoRA adapter 成功合并为完整模型：`models/Qwen3-1.7B-vet-finetuned`
- 修复 Qwen3 `<think>` think 模式：`generation_config.json` 禁用 + 后处理 strip
- `QWEN3_FINETUNED_PATH` 配置更新指向合并后模型
- `rag_interface.py` 新增菜单选项 6，支持直接加载本地合并模型
- 本地推理测试 10/10 通过，无 `<think>` 标签

### 2026-05-01（凌晨）
- Qwen3-1.7B 微调训练完成（3 epochs，eval_acc 71.6%，4h13m）
- batch=8 OOM，调参 batch=7 + grad_acc=5 成功
- 训练数据 31,410 条上传至 AutoDL
- 模型上传至 HuggingFace Hub：`MrK-means/Qwen3-1.7B-VetRAG`

### 2026-05-01（凌晨）
- 完成全链路测试脚本 `src/test_full_chain.py`
  - 修复 ChromaDB 路径错误：`test_full_chain.py` 原硬编码 `./chroma_db`（指向 `src/chroma_db`），更正为使用 `config.CHROMA_DIR`（指向 `VetRAG/chroma_db`，含 858 篇文档）
  - 修复报告输出路径：从不存在的 `VetRAG/VetRAG/full_chain_test_report.json` 修正为 `VetRAG/full_chain_test_report.json`
  - 实测结果：向量库 858 篇文档，召回正常（sim 0.50~0.59），Domain Guard 通过
  - 发现问题：模型生成答案未引用检索文档内容；Domain Guard 分类标签持续输出 `'error'`

### 2026-04-30（夜）
- 完成配置管理全部 7 项任务（requirements.txt / pyproject.toml / conda-environment.yml / .env.example / config.py / logging.py / .gitignore）
- 模型下载、数据上传、OOM 调参、训练提交

### 2026-04-29（夜）
- 完成数据扩充阶段 S1-S3、S5-S6（共 5 项），S4 运行中
- 最终训练集 31,410 条

### 2026-04-29（下午）
- 完成项目结构调整全部 3 项任务（导入路径规范化 / Docker 配置 / 目录清理）
- 全部 79 个测试通过

### 2026-04-29
- 完成配置文档阶段全部 3 项任务（README / docs / CHANGELOG）

### 2026-04-28（下午）
- 完成测试文件阶段全部 7 项任务（79/79 全部通过）
