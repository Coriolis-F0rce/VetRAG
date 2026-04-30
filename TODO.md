# VetRAG 项目任务清单

> 最后更新：2026-05-01（凌晨）

---

## 2026-05-01（凌晨）

### Qwen3-1.7B 微调训练完成

**训练配置**
| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-1.7B |
| 数据 | 训练集 29,839 / 验证集 1,571 |
| batch | 7 × 4 = 28 |
| epoch | 3 |
| 可训练参数 | 17.4M (1.003%) |

**训练结果**
| 指标 | 初始 | 最终 | 变化 |
|------|------|------|------|
| train_loss | 2.82 | ~0.88 | -69% |
| eval_loss | 1.95 | ~1.14 | -42% |
| eval_accuracy | 54% | ~71.6% | +17.6pt |
| entropy | 1.526 | ~0.95 | -38% |
| 训练时长 | — | 4h13m | — |

**结论**
- ✅ 无过拟合迹象，eval_loss 持续下降未触拐点
- ✅ 对比 Qwen3-0.6B 旧数据版（eval_acc 55%），提升显著
- ✅ 泛化能力强，train-eval gap 小
- ⏳ 可考虑继续跑 4-5 epoch 进一步压低 loss

**模型路径**：`/root/autodl-tmp/huggingface/models/qwen3-1.7b-vet-finetuned`

---

## 明日待办（2026-05-01）

### 1. RAG 优化
- [ ] 查询扩展逻辑排查：用户反映"前沿物理化学"等学术问题跑偏到 AI/ML 领域
- [ ] 领域边界过滤：非宠物问题应拒绝回答或友好引导
- [ ] 前端查询扩展/过滤词字典代码定位（如有独立前端项目）

### 2. 训练效果评估
- [x] 模型下载到本地 `VetRAG/models_finetuned/qwen3-1.7b-vet-finetuned/`
- [ ] 本地推理测试：`test_before_finetuning.py` / `test_after_finetuning.py` 对比
- [ ] 关键指标：回答相关性、拒答率、生成质量主观评分
- [ ] 更新 `QWEN3_FINETUNED_PATH` 配置指向本地模型

### 3. 其他
- [ ] 考虑增量训练 4-5 epoch（eval_loss 仍未触拐点）
- [ ] 远期计划参考 TODO.md 远期计划章节

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

## 训练数据扩充 ✅ 已完成

> 完成日期：2026-04-30（凌晨）

### 实施阶段

| 阶段 | 脚本 | 说明 | 条数 |
|------|------|------|------|
| S1 | `merge_and_dedup.py` | 合并增强数据、去重、规则过滤 | 16,552 |
| S2 | `qa_from_diseases.py` | 疾病知识库 → 模板问答对 | 973 |
| S3 | `expand_topics.py` | 品种/行为/手术/日常养护扩充 | 955 |
| S4 | `multi_augment.py` | 指令多样化（API + 规则） | 12,999 |
| S5 | `safety_qa.py` | 通用安全/格式 QA | 56 |
| S6 | `final_merge.py` | 全量合并、去重、导出训练集 | **31,410** |

> S4 正在运行（涉及 API 调用，耗时较长），完成后数据量预计达到 25K+。
> 最终输出：`data_process/final_output/final_training_data.json` + `.jsonl`

### 关键文件

- `data_process/merge_and_dedup.py` — S1：合并去重
- `data_process/qa_from_diseases.py` — S2：疾病→问答对
- `data_process/expand_topics.py` — S3：新话题扩充
- `data_process/multi_augment.py` — S4：指令多样化（运行中）
- `data_process/safety_qa.py` — S5：通用安全 QA
- `data_process/final_merge.py` — S6：最终合并导出

---

## 执行记录

### 2026-04-29（夜）
- 完成数据扩充阶段 S1-S3、S5-S6（共 5 项），S4 运行中
- 创建 `data_process/merge_and_dedup.py`：合并 `augmented_output` + `new_augmented_output`，168 个 JSON 文件从 23,239 条去重至 16,552 条有效数据
- 创建 `data_process/qa_from_diseases.py`：基于 123 种疾病生成 973 条模板问答（8 种模板/病）
- 创建 `data_process/expand_topics.py`：从行为数据/手术列表/品种知识生成 955 条新话题 QA（品种 220 + 行为 126 + 手术 579 + 日常养护 30）
- 创建 `data_process/safety_qa.py`：生成 56 条通用安全/格式 QA（30 条模板 + 6 条格式 + 20 条 API 增强）
- 创建 `data_process/final_merge.py`：合并 S1-S5（不含 S4），去重后最终 **18,524 条**训练数据，输出 JSON + JSONL 格式
- S4 `multi_augment.py` 正在后台运行，采样 500 条数据做 API 增强（同义改写、句式变换、视角转换、情感变化、噪声注入），完成后可再运行 S6 合并

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

## QLoRA 微调流水线 — Qwen3-1.7B 🔄 进行中

> 更新日期：2026-05-01（凌晨）

### 超参数参考（显存安全版）

| 参数 | 值 | 说明 |
|------|----|------|
| `batch_size` | 7 | 极限压榨版（原 8 OOM） |
| `grad_acc` | 5 | 有效 batch = 35 |
| `max_len` | 2048 | 保持最大长度 |
| `epochs` | 3 | |
| `lr` | 2e-4 | |
| `lora_r` | 16 | |
| `lora_alpha` | 32 | |
| `gradient_checkpointing` | True | `use_reentrant=False` |

**显存估算（batch=7, seq_len=2048）：**
- 24层激活峰值 ≈ 8.0 GiB
- 模型+优化器 ≈ 1.07 GiB
- 总计 ≈ 9.47 GiB / 23.52 GiB（剩余 14 GiB 安全垫）

### 2026-05-01（凌晨）

- Step 5 🔄：**训练进行中**，batch=7，显存稳定，未OOM
  - 总 step 数：3198（3 epochs × 1066 steps/epoch）
  - 当前进度：93/3198（epoch 0.09 / 3）
  - 预估剩余：~2.3 小时（训练 2026-04-30 23:42 开始，预计 2026-05-01 02:00 左右完成）
  - 预估费用：~$4.7（余额 20 元，够跑 4 次）
  - 训练集：29,839 条（final_training_data_alpaca.jsonl）
  - 有效 batch：7 × 5 = 35 条/step
  - Loss 趋势：2.816 → 2.458（10 steps）
- Step 6 ⏳：训练完成后
  1. 合并 LoRA adapter → 完整模型（约 3.4 GiB）
  2. 自动上传至 HuggingFace Hub：`MrK-means/Qwen3-1.7B-VetRAG`
     - 逻辑已内置到 `finetune_qlora.py` 末尾，训练结束后自动执行，无需守候
     - 首次使用需先 `huggingface-cli login`（autodl 上执行一次即可）
     - 上传失败时不关机，留给手动补救机会
  3. 本地下载：`git clone https://huggingface.co/MrK-means/Qwen3-1.7B-VetRAG`
  4. 关机（可手动关机或设置 `AUTO_SHUTDOWN_AFTER_UPLOAD = True`）
  - 合并路径：`/root/autodl-tmp/output/Qwen3-1.7B-vetrag-merged`
- Step 7 ⏳：**多轮对话优化**（见下方专题）

### 多轮对话优化思路

当前模型为**单轮 QA 格式**，要支持多轮对话需以下改动：

#### 1. 训练数据格式重构
```json
// 当前：单轮
{"instruction": "狗发烧怎么办？", "output": "..."}

// 目标：多轮
{
  "conversations": [
    {"role": "user", "content": "我的猫最近不爱吃东西"},
    {"role": "assistant", "content": "请问它还有其他症状吗？比如呕吐、腹泻或精神萎靡？"},
    {"role": "user", "content": "还有点打喷嚏"},
    {"role": "assistant", "content": "..."}
  ]
}
```
- 将现有单轮数据**包装成多轮**（系统 prompt → 第一轮用户 → 助手回答）
- 额外生成真实多轮对话数据（3-5轮，模拟追问场景）

#### 2. 训练方式
- 使用 Qwen3 的 `chat_template` 格式化多轮数据
- `max_seq_length` 保持 2048，多轮历史天然更长，注意截断策略
- 可考虑 `group_seq_by_len` packing 优化显存

#### 3. RAG 接口改造
```python
# 当前
def chat(question: str, top_k: int) -> str

# 目标：带 session
def chat(question: str, session_id: str, top_k: int) -> str
# session_id 关联多轮历史，每次检索时拼接 history + question 作为 query
```

#### 4. 优先级建议
1. 先等当前微调完成，测试单轮效果
2. 若单轮达标 → 准备多轮数据 → 第二轮微调（可增量训练）
3. 关键指标：单轮准确率 vs 多轮保持率

### Step 6 补充：模型自动导出到本地方案

> 问题：AutoDL 关机后实例销毁，训练好的模型会丢失。需要关机前把模型传回本地。

**方案 A：rsync 传回本地（推荐）**

```bash
# 训练结束后、关机前，在 autodl 实例上执行
rsync -avz --progress \
    /root/autodl-tmp/output/Qwen3-1.7B-vetrag-merged/ \
    root@本地IP:/d/Backup/PythonProject2/models/Qwen3-1.7B-vetrag/
```

**方案 B：scp 传回本地**

```bash
# 关机前执行
scp -r /root/autodl-tmp/output/Qwen3-1.7B-vetrag-merged/ \
    root@本地IP:/d/Backup/PythonProject2/models/
```

**推荐工作流（Step 6 修正）：**

```
训练结束
  ↓
合并 LoRA adapter → 完整模型
  ↓
rsync/scp 传回本地 models/
  ↓
本地验证模型能加载
  ↓
关机（或保持实例用于后续测试）
```

**rsync/scp 需要本地开启 OpenSSH Server：**

PowerShell 管理员运行：
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```
本地 PowerShell 查 IP：`ipconfig`，确保防火墙开放 22 端口。

---

## 远期计划

### 1. 模型本地化部署
- [x] 训练完成后，模型上传至 HuggingFace Hub（`MrK-means/Qwen3-1.7B-VetRAG`）
  - 上传脚本：`VetRAG/finetune_steps/upload_to_hf.py`
  - 本地下载：`git clone https://huggingface.co/MrK-means/Qwen3-1.7B-VetRAG`
- [x] 本地下载微调后模型：`VetRAG/models_finetuned/qwen3-1.7b-vet-finetuned/`
- [ ] 本地测试加载微调后模型，验证生成质量
- [ ] 配置 `QWEN3_FINETUNED_PATH` 指向本地模型路径
- [ ] 编写本地推理基准测试脚本

### 2. 多轮对话优化
- [ ] 生成多轮对话训练数据集（3-5 轮追问链，模拟真实问诊场景）
- [ ] 将现有单轮数据包装为多轮格式（系统 prompt + 单轮 QA）
- [ ] 修改训练脚本使用 `chat_template` 格式化多轮数据
- [ ] 第二轮微调（可增量训练，复用当前 LoRA adapter）
- [ ] RAG 接口改造：`chat(question, session_id)` 支持多轮历史
- [ ] 多轮对话质量评估（追问相关性、上下文一致性）

### 3. 模型压缩与加速
- [ ] 量化微调后模型（Q4_K_M GGUF），适配本地 CPU 推理
- [ ] 使用 llama.cpp 量化脚本：`quantize` 工具将 fp16 → Q4_K_M
- [ ] 本地 CPU 推理基准测试（生成速度 / 内存占用）
- [ ] 验证量化后模型质量损失可接受

### 4. 上线准备
- [ ] Web UI 多轮对话支持（前端 history 状态管理）
- [ ] Session 管理后端实现（TTL 过期、session 持久化）
- [ ] 压力测试（并发连接数、响应延迟）
- [ ] 部署文档更新（本地部署指南）

---

### 2026-04-30（夜）

- Step 1 ✅：模型已下载至 `/root/autodl-tmp/huggingface/models/Qwen3-1.7B/Qwen/Qwen3-1.7B/`
- Step 2 ✅：数据已上传至 `/root/data/final_training_data_alpaca.jsonl`（31,410 条）
- Step 3 ✅：batch=8 第一次运行 → **OOM**（反向传播峰值 ~23.5 GiB，顶到天花板）
- Step 4 🔄：调参 batch=7 + grad_acc=5，重新提交
  - batch=8 OOM 原因：反向传播峰值 23.51 GiB，刚好在 23.52 GiB 天花板上，碎片化后 5.4 GiB 连续块分配失败
  - batch=7 释放 1.60 GiB → 剩余 1.61 GiB 安全垫
  - 显存监控：`watch -n 2 nvidia-smi`（另开 PS 窗口）
  - 预计耗时：4.7 小时（852 steps/epoch × 6.5s/step × 3 epochs）
  - 若还 OOM：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 降至 batch=5
- 完成配置管理全部 7 项任务
- 创建 `VetRAG/requirements.txt`
- 创建 `VetRAG/pyproject.toml`（含 pytest / ruff / coverage 配置）
- 创建 `VetRAG/conda-environment.yml`
- 创建 `VetRAG/envs/.env.example`
- 创建 `VetRAG/src/core/config.py`（集中配置模块）
- 创建 `VetRAG/src/core/logging.py`（loguru 统一日志）
- 更新 `.gitignore`（补充新增文件、日志、缓存等）
- 更新 `workflow.md`（同步完成项，添加 TODO.md 引用）
