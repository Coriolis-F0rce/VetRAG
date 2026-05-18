# VetRAG 项目任务清单

> 进行中的任务和待办事项。已完成的工作记录见 [accomplishments.md](accomplishments.md)。

---

## 当前进行中

### 1. A/B 实验框架与评估系统

- [x] **评估框架搭建**：`eval/` 目录
- [x] **测试文件整理**：4 个测试脚本已移入 `temp/`
- [x] **文档更新**：knowledge.md 新增第八～十节
- [ ] 实际运行 4 组实验并收集结果
- [ ] 扩充测试集至 30 条（覆盖 6 类）
- [ ] 统计显著性分析（Wilcoxon 符号秩检验）

### 2. RAG 优化

- [x] 混合检索实现（Hybrid Search）：Dense + BM25 + RRF
- [x] 领域边界过滤：Domain Guard 零样本分类
- [x] Domain Guard 分类解析逻辑修复
- [x] Prompt 输出后处理 `_clean_output`
- [ ] 查询扩展逻辑排查："前沿物理化学"等学术问题跑偏

---

## 面试准备与项目重学习（P0）

> 当前最高优先级。下一步工作计划。

### 目标

系统梳理 VetRAG 项目全貌，提炼技术亮点与难点，为面试技术问答做准备。

### 任务清单

- [ ] **项目全貌梳理**：完整阅读并理解 `knowledge.md`，整理出项目的技术链路图
- [ ] **简历内容核对**：对照 `temp/resume_update.md` 与实际代码，确保简历描述与实现一致
- [ ] **技术难点复盘**：Domain Guard 解析逻辑、FastAPI SSE 流式输出、RRF 融合调参 等关键问题的解决方案
- [ ] **RAG 核心问题**：向量检索原理、混合检索设计、Domain Guard 分类机制
- [ ] **QLoRA 微调原理**：4-bit 量化、LoRA Adapter、deepspeed ZeRO
- [ ] **数据工程**：多源数据清洗、ChatML 格式、7 种数据增强方法
- [ ] **代码走读**：重点文件 `rag_interface.py`、`vector_store_chroma.py`、`bm25_index.py`、`domain_guard.py`

### 面试可能问到的问题

| 问题类型 | 示例 |
|----------|------|
| RAG 原理 | 向量检索 vs 关键词检索的区别？混合检索如何融合？RRF 公式？ |
| 模型微调 | QLoRA 的核心思想？为何用 4-bit 量化？LoRA 的 r 和 alpha 如何调？ |
| 系统设计 | 如何提升 RAG 召回质量？Domain Guard 如何实现零样本分类？ |
| 工程问题 | FastAPI 异步流式输出遇到过什么问题？如何解决 SSE 卡顿？ |
| 项目深度 | eval_acc 71.6% 如何得出？NDCG@5 0.9512 意味着什么？数据增强具体做了什么？ |

---

## 未来计划（按优先级排序）

### P1 — 多轮对话支持

> 真实问诊场景需要连续追问，单轮 QA 能力不足。

- [ ] 生成多轮对话训练数据集（3-5 轮追问链）
- [ ] 将现有单轮数据包装为多轮格式（系统 prompt + 单轮 QA）
- [ ] 修改训练脚本使用 `chat_template` 格式化多轮数据
- [ ] 第二轮增量微调（复用当前 LoRA adapter）
- [ ] RAG 接口改造：`chat(question, session_id)` 支持多轮历史
- [ ] 多轮对话质量评估（追问相关性、上下文一致性）

### P1 — DPO 偏好对齐

> 当前 eval_acc 71.6%，生成质量仍有提升空间。

- [ ] 用微调模型生成对比回答（好/差 pair）
- [ ] 通过 DeepSeek API 进行偏好标注
- [ ] 构建约 1000 条偏好对（prompt / chosen / rejected）
- [ ] 使用 DPO 技术进行第二阶段训练
- [ ] 评估 DPO 前后回答安全性与实用性差异

### P2 — Cross-Encoder 重排序

> Hybrid Search 召回 Top-20 后，用更强的模型精排 Top-5。

- [ ] 调研适合中文的 Cross-Encoder 模型（如 `BAAI/bge-reranker-large`）
- [ ] 集成到 `HybridRetriever` 作为 RRF 之后的精排层
- [ ] 评估重排对 NDCG@5 的提升幅度

### P2 — 性能基准测试

> 当前缺少延迟/Latency 量化指标。

- [ ] 混合检索 vs 纯 Dense 的 P50/P95 延迟
- [ ] 端到端响应时间（检索 + 生成）
- [ ] 生成吞吐量（token/s）基准
- [ ] 建立回归测试：每次改动后自动跑基准，防止性能退化

### P3 — 模型压缩与加速

> Qwen3-1.7B 本地推理约 20-30 token/s，可进一步优化。

- [ ] GGUF Q4 量化（llama.cpp）
- [ ] 本地 CPU 推理测试（生成速度 / 内存占用）
- [ ] 验证量化后模型质量损失可接受

### P3 — Query Expansion（HyDE）

> 口语化查询场景中，用户描述与文档用词差异大。

- [ ] 用微调模型生成"假设回答文档"
- [ ] 基于假设文档检索真实文档
- [ ] 对比 HyDE 前后的召回质量

---

## 参考：QLoRA 超参数记录

> 以下为 Qwen3-1.7B 微调的实测超参数，供参考。

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

---

## 参考：多轮对话优化思路

当前模型为**单轮 QA 格式**，要支持多轮对话需以下改动：

### 1. 训练数据格式重构

```json
// 当前：单轮
{"instruction": "狗发烧怎么办？", "output": "..."}

// 目标：多轮
{
  "conversations": [
    {"role": "user", "content": "我的猫最近不爱吃东西"},
    {"role": "assistant", "content": "请问它还有其他症状吗？..."},
    {"role": "user", "content": "还有点打喷嚏"},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 2. RAG 接口改造

```python
# 当前
def chat(question: str, top_k: int) -> str

# 目标：带 session
def chat(question: str, session_id: str, top_k: int) -> str
# session_id 关联多轮历史，每次检索时拼接 history + question 作为 query
```

### 3. 优先级建议

1. 先等当前微调完成，测试单轮效果
2. 若单轮达标 → 准备多轮数据 → 第二轮微调（可增量训练）
3. 关键指标：单轮准确率 vs 多轮保持率
