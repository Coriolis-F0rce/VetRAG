# VetRAG 项目任务清单

> 进行中的任务和待办事项。已完成的工作记录见 [accomplishments.md](accomplishments.md)。

---

## 当前进行中

### 1. RAG 优化

- [x] **混合检索实现（Hybrid Search）**：Dense HNSW + BM25 + RRF 融合
  - 新增 `src/retrievers/bm25_index.py` — BM25 关键词索引（jieba 分词）
  - 新增 `src/retrievers/hybrid_retriever.py` — RRF 融合引擎
  - 重构 `src/vector_store_chroma.py` — 集成混合检索，向后兼容
  - 新增 `USE_HYBRID_SEARCH` 等 5 个配置项（config.py）
  - 新增 5 个测试用例（test_vector_store.py），19/19 通过
  - 依赖：rank-bm25>=0.12.0, jieba>=0.42.1
- [x] **领域边界过滤**：LLM 零样本分类，非宠物狗问题直接拒绝
  - 新增 `src/core/domain_guard.py` — DomainGuard 模块
  - 新增 `USE_DOMAIN_GUARD` 配置项（config.py），默认开启
  - `query()` 和 `query_stream()` 集成 Guard，提前过滤
  - 新增 21 个测试用例（test_domain_guard.py），21/21 通过
- [ ] 查询扩展逻辑排查：用户反映"前沿物理化学"等学术问题跑偏到 AI/ML 领域
- [ ] Domain Guard 分类标签持续输出 `'error'`：分类解析逻辑 bug，模型输出被截断导致解析失败，需修复 `_classify` 方法的解析逻辑
- [ ] Prompt 强制引用检索文档内容：当前 System Prompt 未明确要求模型使用文档数据，模型自说自话且添加乱码 emoji 和自创免责声明

---

## 远期计划

### 1. 多轮对话优化

- [ ] 生成多轮对话训练数据集（3-5 轮追问链，模拟真实问诊场景）
- [ ] 将现有单轮数据包装为多轮格式（系统 prompt + 单轮 QA）
- [ ] 修改训练脚本使用 `chat_template` 格式化多轮数据
- [ ] 第二轮微调（可增量训练，复用当前 LoRA adapter）
- [ ] RAG 接口改造：`chat(question, session_id)` 支持多轮历史
- [ ] 多轮对话质量评估（追问相关性、上下文一致性）

### 2. 模型压缩与加速

- [ ] 量化微调后模型（Q4_K_M GGUF），适配本地 CPU 推理
- [ ] 使用 llama.cpp 量化脚本：`quantize` 工具将 fp16 → Q4_K_M
- [ ] 本地 CPU 推理基准测试（生成速度 / 内存占用）
- [ ] 验证量化后模型质量损失可接受

### 3. 上线准备

- [ ] Web UI 多轮对话支持（前端 history 状态管理）
- [ ] Session 管理后端实现（TTL 过期、session 持久化）
- [ ] 压力测试（并发连接数、响应延迟）
- [ ] 部署文档更新（本地部署指南）

### 4. 模型训练继续

- [ ] 考虑增量训练 4-5 epoch（eval_loss 仍未触拐点，当前 eval_acc 71.6%）

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
    {"role": "assistant", "content": "请问它还有其他症状吗？比如呕吐、腹泻或精神萎靡？"},
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
