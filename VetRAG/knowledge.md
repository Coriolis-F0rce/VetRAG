# 知识库：技术选型与架构决策记录

> 本文件记录 VetRAG 项目中重要的技术选型决策、研究结论和架构设计，供后续参考。

---

## 一、ChromaDB 使用现状分析

> 分析日期：2026-05-01

### 1.1 当前部署情况

| 项目 | 值 |
|------|-----|
| 客户端 | `chromadb.PersistentClient`（持久化模式） |
| 持久化路径 | `VetRAG/chroma_db/` |
| 集合名 | `veterinary_rag` |
| 索引类型 | HNSW，`hnsw:space = "cosine"` |
| Embedding 模型 | `BAAI/bge-large-zh-v1.5`（1024 维，384 max_seq） |
| Fallback 模型 | `paraphrase-multilingual-MiniLM-L12-v2` |
| 向量维度 | 1024 |
| 归一化策略 | L2 归一化（`normalize_embeddings=True`） |

### 1.2 现有检索流程

```
用户查询
    │
    ▼
create_query_embedding(query)      ← BGE 编码（归一化稠密向量）
    │
    ▼
collection.query(query_embedding)  ← ChromaDB HNSW 近似最近邻搜索
    │
    ▼
distance → similarity = 1 - distance  ← 余弦距离转相似度
    │
    ▼
RAGInterface.query_stream()        ← 阈值过滤（默认 0.4）
    │
    ▼
context + prompt → LLM 生成回答
```

### 1.3 已知问题

| 问题 | 影响 | 原因 |
|------|------|------|
| 学术术语查询跑偏 | 非宠物问题被错误检索 | 缺少领域边界过滤 |
| 医学名词无法精确匹配 | "前沿物理化学" 等非宠物词被误检 | 纯语义检索无法精确关键词匹配 |
| 向量检索对短查询不友好 | 问句级查询检索质量不稳 | 短文本 embedding 表达能力弱 |
| 哈希去重存在碰撞风险 | 极少数文档可能被覆盖 | `abs(hash(content)) % 10**8` 碰撞概率约 1e-8 但不归零 |
| metadata filters 参数存在但未使用 | 无法按 content_type 等字段过滤 | `search()` 的 `filters` 参数从未被调用方填充 |

### 1.4 ChromaDB 版本与功能支持

- **ChromaDB 0.4.22+** 引入了实验性的稀疏向量支持（`SparseEmbeddingRecord`），但生态工具链尚不成熟
- ChromaDB 本身不支持 BM25 全文索引
- 当前版本（`import chromadb; print(chromadb.__version__`）需确认

---

## 二、混合检索（Hybrid Search）技术选型

> 研究日期：2026-05-01

### 2.1 为什么要混合检索

单独使用向量检索（dense）或全文检索（sparse）各有局限：

| 检索方式 | 优势 | 劣势 |
|----------|------|------|
| Dense（向量检索） | 语义泛化能力强、同义词理解 | 短查询弱、专有名词精度低 |
| Sparse（BM25/关键词） | 精确关键词匹配、术语精确 | 无法处理同义词/语义漂移 |
| **Hybrid（混合）** | **两者互补，召回率和精度同时提升** | **实现复杂度增加** |

**我们的场景**特别需要混合检索：
- 宠物医疗场景有大量专有病名（犬瘟热、细小病毒、髋关节发育不良）
- 用户查询可能是口语化描述，语义与文档用词有差异
- 当模型被问及非宠物问题时，需要关键词兜底来识别领域不匹配

### 2.2 候选方案对比

| 方案 | 实现方式 | 优点 | 缺点 |
|------|----------|------|------|
| **A. BM25 + Dense RRF** | rank_bm25 独立索引，RRF 融合 | 成熟稳定、无需新模型下载 | 需管理两套索引 |
| B. BGE-M3 多向量 | BGE-M3 支持生成 dense+sparse+colbert | 单一模型统一输出 | 模型更大（567MB），中文 BM25 效果未充分验证 |
| C. ChromaDB 原生 sparse | `SparseEmbeddingRecord` | 与现有 ChromaDB 无缝集成 | 0.4.22+ 实验性功能，API 可能变 |
| D. ColBERT 晚交互 | Multi-Vector Retriever | 精度最高 | 需额外模型，延迟高 |

### 2.3 推荐方案：A. BM25 + Dense RRF

**理由：**
1. `rank_bm25` 库成熟，无需下载新模型
2. RRF（Reciprocal Rank Fusion）简单有效：
   - `RRF_score(d) = Σ 1/(k + rank_i(d))`，其中 k 通常取 60
   - 两个检索系统各返回一个排名列表，RRF 输出最终融合排名
3. 对现有代码改动最小，可在 `ChromaVectorStore` 外部封装
4. 对中文支持良好（BM25 依赖分词，分词器用 jieba）

### 2.4 融合策略

```
用户查询
    │
    ├──► BGE Dense 检索 ──► Top-K dense scores + ranks
    │                              │
    │                              ▼
    └──► BM25 检索 ───► Top-K bm25 scores + ranks
                                      │
                                      ▼
                              RRF 融合（k=60）
                                      │
                                      ▼
                               融合后 Top-K
                                      │
                                      ▼
                                 LLM 生成
```

### 2.5 分词器选型

| 分词器 | 优点 | 缺点 |
|--------|------|------|
| `jieba` | 中文效果最佳、社区成熟 | 需额外依赖 |
| `pkuseg` | 北大出品、学术语料训练 | 性能略低 |
| `chinese-analyzer`（Whoosh） | 与 ChromaDB 生态近 | 仅限英文 |

**决策：使用 `jieba`**，理由：中文分词效果好，pip 一键安装，无版权风险。

### 2.6 RRF 融合参数

| 参数 | 值 | 说明 |
|------|----|------|
| `k` | 60 | RRF 公式中的常量，k=60 是学术界和工业界的经验最优值 |
| `dense_weight` | 0.7 | Dense 检索权重（可配置） |
| `bm25_weight` | 0.3 | BM25 检索权重（可配置） |
| `top_k` | 5 | 每路检索返回的数量，最终融合后再截断 |

---

## 三、重构实施计划

### 3.1 目标

在**不破坏现有 API 兼容性的前提下**，将 `ChromaVectorStore` 的检索能力从纯 Dense 扩展为 Hybrid（Dense + BM25）。

### 3.2 新增文件

| 文件 | 职责 |
|------|------|
| `src/retrievers/hybrid_retriever.py` | 核心：封装 BM25 + Dense 双路检索 + RRF 融合 |
| `src/retrievers/bm25_index.py` | BM25 索引构建与查询 |
| `src/retrievers/__init__.py` | 模块导出 |

### 3.3 修改文件

| 文件 | 改动 |
|------|------|
| `src/vector_store_chroma.py` | `search()` 方法增加 `use_hybrid=True` 参数 |
| `src/core/config.py` | 新增混合检索配置项 |
| `src/rag_interface.py` | `query_stream()` / `query()` 传递 hybrid 参数 |

### 3.4 向后兼容性

- `ChromaVectorStore.search()` 默认 `use_hybrid=False`，保持现有行为不变
- 所有现有测试无需修改
- 新增 `HybridRetrieverTest` 测试用例覆盖混合检索

### 3.5 索引持久化

- BM25 索引保存至 `chroma_db/bm25_index.pkl`（与 ChromaDB 持久化目录一致）
- 增量更新时：只对新增文档更新 BM25 索引

---

## 四、混合检索实现方案

> 完成日期：2026-05-02

### 4.1 核心实现

#### BM25 索引模块 (`src/retrievers/bm25_index.py`)

- `BM25Retriever` 类：封装 BM25Okapi 算法
- jieba 中文分词（`tokenize_lang="zh"`）
- `build_index()`：全量/增量索引构建
- `search(query, top_k)`：返回 `List[BM25Result]`，含 `chunk_id`、`bm25_score`、`rank`
- 持久化：`bm25_index.pkl` + `bm25_corpus.pkl`，保存至 ChromaDB 同目录
- 参数：`k1=1.5`（词频饱和度）、`b=0.75`（文档长度归一化）

#### 混合检索引擎 (`src/retrievers/hybrid_retriever.py`)

- `HybridRetriever` 类：持有 ChromaDB collection + BM25Retriever
- `_search_hybrid()`：双路并行召回（各取 top-20）→ RRF 融合 → 截断至 top-k
- `_rrf_fuse()`：RRF 公式 `RRF = dw/(60+dense_rank) + bw/(60+bm25_rank)`
- `_search_dense_only()`：纯 Dense 模式（向后兼容，不含 rrf_score 字段）

#### ChromaVectorStore 集成

- `use_hybrid=False`（默认）：纯 Dense 检索，原有行为不变
- `use_hybrid=True`：`add_chunks()` 自动增量同步 BM25 索引
- `_ensure_hybrid_retriever()`：延迟初始化，首次 add_chunks 时才创建
- `_build_bm25_from_chroma()`：从已有 ChromaDB 数据迁移构建 BM25 索引

### 4.2 配置项（`src/core/config.py`）

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `USE_HYBRID_SEARCH` | `false` | 是否启用混合检索 |
| `HYBRID_DENSE_WEIGHT` | `0.7` | Dense 检索权重 |
| `HYBRID_BM25_WEIGHT` | `0.3` | BM25 检索权重 |
| `HYBRID_RRF_K` | `60` | RRF 融合常数 |
| `HYBRID_RETRIEVE_K` | `20` | 每路召回数量 |

### 4.3 依赖

```
rank-bm25>=0.12.0
jieba>=0.42.1
```

---

## 五、混合检索效果评估

> 评估日期：2026-05-02
> 评估工具：`temp/eval_hybrid_search.py`
> 语料：12 篇宠物医疗文档，10 条测试查询（人工标注 relevant_doc_ids）
> 环境：BGE-large-zh-v1.5（CUDA），BM25Okapi（jieba 分词）

### 5.1 评估指标说明

| 指标 | 说明 | 取值范围 |
|------|------|----------|
| **HitRate@K** | Top-K 中是否命中至少一条相关文档 | 0 或 1（取平均） |
| **MRR@K** | 首个命中结果排名的倒数 | 0~1，越高越好 |
| **NDCG@K** | 归一化折损累积增益，考虑排序质量 | 0~1，越高越好 |

### 5.2 总体结果

| 配置 | HitRate@5 | MRR@5 | NDCG@5 |
|------|-----------|--------|---------|
| Dense-only（dw=1.0） | 1.0000 | 1.0000 | 0.9000 |
| **Hybrid 默认（dw=0.7, bw=0.3）** | **1.0000** | **1.0000** | **0.9363** |
| Hybrid 最优（dw=0.6, bw=0.4） | 1.0000 | 1.0000 | **0.9512** |
| Pure BM25（dw=0.0） | 1.0000 | 0.9000 | 0.8538 |

**关键发现**：Hybrid 检索在 NDCG@5 上比纯 Dense 提升 **+3.6pt**（0.9363 vs 0.9000），说明融合后排序质量更高。

### 5.3 网格搜索完整结果

| dense_weight | bm25_weight | HitRate@5 | MRR@5 | NDCG@5 |
|-------------|-------------|-----------|--------|---------|
| 0.0 | 1.0 | 1.0000 | 0.9000 | 0.8538 |
| 0.1 | 0.9 | 1.0000 | 0.9000 | 0.8571 |
| 0.2 | 0.8 | 1.0000 | 0.9500 | 0.8838 |
| 0.3 | 0.7 | 1.0000 | 1.0000 | 0.9044 |
| 0.4 | 0.6 | 1.0000 | 1.0000 | 0.9389 |
| **0.5** | **0.5** | **1.0000** | **1.0000** | **0.9483** |
| **0.6** | **0.4** | **1.0000** | **1.0000** | **0.9512** ← 最优 |
| **0.7** | **0.3** | **1.0000** | **1.0000** | **0.9363** ← 默认 |
| 0.8 | 0.2 | 1.0000 | 1.0000 | 0.9182 |
| 0.9 | 0.1 | 1.0000 | 1.0000 | 0.9000 |
| 1.0 | 0.0 | 1.0000 | 1.0000 | 0.9000 |

**调参结论**：NDCG@5 最优区间为 dw=0.5~0.6（等权重或略偏 Dense）。当前默认值 dw=0.7, bw=0.3 是合理的保守配置。实际推荐 dw=0.6, bw=0.4 或 dw=0.5, bw=0.5。

### 5.4 各查询类型分析

| 查询类型 | 示例 | Dense Hit | Hybrid Hit | 说明 |
|----------|------|-----------|------------|------|
| 精确术语查询 | "犬瘟热 症状" | 1.00 | 1.00 | 两者均完美命中，BM25 关键词精确匹配发挥作用 |
| 口语化查询 | "狗狗突然不吃东西没精神" | 1.00 | 1.00 | Dense 语义理解主导，BGE 语义泛化发挥作用 |
| 混合查询 | "犬瘟热早期症状和治疗方案" | 1.00 | 1.00 | Hybrid 将 d001_symptoms 排第1（提升排序质量） |
| 长查询 | 泰迪挠耳朵、耳道黑色分泌物 | 1.00 | 1.00 | Hybrid 将耳道疾病排第1（排序更精确） |
| 症状推断 | "猫咪精神不好一直睡觉" | 1.00 | 1.00 | 两者均命中泛化文档 d005_overview |

**观察**：HitRate 差异不大（语料库较小，相关文档都在 top-5 内），差异主要体现在 **NDCG@5（排序质量）**。Hybrid 在相同命中数量下，将最相关的文档排得更靠前。

### 5.5 典型查询排名对比（q006："犬瘟热的早期症状和治疗方案"）

| 排名 | Dense-only 检索结果 | Hybrid 检索结果 |
|------|---------------------|-----------------|
| 1 | d001_symptoms（症状） | d001_symptoms（症状）✓ |
| 2 | d001_overview（概览） | d001_treatment（治疗）✓ 提升 |
| 3 | d001_treatment（治疗） | d001_overview（概览） 下降 |
| 4 | d003_overview | d003_treatment |
| 5 | d002_overview | d003_overview |

Hybrid 在第2位就放出了 d001_treatment（治疗），更符合查询意图。

### 5.6 当前推荐配置

```bash
# .env
USE_HYBRID_SEARCH=true
HYBRID_DENSE_WEIGHT=0.6   # 从评估数据看 0.5~0.6 最优
HYBRID_BM25_WEIGHT=0.4
```

---

## 六、FastAPI SSE 流式输出问题排查与修复

> 修复日期：2026-05-02

### 6.1 问题现象

前端使用 `EventSource`（SSE）订阅 `/stream` 接口，回答总是等模型完整生成后才一次性显示，而非逐字流式输出。

### 6.2 根本原因

原实现中，`QwenGenerator.async_stream_generate()` 的核心逻辑是：

```python
def async_stream_generate(self, prompt: str, **kwargs):
    streamer = TextIteratorStreamer(self.tokenizer, ...)
    thread = Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
    thread.start()

    # 阻塞迭代：等 streamer 吐出数据
    for text in streamer:
        yield text
```

问题在于这个 `for` 循环是**同步阻塞迭代**——它在 FastAPI 的 `async def event_generator()` 中被 `for token in rag.generator.async_stream_generate(prompt)` 调用时，整个事件循环被卡在 `q.get(timeout=60)` 上。`TextIteratorStreamer` 通过内部队列通信，当队列有新 token 时 `q.get()` 返回，主线程才继续并 `yield`。

但更深层的问题是：当这个生成器在 FastAPI `async` 函数中运行时，Python 的 GIL 使得阻塞迭代与异步事件循环产生竞争。`q.get(timeout=60)` 每 60 秒才触发一次超时，期间 SSE 连接虽然保持但没有任何数据发送给客户端，直到生成全部完成后才一次性 flush 所有数据。

### 6.3 修复方案：asyncio.Queue + async for（最终版）

```python
# rag_interface.py
import asyncio
from queue import Queue
from threading import Thread

class QwenGenerator:
    async def async_stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()

        def generate():
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs = {**self.generation_config, **kwargs, "streamer": streamer}
                thread = Thread(target=self.model.generate, kwargs={**dict(inputs), **gen_kwargs}, daemon=True)
                thread.start()
                try:
                    for token in streamer:
                        q.put_nowait(token)
                finally:
                    thread.join()
            except Exception as e:
                q.put_nowait(e)
            else:
                q.put_nowait(None)

        thread = Thread(target=generate, daemon=True)
        thread.start()

        while True:
            token = await asyncio.wait_for(q.get(), timeout=60)
            if token is None:
                break
            if isinstance(token, Exception):
                raise token
            yield token
```

```python
# web_api.py
async def event_generator():
    async for token in rag.generator.async_stream_generate(prompt):
        clean_token = rag._clean_output(token)
        yield f"data: {json.dumps({'token': clean_token})}\n\n"
    yield "data: [DONE]\n\n"
```

关键点：
1. **`asyncio.Queue` 替代 `queue.Queue`**：跨线程通信 + 事件循环无缝衔接，`put_nowait` 非阻塞放入，`await q.get()` 异步等待
2. **在 `generate()` 中显式迭代 `streamer`**：`streamer` 本身通过内部队列通信，必须在后台线程中显式迭代它并将 token 放入 `asyncio.Queue`
3. **`None` 替代 `StopIteration` 作为结束哨兵**：避免 Python 生成器协议对 `StopIteration` 的特殊处理导致的意外行为
4. **`async for` 直接迭代**：FastAPI `async def` 中用 `async for` 迭代真正的 `AsyncGenerator`，每个 `yield` 不阻塞事件循环

### 6.4 调试经验

- `run_in_executor` + `next(gen)` 的问题：线程池线程无法从 `TextIteratorStreamer` 的阻塞队列中及时获取 token，导致串行化
- `asyncio.from_thread.run()` 的问题：只能在已有事件循环的线程中调用，从普通 Thread 调用会失败
- `queue.Queue` 阻塞迭代的问题：60 秒超时期间整个线程被卡死，FastAPI 事件循环无法介入
- `StopIteration` 作为值的问题：`yield StopIteration` 会触发 PEP 479 转换，且 `_clean_output(StopIteration)` 会报错

```python
from queue import Queue, Empty
from threading import Thread

def async_stream_generate(self, prompt: str, **kwargs):
    q: Queue = Queue()

    def generate():
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", ...).to(self.device)
            streamer = TextIteratorStreamer(...)
            gen_kwargs = {**self.generation_config, **kwargs, "streamer": streamer}
            self.model.generate(**dict(inputs), **gen_kwargs)
        except Exception as e:
            q.put(StopIteration(e))

    thread = Thread(target=generate, daemon=True)
    thread.start()

    while True:
        try:
            token = q.get(timeout=60)
            if isinstance(token, StopIteration):
                raise token.args[0]
            yield token
        except Empty:
            yield StopIteration
            break
```

关键点：
1. **后台线程运行模型生成**，不阻塞主线程
2. **`TextIteratorStreamer` 自动将 token 放入队列**（内部机制），主线程通过 `q.get()` 异步获取
3. **`yield` 不阻塞事件循环**，FastAPI 的 `StreamingResponse` 能及时将每个 token 推送给客户端
4. **`daemon=True`** 确保进程退出时自动回收线程

### 6.4 相关文件

| 文件 | 改动 |
|------|------|
| `src/rag_interface.py` | `async_stream_generate()` 重构为 Queue 模式，新增 `from queue import Queue, Empty` |
| `web_api.py` | `event_generator()` 中 `for token in rag.generator.async_stream_generate(prompt)` 保持不变（生成器同步，迭代不阻塞） |

### 6.5 验证方法

```bash
# 手动测试 SSE 流式输出
curl -N "http://localhost:5002/stream?question=狗狗发烧怎么办&top_k=5&threshold=0.0"
```

正常情况下应看到 `data: {"token": "x"...}` 逐行输出，而非等待数秒后一次性全部返回。

### 6.6 教训总结

- 在 FastAPI 异步上下文中使用同步生成器时，确保生成器内部不阻塞事件循环
- `Thread` + `Queue` 是将 CPU/GPU 密集型任务（模型推理）从异步事件循环中解耦的标准模式
- `TextIteratorStreamer` 本身就是队列生产者，调用方只需从其队列中异步消费即可

---

## 七、开放问题与后续研究

- [ ] **扩大语料评估**：当前仅 12 篇文档 / 10 条查询，建议在正式语料库上验证
- [ ] **Cross-Encoder Reranking**：RRF 融合后用 Cross-Encoder（如 mteb/reranker）进一步重排
- [ ] **Query Expansion**：HyDE（生成假设文档）或其他查询扩展策略
- [ ] **相似度阈值重新校准**：混合检索后相似度分布变化，需重新确定阈值
- [ ] **性能基准**：混合检索 vs 纯 Dense 的延迟对比（BM25 检索本身极快）

---

## 八、评估体系与 A/B 实验框架

> 文档日期：2026-05-04

### 8.1 当前评估现状

#### 检索阶段（有正式指标）

评估工具位于 `temp/eval_hybrid_search.py`，评估指标：

| 指标 | 说明 |
|------|------|
| **HitRate@K** | Top-K 中是否命中至少一条相关文档 |
| **MRR@K** | 首个命中结果排名的倒数 |
| **NDCG@K** | 归一化折损累积增益，考虑排序质量 |

已验证结果（12 篇文档，10 条查询）：

| 配置 | HitRate@5 | MRR@5 | NDCG@5 |
|------|-----------|-------|--------|
| Dense-only | 1.0000 | 1.0000 | 0.9000 |
| **Hybrid 默认（dw=0.7, bw=0.3）** | **1.0000** | **1.0000** | **0.9363** |
| Hybrid 最优（dw=0.6, bw=0.4） | 1.0000 | 1.0000 | **0.9512** |

#### 生成阶段（无自动评估）

- 现有 `test_three_examples.py` / `test_full_chain.py` 依赖人工审查 JSON 报告中的 `final_answer`
- 微调前后对比使用 BGE 语义相似度（`finetune_steps/test_before/after_finetuning.py`），方法粗糙
- **缺失**：无 ROUGE/BERT-Score，无 LLM-as-Judge，无结构化打分体系

### 8.2 A/B 实验设计

#### 实验组定义

| 组 | 模型 | RAG | 目的 |
|----|------|-----|------|
| 组1 | 微调 Qwen3-1.7B | 是 | 验证微调 + 知识增强的联合效果 |
| 组2 | 原始 Qwen3-1.7B | 是 | 验证 RAG 本身对基础模型的效果 |
| 组3 | 微调 Qwen3-1.7B | 否 | 验证微调模型的纯指令遵循能力 |
| 组4 | 原始 Qwen3-1.7B | 否 | 验证基础模型的纯指令遵循能力（基准线） |

#### 文件结构

```
eval/
├── __init__.py
├── llm_judge.py       # LLM-as-Judge 评估器（BGE 语义相似度 + 规则打分）
├── ab_experiment.py   # 4组实验驱动（模型加载 / RAG模式 / 无RAG模式）
├── testset.json       # 测试数据集（10 条，含参考答案）
└── run_ab.py          # 入口脚本：运行实验 → 评分 → 汇总报告
```

#### 评分维度（各 1-5 分）

| 维度 | 说明 | 是否有参考答案 |
|------|------|----------------|
| **accuracy** | 医学准确性 | 有时（有参考答案时 BGE 相似度映射） |
| **relevance** | 回答相关性 | 始终（关键词匹配） |
| **completeness** | 完整性 | 有时（关键词覆盖 + 长度） |
| **format** | 格式规范性 | 始终（纯规则：emoji/代码块/免责声明检测） |

#### 评分方法

**规则打分（默认）**：无需外部模型，利用 BGE 语义相似度 + 关键词覆盖率 + 格式规则引擎，直接计算分数。

**LLM-as-Judge（可选）**：调用更强的本地模型（如 Qwen3-0.6B），通过结构化 prompt 输出 JSON 格式打分，解析失败时降级为规则打分。

### 8.3 使用方法

```bash
# 运行全部 4 组实验 + 规则评分
python eval/run_ab.py

# 只跑 RAG 相关两组
python eval/run_ab.py --modes finetuned_rag,base_rag

# 使用 LLM-as-Judge 评分
python eval/run_ab.py --judge_method llm

# 两者结合（rule + LLM 对比）
python eval/run_ab.py --judge_method hybrid

# 指定测试集
python eval/run_ab.py --testset eval/testset.json
```

### 8.4 输出文件

| 文件 | 内容 |
|------|------|
| `eval/results/raw_*.json` | 各实验组的原始答案和耗时 |
| `eval/results/scored_*.json` | 带评分结果的详细报告 |
| `eval/results/full_report_*.json` | 汇总报告（含各维度均分和排名） |

### 8.5 模型路径配置

| 用途 | 配置项（`src/core/config.py`） |
|------|-------------------------------|
| 基础模型 | `Qwen3_BASE_MODEL_PATH`（默认 `models/Qwen3-1.7B`） |
| 微调模型 | `QWEN3_FINETUNED_PATH`（默认 `models/Qwen3-1.7B-vet-finetuned`） |

**注意**：`models/` 和 `models_finetuned/` 目录已加入 `.gitignore`，微调权重需单独管理。

### 8.6 测试文件整理

根据 `workflow.md` 规范，以下测试脚本已移入 `temp/`：

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `test_three_examples.py` | `temp/test_three_examples.py` | 微调后 3 案例测试 |
| `src/test_full_chain.py` | `temp/test_full_chain.py` | 全链路测试 |
| `finetune_steps/test_before_finetuning.py` | `temp/test_before_finetuning.py` | 微调前语义相似度测试 |
| `finetune_steps/test_after_finetuning.py` | `temp/test_after_finetuning.py` | 微调后语义相似度测试 |

### 8.7 依赖

无新增外部依赖，使用项目已有组件：
- `QwenGenerator` / `RAGInterface`（生成）
- `ChromaVectorStore`（检索）
- `BAAI/bge-large-zh-v1.5`（嵌入 / 评估）

---

## 九、LLM-as-Judge 评估器设计

### 9.1 设计原理

在宠物医疗问答场景中，由于：

1. 需要判断答案的**医学事实正确性**（很难纯规则判断）
2. 需要评估**回答相关性**和**完整性**（可结合语义相似度）
3. 资源限制（无更强的外部 API 模型）

设计了两层评分体系：

#### 第一层：规则打分（默认，始终启用）

- **BGE 语义相似度**：将参考答案和生成答案分别用 `BAAI/bge-large-zh-v1.5` 编码，计算余弦相似度，映射到 1-5 分的 accuracy
- **关键词覆盖率**：提取参考答案中的中文词（jieba 或字符级 n-gram），计算生成答案的覆盖率，映射到 1-5 分的 relevance
- **格式规则引擎**：检测代码块、emoji、勾叉列表、免责声明、Markdown 标题、数字列表，扣分得到 format 分数

#### 第二层：LLM-as-Judge（可选）

当传入更强的本地 Judge 模型（如 Qwen3-0.6B）时，使用结构化 prompt：

```
你是一位专业的宠物医疗问答质量评估员。
请从 accuracy / relevance / completeness / format 四个维度打分（1-5分）。
输出 JSON：{"accuracy": ..., "relevance": ..., "completeness": ..., "format": ...}
```

解析 JSON 成功则使用 LLM 打分，失败则降级为规则打分。

### 9.2 阈值配置

| 维度 | 阈值（低→高映射 1→5） |
|------|------------------------|
| BGE similarity | 0.5 / 0.6 / 0.7 / 0.8 / 0.9 |
| Keyword coverage | 0.2 / 0.35 / 0.5 / 0.65 / 0.8 |
| Completeness | (sim + coverage) / 2 → 0.25 / 0.4 / 0.55 / 0.7 / 0.85 |

### 9.3 无参考答案场景

当测试集无参考答案（`reference` 为空）时：
- accuracy：固定 3 分（中立）
- relevance：基于问题-回答关键词匹配率
- completeness：基于回答长度（字符数 / 80，映射到 1-5）
- format：同规则打分

---

## 十、A/B 实验结果解读指南

### 10.1 预期结果模式

| 假设 | 组间差异模式 | 结论 |
|------|-------------|------|
| 微调有效 | 组3 > 组4（无RAG时） | 微调提升了纯指令遵循能力 |
| RAG 有效 | 组2 > 组4（有RAG时） | RAG 弥补了基础模型知识 |
| 联合最优 | 组1 ≈ 组2 > 组3 > 组4 | 微调和 RAG 互补 |
| 微调损害 RAG | 组1 < 组2 | 微调过度导致领域偏移，检索知识不再被信任 |

### 10.2 维度解读

- **accuracy**：RAG 组显著高于无 RAG 组，说明知识库内容有效
- **relevance**：微调组高于基础组，说明微调改善了指令跟随
- **completeness**：RAG 组显著高于无 RAG 组，说明检索扩展了回答内容
- **format**：基础模型可能生成更多 Markdown/emoji，微调后可改善

### 10.3 统计显著性

当前测试集规模（10 条）不足以做严格的统计显著性检验。建议在初步分析后：
1. 收集至少 30 条覆盖不同类别（emergency / symptom / nutrition / prevention / ethics）的测试用例
2. 使用 Wilcoxon 符号秩检验（非参数）比较组间差异
3. 在汇总报告中标注置信度（当前为探索性结果）

---

## 十一、Ollama 推理迁移

> 决策日期：2026-05-16

### 11.1 迁移原因

原架构通过 transformers 直接加载 Qwen3-0.6B 模型到内存：

| 问题 | 影响 |
|------|------|
| 模型独占 GPU 显存 | 无法同时运行多个模型（Guard + Generator） |
| Python 进程内加载 | 启动慢（~30s），内存占用高 |
| 微调模型部署繁琐 | LoRA merge → GGUF 的手动流程无统一入口 |
| 不支持并发请求 | 单个进程内推理阻塞 |

### 11.2 方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Ollama** | 统一 API、多模型管理、GGUF 原生支持、并发推理 | 额外服务依赖 |
| transformers 本地 | 无外部依赖、完全控制 | 显存独占、部署复杂 |
| vLLM | 高吞吐、PagedAttention | 仅 Linux、对 1.7B 小模型过度设计 |

**决策：Ollama**，理由：小模型场景够用、GGUF 生态契合微调流程、多模型管理方便（Guard + Generator 可同时加载不同模型）。

### 11.3 模型配置

| 用途 | 模型 | Ollama 名称 |
|------|------|------------|
| 答案生成 | Qwen3-1.7B 微调版 | `vetrag-qwen3-1.7b-vet` |
| 领域守卫 | Qwen3-1.7B 基础版 | `qwen3:1.7b` |
| 生成参数 | temperature=0.0, num_predict=512, repeat_penalty=1.2 | — |

### 11.4 架构变化

```
旧: transformers 直接加载 → model.generate()
新: Ollama REST API → POST /api/generate
```

- `QwenGenerator` 改为 Ollama HTTP 客户端
- `DomainGuard` 独立调用 Ollama（不依赖 QwenGenerator）
- Embedding 模型（BGE）不走 Ollama，仍用 sentence-transformers 直接加载

---

## 十二、犬科药学知识 Chunk 设计

> 实施日期：2026-05-16 ~ 2026-05-17

### 12.1 问题

ChromaDB 的 858 个 chunks 中，药物信息隐式嵌入在 diseases.json 的 treatment 数组里（288→331 种药物），没有独立检索能力。用户查询"阿莫西林剂量"时，只能命中疾病块的粗略描述。

### 12.2 方案

**分两阶段**：先从 diseases.json 提取药物做快速索引，再通过 LLM 补全专业字段。

| 阶段 | 输入 | 输出 | 工具 |
|------|------|------|------|
| 提取 | diseases.json | pharmaceuticals_v0.json（8 字段） | `scripts/extract_drugs.py` |
| 补全 | v0 数据 | pharmaceuticals.json（14 字段） | `scripts/enrich_pharmaceuticals.py`（DeepSeek API） |

### 12.3 数据格式（v1）

```json
{
  "drug_name": "阿莫西林克拉维酸钾",
  "drug_name_en": "Amoxicillin/Clavulanate",
  "drug_class": "青霉素类抗生素",
  "mechanism": "抑制细菌细胞壁合成...",
  "indications": ["皮肤感染", "尿路感染"],
  "dosages": ["12.5-25 mg/kg，PO，BID"],
  "routes": ["PO"],
  "frequencies": ["BID"],
  "durations": ["7-14天"],
  "treatment_names": ["浅表脓皮病"],
  "contraindications": ["青霉素过敏史"],
  "side_effects": [{"effect": "胃肠道反应", "frequency": "常见"}],
  "drug_interactions": ["与甲氨蝶呤合用增加毒性"],
  "monitoring": "监测肝肾功能"
}
```

### 12.4 Chunk 策略

- 每种药物 1 个 chunk（~500-1500 字）
- metadata：drug_name、drug_class、indication_count、dosage_available、has_contraindications、has_interactions
- 与疾病 chunk 通过 drug_name 交叉关联，检索时混合召回

### 12.5 最终数据

| 指标 | 值 |
|------|----|
| 药物总数 | 331 |
| 补全率 | 100%（331/331，0 失败） |
| ChromaDB chunks | 1189（药物 331 + 疾病 492 + 手术 198 + 犬种 100 + 行为 49 + 护理 19） |

---

## 十三、DeepSeek LLM-as-Judge 评估设计

> 实施日期：2026-05-17

### 13.1 为什么需要外部 Judge

原评估方案（knowledge.md 第八、九节）使用 BGE 语义相似度 + 规则引擎，局限明显：

- BGE 相似度无法判断医学事实正确性（"阿莫西林剂量 5mg/kg" vs "阿莫西林剂量 500mg/kg" 可能相似度很高）
- 规则打分对格式敏感但对内容质量不敏感
- 无法做多模型对比（只能两两比较）

### 13.2 DeepSeek Judge 设计

- **匿名化**：4 组答案随机映射为 A/B/C/D，Judge 不知道答案来源
- **5 维度**：accuracy、relevance、completeness、format、safety（各 1-5 分）
- **输出格式**：JSON（各维度分数 + 推理 + 对比分析 + 胜者）
- **并行化**：ThreadPoolExecutor（默认 15 workers），50 题 ~14 秒完成

### 13.3 Prompt 设计要点

- 明确 Judge 角色（资深兽医 + 宠物医学评估专家）
- 每维度给出 1-5 分的具体锚点描述
- 要求输出 reasoning、comparison、winner 字段
- 请求 JSON 输出便于程序解析

### 13.4 已知局限

- DeepSeek 自身可能有知识盲区（兽医领域）
- 不同 Judge 模型可能给出不同结论（无绝对基准）
- format 维度的评分标准有时不一致（对结构化回答的偏好）
