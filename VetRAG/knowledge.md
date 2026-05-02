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
