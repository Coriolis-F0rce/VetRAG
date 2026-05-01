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

## 四、开放问题与后续研究

- [ ] **Reranking**：RRF 融合后是否需要 Cross-Encoder reranking 进一步排序？
- [ ] **Query Expansion**：是否需要 HyDE（假设文档）或其他查询扩展策略？
- [ ] **多路召回上限**：Dense 和 BM25 各召回多少条再融合？当前计划各取 top-20 融合至 top-10
- [ ] **相似度阈值适配**：混合检索后，相似度阈值是否需要重新校准？
- [ ] **性能基准**：混合检索 vs 纯 Dense 的延迟和召回率对比测试
