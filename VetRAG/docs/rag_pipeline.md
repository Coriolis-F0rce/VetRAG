# RAG Pipeline 技术文档

本文档描述 VetRAG 检索增强生成（RAG）Pipeline 的完整实现细节。

## 概述

VetRAG Pipeline 包含两个主要阶段：**知识入库**（Ingestion）和 **查询生成**（Query Generation）。

---

## 阶段一：知识入库

```
data/*.json
    │
    ▼
VetRAGDataLoader.load_all_files()
    │
    ├── dispatch by filename
    ▼
_parse_file_based_on_type()
    │
    ├── behaviors.json  → _parse_behaviors()
    ├── breeds.json    → _parse_breeds()
    ├── cares.json     → _parse_cleaned_dog_care()
    ├── diseases.json  → _parse_diseases()
    └── surgeries.json → _parse_surgeries()
    │
    ▼ (List[Dict] chunks)
ChromaVectorStore.add_chunks()
    │
    ├── filter: len(content) >= 10
    ├── deduplicate: hash(content) → chunk_id
    ▼
BGE Embedding Model (BAAI/bge-large-zh-v1.5)
    │
    ├── normalize_embeddings=True
    └── cosine similarity
    │
    ▼
ChromaDB Persistent Storage
```

### 语义分块策略

**行为数据（behaviors）** — 每个行为 1 个 chunk：

```json
{
  "content": "行为名称: 扑跳\n行为类别: 兴奋行为\n描述: ...\n前因: ...\n后果: ...",
  "metadata": {
    "content_type": "behaviors",
    "behavior_name": "扑跳",
    "behavior_category": "兴奋行为",
    "intervention_level": "无/观察"
  },
  "source_file": "behaviors.json",
  "content_type": "behaviors"
}
```

**疾病数据（diseases）** — 每个疾病拆分为 4 个语义块：

| Chunk | 内容 | 元数据字段 |
|-------|------|-----------|
| Chunk 1 | 基本信息 + 症状 + 诊断 | `chunk_type: disease_overview_symptoms` |
| Chunk 2 | 治疗方案 + 预后 + 紧急处理 | `chunk_type: treatment_prognosis_emergency` |
| Chunk 3 | 流行病学 + 预防 + 误诊风险 | `chunk_type: epidemiology_prevention` |
| Chunk 4 | FAQ + 参考文献 | `chunk_type: faq_references` |

**手术数据（surgeries）** — 每个手术拆分为 2 个块：

| Chunk | 内容 |
|-------|------|
| Chunk 1 | 手术概述 + 适应症 + 术前准备 |
| Chunk 2 | 术后护理 + 并发症 + 预后 + 费用 |

### 去重机制

每个 chunk 通过内容 hash 生成唯一 ID：

```python
content_hash = abs(hash(content)) % (10 ** 8)
chunk_id = f"chunk_{content_hash:08d}"
```

已处理的 ID 持久化到 `processed_ids.pkl`，重启后可跳过已存在的 chunk。

### 文档清洗

在向量化和生成前，文档内容会经过清洗：

```python
import re

def _clean_document(text):
    # 移除 JSON 代码块
    text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)
    # 移除其他代码块
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # 移除 Markdown 标题
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()
```

---

## 阶段二：查询生成

```
用户问题
    │
    ▼
ChromaVectorStore.search(query, n_results=5)
    │
    ├── create_query_embedding(query)
    ├── cosine similarity search
    ▼
结果过滤（threshold >= 0.5）
    │
    ▼
文档清洗 (_clean_document)
    │
    ▼
上下文组装
context = "参考资料：\n[相关文档1]\n\n[相关文档2]\n\n..."
    │
    ▼
QwenGenerator.build_chat_prompt()
    │
    ├── system: SYSTEM_PROMPT_VET
    ├── context: 检索到的文档
    └── user: 用户问题
    │
    ▼
Qwen3 LLM 本地推理
    │
    ├── streaming: SSE token 流
    └── non-streaming: 完整字符串
    │
    ▼
返回响应
```

### 相似度阈值

`/stream` 端点支持 `threshold` 参数（默认 0.5），低于该相似度的检索结果不参与生成：

```python
valid_docs = [doc for doc in all_retrieved if doc.get("similarity", 0) >= threshold]
```

### 系统提示词

```python
SYSTEM_PROMPT_VET = """
你是一个专业的兽医助手，同时也需要以温暖、共情的态度回答宠物主人的情感困惑。
要求：
1. 回答应简洁、清晰，直接针对问题，不要添加无关信息。
2. 不要输出参考资料中的原始格式（如 JSON、代码块、Markdown 表格）。
3. 不要添加免责声明、来源说明或注释。
4. 回答应使用自然、流畅的段落，每段不超过 3 句话。
5. 如果参考资料与问题不甚相关，请从你的语料库中进行适当分析，不要自行编造。
"""
```

---

## 增量更新

`increment_manager.py` 提供增量更新能力：

1. **文件监控**：检测 JSON 文件变化（通过 hash）
2. **增量入库**：仅处理新增或变更的 chunk
3. **版本管理**：记录各文件的版本和入库时间

---

## 核心类说明

### `VetRAGDataLoader`

| 方法 | 说明 |
|------|------|
| `load_all_files(file_paths)` | 加载并解析多个 JSON 文件 |
| `_parse_file_based_on_type()` | 根据文件名派发到对应解析器 |
| `_parse_diseases()` | 疾病语义分块（4 chunks/病） |
| `_parse_surgeries()` | 手术语义分块（2 chunks/手术） |

### `ChromaVectorStore`

| 方法 | 说明 |
|------|------|
| `add_chunks(chunks, batch_size=50)` | 批量添加文档 |
| `search(query, n_results, filters)` | 语义检索 |
| `get_collection_stats()` | 获取集合统计 |
| `clear_collection()` | 清空集合 |

### `QwenGenerator`

| 方法 | 说明 |
|------|------|
| `build_chat_prompt(system, user, context)` | 构建聊天模板 |
| `generate(prompt)` | 非流式生成 |
| `generate_stream(prompt)` | 生成器流式 |
| `async_stream_generate(prompt)` | 异步 SSE 流（Web API 用） |
