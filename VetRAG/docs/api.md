# API 参考文档

本文档描述 VetRAG Web API 的所有端点。服务启动后，也可通过 `http://localhost:8000/docs` 访问 Swagger UI 交互式文档。

## 基础信息

| 项目 | 值 |
|------|-----|
| 基础 URL | `http://localhost:8000` |
| API 文档 | `http://localhost:8000/docs` |
| OpenAPI JSON | `http://localhost:8000/openapi.json` |

---

## 端点列表

### `GET /`

返回网页 UI（`index.html`）。

**响应类型：** `text/html`

**响应示例：**

```html
<!DOCTYPE html>
<html>
  <head><title>兽医RAG</title></head>
  <body>...</body>
</html>
```

---

### `GET /stats`

获取系统运行状态，包括向量库文档数量和生成器模型加载状态。

**响应类型：** `application/json`

**响应格式：**

```json
{
  "vector_store": {
    "collection_name": "veterinary_rag",
    "document_count": 1234,
    "processed_ids_count": 1234,
    "persist_directory": "./chroma_db",
    "metadata": { ... }
  },
  "generator_loaded": true,
  "generator_model": "D:\\Backup\\PythonProject2\\VetRAG\\models_finetuned\\qwen3-finetuned"
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `vector_store.document_count` | int | ChromaDB 中的文档总数 |
| `vector_store.collection_name` | str | 集合名称 |
| `generator_loaded` | bool | Qwen 生成器是否已加载 |
| `generator_model` | str | 使用的模型路径 |

---

### `POST /query`

非流式问答接口，一次性返回完整回答。

**请求类型：** `application/json`

**请求体：**

```json
{
  "question": "狗狗发烧怎么办？",
  "top_k": 5
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `question` | string | 是 | - | 查询问题 |
| `top_k` | int | 否 | 5 | 从向量库检索的文档数量 |

**响应（成功）：**

```json
{
  "question": "狗狗发烧怎么办？",
  "answer": "当狗狗体温超过39.5°C时应视为发烧。...",
  "retrieved": [
    {
      "document": "犬瘟热会导致发烧...",
      "similarity": 0.85,
      "metadata": {
        "source_file": "diseases.json",
        "content_type": "diseases_professional"
      }
    }
  ],
  "generated": true
}
```

**响应（问题为空）：**

```json
{
  "error": "问题不能为空"
}
```

---

### `GET /stream`

流式问答接口，通过 Server-Sent Events（SSE）实时推送 token。

**查询参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `question` | string | 是 | - | 查询问题 |
| `top_k` | int | 否 | 3 | 检索文档数量 |
| `threshold` | float | 否 | 0.5 | 相似度阈值（低于此值不参与生成） |

**响应类型：** `text/event-stream`

**响应格式：**

首个 token 随 `docs` 字段附带检索到的文档元数据：

```
data: {"token": "当", "docs": [{"similarity": 0.85, "source": "diseases.json", "preview": "犬瘟热会导致..."}]}
data: {"token": "狗"}
data: {"token": "狗"}
...
data: [DONE]
```

`docs` 字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `similarity` | float | 相似度分值 |
| `source` | str | 来源文件名 |
| `preview` | str | 文档前 120 字预览 |

**错误响应：**

```
data: {"error": "问题不能为空"}
```

**curl 示例：**

```bash
curl -N "http://localhost:8000/stream?question=金毛的寿命有多长"
```

---

## 领域守卫（Domain Guard）

`/stream` 和 `POST /query` 端点在检索前均经过领域守卫检查：

- **宠物相关问题** — 正常进入 RAG 流程
- **非宠物问题** — 直接返回友好拒绝语，不进行检索和生成

`/stream` 端点的拒绝响应示例：

```
data: {"token": "抱"}
data: {"token": "歉"}
...
data: [DONE]
```

---

## 请求/响应代码

| HTTP 状态码 | 说明 |
|------------|------|
| 200 | 请求成功 |
| 422 | 请求参数校验失败（缺少必填参数） |
| 500 | 服务器内部错误（如模型加载失败） |

---

## CORS 配置

API 启用全通 CORS（`Access-Control-Allow-Origin: *`），支持跨域请求。

---

## Web UI

`GET /` 返回的网页 UI 提供以下功能：

- 对话输入框
- 发送按钮
- 流式回答展示区域（打字机动画）
- 相似文档展示区（可折叠，默认收起，收到回答后自动展开）
- 响应时间统计（回答完成后显示耗时和字数）

UI 使用 SSE 与 `/stream` 端点交互，无需额外配置。
