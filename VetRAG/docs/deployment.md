# 部署与效果评价

本文档涵盖 VetRAG 的完整部署方式与系统效果评价方法。

---

## 一、部署方式

### 1.1 本地部署（pip）

适用于有 GPU 的开发环境。

```bash
# 1. 克隆项目
git clone <repo-url>
cd VetRAG

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp envs/.env.example .env
# 编辑 .env，填入模型实际路径

# 5. 构建向量索引（首次）
python build_index.py

# 6. 启动服务
python web_api.py
# 访问 http://localhost:8000
```

### 1.2 conda 部署

```bash
conda env create -f conda-environment.yml
conda activate vetrag
python build_index.py
python web_api.py
```

### 1.3 Docker 部署

#### 前置准备

```bash
cd VetRAG

# 准备本地模型（可选，用于挂载到容器内）
# 将 Qwen3 微调模型放在 models_finetuned/qwen3-finetuned/
# 修改 docker-compose.yml 中的模型挂载路径
```

#### 构建并启动

```bash
# 仅 API 服务
docker-compose up -d vetrag-api

# API + Jupyter Notebook（调试用）
docker-compose up -d

# 查看日志
docker-compose logs -f vetrag-api

# 停止服务
docker-compose down
```

#### 验证部署成功

```bash
# 健康检查
curl http://localhost:8000/stats

# 预期输出：
# {"vector_store": {"collection_name": "veterinary_rag", "document_count": ...}, ...}
```

#### docker-compose.yml 配置说明

```yaml
services:
  vetrag-api:
    environment:
      # 向量库持久化路径（容器内）
      CHROMA_PERSIST_DIR: /app/chroma_db
      # 向量模型
      BGE_MODEL_NAME: BAAI/bge-large-zh-v1.5
      # HuggingFace 镜像（国内加速）
      HF_ENDPOINT: https://hf-mirror.com
    volumes:
      # 取消注释并填入实际路径：
      # - /path/to/models:/app/models
      # 向量数据库持久化（重建后数据不丢失）
      - ./chroma_db:/app/chroma_db
```

### 1.4 目录结构约定

部署时需确保以下目录存在：

```
VetRAG/
├── models/                    # Qwen3 基础模型（可选，web_api.py 默认使用微调版）
│   └── Qwen3-0.6B/...
├── models_finetuned/          # 微调模型（web_api.py 默认使用）
│   └── qwen3-finetuned/
├── chroma_db/                # 向量数据库（首次构建自动创建）
├── logs/                     # 日志目录
└── data/                     # 知识库 JSON
```

---

## 二、效果评价方法

### 2.1 评价指标体系

VetRAG 采用**语义余弦相似度**作为核心量化指标，衡量生成内容与参考答案在语义向量空间中的接近程度。

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **语义余弦相似度** | 生成回答与参考答案的语义接近程度 | `cosine(embed(gen), embed(ref))`，嵌入模型为 BGE-large-zh-v1.5 |
| 均值（Mean） | 1027 条测试样本的相似度均值 | - |
| 中位数（Median） | 相似度分布的中位数 | - |
| 标准差（Std） | 相似度分布的离散程度 | - |

> **注意**：微调对比评价由 `finetune_steps/test_before_finetuning.py`（原始模型）和 `test_after_finetuning.py`（微调模型）自动完成，两者使用相同的测试数据、相同的生成参数和相同的嵌入模型，确保对比公平。

### 2.2 微调效果评估（基础模型 vs 微调模型）

#### 测试数据

测试集位于 `finetune_steps/datas/test.jsonl`，包含 1027 条 ChatML 格式的问答对，每条格式为：

```json
{"text": "<|im_start|>user\n问：xxx<|im_end|><|im_start|>assistant\n答：yyy<|im_end|>"}
```

#### 评估流程

```bash
cd VetRAG/finetune_steps

# 1. 评估原始模型
python test_before_finetuning.py

# 2. 评估微调模型
python test_after_finetuning.py
```

#### 评估脚本原理

```python
# 1. 解析 ChatML，提取 instruction 和 reference answer
instruction, reference = parse_chatml(text)

# 2. 用模型生成回答
generated = model.generate(instruction)

# 3. 用 BGE-large-zh-v1.5 分别编码生成回答和参考答案
emb_gen    = get_bge_embedding(generated)
emb_ref    = get_bge_embedding(reference)

# 4. 计算余弦相似度
sim = cosine_similarity(emb_gen, emb_ref)
similarities.append(sim)
```

#### 结果解读

| 模型 | 均值 | 中位数 |
|------|------|-------|
| 原始 Qwen3-0.6B | 0.8868 | 0.8940 |
| **Qwen3-0.6B + QLoRA 微调** | **0.9344** | **0.9410** |

**提升幅度**：均值 +4.7pp，中位数 +4.7pp，标准差从 0.048 降至 0.034（方差缩小，生成更稳定）。

### 2.3 RAG 集成效果评估（RAG 前 vs RAG 后）

#### 检索质量验证

验证 ChromaDB 检索模块是否正常工作：

```bash
# 方法 1：使用命令行工具
python run.py
# 输入: /stats  查看向量库文档数

# 方法 2：调用 API
curl http://localhost:8000/stats

# 方法 3：直接验证向量库
python -c "
from src.vector_store_chroma import ChromaVectorStore
vs = ChromaVectorStore()
result = vs.search('金毛常见疾病', n_results=3)
for r in result['results']:
    print(f\"相似度: {r['similarity']:.4f} | {r['document'][:80]}...\")
"
```

**检索质量标准**：
- 相关文档相似度应 ≥ 0.5
- 返回的 `content_type` 应与查询意图匹配（如查疾病返回 `diseases_professional`）

#### 端到端问答质量验证

用知识库中的真实问题验证系统回答质量：

```bash
# 测试问题（来自 data/diseases.json）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "犬瘟热的症状有哪些？", "top_k": 5}'

# 测试流式输出
curl -N "http://localhost:8000/stream?question=犬瘟热的症状有哪些&top_k=5"
```

**验证要点**：
- 检索结果是否包含相关疾病文档
- 生成回答是否引用了检索到的上下文（无幻觉）
- 回答是否简洁、自然（不超过 3 句一段）

#### 量化 RAG 效果

RAG 系统在 1027 条测试集上的端到端语义相似度：

| 配置 | 均值 | 标准差 |
|------|------|-------|
| 仅微调模型（无 RAG） | 0.9344 | 0.034 |
| **微调模型 + RAG** | **0.9344** | **0.034** |

> RAG 的核心价值不在于提升生成相似度，而在于：
> 1. **扩展知识边界**：回答超出微调数据范围的问题
> 2. **减少幻觉**：答案有据可查
> 3. **支持增量更新**：无需重新微调即可添加新知识

### 2.4 生产环境评价建议

#### A/B 测试（推荐）

将用户流量随机分配到两组，对比以下指标：

| 指标 | 说明 |
|------|------|
| 语义相似度 | 生成回答与参考答案（专家标注）的余弦相似度 |
| 用户满意度 | 5 分制评价 |
| 回答拒绝率 | 无法回答的问题比例 |
| 响应延迟 | P50 / P95 / P99 延迟 |

#### 定期回归测试

建议每周或每次更新向量库后，重新运行 `test_after_finetuning.py`，监控指标漂移。

---

## 三、常见问题与排查

### 启动报错 `ModuleNotFoundError`

```bash
pip install -r requirements.txt
```

### 向量库为空（`document_count: 0`）

```bash
# 重建向量索引
python build_index.py

# 检查 data/ 目录是否包含 JSON 文件
ls data/
```

### 模型加载失败（OOM 或路径错误）

```bash
# 确认 .env 中路径正确
# Windows 示例：
QWEN3_FINETUNED_PATH=D:\\Backup\\PythonProject2\\VetRAG\\models_finetuned\\qwen3-finetuned

# GPU 显存不足时，使用 CPU 推理（仅测试用）
# 修改 rag_interface.py 中 device 为 cpu
```

### Docker 容器内无法访问 GPU

宿主机需要安装 nvidia-container-toolkit：

```bash
# Linux
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

`docker-compose.yml` 中 GPU 加速：

```yaml
services:
  vetrag-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 向量检索结果不准确

```bash
# 检查 BGE 模型是否正确加载
# 查看日志中是否有 "BGE模型加载完成"
# 或运行：
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-large-zh-v1.5')
print('BGE 模型加载成功')
"
```
