FROM python:3.11-slim

LABEL maintainer="VetRAG Team"
LABEL description="VetRAG - 兽医领域 RAG 问答系统"

WORKDIR /app

# 安装系统依赖（构建 chromium 用于 playwright 等，或中文语言包）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置 HF 镜像（国内加速）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目源码（.dockerignore 过滤掉不需要的文件）
COPY . .

# 创建必要的目录
RUN mkdir -p /app/logs /app/chroma_db

# 下载 BGE 向量化模型（首次启动时由 sentence-transformers 自动下载）
# 此处提前下载以减少冷启动时间
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "\
    from sentence_transformers import SentenceTransformer; \
    print('预下载 BGE 模型...'); \
    SentenceTransformer('BAAI/bge-large-zh-v1.5'); \
    print('模型下载完成')" 2>&1 || echo "BGE 模型下载失败，将在运行时重试"

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/stats || exit 1

# 启动命令：Web API
CMD ["python", "web_api.py"]
