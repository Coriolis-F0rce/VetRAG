"""
集中配置模块
所有配置项统一从此处读取，根目录为 VetRAG/
"""
import os
from pathlib import Path
from typing import Optional

# ---------- 路径 ----------
_VETRAG_ROOT = Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT: Path = _VETRAG_ROOT
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FINETUNED_DIR: Path = PROJECT_ROOT / "models_finetuned"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
STATIC_DIR: Path = PROJECT_ROOT / "static"

# ---------- 模型路径 ----------
# 从环境变量读取，支持覆盖
# ★ Qwen3-1.7B 基础模型（本地）
Qwen3_BASE_MODEL_PATH: Path = Path(os.getenv(
    "Qwen3_BASE_MODEL_PATH",
    str(MODELS_DIR / "Qwen3-1.7B")
))
# ★ Qwen3-1.7B（AutoDL ModelScope）
Qwen3_MODEL_PATH: Path = Path(os.getenv(
    "Qwen3_MODEL_PATH",
    "/root/autodl-tmp/huggingface/models/Qwen3-1.7B"
))
# ★ Qwen3-1.7B 微调后模型（合并后完整权重，本地）
QWEN3_FINETUNED_PATH: Path = Path(os.getenv(
    "QWEN3_FINETUNED_PATH",
    str(MODELS_DIR / "Qwen3-1.7B-vet-finetuned")
))
QWEN3_FINETUNED_PATH_V1: Path = Path(os.getenv(
    "QWEN3_FINETUNED_PATH_V1",
    str(FINETUNED_DIR / "qwen3-finetuned1")
))

# ---------- ChromaDB ----------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "veterinary_rag")

# ---------- Embedding ----------
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
BGE_MODEL_FALLBACK: str = os.getenv("BGE_MODEL_FALLBACK", "paraphrase-multilingual-MiniLM-L12-v2")

# ---------- 混合检索（Hybrid Search）----------
# 是否启用混合检索（Dense HNSW + BM25 + RRF）
USE_HYBRID_SEARCH: bool = os.getenv("USE_HYBRID_SEARCH", "false").lower() in ("true", "1", "yes")
# Dense 向量检索权重（RRF 融合中使用）
HYBRID_DENSE_WEIGHT: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
# BM25 关键词检索权重
HYBRID_BM25_WEIGHT: float = float(os.getenv("HYBRID_BM25_WEIGHT", "0.3"))
# RRF 融合常数（经验最优值）
HYBRID_RRF_K: int = int(os.getenv("HYBRID_RRF_K", "60"))
# 每路检索召回数量（融合前各召回多少条）
HYBRID_RETRIEVE_K: int = int(os.getenv("HYBRID_RETRIEVE_K", "20"))

# ---------- 领域守卫（Domain Guard）----------
# 是否启用 LLM 零样本领域分类（宠物 / 非宠物）
USE_DOMAIN_GUARD: bool = os.getenv("USE_DOMAIN_GUARD", "true").lower() in ("true", "1", "yes")

# ---------- Hugging Face ----------
HF_ENDPOINT: Optional[str] = os.getenv("HF_ENDPOINT")
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ---------- API ----------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_CORS_ORIGINS: str = os.getenv("API_CORS_ORIGINS", "*")

# ---------- 日志 ----------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Optional[Path] = None
_log_file_env = os.getenv("LOG_FILE")
if _log_file_env:
    LOG_FILE = PROJECT_ROOT / _log_file_env

# ---------- 数据文件 ----------
DATA_FILES: list[str] = [
    "behaviors.json",
    "breeds.json",
    "cares.json",
    "diseases.json",
    "surgeries.json",
]

# ---------- 默认对话系统提示词 ----------
SYSTEM_PROMPT_VET: str = (
    "你是一个专业的宠物狗健康问答助手，请严格按照以下规则回答宠物主人的问题。\n"
    "【核心规则】\n"
    "1. 必须基于【相关文档】中的信息进行回答，禁止编造文档中没有的内容。\n"
    "2. 如果文档信息不足以完整回答，可以结合你的知识进行补充，但必须明确说明哪些来自文档、哪些是你的分析。\n"
    "3. 禁止在回答中添加免责声明、来源说明（如\"根据文档\""
    "或\"参考文献\"）、注释或任何结构化标记。\n"
    "4. 禁止使用 emoji、表情符号、特殊符号装饰回答。\n"
    "5. 只输出纯文字段落，不要输出 JSON、代码块、Markdown 表格。\n"
    "【格式要求】\n"
    "回答使用自然、流畅的段落，每段不超过 3 句话。"
)
