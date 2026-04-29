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
Qwen3_MODEL_PATH: Path = Path(os.getenv(
    "Qwen3_MODEL_PATH",
    str(MODELS_DIR / "Qwen3-0.6B" / "qwen" / "Qwen3-0___6B")
))
QWEN3_FINETUNED_PATH: Path = Path(os.getenv(
    "QWEN3_FINETUNED_PATH",
    str(FINETUNED_DIR / "qwen3-finetuned")
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
    "你是一个专业的兽医助手，同时也需要以温暖、共情的态度回答宠物主人的情感困惑。\n"
    "要求：\n"
    "1. 回答应简洁、清晰，直接针对问题，不要添加无关信息。\n"
    "2. 不要输出参考资料中的原始格式（如 JSON、代码块、Markdown 表格）。\n"
    "3. 不要添加免责声明、来源说明或注释。\n"
    "4. 回答应使用自然、流畅的段落，每段不超过 3 句话。\n"
    "5. 如果参考资料与问题不甚相关，请从你的语料库中进行适当分析，不要自行编造。"
)
