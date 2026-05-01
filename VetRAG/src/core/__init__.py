"""VetRAG 核心模块"""
from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    Qwen3_MODEL_PATH,
    QWEN3_FINETUNED_PATH,
    QWEN3_FINETUNED_PATH_V1,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    BGE_MODEL_NAME,
    BGE_MODEL_FALLBACK,
    API_HOST,
    API_PORT,
    API_CORS_ORIGINS,
    LOG_LEVEL,
    LOG_FILE,
    DATA_FILES,
    SYSTEM_PROMPT_VET,
    USE_DOMAIN_GUARD,
)
from .logging import logger
from .domain_guard import DomainGuard

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "Qwen3_MODEL_PATH",
    "QWEN3_FINETUNED_PATH",
    "QWEN3_FINETUNED_PATH_V1",
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION_NAME",
    "BGE_MODEL_NAME",
    "BGE_MODEL_FALLBACK",
    "API_HOST",
    "API_PORT",
    "API_CORS_ORIGINS",
    "LOG_LEVEL",
    "LOG_FILE",
    "DATA_FILES",
    "SYSTEM_PROMPT_VET",
    "USE_DOMAIN_GUARD",
    "logger",
    "DomainGuard",
]
