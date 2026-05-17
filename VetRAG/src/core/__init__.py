"""VetRAG 核心模块"""
from .config import (
    API_CORS_ORIGINS,
    API_HOST,
    API_PORT,
    BGE_MODEL_FALLBACK,
    BGE_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DATA_DIR,
    DATA_FILES,
    LOG_FILE,
    LOG_LEVEL,
    PROJECT_ROOT,
    QWEN3_FINETUNED_PATH,
    QWEN3_FINETUNED_PATH_V1,
    SYSTEM_PROMPT_VET,
    USE_DOMAIN_GUARD,
    Qwen3_MODEL_PATH,
)
from .domain_guard import DomainGuard
from .logging import logger


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
