"""VetRAG 核心模块"""
from .config import (
    API_HOST,
    API_PORT,
    BGE_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    LOG_FILE,
    LOG_LEVEL,
    PROJECT_ROOT,
    SYSTEM_PROMPT_VET,
    USE_DOMAIN_GUARD,
)
from .domain_guard import DomainGuard
from .logging import logger


__all__ = [
    "PROJECT_ROOT",
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION_NAME",
    "BGE_MODEL_NAME",
    "API_HOST",
    "API_PORT",
    "LOG_LEVEL",
    "LOG_FILE",
    "SYSTEM_PROMPT_VET",
    "USE_DOMAIN_GUARD",
    "logger",
    "DomainGuard",
]
