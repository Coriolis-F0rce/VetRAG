"""
统一日志模块
使用 loguru，所有模块统一从此导入 logger
"""
import sys
from pathlib import Path

from loguru import logger as _logger

from .config import LOG_LEVEL, LOG_FILE, LOGS_DIR, PROJECT_ROOT

# 移除 loguru 默认的 handler（会重复输出）
_logger.remove()

# 控制台输出（带颜色）
_log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 控制台 handler
_logger.add(
    sys.stderr,
    format=_log_format,
    level=LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# 文件 handler
if LOG_FILE:
    _log_path = PROJECT_ROOT / LOG_FILE
else:
    _log_path = LOGS_DIR / "vetrag.log"

_log_path.parent.mkdir(parents=True, exist_ok=True)

_logger.add(
    str(_log_path),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level=LOG_LEVEL,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    backtrace=True,
    diagnose=True,
)

# 替换标准库的 root logger，避免重复输出
_logger.info(f"日志系统初始化完成，日志文件: {_log_path}")

# 导出 logger 供其他模块使用
logger = _logger
