"""测试核心配置和日志模块"""
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
# __file__ = D:\Backup\PythonProject2\VetRAG\tests\core\test_config.py
# parents[0] = VetRAG/tests/core/, [1] = VetRAG/tests/, [2] = VetRAG/
sys.path.insert(0, str(project_root))


class TestConfig:
    def test_config_imports(self):
        from src.core import config
        assert config is not None

    def test_project_root_is_path(self):
        from src.core.config import PROJECT_ROOT
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()

    def test_chroma_config_defaults(self):
        from src.core.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
        assert isinstance(CHROMA_PERSIST_DIR, str)
        assert isinstance(CHROMA_COLLECTION_NAME, str)
        assert CHROMA_COLLECTION_NAME == "veterinary_rag"

    def test_bge_model_config(self):
        from src.core.config import BGE_MODEL_NAME
        assert BGE_MODEL_NAME == "BAAI/bge-large-zh-v1.5"

    def test_system_prompt_not_empty(self):
        from src.core.config import SYSTEM_PROMPT_VET
        assert isinstance(SYSTEM_PROMPT_VET, str)
        assert len(SYSTEM_PROMPT_VET) > 10

    def test_api_config_defaults(self):
        from src.core.config import API_HOST, API_PORT
        assert API_HOST == "0.0.0.0"
        assert API_PORT == 5002

    def test_log_level_default(self):
        from src.core.config import LOG_LEVEL
        assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"]


class TestLogging:
    def test_logger_can_be_imported(self):
        from src.core import logger
        assert logger is not None

    def test_logger_has_basic_methods(self):
        from src.core import logger
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_logger_info_does_not_raise(self, capsys):
        from src.core import logger
        logger.info("test info message")

    def test_logger_warning_does_not_raise(self):
        from src.core import logger
        logger.warning("test warning message")

    def test_logger_error_does_not_raise(self):
        from src.core import logger
        logger.error("test error message")
