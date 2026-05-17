---
paths:
  - "tests/**"
  - "**/*.test.py"
  - "**/*_test.py"
---

# 测试规范

- **框架**: pytest >= 8.0
- **位置**: `tests/` 目录，镜像 `src/` 结构
- **命名**: `test_<模块名>.py`，函数名 `test_<功能描述>`
- **运行单个测试**: `python -m pytest tests/api/test_web_api.py::test_query -v`
- **Mock 策略**: 外部依赖（Ollama、ChromaDB）用 `unittest.mock.patch`，内部模块直接调
- **覆盖率**: `python -m pytest tests/ --cov=src --cov-report=term`
- **Web API 测试**: 使用 `httpx.AsyncClient` + `pytest.mark.asyncio`
- **不要**: 为合并/转换脚本写测试（它们是工具脚本，非核心逻辑）
