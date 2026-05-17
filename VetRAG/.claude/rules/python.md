---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "eval/**/*.py"
  - "tests/**/*.py"
---

# Python 编码规范

- **Python 版本**: 3.11+（使用 `Path`、`str | None` 等新语法）
- **类型注解**: 所有函数参数和返回值必须有类型注解
- **导入顺序**: 标准库 → 第三方库 → 项目内模块（`src.`）
- **配置**: 不硬编码路径/密钥，统一从 `src.core.config` 或环境变量读取
- **异步**: FastAPI 端点用 `async def`，Ollama 流式用 `AsyncGenerator`
- **错误处理**: 仅在系统边界（API 端点、文件 I/O）做 try/except，内部函数信任参数
- **不要写**: docstring 长注释、`# TODO`、三引号多行注释（除非解释 WHY 而非 WHAT）
- **Windows 兼容**: 路径用 `Path` 而非字符串拼接；emoji/特殊字符避免在 print 中使用（GBK 编码问题）
