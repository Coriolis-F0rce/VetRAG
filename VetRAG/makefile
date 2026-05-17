.PHONY: help install run-api build-index eval eval-deepseek test clean \
        merge-lora convert-gguf setup-ollama pull-base

# ============================================================
# VetRAG Makefile
# ============================================================

PYTHON      := python
VETRAG_ROOT := .
RESULTS_DIR := eval/results

help:
	@echo "VetRAG 常用命令"
	@echo ""
	@echo "  make run-api          启动 FastAPI Web 服务"
	@echo "  make build-index      构建向量索引"
	@echo "  make eval-ab          运行 A/B 实验（四组生成）"
	@echo "  make eval-deepseek    用 DeepSeek 对四组答案评分"
	@echo "  make test             运行测试套件"
	@echo "  make lint             代码检查（ruff）"
	@echo "  make clean            清理临时文件和缓存"
	@echo ""
	@echo "  Ollama 相关（迁移后新增）："
	@echo "  make merge-lora       合并 LoRA adapter → 完整权重"
	@echo "  make convert-gguf     完整权重 → GGUF 格式"
	@echo "  make setup-ollama     导入模型到 Ollama"
	@echo "  make pull-base        从 Ollama 官方拉取 Qwen3 基础模型"

run-api:
	$(PYTHON) scripts/web_api.py

build-index:
	$(PYTHON) scripts/build_index.py

eval-ab:
	$(PYTHON) eval/scripts/run_ab.py $(if $(MODES),--modes $(MODES),)

eval-deepseek:
	$(PYTHON) eval/scripts/run_deepseek_judge.py

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# ========== Ollama 迁移流程 ==========
merge-lora:
	$(PYTHON) scripts/merge_lora.py

convert-gguf:
	$(PYTHON) scripts/convert_to_gguf.py $(if $(QUANTIZE),--quantize,)

pull-base:
	ollama pull qwen3:0.6b
	ollama pull qwen3:1.7b

setup-ollama:
	$(PYTHON) scripts/setup_ollama.py
