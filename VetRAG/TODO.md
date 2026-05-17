# VetRAG TODO

> 优先级：P0 = 阻塞项 / P1 = 本周 / P2 = 本月 / P3 = 有空再做

---

## P0 — 阻塞项

- [ ] **CI/CD 通过验证**：确认 `.github/workflows/ci.yml` 在 GitHub Actions 上跑通（ruff + pytest + coverage），修复任何 CI-only 失败
- [x] **依赖 lock 文件**：`requirements-lock.txt` 已生成（143 packages），CI 改用 lock 文件 + CPU torch（2026-05-18 已完成）
- [x] **API 端口统一**：`config.py` 和 `web_api.py` 统一为 `API_PORT=5002`（2026-05-17 已修复）

## P1 — 本周

- [x] **数据 schema 版本化**：`pharmaceuticals.json` 包裹为 `{"schema_version": "1.0", "drugs": [...]}`，build_index.py 启动校验（2026-05-17 已完成）
- [ ] **评估全自动化**：一个命令跑完「A/B 实验 → DeepSeek 评分 → 汇总报告」（`make eval-full`）
- [ ] **模型实验追踪**：评估结果文件嵌入模型元数据（名称、base model、日期、Ollama digest），防止结果与模型对不上
- [ ] **测试修复**：Ollama 迁移后 `test_domain_guard.py`（DomainGuard 构造函数签名变更）和 `test_web_api.py`（AutoModelForCausalLM 不存在）需要更新 mock（当前 40 fail + 11 errors）
- [ ] **测试覆盖盲区**：
  - `src/retrievers/` — 混合检索独立测试
  - `eval/scoring/` — Judge 评分解析逻辑测试
- [ ] **安全检查**：FastAPI 加输入长度限制，CORS 加白名单配置项（默认 `*`，可配），加 `slowapi` 限流

## P2 — 本月

- [ ] **可观测性**：
  - `/health` 端点（检查 Ollama + ChromaDB 连通性）
  - Prometheus metrics（`prometheus-fastapi-instrumentator`）
  - request log 加 `X-Request-ID`
- [ ] **CI 扩展**：
  - 加 `mypy` 类型检查（至少核心模块）
  - 加 `bandit` 安全扫描
  - 加定时任务（每周运行 A/B eval，监控得分趋势）
- [ ] **数据校验**：`data/*.json` 加 JSON Schema 校验，CI 中自动检查
- [ ] **微调配置幂等性保护**：`generation_config.json` 被篡改曾导致实验结果崩溃（见旧 TODO.md），需要显式覆盖所有参数或 pin 配置 hash
- [ ] **Docker 镜像版本化**：制定 image tag 策略（`latest` + `git-sha`），推送 CI 自动构建
- [ ] **Ollama 健康监控**：启动时检查 Ollama 连通性 + 模型是否存在，给出明确错误信息而非超时

## P3 — 有空再做

- [ ] **pre-commit hooks**：`.pre-commit-config.yaml`（ruff + mypy + bandit），本地提交前自动检查
- [ ] **负载测试**：`locust` 或 `oha` 对 `/stream` 和 `/query` 做 QPS 基准
- [ ] **结构化日志**：loguru 输出 JSON 格式（生产环境），方便日志采集
- [ ] **知识库数据治理**：`pharmaceuticals.json` 中自动检测重复药名（别名合并）、过期剂量信息
- [ ] **多语言支持**：英文查询也能检索中文知识库（中英跨语言 embedding 评估）
- [ ] **评估结果可视化**：得分趋势图、模型比较雷达图（Grafana 或静态 HTML 报告）
