# 🐾 VetRAG – 兽医知识问答系统

VetRAG 是一个基于 **RAG（检索增强生成）** 架构的智能问答系统，专为宠物健康、疾病、护理等场景设计。它通过检索本地兽医知识库，并结合大语言模型生成准确、共情的回答。

---

## ✨ 主要特性

- **混合检索**：使用 BGE 嵌入模型将文本向量化，通过 Chroma 向量库进行语义检索。
- **大模型生成**：集成 Qwen 系列模型（0.6B 或微调版），支持流式输出。
- **Web 界面**：基于 FastAPI 提供简洁美观的聊天前端，支持逐字打字效果。
- **数据增强**：包含大量数据预处理脚本，支持微调数据的构建与扩充。
- **模块化设计**：核心组件独立，便于二次开发和定制。

---

## 📁 项目结构
VetRAG/
├── data_process/ # 综合性数据处理脚本
│ └── find_faq/ # FAQ 数据清洗与增强
├── finetune_steps/ # 模型微调相关脚本
│ ├── prepare_data.py # 准备微调数据
│ ├── qlora_finetune.py # QLoRA 微调入口
│ ├── test_after_finetuning.py # 微调后测试
│ └── test_before_finetuning.py # 微调前基线测试
├── src/ # 核心 RAG 模块
│ ├── json_loader.py # 加载 JSON 数据并分块
│ ├── vector_store_chroma.py # Chroma 向量库封装
│ ├── rag_pipeline.py # 构建索引管线
│ └── increment_manager.py # 增量更新管理
├── static/ # 前端静态文件
│ └── index.html # 聊天界面
├── build_index.py # 构建向量索引入口
├── rag_interface.py # RAG 核心接口（检索 + 生成）
├── web_api.py # FastAPI Web 服务
├── download_qwen3.py # 下载 Qwen 模型脚本
└── requirements.txt # 依赖列表（建议自行生成）

text

### 关键文件说明

- **`data_process/`**：包含原始数据处理、FAQ 增强等脚本，用于生成高质量的微调和检索数据。
- **`finetune_steps/`**：存放模型微调的全流程脚本，从数据准备到训练、测试。
- **`src/`**：RAG 系统的底层实现，包括数据加载、向量化、向量库操作等。
- **`build_index.py`**：读取 JSON 数据，调用 `src` 中的模块构建 Chroma 向量索引。
- **`rag_interface.py`**：封装了检索和生成逻辑，提供 `RAGInterface` 类供 Web 服务调用。
- **`web_api.py`**：基于 FastAPI 的 Web 服务，提供 `/stream` 流式接口和 `/` 聊天界面。

---

## 🚀 快速开始

### 环境准备

1. **克隆仓库**
   ```bash
   git clone https://github.com/Coriolis-F0rce/VetRAG.git
   cd VetRAG
安装依赖

没有 requirements.txt，需要手动安装主要依赖：

bash
pip install fastapi uvicorn chromadb sentence-transformers transformers torch
下载模型权重
项目使用 Qwen 系列模型，您可以通过 download_qwen3.py 下载（需指定模型路径）或自行放置：

bash
python download_qwen3.py
微调后的模型权重请放置在 models_finetuned/ 目录下。

构建向量索引
确保您已将 JSON 数据文件（如 behaviors.json、diseases.json 等）放入 data/ 目录，然后运行：

bash
python build_index.py
该脚本会读取数据、分块、向量化并保存到 chroma_db/ 目录中。

启动 Web 服务
bash
python web_api.py
服务默认运行在 http://localhost:8000，打开浏览器即可看到聊天界面。
