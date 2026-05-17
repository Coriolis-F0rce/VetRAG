import os
import re
from collections.abc import AsyncGenerator
from typing import Any

import ollama

from src.core.config import (
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    OLLAMA_GENERATOR_MODEL,
    OLLAMA_GUARD_MODEL,
    SYSTEM_PROMPT_VET,
    USE_DOMAIN_GUARD,
    USE_HYBRID_SEARCH,
)
from src.core.domain_guard import DomainGuard
from src.json_loader import VetRAGDataLoader
from src.vector_store_chroma import ChromaVectorStore


# Ollama 生成参数映射（transformers → Ollama）
_GENERATION_PARAM_MAP = {
    "max_new_tokens": "num_predict",
    "temperature": "temperature",
    "repetition_penalty": "repeat_penalty",
    "top_p": "top_p",
    "top_k": "top_k",
}


class QwenGenerator:
    """基于 Ollama 的 LLM 生成器，替代原来的 transformers 直接加载。"""

    def __init__(self, model_name: str, host: str = None):
        self.model_name = model_name
        self.host = host  # Ollama 服务地址，None 表示默认 http://localhost:11434

        print(f"[Ollama] 使用模型: {model_name}")

        # Ollama 生成默认参数（对应原 generation_config）
        self.generation_config = {
            "num_predict": 512,
            "temperature": 0.0,  # 0.0 = greedy decoding (对应 do_sample=False)
            "repeat_penalty": 1.2,
        }

    def _build_options(self, **kwargs) -> dict:
        """将 generation_config + kwargs 转换为 Ollama options 字典"""
        options = dict(self.generation_config)
        for tf_name, ollama_name in _GENERATION_PARAM_MAP.items():
            if tf_name in kwargs:
                options[ollama_name] = kwargs.pop(tf_name)
        # 剩余未知参数直接传入
        options.update(kwargs)
        return options

    def _get_client(self):
        """获取同步 Ollama 客户端"""
        return ollama.Client(host=self.host) if self.host else ollama

    def _get_async_client(self):
        """获取异步 Ollama 客户端"""
        return ollama.AsyncClient(host=self.host) if self.host else ollama.AsyncClient()

    def build_chat_prompt(self, system: str, user: str, context: str = None) -> str:
        """构建对话 prompt（保持与旧版兼容的字符串输出）"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_content = f"参考资料：\n{context}\n\n问题：{user}" if context else user
        messages.append({"role": "user", "content": user_content})
        # 拼接为可读的 prompt 字符串
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def build_chat_messages(self, system: str, user: str, context: str = None) -> list:
        """构建 Ollama chat 格式的消息列表（Qwen3 原生支持）"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_content = f"参考资料：\n{context}\n\n问题：{user}" if context else user
        messages.append({"role": "user", "content": user_content})
        return messages

    # =================================================

    def generate(self, prompt: str, **kwargs) -> str:
        """同步生成（非流式），返回完整回答"""
        options = self._build_options(**kwargs)
        client = self._get_client()
        response = client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=False,
            options=options,
            think=False,
        )
        answer = response["response"].strip()
        # 去除 Qwen3 think 标签残留
        answer = answer.replace("<think>", "").replace("</think>", "")
        return answer.strip()

    def generate_stream(self, prompt: str, **kwargs):
        """同步流式生成（直接打印到控制台）"""
        options = self._build_options(**kwargs)
        client = self._get_client()
        for chunk in client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            options=options,
            think=False,
        ):
            if not chunk.get("done"):
                print(chunk["response"], end="", flush=True)
        print()  # 结尾换行

    async def async_stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        异步流式生成器，通过 Ollama AsyncClient 逐 token 产出，
        供 FastAPI SSE 端点使用。
        """
        options = self._build_options(**kwargs)
        async_client = self._get_async_client()
        response = await async_client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True,
            options=options,
            think=False,
        )
        async for chunk in response:
            if not chunk.get("done"):
                yield chunk["response"]



class RAGInterface:
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        generator_model_name: str | None = None,
        guard_model_name: str | None = None,
        collection_name: str = "veterinary_rag",
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        use_hybrid: bool = USE_HYBRID_SEARCH,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        use_domain_guard: bool = USE_DOMAIN_GUARD,
    ):
        if generator_model_name is None:
            generator_model_name = OLLAMA_GENERATOR_MODEL
        self.chroma_persist_dir = chroma_persist_dir
        self.generator_model_name = generator_model_name
        self.use_hybrid = use_hybrid
        self.use_domain_guard = use_domain_guard

        print(f"加载向量数据库: {chroma_persist_dir}")
        self.vector_store = ChromaVectorStore(
            persist_directory=chroma_persist_dir,
            collection_name=collection_name,
            model_name=embedding_model_name,
            use_hybrid=use_hybrid,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        self.loader = VetRAGDataLoader()
        self.generator = None
        self.generator = QwenGenerator(generator_model_name)

        # 初始化领域守卫（使用 Ollama 基础模型做零样本分类）
        if guard_model_name is None:
            guard_model_name = OLLAMA_GUARD_MODEL
        self.domain_guard = DomainGuard(
            guard_model_name=guard_model_name,
            enabled=use_domain_guard,
        )

        stats = self.vector_store.get_collection_stats()
        print(f"当前向量库包含 {stats['document_count']} 个文档")

    def _clean_document(self, text: str) -> str:
        # 去除 ```json ... ``` 代码块
        text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
        # 去除其他代码块
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # 去除 Markdown 标题符号（可选）
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        return text.strip()

    def _clean_output(self, text: str) -> str:
        """
        对生成结果进行输出后处理，清除残留的 emoji 和乱码。
        作为 System Prompt 约束的 safety net。
        """
        if not text:
            return text
        # 过滤 emoji 和常见特殊符号
        text = re.sub(
            r'[\U0001F000-\U0001F9FF]'  # emoji
            r'|[\U00002702-\U000027B0]'  # dingbats
            r'|[\U0001F600-\U0001F64F]'  # emoticons
            r'|[\U00002600-\U000026FF]'  # misc symbols
            r'|[\U0001F300-\U0001F5FF]'  # symbols & pictographs
            r'|[\U0001F680-\U0001F6FF]'  # transport & map symbols
            r'|[\U0001FA00-\U0001FAFF]'  # chess, symbols
            r'|[\U0001FB00-\U0001FBFF]'  # symbols legacy
            r'|[\U0001F000-\U0001FFFF]'  # full emoji range
            r'|[\U0000200B-\U0000200F]'  # zero-width chars (ZWSP, ZWNJ, ZWJ, etc.)
            r'|[\U0001F1E6-\U0001F1FF]'  # regional indicator symbols (flag emojis)
            r'|[☑☒✓✗✔✘]+',              # 混合勾叉
            '', text
        )
        # 过滤控制字符（换行/空格除外）
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()

    def query_stream(self, question: str, top_k: int = 5, similarity_threshold: float = 0.4):
        # 领域守卫：非宠物问题直接拒绝
        rejection = self.domain_guard.check_and_respond(question)
        if rejection:
            print(f"\n[Domain Guard] {rejection}")
            return

        search_results = self.vector_store.search(question, n_results=top_k)
        all_retrieved = search_results.get("results", [])

        valid_docs = [doc for doc in all_retrieved if doc.get("similarity", 0) >= similarity_threshold]

        print("\n检索结果：")
        for i, doc in enumerate(all_retrieved):
            sim = doc.get("similarity", 0)
            src = doc.get("metadata", {}).get("source_file", "未知")
            valid_flag = "✓" if sim >= similarity_threshold else "✗ (低于阈值)"
            content = doc["document"]
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"{i + 1}. 相似度 {sim:.3f} [{valid_flag}] - {os.path.basename(src)}")
            print(f"   预览: {preview}\n")

        if not valid_docs:
            default_answer = "抱歉，我只擅长回答宠物狗健康、护理、疾病等方面的问题。如果您有关于狗狗的疑问，欢迎提出！"
            print(f"答案：{default_answer}")
            return

        if self.generator is None:
            print("生成模型未加载，仅显示检索结果。")
            return

        context_parts = []
        for doc in valid_docs:
            content = self._clean_document(doc["document"])
            if len(content) > 500:
                content = content[:500] + "…"
            context_parts.append(f"[相关文档] {content}")
        context = "\n\n".join(context_parts)

        system_msg = SYSTEM_PROMPT_VET
        prompt = self.generator.build_chat_prompt(
            system=system_msg,
            user=question,
            context=context
        )
        # =================================================

        print("答案：", end="", flush=True)
        for token in self.generator.generate_stream(prompt):
            print(self._clean_output(token), end="", flush=True)
        print()

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        # 领域守卫：非宠物问题直接拒绝
        rejection = self.domain_guard.check_and_respond(question)
        if rejection:
            return {
                "question": question,
                "answer": rejection,
                "retrieved": [],
                "generated": False,
                "rejected_by_guard": True,
            }

        search_results = self.vector_store.search(question, n_results=top_k)
        retrieved = search_results.get("results", [])

        if not retrieved or self.generator is None:
            return {
                "question": question,
                "answer": None if self.generator is None else "未找到相关文档",
                "retrieved": retrieved,
                "generated": False
            }

        # 构建上下文（清理文档）
        context_parts = []
        for doc in retrieved:
            content = self._clean_document(doc["document"])
            if len(content) > 500:
                content = content[:500] + "…"
            context_parts.append(f"[相关文档] {content}")
        context = "\n\n".join(context_parts)

        system_msg = SYSTEM_PROMPT_VET
        prompt = self.generator.build_chat_prompt(
            system=system_msg,
            user=question,
            context=context
        )
        # =================================================

        answer = self.generator.generate(prompt)
        answer = self._clean_output(answer)

        return {
            "question": question,
            "answer": answer,
            "retrieved": retrieved,
            "generated": True
        }

    def add_new_data(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {"success": False, "error": "文件不存在"}
        result = self.vector_store.add_json_file(file_path, self.loader)
        return result

    def add_chunks(self, chunks: list[dict]) -> dict:
        return self.vector_store.add_chunks(chunks)

    def get_stats(self) -> dict:
        vector_stats = self.vector_store.get_collection_stats()
        return {
            "vector_store": vector_stats,
            "generator_loaded": self.generator is not None,
            "generator_model": self.generator_model_name,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("兽医 RAG 问答系统（检索 + 生成，流式输出）")
    print("=" * 60)

    CHROMA_DIR = "./chroma_db"

    # Ollama 模型选择（需先在 Ollama 中创建这些模型）
    MODEL_OPTIONS = {
        "1": ("vetrag-qwen3-0.6b-base", "vetrag-qwen3-1.7b-base"),
        "2": ("vetrag-qwen3-0.6b-vet", "vetrag-qwen3-1.7b-base"),
        "3": ("vetrag-qwen3-0.6b-vet1", "vetrag-qwen3-1.7b-base"),
        "4": ("vetrag-qwen3-1.7b-base", "vetrag-qwen3-1.7b-base"),
        "5": ("vetrag-qwen3-1.7b-vet", "vetrag-qwen3-1.7b-base"),
    }

    choice = input(
        "请选择生成模型 (Ollama):\n"
        "1：Qwen3-0.6B 基础\n"
        "2：Qwen3-0.6B 微调\n"
        "3：Qwen3-0.6B 微调 v1\n"
        "4：Qwen3-1.7B 基础\n"
        "5：Qwen3-1.7B 微调\n"
    )
    if choice in MODEL_OPTIONS:
        gen_model, guard_model = MODEL_OPTIONS[choice]
    else:
        print("无效选择，使用默认: vetrag-qwen3-0.6b-vet")
        gen_model, guard_model = "vetrag-qwen3-0.6b-vet", "vetrag-qwen3-1.7b-base"

    rag = RAGInterface(
        chroma_persist_dir=CHROMA_DIR,
        generator_model_name=gen_model,
        guard_model_name=guard_model,
    )

    print("\n输入问题开始问答，输入 'quit' 退出。")
    print("其他命令：")
    print("  /stats   - 查看系统状态")
    print("  /add 文件路径 - 添加新 JSON 文件")
    print("=" * 60)

    while True:
        user_input = input("\n问题: ").strip()
        if user_input.lower() in ["quit", "exit", "退出"]:
            break
        elif user_input.startswith("/stats"):
            stats = rag.get_stats()
            print(f"向量库文档数: {stats['vector_store']['document_count']}")
            print(f"生成模型: {stats['generator_model']}")
            continue
        elif user_input.startswith("/add "):
            file_path = user_input[5:].strip()
            result = rag.add_new_data(file_path)
            if result.get("success"):
                print(f"添加成功，新增 {result.get('added', 0)} 个文档")
            else:
                print(f"添加失败: {result.get('error', '未知错误')}")
            continue

        if not user_input:
            continue

        # 使用流式查询
        rag.query_stream(user_input, top_k=3, similarity_threshold=0.5)
