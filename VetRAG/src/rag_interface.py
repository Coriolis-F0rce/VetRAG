import os
import re
import sys
from typing import List, Dict, Any, Optional, Generator
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
)
from peft import PeftModel

from .vector_store_chroma import ChromaVectorStore
from .json_loader import VetRAGDataLoader
from .core.config import (
    SYSTEM_PROMPT_VET,
    Qwen3_BASE_MODEL_PATH,
    QWEN3_FINETUNED_PATH,
    USE_HYBRID_SEARCH,
    HYBRID_DENSE_WEIGHT,
    HYBRID_BM25_WEIGHT,
    USE_DOMAIN_GUARD,
)
from .core.domain_guard import DomainGuard


class QwenGenerator:
    def __init__(self, model_path: str, device: str = None, base_model_path: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path = model_path

        print(f"正在加载生成模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            if base_model_path is None:
                import json
                with open(adapter_config_path, "r") as f:
                    adapter_cfg = json.load(f)
                base_model_path = adapter_cfg.get("base_model_name_or_path", "")
            print(f"检测到 LoRA adapter，加载基础模型: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print("LoRA adapter 加载完成，可训练参数:")
            self.model.print_trainable_parameters()
        else:
            print(f"加载完整模型（非 LoRA）: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                self.model = self.model.to(device)

        self.model.eval()

        self.generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def build_chat_prompt(self, system: str, user: str, context: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            user_content = f"参考资料：\n{context}\n\n问题：{user}"
        else:
            user_content = user
        messages.append({"role": "user", "content": user_content})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    # =================================================

    def generate_stream(self, prompt: str, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        gen_kwargs = {**self.generation_config, **kwargs}
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            self.model.generate(
                **inputs,
                streamer=streamer,
                **gen_kwargs
            )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        gen_kwargs = {**self.generation_config, **kwargs}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # 去除 Qwen3 think 标签内容
        answer = answer.replace("<think>\n\n\n\n\n\n\n\n\n\n</think>", "").replace("<think>", "").replace("</think>", "")
        return answer.strip()

    def async_stream_generate(self, prompt: str, **kwargs):
        """
        异步生成器，逐字返回生成的 token（用于流式输出）
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {**self.generation_config, **kwargs, "streamer": streamer}

        # 在后台线程中运行生成
        thread = Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
        thread.start()

        # 从 streamer 中迭代输出
        for text in streamer:
            yield text

class RAGInterface:
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        generator_model_path: Optional[str] = None,
        generator_base_model_path: Optional[str] = None,
        collection_name: str = "veterinary_rag",
        embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
        use_hybrid: bool = USE_HYBRID_SEARCH,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        use_domain_guard: bool = USE_DOMAIN_GUARD,
    ):
        if generator_model_path is None:
            generator_model_path = str(QWEN3_FINETUNED_PATH)
        self.chroma_persist_dir = chroma_persist_dir
        self.generator_model_path = generator_model_path
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
        self.generator = QwenGenerator(generator_model_path, base_model_path=generator_base_model_path)

        # 初始化领域守卫
        self.domain_guard = DomainGuard(
            generator=self.generator,
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
        self.generator.generate_stream(prompt)
        print()

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
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

        return {
            "question": question,
            "answer": answer,
            "retrieved": retrieved,
            "generated": True
        }

    def add_new_data(self, file_path: str) -> Dict:
        if not os.path.exists(file_path):
            return {"success": False, "error": "文件不存在"}
        result = self.vector_store.add_json_file(file_path, self.loader)
        return result

    def add_chunks(self, chunks: List[Dict]) -> Dict:
        return self.vector_store.add_chunks(chunks)

    def get_stats(self) -> Dict:
        vector_stats = self.vector_store.get_collection_stats()
        return {
            "vector_store": vector_stats,
            "generator_loaded": self.generator is not None,
            "generator_model": self.generator_model_path
        }


if __name__ == "__main__":
    print("=" * 60)
    print("兽医 RAG 问答系统（检索 + 生成，流式输出）")
    print("=" * 60)

    # ========== 配置路径（请根据实际情况修改） ==========
    CHROMA_DIR = "./chroma_db"

    path_comfirm = input(
        "请选择后端模型:\n"
        "1：原 Qwen3-0.6B\n"
        "2：微调的 Qwen3-0.6B\n"
        "3：Qwen3-1.7B（AutoDL）\n"
        "4：微调的 Qwen3-1.7B（本地 LoRA，需先下载基础模型）\n"
        "5：Qwen3-1.7B 基础模型（本地，无微调）\n"
        "6：微调的 Qwen3-1.7B（本地合并后完整权重）"
    )
    BASE_PATH = str(Qwen3_BASE_MODEL_PATH)
    FINETUNED_PATH = str(QWEN3_FINETUNED_PATH)
    if path_comfirm == "1":
        MODEL_PATH = r"D:\Backup\PythonProject2\VetRAG\models\Qwen3-0.6B\qwen\Qwen3-0___6B"
        BASE_MODEL_PATH = None
    elif path_comfirm == "2":
        MODEL_PATH = r"D:\Backup\PythonProject2\VetRAG\models_finetuned\qwen3-finetuned"
        BASE_MODEL_PATH = None
    elif path_comfirm == "3":
        MODEL_PATH = "/root/autodl-tmp/huggingface/models/Qwen3-1.7B"
        BASE_MODEL_PATH = None
    elif path_comfirm == "4":
        MODEL_PATH = r"D:\Backup\PythonProject2\VetRAG\models_finetuned\qwen3-1.7b-vet-finetuned"
        BASE_MODEL_PATH = BASE_PATH
    elif path_comfirm == "5":
        MODEL_PATH = BASE_PATH
        BASE_MODEL_PATH = None
    elif path_comfirm == "6":
        MODEL_PATH = FINETUNED_PATH
        BASE_MODEL_PATH = None
    else:
        print("invalid path!")
        sys.exit()
    # =================================================

    # 初始化接口
    rag = RAGInterface(
        chroma_persist_dir=CHROMA_DIR,
        generator_model_path=MODEL_PATH,
        generator_base_model_path=BASE_MODEL_PATH
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
            print(f"生成模型已加载: {stats['generator_loaded']}")
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
