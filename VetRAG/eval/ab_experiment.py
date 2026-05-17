"""
A/B 实验驱动模块
================
在 4 种实验配置下运行同一组问题，统一收集答案和评估指标：

组1: 微调Qwen3-1.7B + RAG
组2: 原始Qwen3-1.7B + RAG
组3: 微调Qwen3-1.7B（无 RAG，纯生成）
组4: 原始Qwen3-1.7B（无 RAG，纯生成）

用法：
    python eval/ab_experiment.py
"""

import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path


# 确保项目路径可用
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import ollama

from src.core.config import (
    CHROMA_DIR,
    OLLAMA_GENERATOR_MODEL,
    OLLAMA_GUARD_MODEL,
    SYSTEM_PROMPT_VET,
)
from src.vector_store_chroma import ChromaVectorStore


# ---------- 实验配置枚举 ----------
class ExperimentMode(Enum):
    """4 种实验模式。"""
    FINETUNED_RAG = ("微调模型 + RAG", True, True)
    BASE_RAG = ("基础模型 + RAG", False, True)
    FINETUNED_NO_RAG = ("微调模型 无RAG", True, False)
    BASE_NO_RAG = ("基础模型 无RAG", False, False)

    def __init__(self, label: str, use_finetuned: bool, use_rag: bool):
        self.label = label
        self.use_finetuned = use_finetuned
        self.use_rag = use_rag


# ---------- 模型名称（Ollama）----------
# 微调模型和基础模型在 Ollama 中的名称
FINETUNED_MODEL = OLLAMA_GENERATOR_MODEL  # 默认微调模型
BASE_MODEL = OLLAMA_GUARD_MODEL           # 基础模型（护卫模型）


def resolve_model_name(use_finetuned: bool) -> str:
    """返回 Ollama 模型名。"""
    name = FINETUNED_MODEL if use_finetuned else BASE_MODEL
    print(f"[模型] 使用 Ollama 模型: {name}")
    return name


# ---------- 纯生成（无 RAG） ----------
def generate_no_rag(
    model_name: str,
    question: str,
    system_prompt: str = SYSTEM_PROMPT_VET,
) -> tuple[str, float]:
    """不使用 RAG，直接用 Ollama 模型生成回答。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    start = time.perf_counter()
    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={
            "num_predict": 768,
            "temperature": 0.7,
            "repeat_penalty": 1.3,
        },
        think=False,
    )
    elapsed = time.perf_counter() - start
    answer = _clean_text(response["message"]["content"])
    return answer, elapsed


# ---------- RAG 模式 ----------
def generate_with_rag(
    mode: ExperimentMode,
    question: str,
    vector_store,
    model_name: str,
    top_k: int = 5,
    similarity_threshold: float = 0.4,
) -> tuple[str, float, dict]:
    """使用 RAG 生成回答，返回 (答案, 耗时, 检索信息)。"""
    retrieval_info = {"docs_retrieved": 0, "docs_valid": 0, "context_preview": ""}

    # 检索
    search_results = vector_store.search(question, n_results=top_k)
    all_docs = search_results.get("results", [])
    valid_docs = [d for d in all_docs if d.get("similarity", 0) >= similarity_threshold]
    retrieval_info["docs_retrieved"] = len(all_docs)
    retrieval_info["docs_valid"] = len(valid_docs)

    # 构建 context（按 source_file 去重）
    if not valid_docs:
        context = None
        retrieval_info["context_preview"] = "[无可用文档]"
    else:
        seen_sources = set()
        context_parts = []
        for doc in valid_docs:
            source = doc.get("metadata", {}).get("source_file", id(doc))
            if source in seen_sources:
                continue
            seen_sources.add(source)
            content = doc["document"]
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"[相关文档] {content}")
        context = "\n\n".join(context_parts)
        retrieval_info["context_preview"] = context[:300]

    # 构建消息
    user_content = f"参考资料：\n{context}\n\n问题：{question}" if context else question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_VET},
        {"role": "user", "content": user_content},
    ]

    start = time.perf_counter()
    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={
            "num_predict": 768,
            "temperature": 0.7,
            "repeat_penalty": 1.3,
        },
        think=False,
    )
    elapsed = time.perf_counter() - start

    answer = _clean_text(response["message"]["content"])
    return answer, elapsed, retrieval_info


# ---------- 清理输出 ----------
def _clean_text(text: str) -> str:
    """清理生成结果中的格式违规内容。"""
    # 1. 移除思考标签
    text = re.sub(r"<remo>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?thought>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<THINKING[\s\S]*?</THINKING>", "", text, flags=re.IGNORECASE)

    # 2. 移除代码块（``` ... ```）
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 3. 移除 JSON 块（{ ... } 包含多个键值对的）
    text = re.sub(r"\{[^{}]*\"\w+\"\s*:\s*[^{}]+\}", "", text)

    # 4. 移除 Markdown 标题（## ...）和分隔线（---, ***）
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)

    # 5. 移除 Markdown 列表标记（- item, * item → item）
    text = re.sub(r"^[\s]*[-*]\s+", "", text, flags=re.MULTILINE)
    # 移除编号列表标记（1. 2. 或 1、2、）
    text = re.sub(r"^[\s]*\d+[\.\)、]\s*", "", text, flags=re.MULTILINE)

    # 6. 移除所有 emoji（扩展 Unicode 范围）
    text = re.sub(r"[\U0001F000-\U0001FFFF]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAD6]", "", text)
    text = re.sub(r"[\U00002600-\U000027BF]", "", text)  # Misc Symbols
    text = re.sub(r"[\U0000FE00-\U0000FE0F]", "", text)  # Variation Selectors
    text = re.sub(r"[\U0000200B-\U0000200F]", "", text)  # zero-width
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # 7. 移除常见格式违规字符：✓✗✔✘✅❌⚠️
    text = re.sub(r"[✓✗✔✘✅❌⚠️Ⓜⓛ]", "", text)

    # 8. 移除免责声明整行（含上下文一行）
    disclaimer_lines = text.split("\n")
    cleaned_lines = []
    for line in disclaimer_lines:
        if re.search(r"(免责声明|参考文献|数据来源|参考指南|免责提示|不能替代|仅供参考|具体.*请咨询|请务必.*兽医)", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 9. 去重：移除相邻重复行（保留首条）
    lines = text.split("\n")
    deduped = []
    for line in lines:
        s = line.strip()
        if not s:
            deduped.append(line)
        elif deduped and deduped[-1].strip() == s:
            continue  # 跳过与上一行完全相同的行
        else:
            deduped.append(line)
    text = "\n".join(deduped)

    # 10. 全局去重：相同句子在不同段落里重复超过一次就删
    sentences = re.split(r"[。！？\n]", text)
    seen = {}
    filtered = []
    for s in sentences:
        s = s.strip()
        if len(s) < 8:  # 短句放行
            filtered.append(s)
            continue
        if s in seen:
            continue  # 重复句子跳过
        seen[s] = True
        filtered.append(s)
    text = "。".join(filtered)

    # 11. 清理多余空行
    text = re.sub(r"\n{4,}", "\n\n", text)
    text = re.sub(r"\n{3}", "\n\n", text)

    # 12. 去除首尾空白
    text = text.strip()

    # 13. 结尾重复截断：末尾 150 字符内同一行出现 2 次以上就截断
    last_150 = text[-150:] if len(text) > 150 else text
    tail_lines = last_150.split("\n")
    if len(tail_lines) >= 3:
        counts = {}
        for line in tail_lines:
            ls = line.strip()
            if len(ls) > 5:
                counts[ls] = counts.get(ls, 0) + 1
        for ls, count in counts.items():
            if count >= 2:
                idx = text.rfind(ls)
                if idx >= 0:
                    # 找到倒数第二次出现的位置截断
                    second_last = text.rfind(ls, 0, idx)
                    if second_last >= 0:
                        text = text[:second_last].strip()
                break

    return text


# ---------- 主实验循环 ----------
def run_single_experiment(
    mode: ExperimentMode,
    questions: list[dict],
    vector_store=None,
    model_name: str = None,
) -> list[dict]:
    """
    对一组问题运行特定实验模式。

    questions: [{"id": int, "question": str, "reference": str, "category": str}, ...]
    """
    results = []
    is_rag = mode.use_rag

    for sample in questions:
        qid = sample["id"]
        question = sample["question"]
        reference = sample.get("reference", "")

        print(f"\n  [{mode.label}] Q{qid}: {question[:40]}...")

        try:
            if is_rag:
                answer, elapsed, ret_info = generate_with_rag(
                    mode, question, vector_store, model_name
                )
            else:
                answer, elapsed = generate_no_rag(model_name, question)
                ret_info = {}

            result = {
                "id": qid,
                "question": question,
                "reference": reference,
                "category": sample.get("category", ""),
                "mode": mode.label,
                "answer": answer,
                "answer_chars": len(answer),
                "elapsed_s": round(elapsed, 2),
                "retrieval": ret_info,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            result = {
                "id": qid,
                "question": question,
                "reference": reference,
                "category": sample.get("category", ""),
                "mode": mode.label,
                "answer": f"[ERROR] {e}",
                "elapsed_s": 0.0,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

        results.append(result)
        print(f"    -> 答案长度={len(result.get('answer', ''))}字, 耗时={result.get('elapsed_s', 0):.2f}s")

    return results


def summarize(results: list[dict]) -> dict:
    """生成汇总统计。"""
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    avg_chars = sum(len(r.get("answer", "")) for r in results) / total if total else 0
    avg_time = sum(r.get("elapsed_s", 0) for r in results) / total if total else 0

    return {
        "total_samples": total,
        "errors": errors,
        "avg_answer_chars": round(avg_chars, 1),
        "avg_elapsed_s": round(avg_time, 2),
    }


# ---------- 入口 ----------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="VetRAG A/B 实验")
    parser.add_argument("--questions", type=str, default="",
                        help="JSON 格式的问题列表，或指向 JSONL 文件的路径")
    parser.add_argument("--modes", type=str, default="all",
                        help="运行的模式，逗号分隔，可用值: all, finetuned_rag, base_rag, finetuned_no_rag, base_no_rag")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="eval/results",
                        help="结果输出目录（相对于项目根目录）")
    args = parser.parse_args()

    # 加载问题
    if args.questions:
        if os.path.exists(args.questions):
            with open(args.questions, encoding="utf-8") as f:
                questions = json.load(f)
        else:
            questions = json.loads(args.questions)
    else:
        # 默认测试问题（来自 finetune_steps/datas/test.jsonl 的前 3 条解析）
        questions = [
            {
                "id": 1,
                "question": "我的2岁公金毛前爪骨折了，我应该怎么处理",
                "reference": "",
                "category": "emergency",
            },
            {
                "id": 2,
                "question": "我不想养我的狗狗了，我能给他安乐死吗",
                "reference": "",
                "category": "ethics",
            },
            {
                "id": 3,
                "question": "我的博美身患晚期癌症，医生建议我安乐死，我好难受...",
                "reference": "",
                "category": "ethics",
            },
        ]

    # 解析模式
    mode_map = {
        "finetuned_rag": [ExperimentMode.FINETUNED_RAG],
        "base_rag": [ExperimentMode.BASE_RAG],
        "finetuned_no_rag": [ExperimentMode.FINETUNED_NO_RAG],
        "base_no_rag": [ExperimentMode.BASE_NO_RAG],
    }
    if args.modes == "all":
        active_modes = list(ExperimentMode)
    else:
        active_modes = []
        for m in args.modes.split(","):
            m = m.strip()
            if m in mode_map:
                active_modes.extend(mode_map[m])

    output_dir = _project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}

    for mode in active_modes:
        print(f"\n{'=' * 60}")
        print(f"开始实验: {mode.label}")
        print(f"{'=' * 60}")

        # 加载向量库（RAG 模式需要）
        vector_store = None
        if mode.use_rag:
            print("加载向量数据库...")
            vector_store = ChromaVectorStore(
                persist_directory=str(CHROMA_DIR),
                collection_name="veterinary_rag",
                model_name="BAAI/bge-large-zh-v1.5",
            )

        # 确定 Ollama 模型名
        model_name = resolve_model_name(use_finetuned=mode.use_finetuned)

        # 运行实验
        results = run_single_experiment(mode, questions, vector_store, model_name)
        all_results[mode.label] = results

        # 保存结果
        out_file = output_dir / f"{mode.name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"mode": mode.label, "results": results, "summary": summarize(results)}, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {out_file}")

        # 清理
        if vector_store is not None:
            del vector_store

    # 生成汇总报告
    summary_file = output_dir / f"summary_{timestamp}.json"
    full_summary = {}
    for label, results in all_results.items():
        full_summary[label] = summarize(results)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("实验完成！汇总结果：")
    print(f"{'=' * 60}")
    for label, stats in full_summary.items():
        print(f"\n  {label}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")
    print(f"\n汇总报告: {summary_file}")


if __name__ == "__main__":
    main()
