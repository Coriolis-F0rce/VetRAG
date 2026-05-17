"""
全流程测试脚本 - 测试微调后的 qwen3-1.7B 模型

用法: python test_three_examples.py
默认使用微调后合并权重的 Qwen3-1.7B，DomainGuard 开启
"""
import sys
import io
import time
import json
import traceback
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = str(Path(__file__).resolve().parent / "src")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import QWEN3_FINETUNED_PATH, SYSTEM_PROMPT_VET, USE_DOMAIN_GUARD, CHROMA_DIR
from rag_interface import RAGInterface

MODEL_PATH = str(QWEN3_FINETUNED_PATH)
REPORT_FILE = Path(__file__).resolve().parent.parent / "three_examples_test_report.json"


def print_separator(title: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print('=' * width)


def run_full_chain(question: str, top_k: int = 5, similarity_threshold: float = 0.4):
    report = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "steps": {},
        "final_answer": None,
        "guard_passed": False,
        "retrieved_docs_count": 0,
        "valid_docs_count": 0,
        "error": None,
    }

    try:
        # Step 0: 初始化
        print_separator("Step 0 - 初始化")
        init_start = time.perf_counter()
        rag = RAGInterface(
            chroma_persist_dir=str(CHROMA_DIR),
            generator_model_path=MODEL_PATH,
            generator_base_model_path=None,
            use_domain_guard=True,
        )
        init_elapsed = time.perf_counter() - init_start
        print(f"  模型加载耗时: {init_elapsed:.1f}s")
        print(f"  模型路径: {rag.generator_model_path}")
        print(f"  DomainGuard 启用: {rag.domain_guard.enabled}")
        stats = rag.vector_store.get_collection_stats()
        print(f"  向量库文档数: {stats['document_count']}")
        report["steps"]["init"] = {
            "elapsed_s": round(init_elapsed, 2),
            "model_path": str(rag.generator_model_path),
            "domain_guard_enabled": rag.domain_guard.enabled,
        }

        # Step 1: Domain Guard
        print_separator("Step 1 - Domain Guard（零样本分类）")
        guard_start = time.perf_counter()
        guard_result = rag.domain_guard.check_and_respond(question)
        guard_elapsed = time.perf_counter() - guard_start
        guard_passed = guard_result is None
        guard_classification = rag.domain_guard._classify(question)
        print(f"  Guard 判断: {'通过' if guard_passed else '拒绝'}")
        print(f"  分类输出: '{guard_classification}'")
        print(f"  耗时: {guard_elapsed:.3f}s")
        if guard_result:
            print(f"  拒绝语: {guard_result}")
        report["steps"]["domain_guard"] = {
            "elapsed_s": round(guard_elapsed, 3),
            "classification": guard_classification,
            "passed": guard_passed,
            "rejection_message": guard_result,
        }
        report["guard_passed"] = guard_passed

        if not guard_passed:
            report["final_answer"] = guard_result
            return report, rag

        # Step 2: 检索
        print_separator("Step 2 - 向量检索（Hybrid Search）")
        retrieval_start = time.perf_counter()
        search_results = rag.vector_store.search(question, n_results=top_k)
        retrieval_elapsed = time.perf_counter() - retrieval_start
        all_retrieved = search_results.get("results", [])
        valid_docs = [d for d in all_retrieved if d.get("similarity", 0) >= similarity_threshold]
        print(f"  召回文档数: {len(all_retrieved)}")
        print(f"  有效文档数 (sim >= {similarity_threshold}): {len(valid_docs)}")
        print(f"  耗时: {retrieval_elapsed:.3f}s")

        doc_details = []
        for i, doc in enumerate(all_retrieved):
            sim = doc.get("similarity", 0)
            src = doc.get("metadata", {}).get("source_file", "未知")
            is_valid = "✓" if sim >= similarity_threshold else "✗"
            content = doc["document"]
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"  [{i+1}] sim={sim:.3f} [{is_valid}] src={Path(src).name}")
            print(f"       {preview}")
            doc_details.append({
                "rank": i + 1,
                "similarity": round(sim, 4),
                "is_valid": sim >= similarity_threshold,
                "source": Path(src).name,
                "content_preview": preview,
                "full_content": content,
            })

        report["steps"]["retrieval"] = {
            "elapsed_s": round(retrieval_elapsed, 3),
            "total_retrieved": len(all_retrieved),
            "valid_count": len(valid_docs),
            "similarity_threshold": similarity_threshold,
            "docs": doc_details,
        }
        report["retrieved_docs_count"] = len(all_retrieved)
        report["valid_docs_count"] = len(valid_docs)

        if not valid_docs:
            print("\n  [无可用文档]")
            report["final_answer"] = "未找到相关文档"
            return report, rag

        # Step 3: Prompt 构建
        print_separator("Step 3 - Prompt 构建")
        context_parts = []
        for doc in valid_docs:
            content = rag._clean_document(doc["document"])
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"[相关文档] {content}")
        context = "\n\n".join(context_parts)

        full_prompt = rag.generator.build_chat_prompt(
            system=SYSTEM_PROMPT_VET,
            user=question,
            context=context,
        )
        print(f"  Context 片段数: {len(context_parts)}")
        print(f"  Context 总字符数: {len(context)}")
        print(f"  Full prompt 总字符数: {len(full_prompt)}")

        report["steps"]["prompt"] = {
            "system_prompt": SYSTEM_PROMPT_VET,
            "user_question": question,
            "context_snippets": len(context_parts),
            "context_total_chars": len(context),
            "full_prompt_length": len(full_prompt),
            "context_preview": context[:500],
        }

        # Step 4: 生成
        print_separator("Step 4 - LLM 生成")
        gen_start = time.perf_counter()
        answer = rag.generator.generate(full_prompt)
        gen_elapsed = time.perf_counter() - gen_start
        answer = rag._clean_output(answer)
        print(f"  生成耗时: {gen_elapsed:.1f}s")
        print(f"  生成字符数: {len(answer)}")
        print(f"\n  答案:")
        print(f"  {answer}")

        report["steps"]["generation"] = {
            "elapsed_s": round(gen_elapsed, 1),
            "answer_chars": len(answer),
            "answer": answer,
        }
        report["final_answer"] = answer

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n  [ERROR] {e}")
        print(tb)
        report["error"] = str(e)
        report["traceback"] = tb
        rag = None

    return report, rag


def main():
    questions = [
        "我的2岁公金毛前爪骨折了，我应该怎么处理",
        "我不想养我的狗狗了，我能给他安乐死吗",
        "我的博美身患晚期癌症，医生建议我安乐死，我好难受...",
    ]

    all_reports = []

    for i, q in enumerate(questions):
        print_separator(f"全链路测试 #{i+1} / {len(questions)}")
        print(f"问题: {q}")
        print(f"模型: {MODEL_PATH}")
        report, rag = run_full_chain(q)
        all_reports.append(report)
        print_separator(f"测试 #{i+1} 完成")
        print()

    # 保存报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"报告已保存至: {REPORT_FILE}")

    # 打印汇总
    print_separator("测试结果汇总")
    for i, r in enumerate(all_reports):
        status = "通过" if r.get("final_answer") and not r.get("error") else "失败"
        gen_info = r.get("steps", {}).get("generation", {})
        print(f"\n例{i+1}: {questions[i]}")
        print(f"  状态: {status}")
        print(f"  Guard: {r.get('steps', {}).get('domain_guard', {}).get('classification', 'N/A')}")
        print(f"  检索命中: {r.get('retrieved_docs_count', 0)} 条, 有效 {r.get('valid_docs_count', 0)} 条")
        if gen_info:
            print(f"  生成耗时: {gen_info.get('elapsed_s', 'N/A')}s")
            print(f"  答案长度: {gen_info.get('answer_chars', 0)} 字")
        if r.get("error"):
            print(f"  错误: {r.get('error')}")

    return all_reports


if __name__ == "__main__":
    main()
