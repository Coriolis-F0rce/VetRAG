"""
重新生成微调模型答案（带格式清洗）+ 重跑 5 组 Benchmark
========================================================
1. 使用 QwenGenerator（含 _clean_format + temperature=0.05）重新生成 A/C 组答案
2. B/D/E 组复用已有文件
3. 重跑 5 组 listwise 评分

用法：
  python eval/scripts/rerun_cleaned_benchmark.py
  python eval/scripts/rerun_cleaned_benchmark.py --skip_gen  # 跳过生成，直接评分
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from src.core.config import (
    CHROMA_PERSIST_DIR,
    OLLAMA_GENERATOR_MODEL,
    SYSTEM_PROMPT_VET,
)
from src.rag_interface import QwenGenerator
from src.vector_store_chroma import ChromaVectorStore

# reuse the 5-way judge from benchmark script
from eval.scripts.run_benchmark_vs_deepseek import (
    DeepSeekJudge5,
    GROUP_META,
    auto_load_results,
    summarize_5way,
    print_report_5way,
    save_csv_5way,
)


def regenerate_group(
    model_name: str,
    testset: list[dict],
    use_rag: bool,
    vector_store=None,
) -> list[dict]:
    """用 QwenGenerator 重新生成一组答案（含格式清洗）。"""
    label = "RAG" if use_rag else "无RAG"
    print(f"\n  生成: {model_name} + {label} ({len(testset)} 题)")
    generator = QwenGenerator(model_name)

    results = []
    for sample in testset:
        qid = sample["id"]
        question = sample["question"]

        retrieval_info = None
        if use_rag and vector_store:
            sr = vector_store.search(question, n_results=5)
            docs = sr.get("results", [])
            valid = [d for d in docs if d.get("similarity", 0) >= 0.4]
            parts = []
            for d in (valid or docs[:3]):
                content = d["document"]
                if len(content) > 500:
                    content = content[:500] + "…"
                parts.append(f"[相关文档] {content}")
            context = "\n\n".join(parts) if parts else None
            retrieval_info = {
                "docs_retrieved": len(docs),
                "docs_valid": len(valid),
                "context_preview": (parts[0][:300] if parts else "[无]"),
            }
        else:
            context = None

        prompt = generator.build_chat_prompt(
            system=SYSTEM_PROMPT_VET,
            user=question,
            context=context,
        )

        start = time.time()
        answer = generator.generate(prompt)
        elapsed = time.time() - start

        results.append({
            "id": qid,
            "question": question,
            "reference": sample.get("reference", ""),
            "category": sample.get("category", ""),
            "mode": f"{model_name} + {label}",
            "answer": answer,
            "answer_chars": len(answer),
            "elapsed_s": round(elapsed, 2),
            "retrieval": retrieval_info,
            "timestamp": datetime.now().isoformat(),
        })
        sys.stdout.write(f"\r    Q{qid:2d}/{len(testset)}  ({elapsed:.1f}s)")
        sys.stdout.flush()
    print()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_gen", action="store_true", help="跳过生成，直接用已有文件评分")
    parser.add_argument("--workers", type=int, default=15, help="评分并发数")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    results_dir = _project_root / "eval" / "results"
    testset_path = _project_root / "eval" / "datasets" / "testset_50.json"
    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.skip_gen:
        print("=" * 60)
        print("  重新生成微调模型答案（QwenGenerator + _clean_format）")
        print(f"  模型: {OLLAMA_GENERATOR_MODEL}  |  temperature=0.05")
        print("=" * 60)

        # 加载向量库
        vs = ChromaVectorStore(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="veterinary_rag",
            model_name="BAAI/bge-large-zh-v1.5",
        )

        # A: 微调 + RAG
        group_a = regenerate_group(OLLAMA_GENERATOR_MODEL, testset, use_rag=True, vector_store=vs)
        path_a = results_dir / f"raw_FINETUNED_RAG_CLEANED_{timestamp}.json"
        with open(path_a, "w", encoding="utf-8") as f:
            json.dump(group_a, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {path_a}")

        # C: 微调 无RAG
        group_c = regenerate_group(OLLAMA_GENERATOR_MODEL, testset, use_rag=False)
        path_c = results_dir / f"raw_FINETUNED_NO_RAG_CLEANED_{timestamp}.json"
        with open(path_c, "w", encoding="utf-8") as f:
            json.dump(group_c, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {path_c}")

    # ---------- 加载 5 组 ----------
    print(f"\n{'=' * 60}")
    print("  加载 5 组答案")
    print(f"{'=' * 60}")

    results_by_group = {}

    # A: 最新 cleaned 文件，fallback 到旧文件
    for prefix, group_key in [("FINETUNED_RAG_CLEANED", "A"), ("FINETUNED_NO_RAG_CLEANED", "C")]:
        files = sorted(glob.glob(str(results_dir / f"raw_{prefix}_*.json")), key=os.path.getmtime, reverse=True)
        if files:
            path = Path(files[0])
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            results_by_group[group_key] = data.get("results", data) if isinstance(data, dict) else data
            print(f"  [{group_key}] CLEANED: {path.name}")
        else:
            # fallback
            meta = GROUP_META[group_key]
            files = sorted(glob.glob(str(results_dir / f"raw_{meta['file_prefix']}_*.json")), key=os.path.getmtime, reverse=True)
            if files:
                path = Path(files[0])
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                results_by_group[group_key] = data.get("results", data) if isinstance(data, dict) else data
                print(f"  [{group_key}] fallback: {path.name}")

    # B, D, E: 最新文件
    for group_key in ("B", "D", "E"):
        meta = GROUP_META[group_key]
        files = sorted(glob.glob(str(results_dir / f"raw_{meta['file_prefix']}_*.json")), key=os.path.getmtime, reverse=True)
        if files:
            path = Path(files[0])
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            results_by_group[group_key] = data.get("results", data) if isinstance(data, dict) else data
            print(f"  [{group_key}] {path.name}")

    if len(results_by_group) != 5:
        print(f"错误: 需要 5 组，实际 {len(results_by_group)} 组")
        sys.exit(1)

    # ---------- 评分 ----------
    print(f"\n{'=' * 60}")
    print("  5 组 Listwise 评分（匿名）")
    print(f"{'=' * 60}")

    judge = DeepSeekJudge5(api_key=api_key)
    all_scores = judge.score_all_parallel(testset, results_by_group, max_workers=args.workers)

    summary = summarize_5way(all_scores)
    print_report_5way(summary, all_scores)

    # 保存
    report = {
        "timestamp": timestamp,
        "model": "deepseek-chat",
        "testset": str(testset_path),
        "n_samples": len(testset),
        "groups": {g: GROUP_META[g]["label"] for g in ("A", "B", "C", "D", "E")},
        "note": "A/C 组使用 QwenGenerator（_clean_format + temperature=0.05）重新生成",
        "summary": summary,
        "per_question": all_scores,
    }
    json_path = results_dir / f"benchmark_5way_cleaned_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {json_path}")

    csv_path = results_dir / f"benchmark_5way_cleaned_{timestamp}.csv"
    save_csv_5way(all_scores, csv_path)

    # vs 旧结果对比
    print(f"\n{'=' * 60}")
    print("  清洗前后 FORMAT 对比")
    print(f"{'=' * 60}")
    # 加载旧 benchmark
    old_files = sorted(glob.glob(str(results_dir / "benchmark_5way_202*.json")), key=os.path.getmtime, reverse=True)
    if old_files:
        # 找第一次 5way 的（不含 cleaned）
        old_path = None
        for f in old_files:
            if "cleaned" not in f:
                old_path = f
                break
        if old_path:
            with open(old_path, encoding="utf-8") as f:
                old = json.load(f)
            print(f"  {'组别':<28s} {'旧 FORMAT':>10s} {'新 FORMAT':>10s} {'变化':>10s}")
            print(f"  {'-'*56}")
            for group in ("group_A", "group_C"):
                old_fmt = old["summary"].get(group, {}).get("avg_format", 0)
                new_fmt = summary.get(group, {}).get("avg_format", 0)
                diff = new_fmt - old_fmt
                label = summary[group]["label"]
                print(f"  {label:<28s} {old_fmt:>10.3f} {new_fmt:>10.3f} {diff:>+10.3f}")


if __name__ == "__main__":
    main()
