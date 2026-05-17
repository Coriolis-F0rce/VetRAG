"""
A/B 实验运行脚本
================
驱动完整流程：4组实验 → LLM-as-Judge 评分 → 汇总报告。

用法：
    python eval/run_ab.py                          # 运行全部 4 组 + 评分
    python eval/run_ab.py --modes finetuned_rag,base_rag  # 只跑特定组
    python eval/run_ab.py --judge_method rule                   # 只用规则打分（不用 LLM）
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


_eval_dir = Path(__file__).resolve().parent.parent
_project_root = _eval_dir.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from eval.ab_experiment import ExperimentMode, run_single_experiment, summarize
from eval.llm_judge import JudgeScorer
from src.core.config import CHROMA_DIR
from src.vector_store_chroma import ChromaVectorStore


def load_testset(path: str | None = None) -> list[dict]:
    """加载测试集。"""
    path = _project_root / "eval" / "datasets" / "testset.json" if path is None else Path(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("questions", data)


def run_judgment(
    all_results: dict,
    scorer: JudgeScorer,
    judge_model_name: str = "vetrag-qwen3-0.6b-base",
    method: str = "rule",
) -> dict:
    """
    对实验结果进行评分。

    method: "rule" = 规则打分（默认），"llm" = LLM-as-Judge，"hybrid" = 两者结合
    """
    scored = {}
    for mode_label, results in all_results.items():
        scored[mode_label] = []
        for r in results:
            q = r.get("question", "")
            ans = r.get("answer", "")
            ref = r.get("reference", "")

            if method in ("rule", "hybrid"):
                if ref:
                    scores = scorer.score_with_reference(q, ref, ans)
                else:
                    scores = scorer.score_without_reference(q, ans)
                scores["method"] = "rule"
            else:
                scores = {"method": "unknown"}

            if method in ("llm", "hybrid"):
                llm_scores = scorer.score_by_llm(q, ref, ans, judge_model_name)
                scores["llm_judge"] = llm_scores
                if method == "llm":
                    scores = llm_scores
                    scores["method"] = "llm"

            # 合并实验结果与评分
            merged = {**r, "evaluation": scores}
            scored[mode_label].append(merged)

    return scored


def generate_report(scored: dict, output_dir: Path, timestamp: str) -> dict:
    """生成汇总报告并保存。"""
    summary = {}

    for mode_label, results in scored.items():
        dims = ["accuracy", "relevance", "completeness", "format"]
        all_scores = {d: [] for d in dims}

        for r in results:
            ev = r.get("evaluation", {})
            for d in dims:
                all_scores[d].append(ev.get(d, 0))

        summary[mode_label] = {
            "n_samples": len(results),
            "avg_scores": {
                d: round(sum(all_scores[d]) / len(all_scores[d]), 4) if all_scores[d] else 0
                for d in dims
            },
            "avg_total": round(
                sum(sum(all_scores[d]) for d in dims) / (len(dims) * len(results)) if results else 0,
                4,
            ),
            "baseline_stats": summarize(results),
        }

    report = {
        "timestamp": timestamp,
        "summary": summary,
        "rankings": _build_rankings(summary),
    }

    report_file = output_dir / f"full_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_report(report)
    print(f"\n完整报告已保存: {report_file}")
    return report


def _build_rankings(summary: dict) -> dict:
    """基于各维度平均分构建排名。"""
    dims = ["accuracy", "relevance", "completeness", "format", "avg_total"]
    rankings = {}
    for dim in dims:
        key = f"avg_{dim}" if dim != "avg_total" else dim
        sorted_modes = sorted(
            [(m, s.get(key, 0)) for m, s in summary.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        rankings[dim] = [{"rank": i + 1, "mode": m, "score": round(s, 4)} for i, (m, s) in enumerate(sorted_modes)]
    return rankings


def _print_report(report: dict):
    """打印报告。"""
    print("\n" + "=" * 70)
    print("  VetRAG A/B 实验汇总报告")
    print("=" * 70)

    for mode_label, stats in report["summary"].items():
        print(f"\n【{mode_label}】")
        for dim, score in stats["avg_scores"].items():
            bar = "#" * int(score) + "-" * (5 - int(score))
            print(f"  {dim:15s}: {score:.3f} [{bar}]")
        bar = "#" * int(stats["avg_total"]) + "-" * (5 - int(stats["avg_total"]))
        print(f"  {'总均分':15s}: {stats['avg_total']:.3f} [{bar}]")

    print("\n" + "-" * 70)
    print("  排名（按总均分）")
    print("-" * 70)
    for entry in report["rankings"]["avg_total"]:
        print(f"  #{entry['rank']}  {entry['mode']:30s}  {entry['score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="VetRAG A/B 实验 + 评估报告")
    parser.add_argument("--modes", type=str, default="all",
                        help="运行模式，逗号分隔: finetuned_rag, base_rag, finetuned_no_rag, base_no_rag, all")
    parser.add_argument("--testset", type=str, default="",
                        help="测试集路径（JSON）")
    parser.add_argument("--judge_method", type=str, default="rule",
                        choices=["rule", "llm", "hybrid"],
                        help="评分方法: rule=规则, llm=LLM-as-Judge, hybrid=两者结合")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="eval/results",
                        help="结果输出目录")
    parser.add_argument("--no_rag_override", type=str, default="",
                        help="无RAG模式的system prompt覆盖（如不需要可留空）")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试集
    testset = load_testset(args.testset or None)
    print(f"加载测试集: {len(testset)} 条问题")
    for item in testset:
        print(f"  Q{item['id']}: {item['question'][:50]}... [{item.get('category', '?')}]")

    # 解析实验模式
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

    # 初始化评估器
    scorer = JudgeScorer()

    # Ollama Judge 模型名（用于 LLM-as-Judge 评分）
    judge_model_name = "qwen3:0.6b"
    # 检查 Ollama 中是否有此模型，没有则退回到规则打分
    if args.judge_method in ("llm", "hybrid"):
        try:
            import ollama
            ollama.show(judge_model_name)
            print(f"\nJudge 模型: {judge_model_name} (Ollama)")
        except Exception:
            print(f"\n  [警告] Ollama 中未找到 {judge_model_name}，退回到规则打分")
            args.judge_method = "rule"

    from eval.ab_experiment import resolve_model_name

    all_results = {}

    for mode in active_modes:
        print(f"\n{'=' * 60}")
        print(f"实验: {mode.label}")
        print(f"{'=' * 60}")

        vector_store = None
        if mode.use_rag:
            print("  加载向量数据库...")
            vector_store = ChromaVectorStore(
                persist_directory=str(CHROMA_DIR),
                collection_name="veterinary_rag",
                model_name="BAAI/bge-large-zh-v1.5",
            )

        model_name = resolve_model_name(use_finetuned=mode.use_finetuned)

        results = run_single_experiment(mode, testset, vector_store, model_name)
        all_results[mode.label] = results

        # 保存原始结果
        raw_file = output_dir / f"raw_{mode.name}_{timestamp}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  原始结果已保存: {raw_file}")

        if vector_store is not None:
            del vector_store

    # 评分
    print(f"\n{'=' * 60}")
    print(f"评分阶段（method={args.judge_method}）")
    print(f"{'=' * 60}")
    scored = run_judgment(all_results, scorer, judge_model_name, args.judge_method)

    # 生成报告
    generate_report(scored, output_dir, timestamp)

    # 保存详细评分结果
    detail_file = output_dir / f"scored_{timestamp}.json"
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"\n详细评分结果: {detail_file}")


if __name__ == "__main__":
    main()
