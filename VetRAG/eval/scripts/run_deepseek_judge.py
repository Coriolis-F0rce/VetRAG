"""
DeepSeek LLM-as-Judge 评分入口
================================
加载四组实验结果和测试集，通过 DeepSeek API 对每道题四组答案做横向对比评分，
输出汇总报告（JSON + CSV）到 eval/results/。

用法：
    # 标准用法（从 eval/results/ 目录自动查找最新的四组 raw JSON）：
    python eval/scripts/run_deepseek_judge.py

    # 手动指定四组结果文件：
    python eval/scripts/run_deepseek_judge.py \
        --group_a results/FINETUNED_RAG_*.json \
        --group_b results/BASE_RAG_*.json \
        --group_c results/FINETUNED_NO_RAG_*.json \
        --group_d results/BASE_NO_RAG_*.json \
        --testset datasets/testset.json

    # 指定 DeepSeek 模型：
    python eval/scripts/run_deepseek_judge.py --model deepseek-chat
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from eval.scoring.deepseek_judge import DeepSeekJudge


# ---------- 四组实验的标签定义 ----------
GROUP_META = {
    "A": {
        "label": "微调模型 + RAG",
        "file_prefix": "FINETUNED_RAG",
        "mode_enum": "FINETUNED_RAG",
    },
    "B": {
        "label": "基础模型 + RAG",
        "file_prefix": "BASE_RAG",
        "mode_enum": "BASE_RAG",
    },
    "C": {
        "label": "微调模型 无RAG",
        "file_prefix": "FINETUNED_NO_RAG",
        "mode_enum": "FINETUNED_NO_RAG",
    },
    "D": {
        "label": "基础模型 无RAG",
        "file_prefix": "BASE_NO_RAG",
        "mode_enum": "BASE_NO_RAG",
    },
}


# ---------- 加载工具 ----------
def load_latest_result(results_dir: Path, prefix: str) -> list[dict]:
    """从 results_dir 中找到最新匹配的 JSON 文件并加载。"""
    pattern = str(results_dir / f"{prefix}_*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"未找到匹配 {pattern} 的结果文件")
    latest = files[0]
    print(f"  使用结果文件: {latest}")
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 支持 run_ab.py 输出格式（顶层有 "results" 键）；也支持顶层为 list 的格式
    return data.get("results", data) if isinstance(data, dict) else data


def load_results(
    results_dir: Path,
    group_a: str | None,
    group_b: str | None,
    group_c: str | None,
    group_d: str | None,
) -> dict[str, list[dict]]:
    """加载四组实验结果。"""
    loaded = {}
    for group, prefix in [("A", group_a), ("B", group_b), ("C", group_c), ("D", group_d)]:
        if prefix:
            # 用户指定了文件路径
            if "*" in prefix:
                files = sorted(glob.glob(prefix), key=os.path.getmtime, reverse=True)
                if not files:
                    raise FileNotFoundError(f"未找到: {prefix}")
                path = files[0]
            else:
                path = prefix
            print(f"  [{group}] 加载: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded[group] = data.get("results", data) if isinstance(data, dict) else data
        else:
            # 自动查找
            meta = GROUP_META[group]
            print(f"  [{group}] 自动查找 {meta['file_prefix']}_*.json ...")
            loaded[group] = load_latest_result(results_dir, meta["file_prefix"])

    return loaded


def load_testset(path: Path) -> list[dict]:
    """加载测试集。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("questions", [])


# ---------- 汇总统计 ----------
def summarize(all_scores: list[dict]) -> dict:
    """计算各组各维度的平均分和总均分。"""
    dims = ["accuracy", "relevance", "completeness", "format", "safety", "avg_score"]
    groups = ["group_A", "group_B", "group_C", "group_D"]

    summary = {}
    for group in groups:
        summary[group] = {"label": GROUP_META[group[6:]]["label"]}
        for dim in dims:
            vals = [
                s[group][dim]
                for s in all_scores
                if group in s and dim in s[group] and "error" not in s
            ]
            summary[group][f"avg_{dim}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    # 总均分排名
    total_scores = {g: summary[g]["avg_avg_score"] for g in groups}
    ranking = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    summary["_ranking"] = [{"rank": i + 1, "group": g, "avg_total": s} for i, (g, s) in enumerate(ranking)]

    return summary


def print_report(summary: dict, all_scores: list[dict]):
    """打印人类可读的评分报告。"""
    print("\n" + "=" * 70)
    print("  VetRAG DeepSeek LLM-as-Judge 评分报告")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    groups = ["group_A", "group_B", "group_C", "group_D"]
    dims = ["accuracy", "relevance", "completeness", "format", "safety"]

    # 打印每个组的分数
    header = f"{'组别':<30s}" + "".join(f"{d.upper():>10s}" for d in dims) + f"{'AVG':>10s}"
    print(f"\n{header}")
    print("-" * 90)
    for group in groups:
        meta = summary[group]
        row = f"{meta['label']:<30s}"
        row += "".join(f"{meta.get(f'avg_{d}', 0):>10.3f}" for d in dims)
        row += f"{meta.get('avg_avg_score', 0):>10.3f}"
        print(row)

    # 打印排名
    print("\n" + "-" * 70)
    print("  总均分排名")
    print("-" * 70)
    for entry in summary["_ranking"]:
        g = entry["group"]
        print(f"  #{entry['rank']}  {summary[g]['label']:<25s}  {entry['avg_total']:.4f}")

    # 打印每道题详情
    print("\n" + "-" * 70)
    print("  各题详情（winner = 总均分最高组）")
    print("-" * 70)
    for score in all_scores:
        if "error" in score:
            print(f"  Q{score['id']:2d}: [评分失败] {score['error']}")
            continue
        winner = score.get("winner", "?")
        avg = score.get("group_A", {}).get("avg_score", 0)
        print(f"  Q{score['id']:2d} [{score.get('category', '?'):10s}] winner={winner}  A_avg={avg:.2f}")


def save_csv(all_scores: list[dict], output_path: Path):
    """将评分结果保存为 CSV。"""
    dims = ["accuracy", "relevance", "completeness", "format", "safety", "avg_score"]
    groups = ["group_A", "group_B", "group_C", "group_D"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 表头
        header = ["id", "question", "category", "winner"]
        for group in groups:
            for dim in dims:
                header.append(f"{group}_{dim}")
        header.append("comparison")
        writer.writerow(header)

        for score in all_scores:
            row = [score.get("id", ""), score.get("question", ""), score.get("category", ""), score.get("winner", "")]
            for group in groups:
                for dim in dims:
                    row.append(score.get(group, {}).get(dim, ""))
            row.append(score.get("comparison", ""))
            writer.writerow(row)

    print(f"  CSV 已保存: {output_path}")


# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser(description="VetRAG DeepSeek LLM-as-Judge 评分")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="实验结果目录（默认: eval/results）",
    )
    parser.add_argument("--testset", type=str, default="",
                        help="测试集路径（默认: eval/datasets/testset.json）")
    parser.add_argument("--group_a", type=str, default="",
                        help="A组（微调+RAG）结果文件（支持 glob）")
    parser.add_argument("--group_b", type=str, default="",
                        help="B组（基础+RAG）结果文件（支持 glob）")
    parser.add_argument("--group_c", type=str, default="",
                        help="C组（微调无RAG）结果文件（支持 glob）")
    parser.add_argument("--group_d", type=str, default="",
                        help="D组（基础无RAG）结果文件（支持 glob）")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="DeepSeek 模型名（默认: deepseek-chat）")
    parser.add_argument("--api_key", type=str, default="",
                        help="DeepSeek API key（默认从环境变量读取）")
    parser.add_argument("--output", type=str, default="",
                        help="输出目录（默认: eval/results）")
    parser.add_argument("--workers", type=int, default=10,
                        help="并行请求数（默认: 10）")
    parser.add_argument("--sequential", action="store_true",
                        help="使用串行模式（默认并行）")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = _project_root / (args.results_dir or "eval/results")
    output_dir = _project_root / (args.output or "eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试集
    testset_path = _project_root / (args.testset or "eval/datasets/testset.json")
    print(f"加载测试集: {testset_path}")
    testset = load_testset(testset_path)
    print(f"  共 {len(testset)} 道题")

    # 加载四组实验结果
    print("\n加载实验结果:")
    results_by_group = load_results(
        results_dir,
        args.group_a or None,
        args.group_b or None,
        args.group_c or None,
        args.group_d or None,
    )

    # 初始化 DeepSeek 评分器
    judge = DeepSeekJudge(
        api_key=args.api_key or None,
        model=args.model,
    )

    # 运行评分
    if args.sequential:
        print(f"\n开始评分（共 {len(testset)} 道题，串行）...")
        all_scores = judge.score_all(testset, results_by_group)
    else:
        print(f"\n开始评分（共 {len(testset)} 道题，{args.workers} 个并行请求）...")
        all_scores = judge.score_all_parallel(testset, results_by_group, max_workers=args.workers)

    # 汇总
    summary = summarize(all_scores)
    print_report(summary, all_scores)

    # 保存 JSON 报告
    report = {
        "timestamp": timestamp,
        "model": args.model,
        "testset": str(testset_path),
        "n_samples": len(testset),
        "summary": summary,
        "per_question": all_scores,
    }
    json_path = output_dir / f"deepseek_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {json_path}")

    # 保存 CSV
    csv_path = output_dir / f"deepseek_scores_{timestamp}.csv"
    save_csv(all_scores, csv_path)


if __name__ == "__main__":
    main()
