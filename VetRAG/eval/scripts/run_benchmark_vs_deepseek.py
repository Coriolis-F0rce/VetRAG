"""
DeepSeek Benchmark 对比评测
============================
两步流程：
  Step 1 — DeepSeek 作为参赛模型：对 50 道题独立回答（带 vet system prompt，无 RAG）
  Step 2 — DeepSeek 作为裁判：5 组答案匿名打乱，listwise 横向对比评分

5 组对比：
  A: 微调 Qwen3-1.7B + RAG
  B: 原始 Qwen3-1.7B + RAG
  C: 微调 Qwen3-1.7B 无RAG
  D: 原始 Qwen3-1.7B 无RAG
  E: DeepSeek API 直接回答

用法：
  # 完整流程（先答题再评分）
  python eval/scripts/run_benchmark_vs_deepseek.py

  # 只跑 Step 1（生成 DeepSeek 答案）
  python eval/scripts/run_benchmark_vs_deepseek.py --step1_only

  # 只跑 Step 2（假设已有 DeepSeek 答案文件）
  python eval/scripts/run_benchmark_vs_deepseek.py --step2_only
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm


_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from src.core.config import SYSTEM_PROMPT_VET


# ============================================================================
# 常量
# ============================================================================

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

# DeepSeek 回答时的 system prompt（简化版，无 RAG 相关指令）
ANSWER_SYSTEM_PROMPT = (
    "你是一个专业的宠物狗健康问答助手。请基于你的知识，对宠物主人的问题给出"
    "专业、准确、有同理心的回答。使用自然流畅的段落，禁止使用 emoji、代码块、"
    "Markdown 表格。如果问题涉及紧急情况，请强调及时就医的必要性。"
)

# 5 组元数据
GROUP_META = {
    "A": {"label": "微调模型 + RAG",       "file_prefix": "FINETUNED_RAG"},
    "B": {"label": "基础模型 + RAG",       "file_prefix": "BASE_RAG"},
    "C": {"label": "微调模型 无RAG",      "file_prefix": "FINETUNED_NO_RAG"},
    "D": {"label": "基础模型 无RAG",      "file_prefix": "BASE_NO_RAG"},
    "E": {"label": "DeepSeek API 直接回答", "file_prefix": "DEEPSEEK_DIRECT"},
}

DIMENSIONS = ["accuracy", "relevance", "completeness", "format", "safety"]

DIMENSION_DESCRIPTIONS = {
    "accuracy": (
        "医学事实准确性：回答中的医学事实是否正确，是否与参考答案一致，是否有明显幻觉。"
        "5分=完全基于参考答案且无错误；1分=存在严重医学错误或大量幻觉。"
    ),
    "relevance": (
        "回答相关性（含情感共鸣）："
        "1. 切题程度：回答是否直接回应了用户的问题，是否偏离主题。"
        "2. 情感共鸣：回答是否考虑了用户的情绪状态（如悲伤、焦虑、担忧），"
        "   是否给予适度的情感支持和安慰。"
        "5分=完全切题且情感关怀到位；1分=答非所问或冷漠无视用户情绪。"
    ),
    "completeness": (
        "信息完整性：回答是否涵盖了参考答案中的关键信息点，是否有重要遗漏。"
        "5分=信息完整无遗漏；1分=严重缺失关键信息。"
    ),
    "format": (
        "格式规范性：回答是否符合纯文字段落要求，不含 emoji/表格/代码块/Markdown标题/免责声明。"
        "5分=完全符合；1分=严重违规。"
    ),
    "safety": (
        "安全性：回答中是否包含对宠物或主人有害的建议，如错误的药物剂量、有害的偏方等。"
        "5分=完全安全；1分=包含明显有害建议。"
    ),
}

N_ANSWERS = 5  # 5 组 listwise


# ============================================================================
# Step 1: DeepSeek 独立回答
# ============================================================================

def _call_deepseek_chat(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> str:
    """调用 DeepSeek chat API，返回回答文本。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(DEEPSEEK_CHAT_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                print(f"    [限流] 等待 {wait}s 后重试...")
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError("DeepSeek API 调用失败，已达最大重试次数")


def _answer_one(sample: dict, api_key: str, model: str) -> dict:
    """对单道题生成 DeepSeek 回答。"""
    qid = sample["id"]
    question = sample["question"]
    start = time.time()
    try:
        answer = _call_deepseek_chat(ANSWER_SYSTEM_PROMPT, question, api_key, model)
    except Exception as e:
        answer = f"[ERROR] {e}"
    elapsed = time.time() - start
    return {
        "id": qid,
        "question": question,
        "reference": sample.get("reference", ""),
        "category": sample.get("category", ""),
        "mode": "DeepSeek API 直接回答",
        "answer": answer,
        "answer_chars": len(answer),
        "elapsed_s": round(elapsed, 2),
        "retrieval": None,
        "timestamp": datetime.now().isoformat(),
    }


def step1_generate_answers(
    testset: list[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_workers: int = 10,
    output_dir: Path | None = None,
) -> Path:
    """Step 1: DeepSeek 对 50 道题独立回答，保存结果文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = _project_root / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Step 1: DeepSeek 独立回答 {len(testset)} 道题")
    print(f"  模型: {model}  |  并发: {max_workers}  |  system prompt: 已加载")
    print(f"{'='*60}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_answer_one, sample, api_key, model): sample["id"]
            for sample in testset
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="DeepSeek 答题", unit="q")
        for future in pbar:
            result = future.result()
            results.append(result)
            pbar.set_postfix_str(f"Q{result['id']} done ({result['elapsed_s']:.1f}s)")

    # 按 id 排序
    results.sort(key=lambda r: r["id"])

    # 统计
    total_time = sum(r["elapsed_s"] for r in results)
    errors = [r for r in results if r["answer"].startswith("[ERROR]")]
    print(f"\n  完成: {len(results)} 题  |  总耗时: {total_time:.1f}s  |  失败: {len(errors)}")

    # 保存
    output_path = output_dir / f"raw_DEEPSEEK_DIRECT_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {output_path}")

    return output_path


# ============================================================================
# Step 2: 5-way Listwise Judge
# ============================================================================

class DeepSeekJudge5:
    """DeepSeek-as-Judge：5 组答案匿名 listwise 横向对比评分。"""

    SYSTEM_PROMPT = """你是一位专业、严谨的宠物医疗问答质量评估员。

你的任务是：对同一道题的五段不同回答进行横向对比评分。
五段回答分别编号为"回答1"、"回答2"、"回答3"、"回答4"、"回答5"（顺序已打乱，与原始组别无关）。
请严格按照评分标准给每段回答打分。

【重要规则】
1. 只输出 JSON，不要输出任何解释、说明或额外文字。
2. 每个维度的分数必须是 1-5 的整数，5分最好，1分最差。
3. 必须对五段回答都给分，不得跳过任何一段。
4. winner 字段填写得分最高那段回答的编号（如 "回答3"）。
"""

    USER_PROMPT_TEMPLATE = """【题目】
{question}

【参考答案】（标准答案，评分时以此为基准判断准确性）
{reference}

【待评估回答】

{answer_blocks}

请从以下五个维度分别对"回答1/回答2/回答3/回答4/回答5"打分：

{dimensions_text}

输出格式（严格按此 JSON 结构输出，不要包含任何其他内容）：
{{
  "回答1": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<简短评分理由，50字以内>"}},
  "回答2": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<简短评分理由，50字以内>"}},
  "回答3": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<简短评分理由，50字以内>"}},
  "回答4": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<简短评分理由，50字以内>"}},
  "回答5": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<简短评分理由，50字以内>"}},
  "comparison": "<五段回答横向对比分析，200字以内>",
  "winner": "<得分最高的那段回答编号，如'回答3'>"
}}
"""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        max_retries: int = 3,
        seed: int | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed

    def score_batch(
        self,
        question: str,
        reference: str,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        """对同一道题的 5 组答案进行横向评分（匿名格式）。

        Args:
            question: 原始问题
            reference: 参考答案
            answers: {"A": ..., "B": ..., "C": ..., "D": ..., "E": ...}

        Returns:
            包含 group_A~E 评分、comparison、winner 的字典
        """
        if len(answers) != N_ANSWERS:
            raise ValueError(f"DeepSeekJudge5 要求恰好 {N_ANSWERS} 组答案，收到 {len(answers)} 组")

        # 随机打乱顺序，匿名化
        rng = random.Random(self.seed)
        ordered = list(answers.items())
        rng.shuffle(ordered)

        # 构建匿名 answer blocks
        blocks = []
        for idx, (_, content) in enumerate(ordered, start=1):
            blocks.append(f"回答{idx}：\n{content}")
        answer_blocks = "\n\n".join(blocks)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            question=question,
            reference=reference if reference else "（无参考答案）",
            answer_blocks=answer_blocks,
            dimensions_text="\n".join(
                f"- {k}: {v}" for k, v in DIMENSION_DESCRIPTIONS.items()
            ),
        )

        raw = self._call_api(self.SYSTEM_PROMPT, user_prompt)
        raw_result = self._parse_response(raw)

        return self._remap(ordered, raw_result)

    def _remap(
        self,
        ordered: list[tuple[str, str]],
        raw_result: dict[str, Any],
    ) -> dict[str, Any]:
        """将 DeepSeek 返回的 回答1~5 映射回原始组别 A~E。"""
        result: dict[str, Any] = {}

        # winner 映射
        winner_raw = raw_result.get("winner", "")
        winner_match = re.search(r"回答(\d)", winner_raw)
        if winner_match:
            idx = int(winner_match.group(1)) - 1
            if 0 <= idx < len(ordered):
                result["winner"] = ordered[idx][0]

        result["comparison"] = raw_result.get("comparison", "")

        for idx, (group_key, _) in enumerate(ordered, start=1):
            label = f"回答{idx}"
            if label in raw_result and isinstance(raw_result[label], dict):
                scores = raw_result[label]
                self._validate_dimensions(scores)
                result[f"group_{group_key}"] = scores
            else:
                result[f"group_{group_key}"] = {
                    "accuracy": 3, "relevance": 3,
                    "completeness": 3, "format": 3, "safety": 3,
                    "reasoning": "评分解析失败",
                }

        # 补全 avg_score
        for group in ("group_A", "group_B", "group_C", "group_D", "group_E"):
            if group in result and isinstance(result[group], dict):
                vals = [result[group].get(d, 3) for d in DIMENSIONS]
                result[group]["avg_score"] = round(sum(vals) / len(vals), 4)

        return result

    def score_all_parallel(
        self,
        testset: list[dict],
        results_by_group: dict[str, list[dict]],
        max_workers: int = 10,
    ) -> list[dict]:
        """并行评分整个测试集。"""
        indexed = {}
        for group, res in results_by_group.items():
            indexed[group] = {r["id"]: r for r in res}

        all_scores: list[dict] = []
        results_map: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for sample in testset:
                qid = sample["id"]
                answers = {}
                for g in ("A", "B", "C", "D", "E"):
                    r = indexed.get(g, {}).get(qid, {})
                    answers[g] = r.get("answer", "")
                futures[executor.submit(self._score_one, sample, answers)] = qid

            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="DeepSeek Judge (5-way)", unit="q")
            for future in pbar:
                result = future.result()
                results_map[result["id"]] = result
                pbar.set_postfix_str(f"Q{result['id']} done")

        for sample in testset:
            qid = sample["id"]
            all_scores.append(results_map.get(qid, {
                "id": qid, "question": sample["question"],
                "category": sample.get("category", ""), "error": "missing",
            }))

        return all_scores

    def _score_one(self, sample: dict, answers: dict) -> dict:
        qid = sample["id"]
        try:
            scores = self.score_batch(
                question=sample["question"],
                reference=sample.get("reference", ""),
                answers=answers,
            )
            return {"id": qid, "question": sample["question"],
                    "category": sample.get("category", ""), **scores}
        except Exception as e:
            return {"id": qid, "question": sample["question"],
                    "category": sample.get("category", ""), "error": str(e)}

    def _validate_dimensions(self, scores: dict) -> None:
        for dim in DIMENSIONS:
            if dim in scores:
                v = scores[dim]
                if not isinstance(v, (int, float)) or not (1 <= v <= 5):
                    scores[dim] = max(1, min(5, int(v)))

    # ---------- API ----------

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(DEEPSEEK_CHAT_URL, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  [限流] 等待 {wait}s 后重试...")
                    time.sleep(wait)
                    continue
                raise
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("DeepSeek API 调用失败")

    def _parse_response(self, raw: str) -> dict[str, Any]:
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            return {"error": f"无法提取 JSON: {raw[:200]}"}
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return {"error": f"JSON 解析失败: {e}, raw: {raw[:200]}"}


# ============================================================================
# 报告工具
# ============================================================================

def load_result_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


def auto_load_results(results_dir: Path) -> dict[str, list[dict]]:
    """自动加载最新四组 + DeepSeek 结果。"""
    loaded = {}
    for group in ("A", "B", "C", "D", "E"):
        prefix = GROUP_META[group]["file_prefix"]
        pattern = str(results_dir / f"raw_{prefix}_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"未找到: {pattern}")
        print(f"  [{group}] {GROUP_META[group]['label']}: {Path(files[0]).name}")
        loaded[group] = load_result_file(Path(files[0]))
    return loaded


def summarize_5way(all_scores: list[dict]) -> dict:
    groups = ["group_A", "group_B", "group_C", "group_D", "group_E"]
    dims = DIMENSIONS + ["avg_score"]

    summary = {}
    for group in groups:
        label = GROUP_META[group[6:]]["label"]
        summary[group] = {"label": label}
        for dim in dims:
            vals = [
                s[group][dim]
                for s in all_scores
                if group in s and isinstance(s.get(group), dict) and dim in s[group] and "error" not in s
            ]
            summary[group][f"avg_{dim}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    total_scores = {g: summary[g]["avg_avg_score"] for g in groups}
    ranking = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    summary["_ranking"] = [
        {"rank": i + 1, "group": g, "label": GROUP_META[g[6:]]["label"], "avg_total": s}
        for i, (g, s) in enumerate(ranking)
    ]
    return summary


def print_report_5way(summary: dict, all_scores: list[dict]):
    print(f"\n{'=' * 80}")
    print("  VetRAG Benchmark: 5 组 Listwise 对比 (含 DeepSeek)")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    groups = ["group_A", "group_B", "group_C", "group_D", "group_E"]

    # 维度均分表
    header = f"{'组别':<28s}" + "".join(f"{d.upper():>10s}" for d in DIMENSIONS) + f"{'AVG':>10s}"
    print(f"\n{header}")
    print("-" * 88)
    for group in groups:
        meta = summary[group]
        row = f"{meta['label']:<28s}"
        row += "".join(f"{meta.get(f'avg_{d}', 0):>10.3f}" for d in DIMENSIONS)
        row += f"{meta.get('avg_avg_score', 0):>10.3f}"
        print(row)

    # 排名
    print(f"\n{'─' * 60}")
    print("  总均分排名")
    print(f"{'─' * 60}")
    for entry in summary["_ranking"]:
        print(f"  #{entry['rank']}  {entry['label']:<28s}  {entry['avg_total']:.4f}")

    # winner 分布
    print(f"\n{'─' * 60}")
    print("  Winner 分布（各题总均分最高组）")
    print(f"{'─' * 60}")
    win_counts: dict[str, int] = {}
    for s in all_scores:
        w = s.get("winner", "?")
        win_counts[w] = win_counts.get(w, 0) + 1
    for g in ("A", "B", "C", "D", "E"):
        label = GROUP_META[g]["label"]
        count = win_counts.get(g, 0)
        bar = "█" * count
        print(f"  [{g}] {label:<28s}  {count:2d} 题  {bar}")

    # 各题详情
    print(f"\n{'─' * 60}")
    print("  各题详情")
    print(f"{'─' * 60}")
    for s in all_scores:
        if "error" in s:
            print(f"  Q{s['id']:2d}: [评分失败] {s['error']}")
            continue
        winner = s.get("winner", "?")
        scores_str = "  ".join(
            f"{g}={s.get(f'group_{g}', {}).get('avg_score', 0):.2f}"
            for g in ("A", "B", "C", "D", "E")
            if f"group_{g}" in s and isinstance(s[f"group_{g}"], dict)
        )
        print(f"  Q{s['id']:2d} [{s.get('category', '?'):10s}] winner={winner}  |  {scores_str}")


def save_csv_5way(all_scores: list[dict], output_path: Path):
    groups = ["group_A", "group_B", "group_C", "group_D", "group_E"]
    score_fields = DIMENSIONS + ["avg_score"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["id", "question", "category", "winner"]
        for group in groups:
            for field in score_fields:
                header.append(f"{group}_{field}")
        header.append("comparison")
        writer.writerow(header)

        for s in all_scores:
            row = [s.get("id"), s.get("question"), s.get("category"), s.get("winner", "")]
            for group in groups:
                grp_data = s.get(group, {}) if isinstance(s.get(group), dict) else {}
                for field in score_fields:
                    row.append(grp_data.get(field, ""))
            row.append(s.get("comparison", ""))
            writer.writerow(row)

    print(f"  CSV 已保存: {output_path}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VetRAG Benchmark vs DeepSeek")
    parser.add_argument("--testset", type=str, default="",
                        help="测试集路径（默认 eval/datasets/testset_50.json）")
    parser.add_argument("--results_dir", type=str, default="",
                        help="结果目录（默认 eval/results）")
    parser.add_argument("--output_dir", type=str, default="",
                        help="输出目录（默认同 results_dir）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="DeepSeek 模型名（默认 deepseek-chat）")
    parser.add_argument("--api_key", type=str, default="",
                        help="DeepSeek API key（默认从 DEEPSEEK_API_KEY 环境变量读取）")
    parser.add_argument("--workers", type=int, default=10,
                        help="并行数（默认 10）")
    parser.add_argument("--step1_only", action="store_true",
                        help="仅运行 Step 1：生成 DeepSeek 答案")
    parser.add_argument("--step2_only", action="store_true",
                        help="仅运行 Step 2：5 组 listwise 评分（需已有 DeepSeek 答案文件）")
    parser.add_argument("--deepseek_results", type=str, default="",
                        help="Step 2 时手动指定 DeepSeek 答案文件路径")
    args = parser.parse_args()

    # 路径
    results_dir = _project_root / (args.results_dir or "eval/results")
    output_dir = _project_root / (args.output_dir or "eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    testset_path = _project_root / (args.testset or "eval/datasets/testset_50.json")
    print(f"测试集: {testset_path}")
    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)
    print(f"  共 {len(testset)} 道题")

    # API key
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请设置 DEEPSEEK_API_KEY 环境变量或通过 --api_key 传入")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Step 1: 生成 DeepSeek 答案
    # ------------------------------------------------------------------
    if not args.step2_only:
        deepseek_path = step1_generate_answers(
            testset, api_key, args.model, args.workers, output_dir,
        )
        if args.step1_only:
            print("\nStep 1 完成。")
            return
    else:
        # 自动查找最新 DeepSeek 答案文件
        if args.deepseek_results:
            deepseek_path = Path(args.deepseek_results)
        else:
            pattern = str(results_dir / "raw_DEEPSEEK_DIRECT_*.json")
            files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if not files:
                print(f"错误: 未找到 DeepSeek 答案文件 ({pattern})，请先运行 Step 1")
                sys.exit(1)
            deepseek_path = Path(files[0])
        print(f"\n使用 DeepSeek 答案: {deepseek_path}")

    # ------------------------------------------------------------------
    # Step 2: 5 组 listwise 评分
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Step 2: 5 组 Listwise LLM-as-Judge 评分")
    print(f"{'=' * 60}")

    # 加载四组 A/B 结果 + DeepSeek 答案
    print("\n加载实验结果:")
    results_by_group = {}
    # A~D: 自动加载
    for group in ("A", "B", "C", "D"):
        prefix = GROUP_META[group]["file_prefix"]
        pattern = str(results_dir / f"raw_{prefix}_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not files:
            # 尝试从 deepseek_report 同批次找
            alt_pattern = str(results_dir / f"raw_{prefix}_*.json")
            files = sorted(glob.glob(alt_pattern), key=os.path.getmtime, reverse=True)
        if not files:
            print(f"  警告: 未找到 {GROUP_META[group]['label']} 的结果文件 ({prefix}_*.json)，跳过")
            continue
        path = Path(files[0])
        print(f"  [{group}] {GROUP_META[group]['label']}: {path.name}")
        results_by_group[group] = load_result_file(path)

    # E: DeepSeek 答案
    print(f"  [E] {GROUP_META['E']['label']}: {deepseek_path.name}")
    results_by_group["E"] = load_result_file(deepseek_path)

    if len(results_by_group) != 5:
        print(f"错误: 需要 5 组结果，实际加载 {len(results_by_group)} 组")
        sys.exit(1)

    # 运行评分
    judge = DeepSeekJudge5(api_key=api_key, model=args.model)
    print(f"\n开始评分（{len(testset)} 题，{args.workers} 并行）...")
    all_scores = judge.score_all_parallel(testset, results_by_group, max_workers=args.workers)

    # 汇总 & 报告
    summary = summarize_5way(all_scores)
    print_report_5way(summary, all_scores)

    # 保存 JSON
    report = {
        "timestamp": timestamp,
        "model": args.model,
        "testset": str(testset_path),
        "n_samples": len(testset),
        "groups": {g: GROUP_META[g]["label"] for g in ("A", "B", "C", "D", "E")},
        "summary": summary,
        "per_question": all_scores,
    }
    json_path = output_dir / f"benchmark_5way_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {json_path}")

    # 保存 CSV
    csv_path = output_dir / f"benchmark_5way_{timestamp}.csv"
    save_csv_5way(all_scores, csv_path)

    # 最终排名摘要
    print(f"\n{'=' * 60}")
    print("  最终排名")
    print(f"{'=' * 60}")
    for entry in summary["_ranking"]:
        print(f"  #{entry['rank']}  {entry['label']:<28s}  {entry['avg_total']:.4f}")


if __name__ == "__main__":
    main()
