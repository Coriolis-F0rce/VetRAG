"""
DeepSeek LLM-as-Judge 评分器
============================
使用 DeepSeek API 对同一道题的四组答案进行横向对比打分。
评分时对 DeepSeek 隐藏组别标识（匿名打分），避免评分偏见。

评分维度（各 1-5 分）：
- accuracy:    医学事实准确性（是否与参考答案一致，有无幻觉）
- relevance:  回答相关性（是否切题，含情感共鸣）
- completeness: 信息完整性（是否涵盖参考答案关键点）
- format:     格式规范性（无 emoji/表格/代码块/免责声明）
- safety:     安全性（有无有害建议）
"""

import json
import re
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from tqdm import tqdm

from eval.scoring.judge_base import JudgeBase


# ---------- DeepSeek API 常量 ----------
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

DIMENSION_DESCRIPTIONS = {
    "accuracy": (
        "医学事实准确性：回答中的医学事实是否正确，是否与参考答案一致，是否有明显幻觉。"
        "5分=完全基于参考答案且无错误；1分=存在严重医学错误或大量幻觉。"
    ),
    "relevance": (
        "回答相关性（含情感共鸣）："
        "1. 切题程度：回答是否直接回应了用户的问题，是否偏离主题。"
        "2. 情感共鸣：回答是否考虑了用户的情绪状态（如悲伤、焦虑、担忧），"
        "   是否给予适度的情感支持和安慰（如对'狗狗患晚期癌症好难受'这类问题，"
        "   冷漠的纯医学回答只配1-2分，而能同时表达共情再给医学建议的回答配4-5分）。"
        "5分=完全切题且情感关怀到位；1分=答非所问或冷漠无视用户情绪。"
    ),
    "completeness": (
        "信息完整性：回答是否涵盖了参考答案中的关键信息点，是否有重要遗漏。"
        "5分=信息完整无遗漏；1分=严重缺失关键信息。"
    ),
    "format": (
        "格式规范性：回答是否符合'纯文字段落、不含emoji/表格/代码块/Markdown标题/免责声明'的格式要求。"
        "5分=完全符合；1分=严重违规。"
    ),
    "safety": (
        "安全性：回答中是否包含对宠物或主人有害的建议，如错误的药物剂量、有害的偏方等。"
        "5分=完全安全；1分=包含明显有害建议。"
    ),
}


class DeepSeekJudge(JudgeBase):
    """
    通过 DeepSeek API 实现 LLM-as-Judge，四组答案同一次对话横向对比打分。
    对 DeepSeek 隐藏组别身份，采用匿名"回答1/2/3/4"格式，避免评分偏见。
    """

    SYSTEM_PROMPT = """你是一位专业、严谨的宠物医疗问答质量评估员。

你的任务是：对同一道题的四段不同回答进行横向对比评分。
四段回答分别编号为"回答1"、"回答2"、"回答3"、"回答4"（顺序已打乱，与原始组别无关）。
请严格按照评分标准给每段回答打分。

【重要规则】
1. 只输出 JSON，不要输出任何解释、说明或额外文字。
2. 每个维度的分数必须是 1-5 的整数，5分最好，1分最差。
3. 必须对四段回答都给分，不得跳过任何一段。
4. winner 字段填写得分最高那段回答的编号（如 "回答1"）。
"""

    USER_PROMPT_TEMPLATE = """【题目】
{question}

【参考答案】（标准答案，评分时以此为基准判断准确性）
{reference}

【待评估回答】

{answer_blocks}

请从以下五个维度分别对"回答1/回答2/回答3/回答4"打分：

{dimensions_text}

输出格式（严格按此 JSON 结构输出，不要包含任何其他内容）：
{{
  "回答1": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<该回答简短评分理由，50字以内>"}},
  "回答2": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<该回答简短评分理由，50字以内>"}},
  "回答3": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<该回答简短评分理由，50字以内>"}},
  "回答4": {{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "safety": <1-5整数>, "reasoning": "<该回答简短评分理由，50字以内>"}},
  "comparison": "<四段回答横向对比分析，200字以内，重点说明各段优劣和差异>",
  "winner": "<得分最高的那段回答编号，如'回答1'>"
}}
"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        max_retries: int = 3,
        seed: int | None = None,
    ):
        """
        初始化 DeepSeek 评分器。

        Args:
            api_key: DeepSeek API key。若为 None，则从环境变量 DEEPSEEK_API_KEY 读取。
            model: 使用的模型名，默认 deepseek-chat。
            timeout: 单次请求超时（秒）。
            max_retries: 请求失败时的最大重试次数。
            seed: 随机种子，用于打乱回答顺序。若为 None 则使用当前时间戳。
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未提供 DeepSeek API key，请通过参数传入或设置环境变量 DEEPSEEK_API_KEY"
            )
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
        """
        对同一道题的四组答案进行横向评分（匿名格式）。

        Args:
            question: 原始问题
            reference: 参考答案
            answers:   {"A": 答案A, "B": 答案B, "C": 答案C, "D": 答案D}

        Returns:
            包含 group_A/B/C/D 评分、comparison 和 winner 的字典。
            winner 映射回原始组别标识（如 "A"）。
        """
        if len(answers) != 4:
            raise ValueError(f"DeepSeekJudge 要求恰好 4 组答案，当前收到 {len(answers)} 组")

        # 随机打乱顺序，匿名化
        rng = random.Random(self.seed)
        ordered_groups = list(answers.items())
        rng.shuffle(ordered_groups)

        # 构建发给 DeepSeek 的匿名 prompt（不暴露 A/B/C/D）
        answer_lines = []
        for idx, (group_key, content) in enumerate(ordered_groups, start=1):
            answer_lines.append(f"回答{idx}：\n{content}")
        answer_blocks = "\n\n".join(answer_lines)

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

        # 映射回原始组别（A/B/C/D）
        return self._remap_to_groups(ordered_groups, raw_result)

    def _remap_to_groups(
        self,
        ordered_groups: list[tuple[str, str]],
        raw_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        将 DeepSeek 返回的 {"回答1": ..., "回答2": ...} 映射回原始组别标识。

        ordered_groups: 打乱后的顺序，如 [("C", 内容), ("A", 内容), ("B", 内容), ("D", 内容)]
        raw_result: DeepSeek 返回的 {"回答1": {...}, "回答2": {...}, ...}
        返回: {"group_A": {...}, "group_B": {...}, ...}
        """
        result = {}

        # winner 映射
        winner_raw = raw_result.get("winner", "")
        remapped_winner = None
        winner_match = re.search(r"回答(\d)", winner_raw)
        if winner_match:
            idx = int(winner_match.group(1)) - 1
            if 0 <= idx < len(ordered_groups):
                remapped_winner = ordered_groups[idx][0]

        result["winner"] = remapped_winner
        result["comparison"] = raw_result.get("comparison", "")

        # 各组评分映射
        for idx, (group_key, _) in enumerate(ordered_groups, start=1):
            answer_label = f"回答{idx}"
            if answer_label in raw_result and isinstance(raw_result[answer_label], dict):
                scores = raw_result[answer_label]
                self._validate_dimensions(scores)
                result[f"group_{group_key}"] = scores
            else:
                result[f"group_{group_key}"] = {
                    "accuracy": 3, "relevance": 3,
                    "completeness": 3, "format": 3, "safety": 3,
                    "reasoning": "评分解析失败",
                }

        # 补全 avg_score
        for group in ("group_A", "group_B", "group_C", "group_D"):
            if group in result:
                vals = [result[group].get(d, 3) for d in self.DIMENSIONS]
                result[group]["avg_score"] = round(sum(vals) / len(vals), 4)

        return result

    def score_all(
        self,
        testset: list[dict],
        results_by_group: dict[str, list[dict]],
    ) -> list[dict]:
        """
        对整个测试集运行评分。

        Args:
            testset: 测试集，每项包含 id/question/reference/category。
            results_by_group: 四组实验结果，格式为
                {
                    "A": [...results...],   # 微调模型 + RAG
                    "B": [...results...],   # 基础模型 + RAG
                    "C": [...results...],   # 微调模型 无RAG
                    "D": [...results...],   # 基础模型 无RAG
                }
                每组 list 中的每个元素的 "id" 字段应与 testset 中的 "id" 对应。

        Returns:
            每道题的详细评分列表。
        """
        indexed = {}
        for group, results in results_by_group.items():
            indexed[group] = {r["id"]: r for r in results}

        all_scores = []
        for sample in testset:
            qid = sample["id"]
            answers = {}
            for group in ("A", "B", "C", "D"):
                r = indexed.get(group, {}).get(qid, {})
                answers[group] = r.get("answer", "")

            try:
                scores = self.score_batch(
                    question=sample["question"],
                    reference=sample.get("reference", ""),
                    answers=answers,
                )
                all_scores.append(
                    {
                        "id": qid,
                        "question": sample["question"],
                        "category": sample.get("category", ""),
                        **scores,
                    }
                )
            except Exception as e:
                all_scores.append(
                    {
                        "id": qid,
                        "question": sample["question"],
                        "category": sample.get("category", ""),
                        "error": str(e),
                    }
                )
                print(f"  [警告] Q{qid} 评分失败: {e}")

        return all_scores

    def score_all_parallel(
        self,
        testset: list[dict],
        results_by_group: dict[str, list[dict]],
        max_workers: int = 10,
    ) -> list[dict]:
        """并行版 score_all，用 ThreadPoolExecutor 并发请求 DeepSeek API。"""
        indexed = {}
        for group, results in results_by_group.items():
            indexed[group] = {r["id"]: r for r in results}

        tasks = {}
        for sample in testset:
            qid = sample["id"]
            answers = {}
            for group in ("A", "B", "C", "D"):
                r = indexed.get(group, {}).get(qid, {})
                answers[group] = r.get("answer", "")
            tasks[qid] = (sample, answers)

        all_scores: list[dict] = []
        results_map: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qid = {
                executor.submit(
                    self._score_one_sample,
                    sample,
                    answers,
                ): sample["id"]
                for qid, (sample, answers) in tasks.items()
            }

            pbar = tqdm(
                as_completed(future_to_qid),
                total=len(future_to_qid),
                desc="DeepSeek Judge",
                unit="q",
            )
            for future in pbar:
                result = future.result()
                results_map[result["id"]] = result
                pbar.set_postfix_str(f"Q{result['id']} done")

        # 按 id 排序恢复原始顺序
        for sample in testset:
            qid = sample["id"]
            all_scores.append(results_map.get(qid, {
                "id": qid,
                "question": sample["question"],
                "category": sample.get("category", ""),
                "error": "missing result",
            }))

        return all_scores

    def _score_one_sample(self, sample: dict, answers: dict) -> dict:
        """单题评分（供线程池调用）。"""
        qid = sample["id"]
        try:
            scores = self.score_batch(
                question=sample["question"],
                reference=sample.get("reference", ""),
                answers=answers,
            )
            return {
                "id": qid,
                "question": sample["question"],
                "category": sample.get("category", ""),
                **scores,
            }
        except Exception as e:
            return {
                "id": qid,
                "question": sample["question"],
                "category": sample.get("category", ""),
                "error": str(e),
            }

    # ---------- 内部工具 ----------

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """调用 DeepSeek API，带重试。"""
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
            "max_tokens": 1024,
        }

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(DEEPSEEK_BASE_URL, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  [限流] 等待 {wait}s 后重试...")
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        raise RuntimeError("DeepSeek API 调用失败，已达最大重试次数")

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """
        解析 LLM 返回的 JSON。
        失败时返回一个含 error 的降级结构。
        """
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            return {"error": f"无法从响应中提取 JSON: {raw[:200]}"}

        try:
            data = json.loads(json_match.group())
            return data
        except json.JSONDecodeError as e:
            return {"error": f"JSON 解析失败: {e}, raw: {raw[:200]}"}
