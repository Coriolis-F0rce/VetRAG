"""
Judge 协议基类
===============
定义 LLM-as-Judge 的抽象接口，所有评分器均实现此协议。
"""

from abc import ABC, abstractmethod
from typing import Any


class JudgeBase(ABC):
    """LLM-as-Judge 的抽象基类。"""

    DIMENSIONS = ["accuracy", "relevance", "completeness", "format", "safety"]

    @abstractmethod
    def score_batch(
        self,
        question: str,
        reference: str,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        """
        对同一道题的多组答案进行横向评分。

        Args:
            question: 原始问题
            reference: 参考答案（或空字符串）
            answers:  {"A": 答案A, "B": 答案B, ...}

        Returns:
            {
                "group_A": {"accuracy": int, "relevance": int, ...},
                "group_B": {...},
                "comparison": str,
                "winner": str,
            }
        """

    def _validate_dimensions(self, scores: dict[str, int]) -> None:
        """校验所有维度分数在 1-5 范围内。"""
        for dim in self.DIMENSIONS:
            if dim in scores:
                v = scores[dim]
                if not isinstance(v, (int, float)) or not (1 <= v <= 5):
                    scores[dim] = max(1, min(5, int(v)))

    def _avg(self, scores: dict[str, int]) -> float:
        """计算所有维度平均分。"""
        vals = [scores[d] for d in self.DIMENSIONS if d in scores]
        return round(sum(vals) / len(vals), 4) if vals else 0.0
