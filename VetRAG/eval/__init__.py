"""VetRAG 评估模块。"""

from eval.scoring.judge_base import JudgeBase
from eval.scoring.deepseek_judge import DeepSeekJudge

__all__ = ["JudgeBase", "DeepSeekJudge"]
