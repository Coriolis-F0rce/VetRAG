"""
LLM-as-Judge 评估器
===================
使用更强的本地模型（BGE 语义相似度 + 规则 + 可选的 LLM 自评）
对 RAG 生成的答案进行多维度自动打分。

评分维度（各 1-5 分）：
- accuracy:    医学准确性（回答中事实与参考答案的一致程度）
- relevance:  回答相关性（回答是否切题解决用户问题）
- completeness: 完整性（是否涵盖问题所需的关键信息点）
- format:     格式规范性（是否符合项目要求的纯文字段落格式）
"""

import json
import re

import numpy as np
import ollama
import torch
from transformers import AutoModel
from transformers import AutoTokenizer as BgeTokenizer


# ---------- BGE 语义相似度（全局单例） ----------
_bge_model: AutoModel | None = None
_bge_tokenizer: BgeTokenizer | None = None


def _load_bge():
    global _bge_model, _bge_tokenizer
    if _bge_model is None:
        print("加载 BGE-large-zh-v1.5 用于评估...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _bge_tokenizer = BgeTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
        _bge_model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5")
        _bge_model.to(device).eval()
    return _bge_model, _bge_tokenizer


def _get_bge_embedding(text: str) -> np.ndarray:
    model, tokenizer = _load_bge()
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return float(np.dot(vec1, vec2))


# ---------- 关键词匹配工具 ----------
def _extract_keywords(text: str) -> set:
    """提取中文词语（简单的字符级 n-gram）。"""
    text = re.sub(r"[^\u4e00-\u9fff\w]", " ", text)
    words = re.findall(r"[\u4e00-\u9fff]+|\w+", text)
    # 过滤单字和常见停用词
    stopwords = {"的", "了", "是", "在", "和", "有", "我", "你", "他", "它", "这", "那", "吗", "呢", "吧", "啊"}
    return {w for w in words if len(w) >= 2 and w not in stopwords}


# ---------- 评分规则引擎 ----------
class JudgeScorer:
    """基于规则的多维度评分器（无需 LLM，适合无参考答案场景）。"""

    DIMENSION_NAMES = ["accuracy", "relevance", "completeness", "format"]

    # 每个维度的 prompt 模板（LLM 自评用）
    JUDGE_PROMPT_TEMPLATE = """你是一位专业的宠物医疗问答质量评估员。
请根据以下【问题】【参考答案】（如有）和【待评估回答】，从四个维度分别打分（1-5分，5分最好）。

评分标准：
- accuracy（医学准确性）: 回答中的医学事实是否正确，是否有严重错误或幻觉。
  5分：完全基于参考答案，无错误；1分：存在严重医学错误。
- relevance（回答相关性）: 回答是否直接回应了用户的问题。
  5分：完全切题；1分：答非所问。
- completeness（完整性）: 回答是否涵盖了参考答案中的关键信息点。
  5分：信息完整；1分：严重缺失。
- format（格式规范性）: 回答是否符合"纯文字段落、不含emoji/表格/代码块/免责声明"的格式要求。
  5分：完全符合；1分：严重违规。

【问题】
{question}

【参考答案】（标准答案）
{reference}

【待评估回答】
{answer}

请严格按以下 JSON 格式输出评分，不要输出任何其他内容：
{{"accuracy": <1-5整数>, "relevance": <1-5整数>, "completeness": <1-5整数>, "format": <1-5整数>, "reasoning": "<简要评分理由（20字以内）>"}}
"""

    def __init__(self):
        self.bge_sim_threshold = 0.7  # BGE 相似度阈值

    def score_with_reference(
        self,
        question: str,
        reference: str,
        answer: str,
    ) -> dict:
        """
        有参考答案时：结合 BGE 语义相似度 + 关键词覆盖 + LLM 格式规则评分。
        """
        # 1. BGE 语义相似度
        ref_vec = _get_bge_embedding(reference)
        ans_vec = _get_bge_embedding(answer)
        bge_sim = cosine_similarity(ref_vec, ans_vec)

        # 2. 关键词覆盖（参考答案中的词有多少出现在回答中）
        ref_kw = _extract_keywords(reference)
        ans_kw = _extract_keywords(answer)
        kw_coverage = len(ref_kw & ans_kw) / len(ref_kw) if ref_kw else 1.0

        # 3. 规则打分
        scores = {}

        # accuracy：基于 BGE 相似度映射到 1-5
        scores["accuracy"] = self._sim_to_score(bge_sim, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9])

        # relevance：基于关键词覆盖
        scores["relevance"] = self._sim_to_score(kw_coverage, thresholds=[0.2, 0.35, 0.5, 0.65, 0.8])

        # completeness：综合 BGE 相似度和关键词覆盖
        completeness_raw = (bge_sim + kw_coverage) / 2
        scores["completeness"] = self._sim_to_score(completeness_raw, thresholds=[0.25, 0.4, 0.55, 0.7, 0.85])

        # format：纯规则检查
        scores["format"] = self._score_format(answer)

        return {
            "accuracy": scores["accuracy"],
            "relevance": scores["relevance"],
            "completeness": scores["completeness"],
            "format": scores["format"],
            "bge_similarity": round(bge_sim, 4),
            "keyword_coverage": round(kw_coverage, 4),
            "has_reference": True,
        }

    def score_without_reference(
        self,
        question: str,
        answer: str,
    ) -> dict:
        """
        无参考答案时：基于回答内容质量的规则评估。
        """
        # relevance：回答长度和关键词匹配
        q_kw = _extract_keywords(question)
        a_kw = _extract_keywords(answer)
        kw_match = len(q_kw & a_kw) / len(q_kw) if q_kw else 0.5

        # 回答长度（过短说明不完整）
        answer_len = len(answer.strip())
        length_score = min(5, max(1, answer_len // 80))

        scores = {}
        scores["relevance"] = self._sim_to_score(kw_match, thresholds=[0.1, 0.25, 0.4, 0.55, 0.7])
        scores["completeness"] = length_score  # 长度直接映射
        scores["format"] = self._score_format(answer)
        # 无参考答案时 accuracy 只能给 3（中立分）
        scores["accuracy"] = 3

        return {
            "accuracy": scores["accuracy"],
            "relevance": scores["relevance"],
            "completeness": scores["completeness"],
            "format": scores["format"],
            "bge_similarity": None,
            "keyword_coverage": round(kw_match, 4),
            "answer_length": answer_len,
            "has_reference": False,
        }

    def score_by_llm(
        self,
        question: str,
        reference: str,
        answer: str,
        judge_model_name: str = "vetrag-qwen3-0.6b-base",
    ) -> dict:
        """
        LLM-as-Judge：通过 Ollama 调用模型进行结构化打分。
        judge_model_name: Ollama 中用于评分的模型名。
        """
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            reference=reference if reference else "（无参考答案）",
            answer=answer,
        )
        try:
            response = ollama.chat(
                model=judge_model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 128, "temperature": 0.0},
                think=False,
            )
            raw = response["message"]["content"].strip()
        except Exception as e:
            print(f"  [警告] Ollama Judge 调用失败: {e}")
            return {
                "accuracy": 3, "relevance": 3, "completeness": 3, "format": 3,
                "reasoning": f"Judge error: {e}",
                "method": "llm_judge",
            }

        # 解析 JSON
        try:
            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return {
                    "accuracy": int(scores.get("accuracy", 3)),
                    "relevance": int(scores.get("relevance", 3)),
                    "completeness": int(scores.get("completeness", 3)),
                    "format": int(scores.get("format", 3)),
                    "reasoning": scores.get("reasoning", ""),
                    "raw_llm_output": raw,
                    "method": "llm_judge",
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 降级：返回规则打分
        return {
            "accuracy": 3,
            "relevance": 3,
            "completeness": 3,
            "format": 3,
            "reasoning": "LLM输出解析失败，降级为规则打分",
            "raw_llm_output": raw,
            "method": "rule_fallback",
        }

    # ---------- 内部工具 ----------
    @staticmethod
    def _sim_to_score(value: float, thresholds: list[float]) -> int:
        """将 0-1 的相似度值映射为 1-5 分。"""
        for i, t in enumerate(thresholds):
            if value <= t:
                return i + 1
        return 5

    @staticmethod
    def _score_format(text: str) -> int:
        """评估格式规范性（1-5 分）。"""
        score = 5
        # 检查禁止项
        if re.search(r"```[\s\S]*?```", text):  # 代码块
            score -= 1
        if re.search(r"[\U0001F000-\U0001FFFF]", text):  # emoji
            score -= 1
        if re.search(r"^\s*[☑☒✓✗✔✘•●○]", text, re.MULTILINE):  # 勾叉列表
            score -= 1
        if re.search(r"(免责声明|来源|参考文献|根据.*文档)", text):  # 免责声明
            score -= 1
        if re.search(r"^#+\s", text, re.MULTILINE):  # Markdown 标题
            score -= 1
        if re.search(r"^\s*[一二三四五六七八九十\d+][.、]", text, re.MULTILINE):  # 数字列表
            score -= 1
        return max(1, score)

    @staticmethod
    def average_scores(results: list[dict]) -> dict:
        """计算一组结果的平均分。"""
        dims = ["accuracy", "relevance", "completeness", "format"]
        out = {}
        for dim in dims:
            vals = [r[dim] for r in results if dim in r]
            out[f"avg_{dim}"] = round(np.mean(vals), 4) if vals else 0
            out[f"std_{dim}"] = round(np.std(vals), 4) if len(vals) > 1 else 0
        out["avg_total"] = round(
            np.mean([out[f"avg_{d}"] for d in dims]), 4
        )
        return out
