"""
domain_guard.py - 领域边界守卫（LLM 零样本分类，基于 Ollama）

在检索之前判断用户 query 是否属于宠物医疗领域。
- 属于领域内 → 继续 RAG 流程
- 不属于 → 直接返回友好拒绝语，跳过检索和生成

迁移说明：原先使用本地 transformers 加载基础模型做分类，
现已改为调用 Ollama，无需加载独立模型实例。
"""

import re
from typing import Optional, Literal

import ollama
from .logging import logger


class DomainGuard:
    """
    LLM 零样本领域分类器（基于 Ollama）。

    使用基础模型做零样本分类，输出只有两个字："是" 或 "否"，
    最小化 token 消耗。

    拒绝示例：
        - "量子力学是什么"
        - "如何学 Python"
        - "推荐一部电影"

    放行示例：
        - "我家狗最近不吃东西"
        - "狗发烧了怎么办"
        - "狗狗疫苗怎么打"
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个内容分类器。判断用户问题是否与宠物狗的健康、疾病、护理相关。"
        "只回答'是'或'否'，不要解释，不要标点符号。"
    )

    DEFAULT_REJECTION = (
        "抱歉，我是一个专注于宠物狗健康和护理的问答助手。"
        "关于猫、鸟类、鱼类或其他非宠物狗相关的问题，我暂时无法帮忙。 "
        "如果您有任何关于狗狗的健康问题，欢迎随时提问！"
    )

    def __init__(
        self,
        guard_model_name: str = "vetrag-qwen3-1.7b-base",
        system_prompt: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Args:
            guard_model_name: Ollama 中用于领域分类的模型名。
                              建议使用未微调的基础模型（如 qwen3:1.7b），
                              避免微调后模型输出格式退化。
            system_prompt: 分类系统提示词，默认使用 DEFAULT_SYSTEM_PROMPT。
            enabled: 是否启用 Guard。为 False 时所有 query 直接放行。
        """
        self.guard_model_name = guard_model_name
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.enabled = enabled

    def _classify(self, query: str) -> Literal["是", "否", "error"]:
        """
        调用 Ollama LLM 进行零样本分类。

        Returns:
            "是"  — query 属于宠物领域
            "否"  — query 不属于宠物领域
            "error" — 分类失败（如 Ollama 不可用），保守放行
        """
        prompt = f"{self.system_prompt}\n\n问题：{query}"

        try:
            response = ollama.generate(
                model=self.guard_model_name,
                prompt=prompt,
                options={"num_predict": 10, "temperature": 0.0},
                think=False,
            )
            raw_output = response["response"].strip()
        except Exception as e:
            logger.error(f"[DomainGuard] Ollama 调用失败: {e}，保守放行")
            return "error"

        result = raw_output.strip()

        # 1. 精确匹配（去掉首尾空白后）
        stripped = result.strip()
        if stripped in ("是", "否"):
            return stripped

        # 2. 去掉首尾成对引号后再匹配
        stripped_quotes = stripped.strip('""\'\'""')
        if stripped_quotes in ("是", "否"):
            return stripped_quotes

        # 3. 去掉首尾标点后精确匹配（全角/半角标点）
        stripped_punct = stripped.strip(".,，.?？!！;；:：~～")
        if stripped_punct in ("是", "否"):
            return stripped_punct

        # 4. 正则提取 <result>是</result> 或 <result>否</result>
        match = re.search(r'<result>\s*(是|否)\s*</result>', result)
        if match:
            return match.group(1)

        # 5. 正则提取行首的 "是"/"否"（排除引号内）
        quote_stripped = re.sub(r'[""\'].*?[""\']', '', result)
        if re.search(r'^\s*是\b', quote_stripped):
            return "是"
        if re.search(r'^\s*不\b', quote_stripped):
            return "是"
        if re.search(r'^\s*否\b', quote_stripped):
            return "否"

        # 6. 在字符串任意位置查找 "是" 或 "否"
        for char in ["是", "否", "不"]:
            if char in result:
                return "是" if char != "否" else "否"

        # 7. 首字判定（fallback）
        first_char = result[0] if result else ""
        if first_char == "是":
            return "是"
        if first_char in ("不", "否"):
            return "否"

        logger.warning(
            f"[DomainGuard] 意外的分类输出：'{result}'，保守放行"
        )
        return "error"

    def is_pet_related(self, query: str) -> bool:
        """
        判断 query 是否属于宠物医疗领域。

        Returns:
            True  — query 属于宠物领域，或 Guard 未启用
            False — query 不属于宠物领域
        """
        if not self.enabled:
            return True

        if not query or not query.strip():
            return True

        result = self._classify(query)
        return result in ("是", "error")

    def check_and_respond(self, query: str) -> Optional[str]:
        """
        检查 query 并返回拒绝语（如果不相关）。

        Returns:
            拒绝语字符串（如果不相关），None 表示放行
        """
        if not self.is_pet_related(query):
            return (
                "抱歉，我是一个专注于宠物狗健康和护理的问答助手。"
                "关于猫、鸟类、鱼类或其他非宠物狗相关的问题，我暂时无法帮忙。 "
                "如果您有任何关于狗狗的健康问题，欢迎随时提问！"
            )
        return None

    def check_and_respond_stream(self, query: str) -> Optional[str]:
        """流式版本的检查（与同步版本行为一致）。"""
        return self.check_and_respond(query)
