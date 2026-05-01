"""
domain_guard.py - 领域边界守卫（LLM 零样本分类）

在检索之前判断用户 query 是否属于宠物医疗领域。
- 属于领域内 → 继续 RAG 流程
- 不属于 → 直接返回友好拒绝语，跳过检索和生成
"""

from typing import Optional, Literal
from .logging import logger


class DomainGuard:
    """
    LLM 零样本领域分类器。

    用微调后的 Qwen 模型判断用户 query 是否属于宠物医疗领域。
    输出只有两个字："是" 或 "否"，最小化 token 消耗。

    拒绝示例：
        - "量子力学是什么"
        - "如何学 Python"
        - "推荐一部电影"

    放行示例：
        - "我家猫最近不吃东西"
        - "狗发烧了怎么办"
        - "猫瘟怎么预防"
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个宠物狗健康问答系统的审核员。"
        "用户将输入一个问题，请判断它是否与宠物狗（包括但不限于犬类品种、狗的健康、护理、疾病、症状、行为、饮食、疫苗、驱虫、训练等）相关。"
        "只回复一个词：'是' 表示属于宠物狗领域，'否' 表示不属于。"
        "不要添加任何解释或标点符号。"
    )

    DEFAULT_REJECTION = (
        "抱歉，我是一个专注于宠物狗健康和护理的问答助手。"
        "关于猫、鸟类、鱼类或其他非宠物狗相关的问题，我暂时无法帮忙。 "
        "如果您有任何关于狗狗的健康问题，欢迎随时提问！"
    )

    def __init__(
        self,
        generator: Optional["QwenGenerator"] = None,
        system_prompt: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Args:
            generator: QwenGenerator 实例。如果为 None，则 Guard 不生效（直接放行）。
            system_prompt: 分类系统提示词，默认使用 DEFAULT_SYSTEM_PROMPT。
            enabled: 是否启用 Guard。为 False 时所有 query 直接放行。
        """
        self.generator = generator
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.enabled = enabled

    def _classify(self, query: str) -> Literal["是", "否", "error"]:
        """
        调用 LLM 进行零样本分类。

        Returns:
            "是"  — query 属于宠物领域
            "否"  — query 不属于宠物领域
            "error" — 分类失败（如模型未加载），保守放行
        """
        if self.generator is None:
            return "error"

        try:
            prompt = f"{self.system_prompt}\n\n问题：{query}"
            raw_output = self.generator.generate(prompt, max_new_tokens=10)
            result = raw_output.strip()
            if result in ("是", "是 "):
                return "是"
            elif result in ("否", "不", "不是", "否 "):
                return "否"
            else:
                logger.warning(f"[DomainGuard] 意外的分类输出: '{result}'，保守放行")
                return "error"
        except Exception as e:
            logger.error(f"[DomainGuard] 分类失败: {e}，保守放行")
            return "error"

    def is_pet_related(self, query: str) -> bool:
        """
        判断 query 是否属于宠物医疗领域。

        Args:
            query: 用户输入的问题

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

        Args:
            query: 用户输入的问题

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
        """
        流式版本的检查（目前与同步版本行为一致）。

        对于 Guard 判断来说不需要流式输出，
        但为保持接口一致性提供此方法。
        """
        return self.check_and_respond(query)
