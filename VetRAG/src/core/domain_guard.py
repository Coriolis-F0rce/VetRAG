"""
domain_guard.py - 领域边界守卫（LLM 零样本分类）

在检索之前判断用户 query 是否属于宠物医疗领域。
- 属于领域内 → 继续 RAG 流程
- 不属于 → 直接返回友好拒绝语，跳过检索和生成
"""

import re
import os
from typing import Optional, Literal
from .logging import logger


class DomainGuard:
    """
    LLM 零样本领域分类器。

    使用未微调的基础模型（Qwen3-1.7B）做零样本分类，
    避免微调后模型输出格式退化的问题。
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
        "只回复一个词：'是' 表示属于宠物狗领域，'否' 表示不属于。"
        "不要添加任何解释或标点符号。"
    )

    DEFAULT_REJECTION = (
        "抱歉，我是一个专注于宠物狗健康和护理的问答助手。"
        "关于其他非宠物狗相关的问题，我暂时无法帮忙。 "
        "如果您有任何关于狗狗的健康问题，欢迎随时提问！"
    )

    def __init__(
        self,
        generator: Optional["QwenGenerator"] = None,
        system_prompt: Optional[str] = None,
        enabled: bool = True,
        base_model_path: Optional[str] = None,
    ):
        """
        Args:
            generator: 微调后的 QwenGenerator 实例（用于主生成流程，非分类）。
            system_prompt: 分类系统提示词，默认使用 DEFAULT_SYSTEM_PROMPT。
            enabled: 是否启用 Guard。为 False 时所有 query 直接放行。
            base_model_path: 基础模型路径，用于 Domain Guard 分类。
                            如果为 None，则从环境变量或默认路径加载。
        """
        self.generator = generator
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.enabled = enabled
        self.base_generator = None
        self._base_model_loaded = False

        # 确定基础模型路径
        if base_model_path:
            self._base_model_path = base_model_path
        else:
            # 从环境变量或默认路径
            default_base = os.getenv(
                "Qwen3_BASE_MODEL_PATH",
                str(
                    (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    / "models" / "Qwen3-1.7B")
                )
            )
            self._base_model_path = default_base

    def _ensure_base_generator(self) -> bool:
        """
        延迟加载基础模型生成器。只在第一次需要时加载。
        Returns:
            True  — 加载成功
            False — 加载失败
        """
        if self._base_model_loaded:
            return self.base_generator is not None

        self._base_model_loaded = True

        if not os.path.exists(self._base_model_path):
            logger.warning(
                f"[DomainGuard] 基础模型路径不存在: {self._base_model_path}，"
                "Domain Guard 将无法正常分类，保守放行所有请求。"
            )
            return False

        try:
            # 延迟导入，避免循环依赖
            from ..rag_interface import QwenGenerator
            self.base_generator = QwenGenerator(
                model_path=self._base_model_path,
                device="cpu",
            )
            logger.info(f"[DomainGuard] 基础模型加载成功: {self._base_model_path}")
            return True
        except Exception as e:
            logger.error(
                f"[DomainGuard] 基础模型加载失败: {e}，保守放行所有请求。"
            )
            self.base_generator = None
            return False

    def _classify(self, query: str) -> Literal["是", "否", "error"]:
        """
        调用 LLM 进行零样本分类。优先使用基础模型。

        Returns:
            "是"  — query 属于宠物领域
            "否"  — query 不属于宠物领域
            "error" — 分类失败（如模型未加载），保守放行
        """
        # 优先尝试基础模型
        if self._ensure_base_generator() and self.base_generator is not None:
            generator = self.base_generator
        elif self.generator is not None:
            generator = self.generator
        else:
            return "error"

        try:
            prompt = f"{self.system_prompt}\n\n问题：{query}"
            raw_output = generator.generate(prompt, max_new_tokens=30)
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
                f"[DomainGuard] 意外的分类输出（已用基础模型 fallback）：'{result}'，保守放行"
            )
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
