"""
测试 DomainGuard 领域守卫模块（Ollama 版本）
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def _mock_ollama_response(response_text: str):
    """构造 ollama.generate 的 mock 返回值。"""
    return {"response": response_text}


class TestDomainGuardUnit:
    """纯单元测试，不依赖 Ollama"""

    def test_guard_error_conservative(self):
        """Ollama 异常时保守放行"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(guard_model_name="nonexistent-model", enabled=True)
        with patch("src.core.domain_guard.ollama.generate", side_effect=RuntimeError("offline")):
            assert guard.is_pet_related("狗感冒了") is True

    def test_guard_disabled_bypasses_all(self):
        """enabled=False 时所有 query 直接放行，不调用 ollama"""
        from src.core.domain_guard import DomainGuard

        with patch("src.core.domain_guard.ollama.generate") as mock_gen:
            guard = DomainGuard(enabled=False)
            result = guard.is_pet_related("量子力学是什么")
            assert result is True
            mock_gen.assert_not_called()

    def test_empty_query_approved(self):
        """空 query 保守放行，不调用 ollama"""
        from src.core.domain_guard import DomainGuard

        with patch("src.core.domain_guard.ollama.generate") as mock_gen:
            guard = DomainGuard(enabled=True)
            assert guard.is_pet_related("") is True
            assert guard.is_pet_related("   ") is True
            mock_gen.assert_not_called()

    def test_whitespace_query_approved(self):
        """纯空白 query 保守放行"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        assert guard.is_pet_related("  \n\t  ") is True

    def test_classify_returns_yes(self):
        """ollama 返回"是"时放行"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")) as mock_gen:
            result = guard.is_pet_related("我家狗发烧了")
            assert result is True
            mock_gen.assert_called_once()

    def test_classify_returns_no(self):
        """ollama 返回"否"时拒绝"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否")):
            result = guard.is_pet_related("量子力学是什么")
            assert result is False

    def test_classify_returns_unexpected_value_conservative(self):
        """ollama 返回意外值时保守放行"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("我不确定")):
            result = guard.is_pet_related("今天吃什么")
            assert result is True

    def test_classify_yes_with_trailing_space(self):
        """ollama 返回'是 '（带空格）时正确处理"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是 ")):
            assert guard.is_pet_related("猫呕吐怎么办") is True

    def test_classify_no_variants(self):
        """各种"否"的变体都能正确识别"""
        from src.core.domain_guard import DomainGuard

        for return_value in ("否", "否 ", "否。"):
            guard = DomainGuard(enabled=True)
            with patch("src.core.domain_guard.ollama.generate",
                       return_value=_mock_ollama_response(return_value)):
                assert guard.is_pet_related("python教程") is False

    def test_classify_error_conservative(self):
        """ollama 调用异常时保守放行"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   side_effect=RuntimeError("模型加载失败")):
            result = guard.is_pet_related("狗感冒了")
            assert result is True

    def test_check_and_respond_returns_none_for_pet_query(self):
        """宠物相关 query 返回 None（放行）"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")):
            result = guard.check_and_respond("狗发烧了怎么办")
            assert result is None

    def test_check_and_respond_returns_rejection_for_off_topic(self):
        """非宠物 query 返回拒绝语"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否")):
            result = guard.check_and_respond("量子力学是什么")
            assert result is not None
            assert "宠物" in result
            assert isinstance(result, str)
            assert len(result) > 0

    def test_check_and_respond_stream_returns_none_for_pet(self):
        """流式版本：宠物 query 返回 None"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")):
            result = guard.check_and_respond_stream("猫不吃东西")
            assert result is None

    def test_check_and_respond_stream_returns_rejection_for_off_topic(self):
        """流式版本：非宠物 query 返回拒绝语"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否")):
            result = guard.check_and_respond_stream("推荐一部电影")
            assert result is not None
            assert len(result) > 0

    def test_default_system_prompt_not_empty(self):
        """默认 system prompt 不为空"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        assert guard.system_prompt is not None
        assert len(guard.system_prompt) > 10
        assert "狗" in guard.system_prompt

    def test_custom_system_prompt(self):
        """支持自定义 system prompt，传递给 ollama"""
        from src.core.domain_guard import DomainGuard

        custom_prompt = "仅允许宠物健康问题。"
        guard = DomainGuard(enabled=True, system_prompt=custom_prompt)
        assert guard.system_prompt == custom_prompt

        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")) as mock_gen:
            guard.is_pet_related("测试")
            call_kwargs = mock_gen.call_args.kwargs
            assert custom_prompt in call_kwargs["prompt"]


class TestClassifyParsing:
    """测试 _classify 方法的解析逻辑（各种边界 case）"""

    def _build_guard(self):
        from src.core.domain_guard import DomainGuard
        return DomainGuard(enabled=True)

    def test_pure_yes(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")):
            assert guard.is_pet_related("测试") is True

    def test_pure_no(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否")):
            assert guard.is_pet_related("测试") is False

    def test_yes_with_trailing_space(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是 ")):
            assert guard.is_pet_related("测试") is True

    def test_no_with_trailing_space(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否 ")):
            assert guard.is_pet_related("测试") is False

    def test_yes_with_whitespace(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("  是  ")):
            assert guard.is_pet_related("测试") is True

    def test_no_with_whitespace(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("  否  ")):
            assert guard.is_pet_related("测试") is False

    def test_yes_with_period(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是。")):
            assert guard.is_pet_related("测试") is True

    def test_yes_with_question_mark(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是？")):
            assert guard.is_pet_related("测试") is True

    def test_yes_with_exclamation(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是！")):
            assert guard.is_pet_related("测试") is True

    def test_no_with_comma(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否，")):
            assert guard.is_pet_related("测试") is False

    def test_yes_with_explanation(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是，这是关于狗的。")):
            assert guard.is_pet_related("测试") is True

    def test_no_with_explanation(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否，跟宠物无关。")):
            assert guard.is_pet_related("测试") is False

    def test_yes_in_xml_tag(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("<result>是</result>")):
            assert guard.is_pet_related("测试") is True

    def test_no_in_xml_tag(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("<result>否</result>")):
            assert guard.is_pet_related("测试") is False

    def test_yes_in_xml_tag_with_spaces(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("  <result>  是  </result>  ")):
            assert guard.is_pet_related("测试") is True

    def test_yes_leading_with_quotes(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("'是'")):
            assert guard.is_pet_related("测试") is True

    def test_no_leading_with_quotes(self):
        guard = self._build_guard()
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response('"否"')):
            assert guard.is_pet_related("测试") is False


class TestDomainGuardIntegration:
    """集成测试（mock ollama 模拟真实 LLM 响应）"""

    def test_pet_queries_pass(self):
        """宠物狗相关 query 均被放行"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")) as mock_gen:
            pet_queries = [
                "我家狗最近不吃东西",
                "狗发烧了怎么办",
                "金毛适合吃什么狗粮",
                "狗狗呕吐是什么原因",
                "边牧怎么训练",
                "柴犬掉毛严重吗",
                "柯基腿太短怎么办",
                "哈士奇精力太旺盛怎么解决",
            ]
            for q in pet_queries:
                result = guard.is_pet_related(q)
                assert result is True, f"Query '{q}' 应该被放行"
            assert mock_gen.call_count == len(pet_queries)

    def test_off_topic_queries_rejected(self):
        """非宠物狗 query 均被拒绝"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("否")) as mock_gen:
            off_topic_queries = [
                "猫感冒了怎么办",
                "量子力学是什么",
                "如何学 Python",
                "推荐一部电影",
                "今天天气怎么样",
                "仓鼠一直睡觉正常吗",
                "兔子拉稀怎么办",
                "鹦鹉学说话",
            ]
            for q in off_topic_queries:
                result = guard.is_pet_related(q)
                assert result is False, f"Query '{q}' 应该被拒绝"
            assert mock_gen.call_count == len(off_topic_queries)

    def test_mixed_query_pet_part_passes(self):
        """混合 query（包含宠物狗部分）被放行"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")):
            result = guard.is_pet_related("我家狗感冒了，顺便问下量子力学")
            assert result is True

    def test_guard_adds_extra_llm_call(self):
        """每次 is_pet_related 额外产生一次 ollama.generate 调用"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard(enabled=True)
        with patch("src.core.domain_guard.ollama.generate",
                   return_value=_mock_ollama_response("是")) as mock_gen:
            guard.is_pet_related("狗发烧了")
            assert mock_gen.call_count == 1
            guard.is_pet_related("量子力学")
            assert mock_gen.call_count == 2

    def test_guard_default_enabled(self):
        """默认启用 Guard（enabled=True）"""
        from src.core.domain_guard import DomainGuard

        guard = DomainGuard()
        assert guard.enabled is True
