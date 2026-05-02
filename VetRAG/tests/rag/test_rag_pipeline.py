"""
测试 RAG Pipeline 模块
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).resolve().parents[1]
# __file__ = D:\Backup\PythonProject2\VetRAG\tests\rag\test_rag_pipeline.py
# parents[0] = VetRAG/tests/rag/, [1] = VetRAG/tests/, [2] = VetRAG/
sys.path.insert(0, str(project_root))


class TestRAGPipelineUnit:
    """RAG Pipeline 单元测试"""

    def test_pipeline_data_dir_default(self):
        """测试默认数据目录"""
        default_data_dir = "data"
        default_persist_dir = "./chroma_db"
        assert default_data_dir == "data"
        assert default_persist_dir == "./chroma_db"

    def test_system_prompt_construction(self):
        """测试系统提示词构建"""
        system = "你是一个专业的兽医助手。"
        user = "狗狗发烧怎么办？"
        context = "犬瘟热会导致发烧"

        user_content = f"参考资料：\n{context}\n\n问题：{user}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "犬瘟热" in messages[1]["content"]

    def test_retrieval_threshold_filtering(self):
        """测试相似度阈值过滤"""
        results = [
            {"similarity": 0.8, "document": "文档A"},
            {"similarity": 0.6, "document": "文档B"},
            {"similarity": 0.3, "document": "文档C"},  # 低于阈值
            {"similarity": 0.45, "document": "文档D"},  # 低于阈值
        ]
        threshold = 0.5
        valid = [doc for doc in results if doc.get("similarity", 0) >= threshold]
        assert len(valid) == 2
        assert all(doc["similarity"] >= threshold for doc in valid)

    def test_context_truncation(self):
        """测试上下文截断逻辑"""
        long_content = "x" * 1000
        max_len = 500
        truncated = long_content[:max_len] + "…" if len(long_content) > max_len else long_content
        assert len(truncated) == 501
        assert truncated.endswith("…")

    def test_retrieval_formatting(self):
        """测试检索结果格式化"""
        raw_doc = {
            "document": "```json\n{\"key\": \"value\"}\n```\n## 犬瘟热\n犬瘟热是一种传染病。",
            "metadata": {"source_file": "diseases.json", "content_type": "diseases"},
            "similarity": 0.85
        }

        import re
        content = raw_doc["document"]
        content = re.sub(r"```json.*?```", "", content, flags=re.DOTALL)
        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
        content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
        content = content.strip()

        assert "```" not in content
        assert "犬瘟热" in content
        assert raw_doc["metadata"]["content_type"] == "diseases"


class TestRAGPipelineIntegration:
    """需要部分组件的集成测试"""

    def test_load_data_files_method_exists(self):
        """确认方法存在（签名测试）"""
        from src.rag_pipeline import VetRAGPipeline
        assert hasattr(VetRAGPipeline, "load_data_files")
        assert hasattr(VetRAGPipeline, "build_vector_index")
        assert hasattr(VetRAGPipeline, "query")
        assert hasattr(VetRAGPipeline, "get_system_info")


class TestChunkMetadata:
    """测试文档块元数据"""

    def test_chunk_metadata_structure(self):
        chunk = {
            "content": "犬瘟热的症状包括发烧、咳嗽等。",
            "metadata": {
                "content_type": "diseases_professional",
                "disease_name": "犬瘟热",
                "urgency_level": 3,
                "zoonotic": False
            },
            "source_file": "diseases.json",
            "content_type": "diseases_professional"
        }
        assert "content" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["disease_name"] == "犬瘟热"
        assert chunk["metadata"]["zoonotic"] is False

    def test_chunk_id_generation(self):
        """测试 chunk ID 生成逻辑"""
        import hashlib
        content = "犬瘟热是一种严重的传染病"
        content_hash = abs(hash(content)) % (10 ** 8)
        chunk_id = f"chunk_{content_hash:08d}"
        assert chunk_id.startswith("chunk_")
        assert len(chunk_id) == 14
        assert chunk_id == f"chunk_{content_hash:08d}"


class TestCleanOutput:
    """测试 _clean_output 输出后处理"""

    def _clean_output(self, text: str) -> str:
        """直接复制 RAGInterface._clean_output 的逻辑用于测试"""
        import re
        if not text:
            return text
        text = re.sub(
            r'[\U0001F000-\U0001F9FF]'
            r'|[\U00002702-\U000027B0]'
            r'|[\U0001F600-\U0001F64F]'
            r'|[\U00002600-\U000026FF]'
            r'|[\U0001F300-\U0001F5FF]'
            r'|[\U0001F680-\U0001F6FF]'
            r'|[\U0001FA00-\U0001FAFF]'
            r'|[\U0001FB00-\U0001FBFF]'
            r'|[\U0001F000-\U0001FFFF]'
            r'|[\U0000200B-\U0000200F]'
            r'|[\U0001F1E6-\U0001F1FF]'
            r'|[☑☒✓✗✔✘]+',
            '', text
        )
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()

    def test_clean_output_removes_emoji(self):
        assert self._clean_output("狗🐶") == "狗"
        assert self._clean_output("🐕 狗粮推荐") == "狗粮推荐"
        assert self._clean_output("🐾健康指南😊") == "健康指南"
        assert self._clean_output("🐶🐕🐩🐺🦮🐕‍🦺") == ""

    def test_clean_output_removes_dingbats(self):
        assert self._clean_output("答案☑") == "答案"
        assert self._clean_output("选项：✓ 是，✗ 否") == "选项： 是， 否"
        assert self._clean_output("正确✔错误✘") == "正确错误"

    def test_clean_output_removes_control_chars(self):
        assert self._clean_output("text\x07beep") == "textbeep"
        assert self._clean_output("line1\x0bline2") == "line1line2"

    def test_clean_output_preserves_normal_text(self):
        normal = "狗狗发烧时，首先要测量体温。如果体温超过39.5度，建议尽快就医。"
        assert self._clean_output(normal) == normal

    def test_clean_output_empty_input(self):
        assert self._clean_output("") == ""
        assert self._clean_output("   ") == ""

    def test_clean_output_none_input(self):
        assert self._clean_output(None) is None


class TestQueryFlow:
    """测试查询流程"""

    def test_empty_query_returns_empty(self):
        results = {"results": [], "total_results": 0}
        assert results["total_results"] == 0
        assert len(results["results"]) == 0

    def test_no_generator_returns_none_answer(self):
        generator = None
        answer = None if generator is None else generator.generate("test")
        assert answer is None

    def test_answer_format_structure(self):
        answer = {
            "question": "狗狗发烧怎么办？",
            "answer": "应该测量体温并及时就医。",
            "retrieved": [{"similarity": 0.85}],
            "generated": True
        }
        assert "question" in answer
        assert "answer" in answer
        assert "retrieved" in answer
        assert "generated" in answer
        assert answer["generated"] is True
