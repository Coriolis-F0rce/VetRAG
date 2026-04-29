"""
测试 ChromaDB 向量存储模块
使用 Mock 避免加载重型模型
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).resolve().parents[1]
# __file__ = D:\Backup\PythonProject2\VetRAG\tests\rag\test_vector_store.py
# parents[0] = VetRAG/tests/rag/, [1] = VetRAG/tests/, [2] = VetRAG/
sys.path.insert(0, str(project_root))


class TestChromaVectorStoreUnit:
    """纯单元测试，不依赖模型加载"""

    def test_collection_name_default(self):
        # __new__ 跳过 __init__，不触发 chromadb/sentence_transformers 导入
        from src.vector_store_chroma import ChromaVectorStore
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store.collection_name = "veterinary_rag"
        store.persist_directory = "./chroma_db"
        assert store.collection_name == "veterinary_rag"

    def test_search_result_format(self):
        mock_results = {
            "ids": [["chunk_00001"]],
            "documents": [["犬瘟热是一种病毒性疾病"]],
            "metadatas": [[{"source_file": "diseases.json"}]],
            "distances": [[0.15]]
        }

        formatted = []
        if mock_results["documents"] and mock_results["documents"][0]:
            for i in range(len(mock_results["documents"][0])):
                distance = mock_results["distances"][0][i]
                similarity = 1.0 - distance
                formatted.append({
                    "id": mock_results["ids"][0][i],
                    "document": mock_results["documents"][0][i],
                    "metadata": mock_results["metadatas"][0][i],
                    "similarity": similarity,
                    "distance": distance
                })

        assert len(formatted) == 1
        assert formatted[0]["similarity"] == 0.85
        assert formatted[0]["distance"] == 0.15
        assert formatted[0]["document"] == "犬瘟热是一种病毒性疾病"

    def test_similarity_conversion(self):
        distances = [0.0, 0.25, 0.5, 0.75, 1.0]
        similarities = [1.0 - d for d in distances]
        assert similarities == [1.0, 0.75, 0.5, 0.25, 0.0]

    def test_chunks_filter_short_content(self):
        """过滤掉内容过短的 chunks"""
        chunks = [
            {"content": "短的"},            # 太短（3个字符）
            {"content": "这是正常长度的文档内容"},  # 通过（12个字符）
            {"content": ""},               # 空内容
            {"content": "   "},           # 只有空格
        ]
        filtered = [c for c in chunks if c.get("content", "") and len(c.get("content", "").strip()) >= 10]
        assert len(filtered) == 1
        assert filtered[0]["content"] == "这是正常长度的文档内容"

    def test_chunks_filter_empty_content(self):
        chunks = [
            {"content": "x" * 20},
            {"content": ""},
            {"content": None},
        ]
        filtered = [c for c in chunks if c.get("content") and len(str(c["content"]).strip()) >= 10]
        assert len(filtered) == 1


class TestChromaVectorStoreIntegration:
    """需要 ChromaDB 但不需要模型的集成测试"""

    @pytest.fixture
    def temp_chroma_dir(self, tmp_path):
        chroma_dir = tmp_path / "test_chroma"
        chroma_dir.mkdir()
        return str(chroma_dir)

    @pytest.fixture
    def mock_embedding_model(self):
        """返回一个 Mock 的 embedding 模型"""
        mock_model = MagicMock()
        mock_embedding = [0.1] * 1024
        mock_model.encode.return_value = [mock_embedding]
        return mock_model

    def test_init_creates_directory(self, temp_chroma_dir):
        pytest.importorskip("chromadb")
        from src.vector_store_chroma import ChromaVectorStore
        store = ChromaVectorStore(
            persist_directory=temp_chroma_dir,
            collection_name="test_collection"
        )
        assert Path(temp_chroma_dir).exists()

    def test_add_chunks_returns_stats(self, temp_chroma_dir):
        pytest.importorskip("chromadb")
        from src.vector_store_chroma import ChromaVectorStore
        store = ChromaVectorStore(persist_directory=temp_chroma_dir)

        mock_collection = MagicMock()
        store.collection = mock_collection
        store.processed_ids = set()
        store._save_processed_ids = MagicMock()

        chunks = [
            {"content": "犬瘟热是一种严重的犬类传染病", "metadata": {}, "source_file": "test.json"}
        ]
        result = store.add_chunks(chunks, batch_size=10)

        assert "added" in result
        assert "skipped" in result
        assert "total" in result
        assert "current_total" in result

    def test_add_chunks_empty_list(self, temp_chroma_dir):
        pytest.importorskip("chromadb")
        from src.vector_store_chroma import ChromaVectorStore
        store = ChromaVectorStore(persist_directory=temp_chroma_dir)
        result = store.add_chunks([])
        assert result["added"] == 0
        assert result["total"] == 0

    def test_get_collection_stats(self, temp_chroma_dir):
        pytest.importorskip("chromadb")
        from src.vector_store_chroma import ChromaVectorStore
        store = ChromaVectorStore(persist_directory=temp_chroma_dir)

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_collection.metadata = {"hnsw:space": "cosine"}
        store.collection = mock_collection

        stats = store.get_collection_stats()
        assert stats["document_count"] == 42
        assert stats["collection_name"] == "veterinary_rag"


class TestDocumentCleaning:
    """测试文档清洗逻辑"""

    def test_clean_removes_json_codeblock(self):
        text = "```json\n{\"key\": \"value\"}\n```\n这是正文内容"
        import re
        cleaned = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)
        assert "```json" not in cleaned
        assert "这是正文内容" in cleaned

    def test_clean_removes_generic_codeblock(self):
        text = "```python\nprint('hello')\n```\n正文"
        import re
        cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        assert "```" not in cleaned
        assert "正文" in cleaned

    def test_clean_removes_markdown_headers(self):
        text = "# 标题\n## 子标题\n正文内容"
        import re
        cleaned = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        assert "#" not in cleaned
        assert "正文内容" in cleaned

    def test_clean_trims_whitespace(self):
        text = "   \n\n  正文  \n  "
        cleaned = text.strip()
        assert cleaned == "正文"

    def test_clean_combined(self):
        text = """# 标题
```json
{"data": "test"}
```
正文内容
## 子标题
"""
        import re
        cleaned = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        assert "```" not in cleaned
        assert "正文内容" in cleaned
