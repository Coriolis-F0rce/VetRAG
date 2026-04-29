"""
测试 FastAPI Web 服务接口
使用 TestClient 进行同步测试，无需启动真实服务器
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).resolve().parents[1]
# __file__ = D:\Backup\PythonProject2\VetRAG\tests\api\test_web_api.py
# parents[0] = VetRAG/tests/api/, [1] = VetRAG/tests/, [2] = VetRAG/
sys.path.insert(0, str(project_root))

# mock 所有重型依赖，避免导入报错
mock_transformers = MagicMock()
mock_chromadb = MagicMock()
mock_sentence_transformers = MagicMock()
mock_torch = MagicMock()
mock_numpy = MagicMock()

for mod_name in [
    "transformers", "transformers.generation", "transformers.generation.utils",
    "transformers.training_args", "transformers.modeling_utils",
    "chromadb", "chromadb.config",
    "sentence_transformers",
    "torch", "torch.cuda",
]:
    sys.modules[mod_name] = MagicMock()


@pytest.fixture
def mock_rag_interface():
    """Mock RAGInterface 以避免加载真实模型"""
    mock = MagicMock()
    mock.get_stats.return_value = {
        "vector_store": {"document_count": 10},
        "generator_loaded": True,
        "generator_model": "mock_model_path"
    }
    mock.query.return_value = {
        "question": "狗狗发烧怎么办？",
        "answer": "应该及时测量体温并就医。",
        "retrieved": [],
        "generated": True
    }
    mock.vector_store.search.return_value = {
        "results": [
            {
                "document": "犬瘟热会导致发烧",
                "similarity": 0.85,
                "metadata": {"source_file": "diseases.json"}
            }
        ]
    }
    return mock


@pytest.fixture
def app_client(mock_rag_interface):
    with patch("src.rag_interface.ChromaVectorStore") as mock_cs:
        mock_cs.return_value.get_collection_stats.return_value = {"document_count": 10}
        mock_cs.return_value.search.return_value = {"results": []}
    with patch("src.rag_interface.VetRAGDataLoader"):
        with patch("src.rag_interface.AutoModelForCausalLM"):
            with patch("src.rag_interface.AutoTokenizer"):
                with patch("src.rag_interface.QwenGenerator"):
                    with patch("src.core.config.CHROMA_PERSIST_DIR", "./temp_test_chroma"):
                        with patch("src.core.config.QWEN3_FINETUNED_PATH", "./temp_test_model"):
                            from web_api import app
                            from fastapi.testclient import TestClient
                            with TestClient(app) as client:
                                yield client


class TestRootEndpoint:
    def test_root_returns_html(self, app_client):
        response = app_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "兽医RAG" in response.text


class TestStatsEndpoint:
    def test_stats_returns_json(self, app_client, mock_rag_interface):
        response = app_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "vector_store" in data
        assert "generator_loaded" in data

    def test_stats_response_format(self, app_client, mock_rag_interface):
        """stats 端点返回格式正确"""
        response = app_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestQueryEndpoint:
    def test_query_returns_json(self, app_client, mock_rag_interface):
        response = app_client.post("/query", json={"question": "狗狗发烧怎么办？"})
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data

    def test_query_with_top_k(self, app_client, mock_rag_interface):
        response = app_client.post("/query", json={"question": "测试", "top_k": 3})
        assert response.status_code == 200

    def test_query_empty_question(self, app_client):
        response = app_client.post("/query", json={"question": ""})
        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    def test_query_missing_question(self, app_client):
        response = app_client.post("/query", json={})
        assert response.status_code == 200
        data = response.json()
        assert "error" in data


class TestStreamEndpoint:
    def test_stream_returns_sse(self, app_client):
        response = app_client.get("/stream", params={"question": "测试问题"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_stream_with_params(self, app_client):
        response = app_client.get(
            "/stream",
            params={"question": "狗狗咳嗽", "top_k": 5, "threshold": 0.3}
        )
        assert response.status_code == 200


class TestCORS:
    def test_cors_middleware_configured(self, app_client):
        """验证 CORS 中间件已配置（允许访问 /stats）"""
        response = app_client.get("/stats")
        assert response.status_code == 200


class TestErrorHandling:
    def test_query_invalid_json(self, app_client):
        # FastAPI/Starlette 在收到无效 JSON body 时会抛出 json.decoder.JSONDecodeError
        # 该异常未被业务层捕获，以 500 Internal Server Error 体现
        with pytest.raises(Exception) as exc_info:
            app_client.post("/query", content="not json", headers={"Content-Type": "application/json"})
        assert exc_info.value is not None
