"""
BM25 稀疏关键词检索模块

基于 rank_bm25 库实现中文 BM25 检索。
依赖：pip install rank-bm25 jieba
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """单条 BM25 检索结果"""
    chunk_id: str
    document: str
    metadata: Dict[str, Any]
    bm25_score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "metadata": self.metadata,
            "bm25_score": self.bm25_score,
            "rank": self.rank,
        }


class BM25Retriever:
    """
    基于 BM25 算法的稀疏关键词检索器。

    使用 rank_bm25 库实现，对中文文本使用 jieba 分词。
    索引可持久化到磁盘，增量更新时无需全量重建。

    使用示例：
        retriever = BM25Retriever(persist_dir="./chroma_db")
        # 构建索引（首次或增量）
        retriever.build_index(documents, chunk_ids, metadatas)
        # 检索
        results = retriever.search("狗狗发烧怎么办", top_k=5)
    """

    # BM25 标准参数
    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        tokenize_lang: str = "zh",
    ):
        """
        初始化 BM25 检索器。

        Args:
            persist_dir: 持久化目录，BM25 索引保存至此
            k1: BM25 k1 参数，控制词频饱和度（默认 1.5）
            b: BM25 b 参数，控制文档长度归一化（默认 0.75）
            tokenize_lang: 分词语言，"zh" 使用 jieba，"en" 使用空格分词
        """
        self.persist_dir = persist_dir
        self.k1 = k1
        self.b = b
        self.tokenize_lang = tokenize_lang

        self._index_file = os.path.join(persist_dir, "bm25_index.pkl")
        self._corpus_file = os.path.join(persist_dir, "bm25_corpus.pkl")

        self._tokenized_corpus: List[List[str]] = []
        self._chunk_ids: List[str] = []
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._bm25: Optional[Any] = None  # rank_bm25.BM25Ok

        self._load_index()

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        if self.tokenize_lang == "zh":
            import jieba
            return [w for w in jieba.cut(text) if w.strip()]
        else:
            return text.lower().split()

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def build_index(
        self,
        documents: List[str],
        chunk_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        incremental: bool = False,
    ) -> Dict[str, int]:
        """
        构建或更新 BM25 索引。

        Args:
            documents: 文档文本列表
            chunk_ids: 对应的 chunk ID 列表（用于去重和结果关联）
            metadatas: 可选，元数据列表
            incremental: 若为 True，增量追加文档；否则重建索引
        """
        if not documents:
            return {"added": 0, "total": 0}

        if metadatas is None:
            metadatas = [{}] * len(documents)

        if incremental and self._bm25 is not None:
            self._add_to_index(documents, chunk_ids, metadatas)
        else:
            self._rebuild_index(documents, chunk_ids, metadatas)

        self._save_index()
        return {
            "added": len(documents),
            "total": len(self._documents),
        }

    def _rebuild_index(
        self,
        documents: List[str],
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """全量重建 BM25 索引"""
        logger.info(f"BM25: 全量重建索引，共 {len(documents)} 篇文档")

        self._chunk_ids = list(chunk_ids)
        self._documents = list(documents)
        self._metadatas = list(metadatas)

        self._tokenized_corpus = [self._tokenize(doc) for doc in documents]

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        self._bm25.corpus = self._tokenized_corpus

        logger.info(f"BM25: 索引构建完成，共 {len(self._documents)} 篇文档")

    def _add_to_index(
        self,
        documents: List[str],
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """增量追加文档到现有索引"""
        logger.info(f"BM25: 增量追加 {len(documents)} 篇文档")

        self._chunk_ids.extend(chunk_ids)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

        new_tokenized = [self._tokenize(doc) for doc in documents]
        self._tokenized_corpus.extend(new_tokenized)

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        self._bm25.corpus = self._tokenized_corpus

    def add_documents(
        self,
        documents: List[str],
        chunk_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """
        增量添加文档（便捷封装，等价于 build_index(incremental=True)）
        """
        return self.build_index(documents, chunk_ids, metadatas, incremental=True)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[BM25Result]:
        """
        执行 BM25 关键词检索。

        Args:
            query: 用户查询文本
            top_k: 返回结果数量

        Returns:
            按 BM25 分数降序排列的检索结果列表
        """
        if self._bm25 is None or not self._documents:
            logger.warning("BM25 索引为空，请先调用 build_index()")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            results.append(BM25Result(
                chunk_id=self._chunk_ids[idx],
                document=self._documents[idx],
                metadata=self._metadatas[idx] if idx < len(self._metadatas) else {},
                bm25_score=float(scores[idx]),
                rank=rank + 1,
            ))

        return results

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load_index(self):
        """从磁盘加载 BM25 索引"""
        if not os.path.exists(self._index_file):
            return

        try:
            with open(self._index_file, "rb") as f:
                index_data = pickle.load(f)
            with open(self._corpus_file, "rb") as f:
                corpus_data = pickle.load(f)

            self._chunk_ids = corpus_data["chunk_ids"]
            self._documents = corpus_data["documents"]
            self._metadatas = corpus_data["metadatas"]
            self._tokenized_corpus = corpus_data["tokenized_corpus"]

            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
            self._bm25.corpus = self._tokenized_corpus

            logger.info(f"BM25: 从磁盘加载索引，共 {len(self._documents)} 篇文档")
        except Exception as e:
            logger.warning(f"BM25: 索引加载失败（{e}），将重建索引")
            self._chunk_ids = []
            self._documents = []
            self._metadatas = []
            self._tokenized_corpus = []
            self._bm25 = None

    def _save_index(self):
        """将 BM25 索引持久化到磁盘"""
        os.makedirs(self.persist_dir, exist_ok=True)

        try:
            with open(self._index_file, "wb") as f:
                pickle.dump({"k1": self.k1, "b": self.b}, f)
            with open(self._corpus_file, "wb") as f:
                pickle.dump({
                    "chunk_ids": self._chunk_ids,
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "tokenized_corpus": self._tokenized_corpus,
                }, f)
            logger.info(f"BM25: 索引已保存至 {self.persist_dir}")
        except Exception as e:
            logger.error(f"BM25: 索引保存失败：{e}")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def clear(self):
        """清空索引（内存中）"""
        self._chunk_ids = []
        self._documents = []
        self._metadatas = []
        self._tokenized_corpus = []
        self._bm25 = None

    def remove_persist(self):
        """删除磁盘上的索引文件"""
        for f in [self._index_file, self._corpus_file]:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"BM25: 已删除 {f}")

    @property
    def document_count(self) -> int:
        """当前索引中的文档数量"""
        return len(self._documents)

    def get_stats(self) -> Dict[str, Any]:
        """返回检索器统计信息"""
        return {
            "document_count": self.document_count,
            "persist_dir": self.persist_dir,
            "k1": self.k1,
            "b": self.b,
            "tokenize_lang": self.tokenize_lang,
            "index_loaded": self._bm25 is not None,
        }
