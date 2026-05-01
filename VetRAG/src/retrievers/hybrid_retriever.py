"""
混合检索器：Dense (ChromaDB HNSW) + Sparse (BM25) + RRF 融合

核心算法：Reciprocal Rank Fusion (RRF)
    RRF_score(d) = Σ 1 / (k + rank_i(d))
    其中 k = 60（经验最优常数），i 为各检索系统的编号

使用方式：
    from src.retrievers import HybridRetriever

    hr = HybridRetriever(chroma_collection=collection, embed_fn=embed_fn)
    hr.build_index(documents, chunk_ids, metadatas)
    results = hr.search("狗狗发烧怎么办", top_k=5, dense_weight=0.7, bm25_weight=0.3)
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass

from src.retrievers.bm25_index import BM25Retriever, BM25Result

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """
    混合检索单条结果。

    同时包含 Dense 和 BM25 两路检索的分数与排名，
    以及 RRF 融合后的综合分数和最终排名。
    """
    chunk_id: str
    document: str
    metadata: Dict[str, Any]
    # Dense 分数
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    # BM25 分数
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    # 融合分数
    rrf_score: float = 0.0
    rank: int = 0
    # 用于调试的各路排名
    _ranks: Dict[str, int] = None

    def __post_init__(self):
        if self._ranks is None:
            self._ranks = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "metadata": self.metadata,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "rrf_score": self.rrf_score,
            "rank": self.rank,
        }


class HybridRetriever:
    """
    混合检索器：封装 Dense (ChromaDB HNSW) + Sparse (BM25) 检索和 RRF 融合。

    设计原则：
    1. **组合优于继承**：内部持有 ChromaDB collection 和 BM25Retriever，各自独立
    2. **延迟初始化**：BM25Retriever 在首次 build_index 时才实例化
    3. **索引持久化**：BM25 索引保存至 persist_dir/bm25_index.pkl
    4. **向后兼容**：search() 保留 use_dense_only / use_bm25_only 参数用于调试

    与 ChromaVectorStore 的关系：
    - HybridRetriever 不直接持有 ChromaVectorStore
    - 构造时传入 chroma_collection（ChromaDB Collection 对象）和 embed_fn（编码函数）
    - ChromaVectorStore 内部会实例化 HybridRetriever 来提供混合检索能力

    使用示例：
        hr = HybridRetriever(
            chroma_collection=chroma_collection,
            embed_fn=lambda texts: model.encode(texts),
            persist_dir="./chroma_db",
        )
        hr.build_index(documents, chunk_ids, metadatas)
        results = hr.search("狗狗发烧怎么办", top_k=5)
    """

    # RRF 公式中的常数，学术界和工业界经验最优值
    RRF_K = 60

    def __init__(
        self,
        chroma_collection: Any,
        embed_fn: Callable[[str], List[float]],
        persist_dir: str = "./chroma_db",
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        default_top_k: int = 20,
    ):
        """
        初始化混合检索器。

        Args:
            chroma_collection: ChromaDB Collection 对象
            embed_fn: 查询编码函数，输入字符串返回归一化向量列表
            persist_dir: BM25 索引持久化目录
            dense_weight: Dense 检索权重（RRF 融合时使用）
            bm25_weight: BM25 权重
            default_top_k: 每路检索的默认召回数量
        """
        self._chroma_collection = chroma_collection
        self._embed_fn = embed_fn
        self._persist_dir = persist_dir
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight
        self._default_top_k = default_top_k

        self._bm25_retriever: Optional[BM25Retriever] = None
        self._index_built: bool = False

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
        构建 BM25 索引（同时需要 ChromaDB 侧已通过 ChromaVectorStore.add_chunks 添加文档）。

        注意：ChromaDB 侧的向量索引由 ChromaVectorStore 管理，
        此处只负责构建 BM25 索引。

        Args:
            documents: 文档文本列表
            chunk_ids: 对应 chunk ID 列表
            metadatas: 可选元数据列表
            incremental: 是否增量更新

        Returns:
            构建结果统计
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)

        self._bm25_retriever = BM25Retriever(
            persist_dir=self._persist_dir,
        )

        result = self._bm25_retriever.build_index(
            documents=documents,
            chunk_ids=chunk_ids,
            metadatas=metadatas,
            incremental=incremental,
        )

        self._index_built = True
        logger.info(
            f"HybridRetriever: BM25 索引构建完成，"
            f"共 {result['total']} 篇文档（增量={incremental}）"
        )
        return result

    def add_documents(
        self,
        documents: List[str],
        chunk_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """
        增量添加文档到 BM25 索引。

        注意：向量侧由 ChromaVectorStore.add_chunks 管理。
        """
        if self._bm25_retriever is None:
            return self.build_index(documents, chunk_ids, metadatas, incremental=False)

        return self._bm25_retriever.add_documents(documents, chunk_ids, metadatas)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        use_hybrid: bool = True,
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        执行混合检索。

        Args:
            query: 用户查询文本
            top_k: 最终返回结果数量
            dense_weight: Dense 检索权重（覆盖默认值）
            bm25_weight: BM25 权重（覆盖默认值）
            use_hybrid: 是否使用混合模式；False 则仅使用 Dense 检索
            return_raw: 若为 True，同时返回 Dense 和 BM25 各自的原始结果

        Returns:
            {
                "query": str,
                "results": List[HybridResult],
                "dense_results": List[Dict],  # 仅当 return_raw=True
                "bm25_results": List[Dict],  # 仅当 return_raw=True
            }
        """
        if use_hybrid:
            return self._search_hybrid(query, top_k, dense_weight, bm25_weight, return_raw)
        else:
            return self._search_dense_only(query, top_k, return_raw)

    def _search_dense_only(
        self,
        query: str,
        top_k: int,
        return_raw: bool,
    ) -> Dict[str, Any]:
        """仅使用 Dense 向量检索（保持与原 ChromaVectorStore.search 兼容）"""
        query_embedding = self._embed_fn(query)

        results = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted = self._format_chroma_results(results)
        out = {
            "query": query,
            "total_results": len(formatted),
            "results": formatted,
        }
        if return_raw:
            out["dense_results"] = formatted
            out["bm25_results"] = []
        return out

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        dense_weight: Optional[float],
        bm25_weight: Optional[float],
        return_raw: bool,
    ) -> Dict[str, Any]:
        """
        混合检索核心流程：

        1. 两路并行召回（Dense top-k + BM25 top-k）
        2. RRF 融合
        3. 返回 top-k 结果
        """
        dw = dense_weight if dense_weight is not None else self._dense_weight
        bw = bm25_weight if bm25_weight is not None else self._bm25_weight

        retrieve_k = max(top_k * 2, 20)

        # Dense 检索
        query_embedding = self._embed_fn(query)
        dense_results = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieve_k,
            include=["documents", "metadatas", "distances"],
        )
        dense_formatted = self._format_chroma_results(dense_results, include_rrf_fields=True)

        # BM25 检索
        bm25_hits: List[BM25Result] = []
        if self._bm25_retriever is not None and self._bm25_retriever.document_count > 0:
            bm25_hits = self._bm25_retriever.search(query, top_k=retrieve_k)
        else:
            logger.warning("BM25 索引为空，退化为纯 Dense 检索")

        # 构建各路排名字典（用于 RRF）
        dense_ranks = self._build_dense_rank_dict(dense_formatted)
        bm25_ranks = self._build_bm25_rank_dict(bm25_hits)

        # RRF 融合
        fused = self._rrf_fuse(
            dense_formatted, dense_ranks,
            bm25_hits, bm25_ranks,
            dw, bw,
        )

        # 截断至 top_k
        final_results = fused[:top_k]

        out = {
            "query": query,
            "total_results": len(final_results),
            "results": final_results,
        }
        if return_raw:
            out["dense_results"] = dense_formatted
            out["bm25_results"] = [h.to_dict() for h in bm25_hits]
        return out

    # ------------------------------------------------------------------
    # RRF 融合
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_results: List[Dict],
        dense_ranks: Dict[str, int],
        bm25_hits: List[BM25Result],
        bm25_ranks: Dict[str, int],
        dense_weight: float,
        bm25_weight: float,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion 实现。

        公式：RRF_score(d) = w1/(k + r1(d)) + w2/(k + r2(d))
        - d: 文档
        - k: RRF_K = 60
        - r1(d): Dense 检索中的排名（从 1 开始）
        - r2(d): BM25 检索中的排名（从 1 开始）
        - w1, w2: 各自权重

        若某文档只在一路中出现，未出现的路默认排名为 retrieve_k + 1
        """
        all_chunk_ids: set = set()

        for r in dense_results:
            all_chunk_ids.add(r["id"])
        for h in bm25_hits:
            all_chunk_ids.add(h.chunk_id)

        retrieve_k = max(len(dense_results), len(bm25_hits), 1)
        missing_rank = retrieve_k + 1

        scored: Dict[str, float] = {}
        for chunk_id in all_chunk_ids:
            rrf = 0.0
            if chunk_id in dense_ranks:
                dr = dense_ranks[chunk_id]
                rrf += dense_weight / (self.RRF_K + dr)
            else:
                rrf += dense_weight / (self.RRF_K + missing_rank)

            if chunk_id in bm25_ranks:
                br = bm25_ranks[chunk_id]
                rrf += bm25_weight / (self.RRF_K + br)
            else:
                rrf += bm25_weight / (self.RRF_K + missing_rank)

            scored[chunk_id] = rrf

        sorted_ids = sorted(scored, key=lambda cid: scored[cid], reverse=True)

        # 构建 id → document 的映射
        id_to_doc = {r["id"]: r for r in dense_results}
        for h in bm25_hits:
            if h.chunk_id not in id_to_doc:
                id_to_doc[h.chunk_id] = {
                    "id": h.chunk_id,
                    "document": h.document,
                    "metadata": h.metadata,
                    "similarity": None,
                    "distance": None,
                    "bm25_score": h.bm25_score,
                }

        # 组装最终结果
        final = []
        for rank, chunk_id in enumerate(sorted_ids, start=1):
            entry = id_to_doc[chunk_id].copy()
            entry["rrf_score"] = scored[chunk_id]
            entry["rank"] = rank

            if chunk_id in dense_ranks:
                entry["dense_rank"] = dense_ranks[chunk_id]
                entry["dense_score"] = id_to_doc[chunk_id].get("similarity")
            else:
                entry["dense_rank"] = None
                entry["dense_score"] = None

            if chunk_id in bm25_ranks:
                entry["bm25_rank"] = bm25_ranks[chunk_id]
                entry["bm25_score"] = id_to_doc[chunk_id].get("bm25_score")
            else:
                entry["bm25_rank"] = None
                entry["bm25_score"] = None

            final.append(entry)

        return final

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _embed_fn_single(self, query: str) -> List[float]:
        """包装 _embed_fn，支持单字符串输入"""
        if isinstance(query, str):
            return self._embed_fn(query)
        return self._embed_fn(query)

    def _format_chroma_results(self, results: Dict, include_rrf_fields: bool = False) -> List[Dict]:
        """将 ChromaDB query 结果格式化为统一字典列表

        Args:
            results: ChromaDB query 返回结果
            include_rrf_fields: 是否包含 RRF 融合字段；dense-only 模式不包含
        """
        formatted = []
        if not (results.get("documents") and results["documents"][0]):
            return formatted

        docs = results["documents"][0]
        dists = results.get("distances", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []

        for i, doc in enumerate(docs):
            dist = dists[i] if i < len(dists) else 0
            entry = {
                "id": ids[i] if i < len(ids) else f"unknown_{i}",
                "document": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "similarity": 1.0 - dist,
                "distance": dist,
                "dense_rank": i + 1,
            }
            if include_rrf_fields:
                entry["bm25_score"] = None
                entry["rrf_score"] = 0.0
                entry["rank"] = 0
                entry["bm25_rank"] = None
            formatted.append(entry)
        return formatted

    def _build_dense_rank_dict(self, results: List[Dict]) -> Dict[str, int]:
        return {r["id"]: r["dense_rank"] for r in results}

    def _build_bm25_rank_dict(self, hits: List[BM25Result]) -> Dict[str, int]:
        return {h.chunk_id: h.rank for h in hits}

    def get_stats(self) -> Dict[str, Any]:
        """返回检索器统计信息"""
        return {
            "index_built": self._index_built,
            "bm25_stats": self._bm25_retriever.get_stats() if self._bm25_retriever else None,
            "dense_weight": self._dense_weight,
            "bm25_weight": self._bm25_weight,
            "default_top_k": self._default_top_k,
            "chroma_doc_count": self._chroma_collection.count()
                if self._chroma_collection else 0,
        }
