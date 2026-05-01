# vector_store_chroma.py

import os
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import pickle

from retrievers import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:

    def __init__(self,
                 collection_name="veterinary_rag",
                 persist_directory="./chroma_db",
                 model_name="BAAI/bge-large-zh-v1.5",
                 use_hybrid: bool = False,
                 dense_weight: float = 0.7,
                 bm25_weight: float = 0.3):
        """
        初始化BGE向量存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            model_name: BGE模型名称
            use_hybrid: 是否启用混合检索（Dense + BM25 + RRF）
            dense_weight: Dense检索权重（use_hybrid=True时生效）
            bm25_weight: BM25权重（use_hybrid=True时生效）
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        os.makedirs(persist_directory, exist_ok=True)

        print(f"初始化ChromaDB，持久化目录: {persist_directory}")

        # 初始化ChromaDB持久化客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        print(f"加载BGE模型: {model_name}")
        try:
            # 使用sentence-transformers加载BGE模型
            self.embedding_model = SentenceTransformer(model_name)
            print("✓ BGE模型加载完成")

            # 测试模型
            test_embedding = self.embedding_model.encode(["测试"], normalize_embeddings=True)
            print(f"模型维度: {test_embedding.shape[1]}")

        except Exception as e:
            print(f"✗ BGE模型加载失败: {e}")
            # 回退到MiniLM
            print("尝试加载MiniLM模型...")
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✓ 回退到MiniLM模型")

        # 创建或获取集合
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "model": model_name,
                    "created_at": datetime.now().isoformat()
                }
            )
            print(f"✓ 集合 '{collection_name}' 已加载/创建")
        except Exception as e:
            print(f"✗ 集合创建失败: {e}")
            raise

        # 加载已处理的文档ID集合
        self.processed_ids_file = os.path.join(persist_directory, "processed_ids.pkl")
        self.processed_ids = self._load_processed_ids()

        # 初始化混合检索器（延迟初始化，首次 build_index 时才真正创建）
        self._hybrid_retriever: Optional[HybridRetriever] = None

        print(f"初始化完成，当前文档数: {self.collection.count()}")
        if self.use_hybrid:
            print(f"✓ 混合检索已启用（Dense={dense_weight}, BM25={bm25_weight}）")

    def _load_processed_ids(self) -> set:
        """加载已处理的文档ID集合"""
        if os.path.exists(self.processed_ids_file):
            try:
                with open(self.processed_ids_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return set()

    def _save_processed_ids(self):
        """保存已处理的文档ID集合"""
        try:
            with open(self.processed_ids_file, 'wb') as f:
                pickle.dump(self.processed_ids, f)
        except Exception as e:
            print(f"警告: 保存处理ID失败: {e}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        批量创建嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        try:
            # 使用sentence-transformers编码，支持normalize_embeddings参数
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,  # 归一化用于余弦相似度
                convert_to_numpy=True
            )

            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)

            return embeddings.tolist()

        except Exception as e:
            print(f"嵌入创建失败: {e}")
            # 尝试不使用归一化
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # 手动归一化
            import numpy as np
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            return embeddings.tolist()

    def create_query_embedding(self, query: str) -> List[float]:
        """
        创建查询嵌入向量

        Args:
            query: 查询文本

        Returns:
            查询嵌入向量
        """
        try:
            embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            print(f"查询嵌入创建失败: {e}")
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            import numpy as np
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()

    def add_chunks(self, chunks: List[Dict], batch_size: int = 50) -> Dict:
        """
        添加文档块到向量库

        Args:
            chunks: 文档块列表
            batch_size: 批处理大小

        Returns:
            添加统计信息
        """
        if not chunks:
            return {"added": 0, "skipped": 0, "total": 0, "current_total": self.collection.count()}

        total_chunks = len(chunks)
        added_count = 0
        skipped_count = 0

        print(f"开始处理 {total_chunks} 个文档块，批次大小: {batch_size}")

        # 收集有效文档用于 BM25 索引
        bm25_docs = []
        bm25_ids = []
        bm25_metas = []

        # 分批处理
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = []
            batch_ids = []
            batch_metadatas = []
            batch_documents = []

            # 准备批处理数据
            for chunk in batch:
                content = chunk.get("content", "")
                if not content or len(content.strip()) < 10:
                    skipped_count += 1
                    continue

                # 生成唯一ID
                content_hash = abs(hash(content)) % (10 ** 8)
                chunk_id = f"chunk_{content_hash:08d}"

                # 检查是否已处理
                if chunk_id in self.processed_ids:
                    skipped_count += 1
                    continue

                # 准备数据
                batch_texts.append(content)
                batch_ids.append(chunk_id)
                batch_documents.append(content)

                # 构建元数据
                metadata = {
                    **chunk.get("metadata", {}),
                    "source_file": chunk.get("source_file", ""),
                    "content_type": chunk.get("content_type", ""),
                    "text_length": len(content),
                    "added_at": datetime.now().isoformat(),
                    "chunk_id": chunk_id
                }
                batch_metadatas.append(metadata)

                # 收集用于 BM25 索引
                bm25_docs.append(content)
                bm25_ids.append(chunk_id)
                bm25_metas.append(metadata)

            if not batch_texts:
                print(f"批次 {i // batch_size + 1}: 没有有效文本")
                continue

            try:
                print(f"批次 {i // batch_size + 1}: 为 {len(batch_texts)} 个文档生成嵌入...")

                # 生成嵌入向量
                embeddings = self.create_embeddings(batch_texts)

                # 添加到集合
                self.collection.add(
                    embeddings=embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

                # 更新已处理ID集合
                self.processed_ids.update(batch_ids)
                added_count += len(batch_ids)

                print(f"  批次 {i // batch_size + 1}: 成功添加 {len(batch_ids)} 个文档")

            except Exception as e:
                print(f"✗ 批次处理失败: {e}")
                skipped_count += len(batch_texts)
                continue

        # 保存更新后的ID集合
        self._save_processed_ids()

        # 增量更新 BM25 索引
        if self.use_hybrid and bm25_docs:
            self._ensure_hybrid_retriever()
            self._hybrid_retriever.add_documents(
                documents=bm25_docs,
                chunk_ids=bm25_ids,
                metadatas=bm25_metas,
            )
            print(f"✓ BM25 索引已更新（+{len(bm25_docs)} 篇文档）")

        current_total = self.collection.count()
        print(f"✓ 添加完成: {added_count} 新增, {skipped_count} 跳过, 当前总数: {current_total}")

        return {
            "added": added_count,
            "skipped": skipped_count,
            "total": total_chunks,
            "current_total": current_total
        }

    def add_json_file(self, file_path: str, loader) -> Dict:
        """
        添加单个JSON文件到向量库

        Args:
            file_path: JSON文件路径
            loader: 数据加载器实例

        Returns:
            处理结果
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": "文件不存在"}

        try:
            print(f"加载文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)

            # 解析为chunks
            filename = os.path.basename(file_path)
            chunks = loader._parse_file_based_on_type(filename, data, file_path)

            if not chunks:
                return {"success": False, "error": "无法解析文件内容"}

            print(f"解析得到 {len(chunks)} 个chunks")

            # 添加chunks到向量库
            result = self.add_chunks(chunks)

            return {
                "success": True,
                "file": file_path,
                "chunks_parsed": len(chunks),
                **result
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, query: str, n_results: int = 5, filters: Dict = None,
               use_hybrid: Optional[bool] = None,
               dense_weight: Optional[float] = None,
               bm25_weight: Optional[float] = None) -> Dict:
        """
        语义搜索

        Args:
            query: 查询文本
            n_results: 返回结果数量
            filters: 过滤条件（仅 Dense 检索模式支持）
            use_hybrid: 是否启用混合检索（覆盖实例默认值）
            dense_weight: Dense 检索权重（use_hybrid=True时生效）
            bm25_weight: BM25 权重（use_hybrid=True时生效）

        Returns:
            搜索结果
        """
        # 决定使用混合还是纯 Dense
        do_hybrid = self.use_hybrid if use_hybrid is None else use_hybrid

        if do_hybrid:
            return self._search_hybrid(query, n_results, dense_weight, bm25_weight, filters)

        # 原有 Dense-only 逻辑
        query_embedding = self.create_query_embedding(query)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1.0 - distance

                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'similarity': similarity,
                        'distance': distance
                    })

            return {
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results
            }

        except Exception as e:
            print(f"搜索失败: {e}")
            return {
                "query": query,
                "total_results": 0,
                "results": [],
                "error": str(e)
            }

    def _search_hybrid(
        self,
        query: str,
        n_results: int,
        dense_weight: Optional[float],
        bm25_weight: Optional[float],
        filters: Dict = None,
    ) -> Dict:
        """混合检索（Dense HNSW + BM25 + RRF 融合）"""
        if self._hybrid_retriever is None:
            print("混合检索器未初始化，退化为纯 Dense 检索")
            return self.search(query, n_results, filters, use_hybrid=False)

        dw = dense_weight if dense_weight is not None else self.dense_weight
        bw = bm25_weight if bm25_weight is not None else self.bm25_weight

        raw = self._hybrid_retriever.search(
            query=query,
            top_k=n_results,
            dense_weight=dw,
            bm25_weight=bw,
            use_hybrid=True,
        )

        return {
            "query": query,
            "total_results": raw["total_results"],
            "results": raw["results"],
        }

    def _ensure_hybrid_retriever(self):
        """延迟初始化 HybridRetriever（需要 ChromaDB collection 已就绪）"""
        if self._hybrid_retriever is not None:
            return

        self._hybrid_retriever = HybridRetriever(
            chroma_collection=self.collection,
            embed_fn=lambda q: self.create_query_embedding(q),
            persist_dir=self.persist_directory,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )

        # 如果 ChromaDB 中已有数据，构建 BM25 索引
        if self.collection.count() > 0:
            self._build_bm25_from_chroma()

    def _build_bm25_from_chroma(self):
        """从 ChromaDB 中提取所有文档，构建 BM25 索引（用于已有数据迁移）"""
        print("从 ChromaDB 提取数据构建 BM25 索引...")
        all_data = self.collection.get(
            include=["documents", "metadatas"]
        )

        if not all_data.get("ids"):
            return

        docs = all_data["documents"]
        ids = all_data["ids"]
        metas = all_data.get("metadatas", []) or []

        self._hybrid_retriever.build_index(
            documents=docs,
            chunk_ids=ids,
            metadatas=metas,
            incremental=False,
        )
        print(f"✓ BM25 索引构建完成（{len(docs)} 篇文档）")

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        stats = {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "processed_ids_count": len(self.processed_ids),
            "persist_directory": self.persist_directory,
            "metadata": self.collection.metadata,
            "use_hybrid": self.use_hybrid,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
        }
        if self._hybrid_retriever is not None:
            stats["hybrid_stats"] = self._hybrid_retriever.get_stats()
        return stats

    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"集合 '{self.collection_name}' 已删除")

            # 重新创建空集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "model": "BAAI/bge-large-zh-v1.5",
                    "created_at": datetime.now().isoformat(),
                    "reset_at": datetime.now().isoformat()
                }
            )

            # 清空已处理ID集合
            self.processed_ids.clear()
            self._save_processed_ids()

            # 清空混合检索器的 BM25 索引
            if self._hybrid_retriever is not None:
                self._hybrid_retriever._bm25_retriever.clear()
                self._hybrid_retriever._bm25_retriever.remove_persist()
                print("✓ BM25 索引已清空")

            print("✓ 集合已清空并重新创建")
            return True

        except Exception as e:
            print(f"✗ 清空集合失败: {e}")
            return False

    def cleanup(self, remove_persist_dir: bool = False) -> Dict:
        """
        清理向量库

        Args:
            remove_persist_dir: 是否删除持久化目录

        Returns:
            清理结果
        """
        try:
            persist_dir = self.persist_directory

            if os.path.exists(self.processed_ids_file):
                os.remove(self.processed_ids_file)

            # 删除 BM25 索引文件
            bm25_files = [
                os.path.join(persist_dir, "bm25_index.pkl"),
                os.path.join(persist_dir, "bm25_corpus.pkl"),
            ]
            for f in bm25_files:
                if os.path.exists(f):
                    os.remove(f)

            if remove_persist_dir and os.path.exists(persist_dir):
                import shutil
                shutil.rmtree(persist_dir)
                return {
                    "success": True,
                    "message": f"持久化目录已删除: {persist_dir}"
                }
            else:
                return {
                    "success": True,
                    "message": f"处理ID文件和BM25索引已删除，持久化目录保留: {persist_dir}"
                }

        except Exception as e:
            return {"success": False, "error": str(e)}