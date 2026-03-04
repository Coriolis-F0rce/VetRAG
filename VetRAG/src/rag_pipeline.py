# simple_rag_pipeline.py - 简化版RAG管道

import os
import json
from datetime import datetime
from typing import List, Dict, Any
import pickle


class VetRAGPipeline:
    """简化版兽医RAG系统"""

    def __init__(self, data_dir="data", persist_dir="./chroma_db"):
        """
        初始化简化版RAG系统

        Args:
            data_dir: 数据目录
            persist_dir: 向量库持久化目录
        """
        self.data_dir = data_dir
        self.persist_dir = persist_dir

        # 导入模块
        print("导入模块...")
        try:
            from json_loader import VetRAGDataLoader
            from vector_store_chroma_bge import ChromaVectorStore

            self.loader = VetRAGDataLoader()
            self.vector_store = ChromaVectorStore(
                persist_directory=persist_dir,
                model_name="BAAI/bge-large-zh-v1.5"
            )

            print("✓ 模块导入成功")
        except ImportError as e:
            print(f"✗ 模块导入失败: {e}")
            raise

        # 状态文件
        self.state_file = "./rag_state.json"
        self.chunks_cache_file = "./chunks_cache.pkl"

        # 加载状态
        self.state = self._load_state()
        self.chunks_cache = self._load_chunks_cache()

    def _load_state(self) -> Dict:
        """加载系统状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        # 初始状态
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "processed_files": [],
            "total_chunks": 0
        }

    def _save_state(self):
        """保存系统状态"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _load_chunks_cache(self) -> List[Dict]:
        """加载chunks缓存"""
        if os.path.exists(self.chunks_cache_file):
            try:
                with open(self.chunks_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return []

    def _save_chunks_cache(self):
        """保存chunks缓存"""
        try:
            with open(self.chunks_cache_file, 'wb') as f:
                pickle.dump(self.chunks_cache, f)
        except Exception as e:
            print(f"警告: 保存缓存失败: {e}")

    def load_data_files(self, file_names: List[str] = None) -> Dict:
        """
        加载数据文件

        Args:
            file_names: 文件名称列表，None表示加载所有文件

        Returns:
            加载结果统计
        """
        if file_names is None:
            file_names = ["behaviors.json", "breeds.json", "cares.json", "diseases.json", "surgeries.json"]

        file_paths = [os.path.join(self.data_dir, fname) for fname in file_names]

        print("开始加载数据...")

        all_chunks = []
        content_stats = {}

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"✗ 文件不存在: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 解析文件
                filename = os.path.basename(file_path)
                chunks = self.loader._parse_file_based_on_type(filename, data, file_path)

                # 统计内容类型
                for chunk in chunks:
                    content_type = chunk.get("content_type", "unknown")
                    content_stats[content_type] = content_stats.get(content_type, 0) + 1

                all_chunks.extend(chunks)
                print(f"✓ {filename}: {len(chunks)} 个chunks")

            except Exception as e:
                print(f"✗ 加载失败 {file_path}: {e}")

        # 保存到缓存
        self.chunks_cache = all_chunks
        self._save_chunks_cache()

        # 更新状态
        self.state["processed_files"] = file_paths
        self.state["total_chunks"] = len(all_chunks)
        self._save_state()

        print(f"✓ 成功加载 {len(all_chunks)} 个文本块")

        # 显示内容类型分布
        print("\n内容类型分布:")
        for content_type, count in content_stats.items():
            print(f"  {content_type}: {count}个")

        return {
            "total_chunks": len(all_chunks),
            "content_stats": content_stats,
            "chunks": all_chunks
        }

    def build_vector_index(self, chunks: List[Dict] = None) -> Dict:
        """
        构建向量索引

        Args:
            chunks: 文档块列表，None表示使用缓存的chunks

        Returns:
            构建结果统计
        """
        if chunks is None:
            chunks = self.chunks_cache

        if not chunks:
            print("没有可用的chunks，请先加载数据")
            return {"success": False, "error": "没有数据"}

        print("开始向量化...")

        # 添加chunks到向量库
        result = self.vector_store.add_chunks(chunks)

        # 更新状态
        self.state["vector_index_built"] = True
        self.state["vector_index_time"] = datetime.now().isoformat()
        self.state["vector_index_stats"] = result
        self._save_state()

        print(f"✓ 成功向量化 {result}")

        return result

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        查询

        Args:
            question: 查询问题
            top_k: 返回结果数量

        Returns:
            查询结果
        """
        print(f"查询: {question}")

        results = self.vector_store.search(question, n_results=top_k)

        if "error" in results:
            print(f"✗ 查询失败: {results['error']}")
            return results

        print(f"找到 {results['total_results']} 个相关结果:")

        for i, result in enumerate(results["results"]):
            print(f"\n{i + 1}. [相似度: {result['similarity']:.3f}]")

            # 显示来源信息
            metadata = result.get('metadata', {})
            source_file = metadata.get('source_file', '未知')
            content_type = metadata.get('content_type', '未知')
            print(f"   来源: {os.path.basename(source_file)} ({content_type})")

            # 显示内容摘要
            content = result['document']
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   内容: {content}")

        return results

    def add_new_json_file(self, file_path: str) -> Dict:
        """
        添加新的JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            添加结果
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": "文件不存在"}

        print(f"添加新文件: {file_path}")

        # 加载文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 解析为chunks
            filename = os.path.basename(file_path)
            chunks = self.loader._parse_file_based_on_type(filename, data, file_path)

            if not chunks:
                return {"success": False, "error": "无法解析文件内容"}

            print(f"解析得到 {len(chunks)} 个chunks")

            # 添加到向量库
            result = self.vector_store.add_chunks(chunks)

            # 更新缓存和状态
            self.chunks_cache.extend(chunks)
            self._save_chunks_cache()

            if file_path not in self.state["processed_files"]:
                self.state["processed_files"].append(file_path)

            self.state["total_chunks"] += result.get("added", 0)
            self._save_state()

            return {
                "success": True,
                "file": file_path,
                "chunks_parsed": len(chunks),
                **result
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def batch_add_json_files(self, file_paths: List[str]) -> List[Dict]:
        """
        批量添加JSON文件

        Args:
            file_paths: 文件路径列表

        Returns:
            添加结果列表
        """
        results = []
        for file_path in file_paths:
            result = self.add_new_json_file(file_path)
            results.append(result)
            print(f"  {file_path}: {result.get('added', 0)} 个新增")

        return results

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        vector_stats = self.vector_store.get_collection_stats()

        return {
            "system": {
                "version": self.state.get("version", "1.0"),
                "created_at": self.state.get("created_at"),
                "last_updated": self.state.get("last_updated"),
                "total_files": len(self.state.get("processed_files", [])),
                "total_chunks": self.state.get("total_chunks", 0),
                "cached_chunks": len(self.chunks_cache)
            },
            "vector_store": vector_stats
        }

    def cleanup(self, remove_all: bool = False) -> Dict:
        """
        清理系统

        Args:
            remove_all: 是否删除所有数据

        Returns:
            清理结果
        """
        print("开始清理系统...")

        results = []

        # 清理向量库
        if remove_all:
            vector_result = self.vector_store.clear_collection()
            results.append({"action": "clear_vector_store", "success": vector_result})

        # 清理缓存文件
        if os.path.exists(self.chunks_cache_file):
            os.remove(self.chunks_cache_file)
            results.append({"action": "remove_chunks_cache", "success": True})

        # 清理状态文件
        if remove_all and os.path.exists(self.state_file):
            os.remove(self.state_file)
            results.append({"action": "remove_state_file", "success": True})

        # 清理持久化目录
        if remove_all and os.path.exists(self.persist_dir):
            import shutil
            shutil.rmtree(self.persist_dir)
            results.append({"action": "remove_persist_dir", "success": True})

        print("✓ 系统清理完成")

        return {
            "success": True,
            "results": results,
            "removed_all": remove_all
        }


def main():
    """主函数"""
    print("=" * 60)
    print("简化版兽医RAG系统")
    print("=" * 60)

    # 检查依赖
    print("检查依赖...")
    try:
        import chromadb
        print("✓ chromadb")
    except:
        print("✗ chromadb - 请运行: pip install chromadb==0.4.22")
        return

    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers")
    except:
        print("✗ sentence-transformers - 请运行: pip install sentence-transformers")
        return

    try:
        import tqdm
        print("✓ tqdm")
    except:
        print("✗ tqdm - 请运行: pip install tqdm")
        return

    try:
        import numpy as np
        print("✓ numpy")
    except:
        print("✗ numpy - 请运行: pip install numpy")
        return

    # 初始化系统
    print("\n导入模块...")
    try:
        rag = VetRAGPipeline(data_dir="data", persist_dir="./chroma_db")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return

    # 检查是否已有向量索引
    vector_stats = rag.vector_store.get_collection_stats()
    if vector_stats["document_count"] > 0:
        print(f"\n检测到现有向量索引 ({vector_stats['document_count']} 个文档)")
        use_existing = input("是否使用现有索引？(y/n): ").lower()

        if use_existing == 'n':
            # 重新构建
            print("重新构建向量索引...")
            rag.cleanup(remove_all=True)
            data_result = rag.load_data_files()
            vector_result = rag.build_vector_index()
        else:
            # 使用现有索引
            print("使用现有索引...")
            # 只加载数据到缓存（不重建向量）
            rag.load_data_files()
    else:
        # 首次构建
        print("\n首次构建向量索引...")
        data_result = rag.load_data_files()
        vector_result = rag.build_vector_index()

    # 显示系统信息
    print("\n" + "=" * 60)
    print("系统信息:")
    info = rag.get_system_info()
    print(f"  文档总数: {info['system']['total_chunks']}")
    print(f"  向量库文档数: {info['vector_store']['document_count']}")
    print(f"  已处理文件: {info['system']['total_files']} 个")

    # 交互模式
    print("\n" + "=" * 60)
    print("进入交互模式，可用命令:")
    print("  '查询 问题' - 执行查询")
    print("  '添加 文件路径' - 添加新JSON文件")
    print("  '批量添加 目录路径' - 批量添加目录下的JSON文件")
    print("  '状态' - 查看系统状态")
    print("  '清理' - 清理系统")
    print("  '退出' - 退出程序")
    print("=" * 60)

    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("再见！")
            break

        elif user_input.startswith('查询 '):
            question = user_input[3:].strip()
            if question:
                rag.query(question, top_k=3)
            else:
                print("请输入查询问题")

        elif user_input.startswith('添加 '):
            file_path = user_input[3:].strip()
            if os.path.exists(file_path):
                result = rag.add_new_json_file(file_path)
                if result["success"]:
                    print(f"✓ 添加成功: {result.get('added', 0)} 个新增文档")
                else:
                    print(f"✗ 添加失败: {result.get('error', '未知错误')}")
            else:
                print(f"✗ 文件不存在: {file_path}")

        elif user_input.startswith('批量添加 '):
            dir_path = user_input[5:].strip()
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # 查找所有JSON文件
                json_files = []
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))

                if json_files:
                    print(f"找到 {len(json_files)} 个JSON文件")
                    results = rag.batch_add_json_files(json_files)

                    total_added = sum(r.get('added', 0) for r in results if r.get('success', False))
                    print(f"✓ 批量添加完成: 总计新增 {total_added} 个文档")
                else:
                    print("没有找到JSON文件")
            else:
                print(f"✗ 目录不存在: {dir_path}")

        elif user_input == '状态':
            info = rag.get_system_info()
            print("\n系统状态:")
            print(f"  版本: {info['system']['version']}")
            print(f"  创建时间: {info['system']['created_at']}")
            print(f"  最后更新: {info['system']['last_updated']}")
            print(f"  已处理文件数: {info['system']['total_files']}")
            print(f"  总chunks数: {info['system']['total_chunks']}")
            print(f"  向量库文档数: {info['vector_store']['document_count']}")

        elif user_input == '清理':
            print("清理选项:")
            print("  1. 只清理缓存文件")
            print("  2. 完全清理（删除所有数据）")
            choice = input("请选择 (1/2): ").strip()

            if choice == '1':
                result = rag.cleanup(remove_all=False)
                print(f"缓存清理完成")
            elif choice == '2':
                confirm = input("确认完全清理？这将删除所有向量数据和缓存 (y/n): ").lower()
                if confirm == 'y':
                    result = rag.cleanup(remove_all=True)
                    print(f"完全清理完成")
                else:
                    print("取消清理")
            else:
                print("无效选择")

        else:
            print("未知命令，请输入: 查询/添加/批量添加/状态/清理/退出")


if __name__ == "__main__":
    main()