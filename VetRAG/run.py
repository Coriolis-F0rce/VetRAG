import os
import sys
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def cleanup_before_run():
    chroma_dir = os.path.join(project_root, "bge")
    if os.path.exists(chroma_dir):
        print("检测到旧的向量数据库，正在清理...")
        try:
            shutil.rmtree(chroma_dir)
            print("✓ 已清理旧的向量数据库")
        except Exception as e:
            print(f"✗ 清理失败: {e}")
            print("请手动删除 chroma_db 目录后重试")
            return False
    return True


def install_requirements():
    requirements = [
        "chromadb",
        "sentence-transformers",
        "tqdm",
        "numpy"
    ]

    for req in requirements:
        try:
            __import__(req.replace("-", "_"))
            print(f"✓ {req}")
        except ImportError:
            print(f"✗ 缺少 {req}，正在安装...")
            os.system(f"pip install {req}")


def main():
    print("兽医RAG系统")

    install_requirements()

    if not cleanup_before_run():
        return

    try:
        print("\n导入模块...")
        from src.json_loader import VetRAGDataLoader
        from src.vector_store_chroma import ChromaVectorStore
        print("✓ 模块导入成功")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("\n请确保:")
        print("1. 在VetRAG目录下运行此脚本")
        print("2. src目录下有正确的文件")
        return

    data_dir = "data"
    file_paths = [
        os.path.join(data_dir, "behaviors.json"),
        os.path.join(data_dir, "breeds.json"),
        os.path.join(data_dir, "cares.json"),
        os.path.join(data_dir, "diseases.json"),
        os.path.join(data_dir, "surgeries.json")
    ]

    missing_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"\n缺少 {len(missing_files)} 个文件，请确保data目录包含所有JSON文件")
        return

    print("\n" + "=" * 60)
    print("开始加载数据...")
    loader = VetRAGDataLoader()
    chunks = loader.load_all_files(file_paths)
    print(f"✓ 成功加载 {len(chunks)} 个文本块")

    content_stats = {}
    for chunk in chunks:
        content_type = chunk.get("content_type", "unknown")
        content_stats[content_type] = content_stats.get(content_type, 0) + 1

    print("\n内容类型分布:")
    for content_type, count in content_stats.items():
        print(f"  {content_type}: {count}个")

    print("\n" + "=" * 60)
    print("开始向量化...")
    try:
        vector_store = ChromaVectorStore()
        num_added = vector_store.add_chunks(chunks)
        print(f"✓ 成功向量化 {num_added} 个文档")
    except Exception as e:
        print(f"✗ 向量化失败: {e}")
        print("\n可能的原因:")
        print("1. 缺少依赖（确保tqdm已安装）")
        print("2. 内存不足")
        print("3. 模型下载失败")
        return

    print("\n" + "=" * 60)
    print("向量数据库构建完成！")
    print("现在可以开始查询了，输入 'quit' 或 '退出' 结束")
    print("=" * 60)

    while True:
        query = input("\n请输入问题: ").strip()
        if query.lower() in ['quit', '退出', 'exit']:
            break

        if not query:
            continue

        try:
            results = vector_store.search(query, n_results=5)

            if results and 'results' in results and results['results']:
                formatted_results = results['results']
                print(f"\n找到 {len(formatted_results)} 个相关结果:")

                for i, result in enumerate(formatted_results):
                    similarity = result['similarity']
                    doc = result['document']
                    meta = result['metadata']

                    print(f"\n{i + 1}. [相似度: {similarity:.3f}]")
                    print(f"   类型: {meta.get('content_type', 'unknown')}")
                    print(f"   来源: {meta.get('source_file', 'unknown')}")
                    if 'sub_category' in meta:
                        print(f"   子类: {meta.get('sub_category')}")
                    preview = doc[:300] + "..." if len(doc) > 300 else doc
                    print(f"   内容: {preview}")
            else:
                print("未找到相关结果")
        except Exception as e:
            print(f"查询出错: {e}")


if __name__ == "__main__":
    main()